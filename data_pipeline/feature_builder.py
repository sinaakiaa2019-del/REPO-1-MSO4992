"""Predictors: causal market features, topology groups and the forward-loss label."""

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Feature-group column names shared by the improvement-study phases in
# robustness_builder.py / international_builder.py / run_ml.py.
MARKET = ["log_vix", "vol20", "drawdown_60", "yield_curve"]
TOPOLOGY = ["h1_entropy_normalized", "h1_total_per_asset", "w2_per_sqrt_asset"]
CONTROL = ["eigenvalue_ratio", "average_correlation"]
TOPOLOGY_Z = [name + "_z252" for name in TOPOLOGY]
TOPOLOGY_EWS = ["landscape_cv60", "landscape_tau60"]
COVERAGE_ARTEFACTS = ["eligible_assets", "eligible_fraction"]
TARGETS = {"loss10": 0.10, "loss075": 0.075, "loss05": 0.05}  # SPY loss thresholds tested in the improvement study
# The two pre-declared primary comparisons (the declared improvement protocol). Every phase's
# paired_intervals() call reports Bonferroni bounds sized for this family (2 tests),
# not just phase 1's own comparisons, so the correction is comparable everywhere.
PRIMARY_FAMILY = [("combined_z", "market"), ("combined_ews", "market")]


def regime_topology(top):
    """Causal regime-relative topology: today's level vs. its own trailing 252-session history."""
    result = pd.DataFrame(index=top.index)
    for name in TOPOLOGY:
        history = top[name].rolling(252, min_periods=126)
        mean, sd = history.mean().shift(1), history.std().shift(1)
        result[name + "_z252"] = ((top[name] - mean) / sd.replace(0, np.nan)).clip(-10, 10)
    # Early-warning stats on the landscape L1 norm (Gidea & Katz, 2018): coefficient
    # of variation and trend, both over a trailing 60-session window.
    norm = top.landscape1_l1_per_asset
    result["landscape_cv60"] = norm.rolling(60).std() / norm.rolling(60).mean()
    order = np.arange(60)
    result["landscape_tau60"] = norm.rolling(60).apply(lambda w: kendalltau(order, w)[0], raw=True)
    return result


def orthogonalize_topology(df, topology_cols, market_cols, window=60, minimum=40):
    """Trailing-window OLS residual of each topology column against the market columns.

    Causal: the regression at date t only uses the `window` sessions strictly before t.
    What is left over is topology's own signal, with anything a linear market model
    could already explain removed.
    """
    X = df[market_cols].to_numpy()
    Y = df[topology_cols].to_numpy()
    n = len(df)
    resid = np.full_like(Y, np.nan, dtype=float)
    design = np.column_stack([np.ones(n), X])
    for t in range(minimum, n):
        lo = max(0, t - window)
        coef, *_ = np.linalg.lstsq(design[lo:t], Y[lo:t], rcond=None)
        resid[t] = Y[t] - design[t] @ coef
    return pd.DataFrame(resid, index=df.index, columns=[c + "_orth" for c in topology_cols])


def regime_normalize(frame, cols, window=252, minimum=126):
    """Causal trailing z-score for each column: today vs. its own trailing history.

    A column that is constant for long stretches (e.g. many persistence-image pixels
    are exactly zero) gives a zero trailing std; the z-score is undefined there and is
    filled with 0 (no regime signal detected) instead of dropping the day.
    """
    history = frame[cols].rolling(window, min_periods=minimum)
    mean, sd = history.mean().shift(1), history.std().shift(1)
    z = ((frame[cols] - mean) / sd.replace(0, np.nan)).clip(-10, 10).fillna(0.0)
    z.columns = [c + "_z" for c in cols]
    return z


def horizon_label(spy, loss, horizon=20):
    """One if SPY falls at least `loss` below today's close within `horizon` sessions."""
    forward = pd.concat([spy.shift(-i) for i in range(1, horizon + 1)], axis=1)
    label = (forward.min(axis=1) / spy - 1 <= -loss).astype(float)
    label.iloc[-horizon:] = np.nan
    label_end = pd.Series(spy.index, index=spy.index).shift(-horizon)
    return label, label_end


def enhanced_features(spy, vix, coverage):
    """Describe recent stress using information available at the forecast close."""
    result = pd.DataFrame(index=spy.index)
    daily = np.log(spy).diff()
    for window in [5, 20, 60]:
        result[f"momentum_{window}"] = np.log(spy / spy.shift(window))
    for window in [60, 252]:
        result[f"drawdown_{window}"] = spy / spy.rolling(window).max() - 1
    result["downside_vol20"] = daily.clip(upper=0).rolling(20).std() * np.sqrt(252)
    result["vix_change5"] = vix.diff(5)
    result["vix_ratio60"] = vix / vix.rolling(60).mean()
    result["eligible_assets"] = coverage.eligible.reindex(spy.index)
    result["eligible_fraction"] = (coverage.eligible / coverage.expected_members).reindex(spy.index)
    return result


def data():
    """Build the target and all feature groups before choosing shared dates."""
    t = pd.read_pickle(HERE / "topology.pkl")
    markets = pd.read_pickle(ROOT / "cache/markets.pkl")
    spy = markets.SPY.dropna()
    idx = spy.index
    forward = pd.concat([spy.shift(-i) for i in range(1, 21)], axis=1)
    y = (forward.min(axis=1) / spy - 1 <= -0.1).astype(float)
    y.iloc[-20:] = np.nan
    end = pd.Series(idx, index=idx).shift(-20)
    v = pd.read_csv(ROOT / "cache/vix.csv")
    v = pd.Series(v.CLOSE.to_numpy(), index=pd.to_datetime(v.DATE))
    yc = pd.read_csv(ROOT / "cache/t10y2y_real.csv", index_col=0, parse_dates=True).iloc[:, 0]
    base = pd.DataFrame(index=idx)
    base["vix"] = v.reindex(idx).shift(1)
    base["yield_curve"] = pd.to_numeric(yc, errors="coerce").reindex(idx).ffill(limit=5).shift(1)
    base["vol20"] = np.log(spy).diff().rolling(20).std() * np.sqrt(252)
    base["eigenvalue_ratio"] = t.eigenvalue_ratio.reindex(idx)
    coverage = pd.read_csv(OUT / "coverage.csv", index_col=0, parse_dates=True)
    enhanced = enhanced_features(spy, base.vix, coverage)
    # Explicit exclusion list, not "everything else": the locked baseline ladder's
    # topology representation must stay fixed even as topology_analyser adds new,
    # separately-gated research-only summary columns (persistence landscapes,
    # network centrality) to the same underlying topology.pkl frame.
    compact = t.drop(
        columns=[
            "n_assets",
            "eigenvalue_ratio",
            "average_correlation",
            "landscape1_l1_per_asset",
            "landscape1_l2_per_asset",
            "network_clustering",
            "network_algebraic_connectivity",
        ]
    )
    dynamics = pd.DataFrame(index=t.index)
    for c in [
        "h1_entropy_normalized",
        "h1_total_per_asset",
        "h1_integrated_l2_per_asset",
        "w2_per_sqrt_asset",
    ]:
        dynamics[c + "_change5"] = compact[c].diff(5)
        mean = compact[c].rolling(60, min_periods=40).mean().shift(1)
        std = compact[c].rolling(60, min_periods=40).std().shift(1)
        dynamics[c + "_z60"] = ((compact[c] - mean) / std.replace(0, np.nan)).clip(-10, 10)
    with np.load(OUT / "betti_curves.npz") as b:
        full = pd.DataFrame(
            b["betti"][:, :, 1] / t.n_assets.to_numpy()[:, None],
            index=t.index,
            columns=[f"betti_{j:03}" for j in range(200)],
        )
    allx = (
        base.join(compact)
        .join(dynamics)
        .join(full)
        .join(enhanced)
        .join(y.rename("label"))
        .join(end.rename("label_end"))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    groups = {
        "base": list(base.columns),
        "compact": list(compact.columns),
        "combined": list(base.columns) + list(compact.columns),
        "dynamic_combined": list(base.columns) + list(compact.columns) + list(dynamics.columns),
        "full": list(full.columns) + list(compact.columns),
    }
    # eligible_assets/eligible_fraction stay in `enhanced` (and in the saved
    # features_and_labels.csv) as audit columns, but are excluded here: both are
    # near-perfectly correlated with calendar date as the public universe grows,
    # so a model trained on them learns a time trend rather than market information.
    groups["enhanced_base"] = groups["base"] + [
        c for c in enhanced.columns if c not in COVERAGE_ARTEFACTS
    ]
    groups["enhanced_combined"] = groups["enhanced_base"] + groups["compact"]
    allx.to_pickle(HERE / "features.pkl")
    allx.to_csv(OUT / "features_and_labels.csv")
    (OUT / "feature_registry.json").write_text(json.dumps(groups, indent=2))
    return (allx, groups)

"""ML helpers: fair tuning, model fitting, thresholds and alarm metrics."""

from config import ROOT, HERE, OUT
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


def alarm_threshold(y, p, budget=0.05):
    """Use earlier negative days to set a strict greater-than alarm rule."""
    negatives = np.asarray(p)[np.asarray(y) == 0]
    if not len(negatives):
        raise ValueError("Calibration has no negative days")
    return float(np.quantile(negatives, 1 - budget, method="higher"))


def fit(X, y, kind, param):
    """Fit one declared classifier to a training sample."""
    if y.nunique() < 2:
        raise ValueError("Cannot fit a single-class training set")
    if kind == "lr":
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=param, max_iter=1500, solver="lbfgs", random_state=42),
        )
    elif kind == "lasso":
        # L1-penalised logistic regression: automatic feature selection instead of
        # ridge shrinkage, useful when most columns (e.g. persistence-image pixels)
        # are uninformative most of the time.
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=param, penalty="l1", solver="liblinear", max_iter=2000, random_state=42),
        )
    elif kind == "rf":
        # Bagged trees: a different nonlinear family from XGBoost's boosting, to check
        # whether XGBoost's edge on topology is about tree ensembles generally or
        # boosting specifically. min_samples_leaf=10 mirrors XGBoost's min_child_weight=10.
        m = RandomForestClassifier(
            n_estimators=200,
            max_depth=param,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        )
    elif kind == "xgb":
        m = XGBClassifier(
            n_estimators=100,
            max_depth=param,
            learning_rate=0.04,
            reg_lambda=10,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    m.fit(X, y)
    return m


def select(X, y, end, kind):
    """Compare two settings using two earlier, non-overlapping time blocks."""
    candidates = [0.1, 1.0] if kind == "lr" else [2, 3]
    records = []
    edges = np.linspace(int(0.4 * len(X)), len(X), 3).astype(int)
    for param in candidates:
        truth = []
        pred = []
        for a, b in zip(edges[:-1], edges[1:]):
            train = end < X.index[a]
            if y.loc[train].nunique() < 2:
                continue
            assert end.loc[train].max() < X.index[a]
            model = fit(X.loc[train], y.loc[train], kind, param)
            truth.extend(y.iloc[a:b])
            pred.extend(model.predict_proba(X.iloc[a:b])[:, 1])
        score = average_precision_score(truth, pred) if sum(truth) > 0 else float("-inf")
        records.append(dict(parameter=param, inner_ap=score, inner_n=len(truth)))
    return (max(records, key=lambda x: x["inner_ap"])["parameter"], records)


def metrics(y, p, alarm):
    """Calculate ranking and warning metrics from probabilities."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    alarm = np.asarray(alarm, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    starts = alarm & ~np.r_[False, alarm[:-1]]
    episode_false = 0
    for s in np.flatnonzero(starts):
        e = s
        while e < len(alarm) and alarm[e]:
            e += 1
        episode_false += int(y[s:e].sum() == 0)
    return dict(
        n=len(y),
        positives=int(y.sum()),
        prevalence=float(y.mean()),
        auroc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        false_positive_days=int(fp),
        true_positive_days=int(tp),
        false_negative_days=int(fn),
        false_positive_rate=float(fp / max(fp + tn, 1)),
        recall=float(tp / max(tp + fn, 1)),
        precision=float(tp / max(tp + fp, 1)),
        time_under_warning=float(alarm.mean()),
        alarm_episodes=int(starts.sum()),
        false_alarm_episodes=episode_false,
    )


def stratified(predictions):
    """Positive-weighted within-year AUROC/AP: removes between-year score-level effects."""
    rows = []
    for (model, budget), g in predictions.groupby(["model", "budget"]):
        years = [annual for _, annual in g.groupby("year") if annual.label.nunique() == 2]
        weights = [annual.label.sum() for annual in years]
        rows.append(dict(
            model=model, budget=budget,
            within_year_auroc=float(np.average([roc_auc_score(a.label, a.score) for a in years], weights=weights)),
            within_year_ap=float(np.average([average_precision_score(a.label, a.score) for a in years], weights=weights)),
            years_with_both_classes=len(years),
        ))
    return pd.DataFrame(rows)


def paired_intervals(predictions, pairs, primary_family=(), blocks=(20, 60, 120)):
    """Paired moving-block AP-difference intervals (same draws as robustness_builder.paired_comparison).

    `primary_family` optionally flags a subset of `pairs` as a pre-declared primary
    hypothesis family, for which a Bonferroni-corrected interval is also reported.
    """
    pivot = predictions.pivot(index="date", columns="model", values="score").sort_index()
    assert pivot.notna().all().all(), "Comparisons require identical dates"
    labels = (predictions.drop_duplicates("date").set_index("date").label
              .reindex(pivot.index).astype(int).to_numpy())
    alpha = 0.05 / max(len(primary_family), 1)
    rows = []
    for block in blocks:
        block = min(block, len(labels))
        rng = np.random.default_rng(42)
        samples = []
        for _ in range(500):
            starts = rng.integers(0, len(labels) - block + 1, size=int(np.ceil(len(labels) / block)))
            ix = np.concatenate([np.arange(s, s + block) for s in starts])[: len(labels)]
            if 0 < labels[ix].sum() < len(ix):
                samples.append(ix)
        names = sorted({n for pair in pairs for n in pair})
        values = {n: np.array([average_precision_score(labels[ix], pivot[n].to_numpy()[ix]) for ix in samples])
                  for n in names}
        for a, b in pairs:
            delta = values[a] - values[b]
            rows.append(dict(
                model=a, reference=b, block_sessions=block, draws=len(delta),
                ap_difference=average_precision_score(labels, pivot[a]) - average_precision_score(labels, pivot[b]),
                lower95=float(np.quantile(delta, 0.025)), upper95=float(np.quantile(delta, 0.975)),
                primary_family=(a, b) in primary_family,
                lower_bonferroni=float(np.quantile(delta, alpha / 2)),
                upper_bonferroni=float(np.quantile(delta, 1 - alpha / 2)),
                share_draws_favouring_model=float((delta > 0).mean()),
            ))
    return pd.DataFrame(rows)


def benjamini_hochberg(p, alpha=0.05):
    """Standard BH step-up multiple-testing correction. Returns (q_values, rejected_at_alpha)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_raw = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q_raw[::-1])[::-1]  # enforce monotone non-decreasing q
    q_out = np.empty(m)
    q_out[order] = np.clip(q, 0, 1)
    return q_out, q_out <= alpha


def research_train_index(end, boundary, history):
    """Only known labels; optionally retain the most recent observed sessions."""
    indices = np.flatnonzero((end < boundary).to_numpy())
    return indices[-history:] if history else indices


def research_select(df, candidates):
    """Select every modelling choice using two purged earlier blocks only.

    Validation dates are identical across candidates. Single-class training
    folds use their training prevalence, so a rolling window never borrows
    an older crisis or silently drops a difficult validation period. Each
    candidate may declare its own learner family via an optional "kind" key
    (default "lr"), so the same purge/selection machinery serves logistic
    and tree-based candidates alike; every existing caller that never sets
    "kind" is unaffected byte-for-byte.
    """
    edges = np.linspace(int(0.4 * len(df)), len(df), 3).astype(int)
    records = []
    for candidate in candidates:
        truth, scores = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            train = research_train_index(df.label_end, df.index[a], candidate["history"])
            assert len(train) and df.label_end.iloc[train].max() < df.index[a]
            y = df.label.iloc[train].astype(int)
            if y.nunique() < 2:
                p = np.full(b-a, y.mean())
            else:
                model = fit(df.iloc[train][candidate["columns"]], y, candidate.get("kind", "lr"), candidate["C"])
                p = model.predict_proba(df.iloc[a:b][candidate["columns"]])[:, 1]
            truth.extend(df.label.iloc[a:b].astype(int))
            scores.extend(p)
        ap = float(average_precision_score(truth, scores)) if sum(truth) else None
        records.append(dict(**candidate, inner_ap=ap, inner_n=len(truth), inner_positives=sum(truth)))
    # A declared deterministic default handles all-negative inner validation.
    best = max(records, key=lambda r: -1 if r["inner_ap"] is None else r["inner_ap"])
    return best, records


def evaluate_groups(frame, groups, budgets=(0.01, 0.05, 0.10), kinds=None):
    """Equal three-candidate search budgets; save calibration scores and every test day.

    All groups share rows, inner folds and outer years. A single-class fit
    predicts its training prevalence rather than dropping a difficult fold.

    `kinds` is an optional {group_name: "lr"|"xgb"} map; any group absent
    from it defaults to "lr", exactly as before this parameter existed, so
    every original call site (which never passes `kinds`) is byte-for-byte
    unaffected. An "xgb" group gets the same three-candidate discipline as
    an "lr" one, just over max_depth instead of C.
    """
    kinds = kinds or {}
    grid = {'lr': [0.01, 0.1, 1.0], 'xgb': [2, 3, 4], 'lasso': [0.01, 0.1, 1.0], 'rf': [3, 5, 8]}
    columns = sorted({c for names in groups.values() for c in names})
    df = frame[columns + ['label', 'label_end']].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    predictions, calibrations, tuning, selections, splits = [], [], [], [], []
    models = {}
    candidates = [dict(id=f'{name}/C={c}', group=name, columns=names, C=c, history=0, kind=kinds.get(name, 'lr'))
                  for name, names in groups.items() for c in grid[kinds.get(name, 'lr')]]
    for year in range(2007, 2025):
        start = pd.Timestamp(f'{year}-01-01')
        test = (df.index >= start) & (df.index < pd.Timestamp(f'{year+1}-01-01'))
        known = np.flatnonzero((df.label_end < start).to_numpy())
        if not test.any() or len(known) < 525:
            continue
        cal = known[-252:]
        train = np.flatnonzero((df.label_end < df.index[cal[0]]).to_numpy())
        if len(train) < 252:
            continue
        assert df.label_end.iloc[train].max() < df.index[cal[0]]
        assert df.label_end.iloc[cal].max() < df.index[test].min()
        _, records = research_select(df.iloc[train], candidates)
        tuning.extend(dict(year=year, **r) for r in records)
        splits.append(dict(year=year, fit_label_end=df.label_end.iloc[train].max(),
                           calibration_start=df.index[cal[0]], calibration_label_end=df.label_end.iloc[cal].max(),
                           test_start=df.index[test].min(), test_end=df.index[test].max()))
        for name, names in groups.items():
            best = max((r for r in records if r['group'] == name),
                       key=lambda r: -1 if r['inner_ap'] is None else r['inner_ap'])
            y = df.label.iloc[train].astype(int)
            model = fit(df.iloc[train][names], y, kinds.get(name, 'lr'), best['C']) if y.nunique() > 1 else None
            pc = model.predict_proba(df.iloc[cal][names])[:,1] if model else np.full(len(cal), y.mean())
            pt = model.predict_proba(df.loc[test,names])[:,1] if model else np.full(test.sum(), y.mean())
            selections.append(dict(year=year, model=name, C=best['C'], inner_ap=best['inner_ap'],
                                   constant_fallback=model is None))
            calibrations.append(pd.DataFrame(dict(date=df.index[cal],year=year,model=name,
                                                  label=df.label.iloc[cal].to_numpy(),score=pc)))
            thresholds = {}
            for budget in budgets:
                cut = alarm_threshold(df.label.iloc[cal], pc, budget)
                assert (pc[df.label.iloc[cal].to_numpy()==0] > cut).mean() <= budget + 1e-12
                thresholds[budget] = cut
                predictions.append(pd.DataFrame(dict(date=df.index[test],year=year,model=name,budget=budget,
                    label=df.label.loc[test].to_numpy(), score=pt, threshold=cut, alarm=pt>cut)))
            models[year,name] = (model, names, float(y.mean()), thresholds)
        # VIX is an explicit raw-score benchmark, with the same earlier negatives.
        if 'vix' in frame:
            pc = frame.vix.reindex(df.index[cal]).to_numpy()
            pt = frame.vix.reindex(df.index[test]).to_numpy()
            calibrations.append(pd.DataFrame(dict(date=df.index[cal],year=year,model='VIX',label=df.label.iloc[cal].to_numpy(),score=pc)))
            for budget in budgets:
                cut=alarm_threshold(df.label.iloc[cal],pc,budget)
                predictions.append(pd.DataFrame(dict(date=df.index[test],year=year,model='VIX',budget=budget,
                    label=df.label.loc[test].to_numpy(),score=pt,threshold=cut,alarm=pt>cut)))
        print('Equal-budget evaluation',year,len(groups),'groups',flush=True)
    if not predictions:
        raise ValueError('Insufficient history for the declared evaluation')
    pred=pd.concat(predictions,ignore_index=True).sort_values(['model','budget','date'])
    aggregate=pd.DataFrame([dict(model=name,budget=budget,**metrics(g.label,g.score,g.alarm))
                            for (name,budget),g in pred.groupby(['model','budget'])])
    return dict(predictions=pred,calibration=pd.concat(calibrations,ignore_index=True),
                metrics=aggregate,selected=pd.DataFrame(selections),tuning=pd.DataFrame(tuning),
                splits=pd.DataFrame(splits),models=models)

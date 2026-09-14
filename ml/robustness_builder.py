"""Robustness: universe-size controls, normalisation and training-only shrinkage.
Also holds the post-hoc improvement-study phases (2, 3, 5, 6, 7)."""

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd
from model_training import select, fit, alarm_threshold, metrics, evaluate_groups, stratified, paired_intervals, benjamini_hochberg
from matrix_builder import correlation
from scipy.linalg import cho_factor, cho_solve


def drawdown_events(price, loss=.10):
    """One event from a 10% high-water drawdown until recovery of its peak.

    Peaks and recoveries are evaluation annotations, never model predictors.
    Open events at the end of the sample are explicitly censored.
    """
    high=float(price.iloc[0]); peak=price.index[0]; active=None; rows=[]
    for day,value in price.items():
        if active is not None:
            if value >= high:
                active.update(recovery=day,censored=False); rows.append(active);active=None
                high=float(value);peak=day
            continue
        if value >= high:
            high=float(value);peak=day
        elif value/high-1 <= -loss:
            active=dict(event=str(day.date()),peak=peak,onset=day,peak_price=high)
    if active is not None:
        active.update(recovery=price.index[-1],censored=True);rows.append(active)
    return pd.DataFrame(rows)


def event_leads(predictions, events, calendar, group_columns=('model','budget')):
    """Warnings in the 20 observed sessions strictly before event onset."""
    rows=[]
    for key,group in predictions.groupby(list(group_columns)):
        if not isinstance(key,tuple):key=(key,)
        info=dict(zip(group_columns,key)); group=group.set_index('date').sort_index()
        for event in events.itertuples():
            if event.onset < group.index.min() or event.onset > group.index.max():continue
            position=calendar.get_loc(event.onset)
            window=calendar[max(0,position-20):position]
            observed=group.reindex(window)
            alarms=observed[observed.alarm.fillna(False).astype(bool)]
            first=alarms.index.min() if len(alarms) else pd.NaT
            rows.append(dict(**info,event=event.event,onset=event.onset,peak=event.peak,
                fully_observed=int(observed.score.notna().sum())==20,
                first_alarm=first,lead_sessions=position-calendar.get_loc(first) if pd.notna(first) else np.nan,
                alarm_days=len(alarms),censored=event.censored))
    return pd.DataFrame(rows)


def paired_comparison(predictions,pairs,blocks=(20,60,120)):
    """Paired AP intervals under several declared dependence lengths."""
    from sklearn.metrics import average_precision_score
    pivot=predictions.pivot(index='date',columns='model',values='score').sort_index()
    assert pivot.notna().all().all(), 'Comparisons require identical dates'
    labels=predictions.drop_duplicates('date').set_index('date').label.reindex(pivot.index).astype(int).to_numpy()
    rows=[]
    if not labels.sum():return pd.DataFrame()
    for block in blocks:
        block=min(block,len(labels));rng=np.random.default_rng(42)
        samples=[]
        for _ in range(500):
            starts=rng.integers(0,len(labels)-block+1,size=int(np.ceil(len(labels)/block)))
            ix=np.concatenate([np.arange(s,s+block) for s in starts])[:len(labels)]
            if 0<labels[ix].sum()<len(ix):samples.append(ix)
        names=sorted({n for p in pairs for n in p})
        values={n:np.array([average_precision_score(labels[ix],pivot[n].to_numpy()[ix]) for ix in samples]) for n in names}
        for a,b in pairs:
            delta=values[a]-values[b]
            rows.append(dict(model=a,reference=b,block_sessions=block,draws=len(delta),
                ap_difference=average_precision_score(labels,pivot[a])-average_precision_score(labels,pivot[b]),
                lower95=float(np.quantile(delta,.025)),upper95=float(np.quantile(delta,.975))))
    return pd.DataFrame(rows)


def run_defence():
    """Run the predeclared balanced comparisons and complete the RQ evidence."""
    from international_builder import transfer_controls
    df=pd.read_pickle(HERE/'features.pkl').copy()
    top=pd.read_pickle(HERE/'topology.pkl')
    df['log_vix']=np.log(df.vix)
    df['average_correlation']=top.average_correlation.reindex(df.index)
    market=['log_vix','vol20','drawdown_60','yield_curve']
    topology=['h1_entropy_normalized','h1_total_per_asset','w2_per_sqrt_asset']
    control=['eigenvalue_ratio','average_correlation']
    groups=dict(market=market,topology=topology,combined=market+topology,
                market_correlation=market+control,combined_correlation=market+control+topology)
    for j in [80,90,100,110]:groups[f'scale_{j:03}']=market+[f'betti_{j:03}']
    result=evaluate_groups(df,groups)
    models=result.pop('models')
    for name,frame in result.items():
        if 'columns' in frame:frame['columns']=frame['columns'].map(json.dumps)
        frame.to_csv(OUT/f'defence_{name}.csv',index=False)
    primary=result['predictions'][result['predictions'].budget==.05]
    comparisons=paired_comparison(primary,[('combined','market'),('combined_correlation','market_correlation'),
        ('combined','VIX'),('topology','VIX')]+[(f'scale_{j:03}','market') for j in [80,90,100,110]])
    comparisons.to_csv(OUT/'defence_comparisons.csv',index=False)
    spy=pd.read_pickle(HERE/'markets.pkl').SPY.dropna()
    events=drawdown_events(spy)
    events.to_csv(OUT/'defence_events.csv',index=False)
    event_leads(result['predictions'],events,spy.index).to_csv(OUT/'defence_event_leads.csv',index=False)
    selected=[]
    for year,tuning in result['tuning'].groupby('year'):
        scales=tuning[tuning.group.str.startswith('scale_')]
        best=scales.iloc[scales.inner_ap.fillna(-1).to_numpy().argmax()]
        j=int(best.group.split('_')[1]);eps=np.linspace(0,2,200)[j]
        selected.append(dict(year=year,scale_index=j,epsilon=eps,shrunk_correlation_threshold=1-eps**2/2,
                             C=best.C,inner_ap=best.inner_ap))
    pd.DataFrame(selected).to_csv(OUT/'defence_scale_choices.csv',index=False)
    transfer=transfer_controls(models)
    rows=[]
    for universe,g in transfer.groupby('universe'):
        comp=paired_comparison(g,[('frozen_US_topology','target_volatility'),('frozen_US_topology','target_momentum')])
        comp['universe']=universe;rows.append(comp)
    pd.concat(rows).to_csv(OUT/'defence_transfer_comparisons.csv',index=False)
    sectors=pd.read_csv(OUT/'rq5_predictions.csv',parse_dates=['date'])
    event_leads(sectors,events,spy.index,('symbol','sector_proxy','model','budget')).to_csv(OUT/'rq5_event_leads.csv',index=False)
    sector_comp=[]
    for symbol,g in sectors.groupby('symbol'):
        comp=paired_comparison(g,[('sector_combined','sector_market'),('sector_topology','sector_market')],blocks=(60,))
        comp['symbol']=symbol;sector_comp.append(comp)
    pd.concat(sector_comp).to_csv(OUT/'rq5_comparisons.csv',index=False)
    (OUT/'defence_protocol.json').write_text(json.dumps(dict(
        status='Retrospective development, no untouched holdout',groups=groups,C=[.01,.1,1.0],
        budgets=[.01,.05,.10],calibration_sessions=252,inner_blocks=2,test_years=[2007,2024],
        event_rule='10% SPY high-water drawdown to peak recovery; 20-session warning window',
        sector_rule='Nine pre-2000 sector funds; 60-return window, 5-delay embedding, scalar within-window standardisation',
        limitations='Public delisting/identity coverage incomplete; ETF proxy scope differs from firm-sector RQ5; no multiple-search correction'),indent=2))
    verify_defence()
    from report_builder import make_defence_report
    make_defence_report()


def verify_defence():
    """Recompute saved metrics, calibration thresholds and training selections."""
    from model_training import metrics,alarm_threshold
    pred=pd.read_csv(OUT/'defence_predictions.csv',parse_dates=['date'])
    saved=pd.read_csv(OUT/'defence_metrics.csv').set_index(['model','budget'])
    cal=pd.read_csv(OUT/'defence_calibration.csv')
    reference=None
    for (name,budget),g in pred.groupby(['model','budget']):
        dates=g.sort_values('date').date.tolist()
        if reference is None:reference=dates
        assert dates==reference and not g.date.duplicated().any()
        for k,v in metrics(g.label,g.score,g.alarm).items():
            assert v is None or abs(saved.loc[(name,budget),k]-v)<1e-10,(name,k)
        for year,annual in g.groupby('year'):
            c=cal[(cal.model==name)&(cal.year==year)]
            cut=alarm_threshold(c.label,c.score,budget)
            assert np.allclose(annual.threshold,cut,atol=1e-12)
            assert ((annual.score>annual.threshold)==annual.alarm).all()
    splits=pd.read_csv(OUT/'defence_splits.csv')
    assert (splits.fit_label_end<splits.calibration_start).all()
    assert (splits.calibration_label_end<splits.test_start).all()
    tuning=pd.read_csv(OUT/'defence_tuning.csv');selected=pd.read_csv(OUT/'defence_selected.csv')
    assert len(tuning)==18*9*3 and len(selected)==18*9
    assert set(tuning.year)==set(range(2007,2025))
    for row in selected.itertuples():
        candidates=tuning[(tuning.year==row.year)&(tuning.group==row.model)]
        best=candidates.iloc[candidates.inner_ap.fillna(-1).to_numpy().argmax()]
        assert row.C==best.C
    (OUT/'defence_verification.json').write_text(json.dumps(dict(
        metrics_recalculated=True,shared_test_dates=True,calibration_recalculated=True,
        purged_splits=True,equal_candidate_budgets=True,complete_18_years=True),indent=2))
    print('DEFENCE VERIFICATION PASSED',flush=True)


def research_inputs():
    """Build the predeclared TDA grid and a shared, causal feature panel."""
    from topology_analyser import research_topology, checksum
    import hashlib

    destination = HERE / "research"
    destination.mkdir(exist_ok=True)
    specs = [
        dict(name="short", window=40, **{"lambda": 0.94}, shrink=0.1),
        dict(name="long", window=120, **{"lambda": 0.98}, shrink=0.1),
        dict(name="shrink", window=60, **{"lambda": 0.94}, shrink=0.2),
        dict(
            name="quality", window=60, **{"lambda": 0.94}, shrink=0.1, mask_extreme=True
        ),
    ]
    inputs = {
        str(p.relative_to(ROOT)): checksum(p)
        for p in [
            HERE / "returns.pkl",
            HERE / "membership_fja.csv",
            HERE / "features.pkl",
            *sorted((ROOT / "topology_analyser").glob("*.py")),
            ROOT / "matrix_builder.py",
        ]
    }
    frames = {"original": pd.read_pickle(HERE / "features.pkl")}
    for spec in specs:
        key = hashlib.sha256(
            json.dumps([spec, inputs], sort_keys=True).encode()
        ).hexdigest()[:20]
        target = destination / key
        target.mkdir(exist_ok=True)
        table = target / "features.pkl"
        if not table.exists():
            frame = research_topology(spec, target)
            frame.to_pickle(table)
        frames[spec["name"]] = pd.read_pickle(table)
    df = frames.pop("original").copy()
    compact = ["h1_entropy_normalized", "h1_total_per_asset", "w2_per_sqrt_asset"]
    landscape = ["landscape1_l1_per_asset", "landscape1_l2_per_asset"]
    network = ["network_clustering", "network_algebraic_connectivity"]
    # The locked baseline's "compact" topology group deliberately excludes these
    # research-only columns (see feature_builder.py), so features.pkl (the source
    # of df.copy() below) does not carry them for the default configuration; read
    # them from the unfiltered per-window summary instead. The other four
    # configurations' frames come straight from research_topology() and already
    # carry every summary column, landscape/network included.
    default_raw = pd.read_pickle(HERE / "topology.pkl")
    topology = {}
    topology_landscape = {}
    topology_network = {}
    for name, frame in [("default", df.copy()), *frames.items()]:
        columns = []
        for col in compact + [f"betti_{j:03}" for j in [80, 90, 100, 110]]:
            new = name + "_" + col
            df[new] = frame[col].reindex(df.index)
            columns.append(new)
        topology[name] = columns
        extra_source = default_raw if name == "default" else frame
        lcols = []
        for col in landscape:
            new = name + "_" + col
            df[new] = extra_source[col].reindex(df.index)
            lcols.append(new)
        topology_landscape[name] = lcols
        ncols = []
        for col in network:
            new = name + "_" + col
            df[new] = extra_source[col].reindex(df.index)
            ncols.append(new)
        topology_network[name] = ncols
    # Positive VIX values permit a stable, causal nonlinear transform.
    df["log_vix"] = np.log(df.vix)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.to_pickle(destination / "features.pkl")
    coverage = []
    for name, frame in [
        ("default", pd.read_pickle(HERE / "topology.pkl")),
        *frames.items(),
    ]:
        coverage.append(
            dict(
                configuration=name,
                daily_windows=len(frame),
                minimum_assets=int(frame.n_assets.min()),
                mean_assets=float(frame.n_assets.mean()),
                maximum_assets=int(frame.n_assets.max()),
                first_date=str(frame.index.min().date()),
                last_date=str(frame.index.max().date()),
            )
        )
    pd.DataFrame(coverage).to_csv(destination / "tda_coverage.csv", index=False)
    return df, topology, topology_landscape, topology_network, specs, inputs


def research_comparisons(predictions):
    """Paired blocks quantify sampling uncertainty, not research-selection bias."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(42)
    pivot = predictions.pivot(
        index="date", columns="model", values="score"
    ).sort_index()
    y = (
        predictions.drop_duplicates("date")
        .set_index("date")
        .label.reindex(pivot.index)
        .to_numpy()
    )
    pairs = [(name, "VIX") for name in pivot if name != "VIX"]
    pairs += [
        ("selected_combined", "selected_market"),
        ("selected_quality", "selected_combined"),
        ("selected_topology", "original_LR_topology"),
        ("selected_quality_topology", "selected_topology"),
        ("selected_market", "original_LR_base"),
        ("selected_landscape", "selected_topology"),
        ("selected_network", "selected_topology"),
        ("selected_ensemble", "selected_market"),
        ("selected_ensemble", "selected_topology"),
    ]
    rows = []
    samples = []
    for _ in range(500):
        starts = rng.integers(0, len(y) - 60 + 1, size=int(np.ceil(len(y) / 60)))
        sample = np.concatenate([np.arange(s, s + 60) for s in starts])[: len(y)]
        if y[sample].sum():
            samples.append(sample)
    aps = {
        name: [
            average_precision_score(y[ix], pivot[name].to_numpy()[ix]) for ix in samples
        ]
        for name in pivot
    }
    for a, b in pairs:
        differences = np.asarray(aps[a]) - np.asarray(aps[b])
        rows.append(
            dict(
                model=a,
                reference=b,
                ap_difference=average_precision_score(y, pivot[a])
                - average_precision_score(y, pivot[b]),
                lower_95=np.quantile(differences, 0.025),
                upper_95=np.quantile(differences, 0.975),
                draws=len(samples),
                block_sessions=60,
            )
        )
    return pd.DataFrame(rows)


def run_sensitivity():
    """Run the declared sensitivity analyses and save their results."""
    df = pd.read_pickle(HERE / "features.pkl")
    groups = json.loads((OUT / "feature_registry.json").read_text())
    top = pd.read_pickle(HERE / "topology.pkl")
    df["universe_size"] = top.n_assets.reindex(df.index)
    compact = groups["compact"]
    raw = []
    for name in compact:
        col = "unnormalized_" + name
        raw.append(col)
        if "per_asset" in name:
            df[col] = df[name] * df.universe_size
        elif name == "w2_per_sqrt_asset":
            df[col] = df[name] * np.sqrt(df.universe_size)
        elif name == "h1_entropy_normalized":
            df[col] = df[name] * np.log(
                np.maximum(df.h1_count_per_asset * df.universe_size, 2)
            )
        else:
            df[col] = df[name]
    specs = {
        "LR_base_size_control": groups["base"] + ["universe_size"],
        "LR_combined_size_control": groups["combined"] + ["universe_size"],
        "LR_unnormalized_topology": raw,
    }
    predictions = []
    rows = []
    y = df.label.astype(int)
    end = df.label_end
    for year in range(2007, 2025):
        start = pd.Timestamp(f"{year}-01-01")
        test = (df.index >= start) & (df.index < pd.Timestamp(f"{year + 1}-01-01"))
        known = np.flatnonzero((end < start).to_numpy())
        cal = known[-252:]
        train = end < df.index[cal[0]]
        for name, cols in specs.items():
            param, _ = select(df.loc[train, cols], y.loc[train], end.loc[train], "lr")
            m = fit(df.loc[train, cols], y.loc[train], "lr", param)
            pc = m.predict_proba(df.iloc[cal][cols])[:, 1]
            p = m.predict_proba(df.loc[test, cols])[:, 1]
            cut = alarm_threshold(y.iloc[cal], pc)
            alarm = p > cut
            rows.append(dict(year=year, model=name, **metrics(y.loc[test], p, alarm)))
            predictions.append(
                pd.DataFrame(
                    dict(
                        date=df.index[test],
                        year=year,
                        model=name,
                        label=y.loc[test].values,
                        score=p,
                        threshold=cut,
                        alarm=alarm,
                    )
                )
            )
        print("Sensitivity year", year, flush=True)
    p = pd.concat(predictions)
    p.to_csv(OUT / "sensitivity_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_annual.csv", index=False)
    pd.DataFrame(
        [
            dict(model=name, **metrics(g.label, g.score, g.alarm))
            for name, g in p.groupby("model")
        ]
    ).to_csv(OUT / "sensitivity_aggregate.csv", index=False)
    returns = pd.read_pickle(HERE / "returns.pkl")
    a = returns.to_numpy()
    records = []
    membership = (
        pd.read_csv(ROOT / "cache/membership_fja.csv", parse_dates=["date"])
        .set_index("date")
        .tickers.sort_index()
        .reindex(returns.index, method="ffill")
    )
    for t in range(60, len(returns) - 1, 63):
        if returns.index[t + 1] >= pd.Timestamp("2007-01-01"):
            break
        block = a[t - 59 : t + 1]
        eligible = (
            np.isfinite(block).all(axis=0)
            & np.isfinite(a[t + 1])
            & (np.nanstd(block, axis=0) > 1e-10)
        )
        members = {s.replace(".", "-") for s in membership.iloc[t].split(",")}
        eligible &= np.array([s.replace(".", "-") in members for s in returns.columns])
        x = block[:, eligible]
        if x.shape[1] < 3:
            continue
        weights = 0.94 ** np.arange(59, -1, -1)
        weights /= weights.sum()
        mu = weights @ x
        sd = np.sqrt(weights @ (x - mu) ** 2)
        future = (a[t + 1, eligible] - mu) / sd
        for shrink in [0.01, 0.1, 0.2]:
            c = correlation(x, shrink=shrink)
            ch = cho_factor(c, lower=True)
            score = (
                2 * np.log(np.diag(ch[0])).sum() + future @ cho_solve(ch, future)
            ) / len(future)
            records.append(
                dict(
                    date=str(returns.index[t].date()),
                    shrinkage=shrink,
                    assets=len(future),
                    next_session_gaussian_score_per_asset=score,
                    role="training-only numerical sensitivity, lower is better; does not retune locked primary estimator",
                )
            )
    pd.DataFrame(records).to_csv(
        OUT / "training_only_covariance_sensitivity.csv", index=False
    )
    print("SENSITIVITY COMPLETE", flush=True)


# ---------------------------------------------------------------------------
# Improvement study: post-hoc phases run after the primary defence release.
# Every phase writes only to results/improved and is declared in docs/ or in
# this file's own docstrings before it ran. Negative findings are kept and
# reported the same way as positive ones -- that is this project's own rule.
# ---------------------------------------------------------------------------


def load_panel(connection):
    """The shared feature panel plus regime-relative topology (today vs. trailing 252 sessions)."""
    from feature_builder import TOPOLOGY_Z, TOPOLOGY_EWS, regime_topology

    df = pd.read_sql_query("SELECT * FROM features_and_labels", connection)
    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["label_end"] = pd.to_datetime(df["label_end"])
    top = pd.read_sql_query("SELECT * FROM topology_features", connection, parse_dates=["date"])
    top = top.set_index("date").sort_index()
    df["log_vix"] = np.log(df.vix)
    df["average_correlation"] = top.average_correlation.reindex(df.index)
    regime = regime_topology(top)
    df = df.join(regime)
    assert not df[TOPOLOGY_Z + TOPOLOGY_EWS].isna().any().any()
    return df, regime


def feature_registry(connection):
    """Return the documented feature groups stored in the evidence database."""
    row = connection.execute("SELECT content FROM documents WHERE name='feature_registry.json'").fetchone()
    if row is None:
        raise ValueError("feature_registry.json is missing from the evidence database")
    return json.loads(row[0])


def summarise_events(leads, columns):
    """Event recall and median lead time, grouped by the given columns."""
    observed = leads[leads.fully_observed.astype(bool)]
    summary = (
        observed.groupby(list(columns))
        .agg(events=("event", "size"), detected=("lead_sessions", "count"),
             median_lead_when_detected=("lead_sessions", "median"))
        .reset_index()
    )
    summary["event_recall"] = summary.detected / summary.events
    return summary


def run_improvement_study():
    """Phase 1: regime-relative topology, TDA early-warning stats, and a coverage-bug fix.

    Declared in the declared improvement protocol before it ran. Reproduces every published
    defence_* score to machine precision first, then tests: (a) z-scoring the three
    topology summaries against their own trailing 252-session history instead of using
    raw levels, (b) two landscape-based early-warning statistics (coefficient of
    variation and trend, Gidea & Katz 2018), and (c) a bug fix to the classic ladder --
    "eligible_assets"/"eligible_fraction" track calendar date almost exactly, so those
    models were partly learning "what year is it."
    """
    import sqlite3
    import platform
    import time
    from importlib.metadata import version
    from feature_builder import MARKET, TOPOLOGY, CONTROL, TOPOLOGY_Z, TOPOLOGY_EWS, TARGETS, PRIMARY_FAMILY, horizon_label
    from run_ml import classic_correction
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    groups = dict(
        market=MARKET, market_correlation=MARKET + CONTROL, topology=TOPOLOGY,
        combined=MARKET + TOPOLOGY, combined_correlation=MARKET + CONTROL + TOPOLOGY,
        topology_z=TOPOLOGY_Z, combined_z=MARKET + TOPOLOGY_Z,
        combined_z_correlation=MARKET + CONTROL + TOPOLOGY_Z,
        topology_ews=TOPOLOGY_EWS, combined_ews=MARKET + TOPOLOGY_EWS,
        combined_ews_correlation=MARKET + CONTROL + TOPOLOGY_EWS,
    )
    primary_family = PRIMARY_FAMILY
    comparisons_wanted = primary_family + [
        ("combined", "market"), ("combined_z_correlation", "market_correlation"),
        ("combined_ews_correlation", "market_correlation"), ("combined_correlation", "market_correlation"),
        ("combined_z", "VIX"), ("combined_ews", "VIX"), ("combined", "VIX"), ("market", "VIX"),
        ("topology_z", "topology"), ("topology_ews", "topology"),
    ]

    def run_target(panel, spy, events, target, loss):
        """Evaluate one loss threshold using the declared annual protocol."""
        frame = panel.copy()
        label = horizon_label(spy, loss, 20)[0].reindex(frame.index)
        if target == "loss10":
            assert np.array_equal(label.to_numpy(), frame.label.to_numpy()), "Primary label changed"
        frame["label"] = label
        result = evaluate_groups(frame, groups)
        result.pop("models")
        predictions = result["predictions"]
        reference = None
        for _, g in predictions.groupby(["model", "budget"]):
            if reference is None:
                reference = g.date.tolist()
            assert g.date.tolist() == reference, "Models must share test dates"
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        scores["ap_lift"] = scores.average_precision / scores.prevalence
        primary = predictions[predictions.budget == 0.05]
        comparisons = paired_intervals(primary, comparisons_wanted, primary_family=primary_family)
        # Cross-check against this file's own paired_comparison(), used by the published
        # defence report -- the two must agree on every shared field.
        check = paired_comparison(primary, primary_family).merge(
            comparisons, on=["model", "reference", "block_sessions"], suffixes=("_check", "")
        )
        for field in ["ap_difference", "lower95", "upper95"]:
            assert np.allclose(check[field], check[field + "_check"], atol=1e-12)
        leads = event_leads(predictions, events, spy.index)
        tuning = result["tuning"]
        tuning["columns"] = tuning["columns"].map(json.dumps)
        tables = dict(metrics=scores, comparisons=comparisons, event_leads=leads, predictions=predictions,
                      calibration=result["calibration"], tuning=tuning, selected=result["selected"], splits=result["splits"])
        for table in tables.values():
            table.insert(0, "target", target)
        tables["event_summary"] = summarise_events(leads, ("target", "model", "budget"))
        print("TARGET COMPLETE", target, flush=True)
        return tables

    started = time.time()
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        panel, regime = load_panel(connection)
        registry = feature_registry(connection)
        published = pd.read_sql_query(
            "SELECT date, model, budget, score, alarm FROM defence_predictions", connection, parse_dates=["date"]
        )
        classic_published = pd.read_sql_query(
            "SELECT date, model, score, alarm FROM predictions", connection, parse_dates=["date"]
        )
    connection.close()
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    events = drawdown_events(spy)
    outputs = [run_target(panel, spy, events, target, loss) for target, loss in TARGETS.items()]
    tables = {name: pd.concat([o[name] for o in outputs], ignore_index=True) for name in outputs[0]}

    # The primary (loss10) predictions must be byte-identical to the published release.
    primary = tables["predictions"][tables["predictions"].target == "loss10"]
    joined = primary.merge(published, on=["date", "model", "budget"], suffixes=("", "_published"))
    expected = published[published.model.isin(primary.model.unique())]
    assert len(joined) == len(expected)
    baseline_difference = float((joined.score - joined.score_published).abs().max())
    assert baseline_difference < 1e-10, "Adding groups changed a published baseline"

    classic_predictions, classic, reproduction = classic_correction(panel, registry, classic_published)
    database = destination / "improved_evidence.sqlite"
    database.unlink(missing_ok=True)
    detail = dict(
        predictions=tables["predictions"], calibration=tables["calibration"], tuning=tables["tuning"],
        selected=tables["selected"], splits=tables["splits"], event_leads=tables["event_leads"],
        classic_predictions=classic_predictions, regime_topology_features=regime.reset_index(),
    )
    with sqlite3.connect(database) as connection:
        for name, frame in detail.items():
            frame.to_sql(name, connection, index=False, if_exists="replace")
    connection.close()
    for name in ["metrics", "comparisons", "event_summary"]:
        tables[name].to_csv(destination / f"{name}.csv", index=False)
    classic.to_csv(destination / "classic_correction.csv", index=False)

    shown = ["model", "auroc", "average_precision", "ap_lift", "within_year_auroc", "within_year_ap",
             "false_positive_rate", "recall", "false_alarm_episodes"]
    text = ["# Improvement study -- generated tables", "",
            "Generated by robustness_builder.run_improvement_study(). See the declared improvement protocol.", ""]
    for target, loss in TARGETS.items():
        m = tables["metrics"]
        m = m[(m.target == target) & (m.budget == 0.05)].sort_values("average_precision", ascending=False)
        c = tables["comparisons"]
        c = c[(c.target == target) & (c.block_sessions == 60)]
        text += [f"## Target: SPY loss of at least {loss:.1%} within 20 sessions", "",
                 f"Test sessions {int(m.n.iloc[0]):,}; positive days {int(m.positives.iloc[0])}; prevalence {m.prevalence.iloc[0]:.4f}.", "",
                 md(m[shown]), "", "Paired AP differences, 60-session blocks (Bonferroni bounds apply to the primary family only):", "",
                 md(c[["model", "reference", "ap_difference", "lower95", "upper95", "lower_bonferroni",
                       "upper_bonferroni", "share_draws_favouring_model", "primary_family"]]), ""]
        e = tables["event_summary"]
        e = e[(e.target == target) & (e.budget == 0.05)]
        text += ["Mechanical drawdown events, 5% calibration budget:", "",
                 md(e[["model", "events", "detected", "median_lead_when_detected", "event_recall"]]), ""]
    text += ["## Classic ladder with coverage time trends removed", "",
             md(classic[["model", "coverage_trend_variables", "auroc", "average_precision", "within_year_auroc",
                         "false_positive_rate", "recall", "false_alarm_episodes"]]), ""]
    (destination / "SUMMARY_TABLES.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    (destination / "protocol.json").write_text(json.dumps(dict(
        status="Retrospective development declared in the declared improvement protocol before execution",
        groups=groups, primary_family=primary_family, comparisons=comparisons_wanted, targets=TARGETS,
        z_score=dict(window=252, minimum=126, excludes_current=True, clip=10),
        early_warning=dict(series="landscape1_l1_per_asset", window=60, statistics=["coefficient_of_variation", "kendall_tau_trend"]),
        classic_correction=dict(removed=["eligible_assets", "eligible_fraction"]),
        bootstrap=dict(blocks=[20, 60, 120], draws=500, seed=42),
    ), indent=2))
    (destination / "verification.json").write_text(json.dumps(dict(
        primary_label_identical_to_published=True, new_features_missing_on_shared_dates=0,
        shared_test_dates_every_target_and_model=True, published_baseline_max_score_difference=baseline_difference,
        paired_intervals_match_robustness_builder=True, calibration_budgets_asserted_by_evaluate_groups=True,
        classic_reproduction=reproduction, python=platform.python_version(),
        packages={n: version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]},
        seconds=round(time.time() - started, 1),
    ), indent=2))
    print("IMPROVEMENT STUDY COMPLETE", round(time.time() - started, 1), "seconds", flush=True)


def run_orthogonal_addendum():
    """Phase 2 (post-hoc): is topology's own signal separate from market, or a proxy for it?

    Added after inspecting phase 1's results, so this is explicitly not part of the
    pre-declared plan. Each topology feature is replaced by its residual after a
    trailing 60-session regression on the market group -- what's left over once a
    linear market model's explanation is removed. Runs its own evaluate_groups() call,
    so phase 1's verified reproduction of the published baseline is untouched.
    """
    import sqlite3
    from feature_builder import MARKET, TOPOLOGY, TARGETS, PRIMARY_FAMILY, horizon_label, orthogonalize_topology
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)

    def load():
        """Load and align the inputs needed for this extension."""
        with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
            df = pd.read_sql_query("SELECT * FROM features_and_labels", connection)
            top = pd.read_sql_query("SELECT * FROM topology_features", connection, parse_dates=["date"])
        connection.close()
        df = df.rename(columns={"Unnamed: 0": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["label_end"] = pd.to_datetime(df["label_end"])
        top = top.set_index("date").sort_index()
        df["log_vix"] = np.log(df.vix)
        df["average_correlation"] = top.average_correlation.reindex(df.index)
        orth = orthogonalize_topology(df, TOPOLOGY, MARKET)
        return df.join(orth), orth

    def run_target(df, spy, events, target, loss):
        """Evaluate one target definition with identical test dates."""
        frame = df.dropna(subset=[c + "_orth" for c in TOPOLOGY]).copy()
        label = horizon_label(spy, loss, 20)[0].reindex(frame.index)
        if target == "loss10":
            assert np.array_equal(label.dropna().to_numpy(), frame.label.reindex(label.dropna().index).to_numpy())
        frame["label"] = label
        orth_cols = [c + "_orth" for c in TOPOLOGY]
        groups = dict(market=MARKET, topology=TOPOLOGY, combined=MARKET + TOPOLOGY,
                      topology_orthogonal=orth_cols, combined_orthogonal=MARKET + orth_cols)
        result = evaluate_groups(frame, groups)
        result.pop("models")
        predictions = result["predictions"]
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        primary = predictions[predictions.budget == 0.05]
        pairs = [("combined_orthogonal", "market"), ("topology_orthogonal", "topology"), ("combined_orthogonal", "combined")]
        comparisons = paired_intervals(primary, pairs, primary_family=PRIMARY_FAMILY)
        leads = event_leads(predictions, events, spy.index)
        for table in [scores, comparisons, leads]:
            table.insert(0, "target", target)
        print("ORTHOGONAL ADDENDUM TARGET COMPLETE", target, flush=True)
        return dict(metrics=scores, comparisons=comparisons, event_leads=leads, predictions=primary)

    df, orth = load()
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    events = drawdown_events(spy)
    outputs = [run_target(df, spy, events, target, loss) for target, loss in TARGETS.items()]
    tables = {name: pd.concat([o[name] for o in outputs], ignore_index=True) for name in outputs[0]}

    # Sanity check: refitting market/topology/combined on this addendum's slightly
    # shorter shared-date panel should reproduce phase 1's numbers closely.
    phase1 = pd.read_csv(destination / "metrics.csv")
    phase1 = phase1[(phase1.budget == 0.05) & phase1.model.isin(["market", "topology", "combined"])]
    here = tables["metrics"][(tables["metrics"].budget == 0.05) & tables["metrics"].model.isin(["market", "topology", "combined"])]
    check = here.merge(phase1, on=["target", "model", "budget"], suffixes=("_addendum", "_phase1"))
    check["auroc_difference"] = check.auroc_addendum - check.auroc_phase1
    check["ap_difference"] = check.average_precision_addendum - check.average_precision_phase1
    reproduction_note = (
        "Exact match to phase 1 for market/topology/combined."
        if np.allclose(check.auroc_difference, 0, atol=1e-9) and np.allclose(check.ap_difference, 0, atol=1e-9)
        else "Small differences from phase 1 (see orthogonal_reproduction_check.csv): this addendum's shared "
        "dates start ~40 sessions later because the orthogonalization regression needs that much prior "
        "history, which can shift the earliest purged split boundaries."
    )
    check.to_csv(destination / "orthogonal_reproduction_check.csv", index=False)

    # Multiple-testing correction: Benjamini-Hochberg FDR across phase 1's and this
    # addendum's 60-session-block comparisons -- the full family so far.
    phase1_comp = pd.read_csv(destination / "comparisons.csv")
    phase1_comp = phase1_comp[phase1_comp.block_sessions == 60].copy()
    phase1_comp["source"] = "phase1_improvement_study"
    addendum_comp = tables["comparisons"][tables["comparisons"].block_sessions == 60].copy()
    addendum_comp["source"] = "phase2_orthogonal_addendum"
    family = pd.concat([phase1_comp, addendum_comp], ignore_index=True)
    family["p_approx"] = (2 * np.minimum(family.share_draws_favouring_model, 1 - family.share_draws_favouring_model)).clip(1.0 / 501, 1.0)
    family["q_value_bh"], family["significant_fdr_5pct"] = benjamini_hochberg(family.p_approx.to_numpy())
    family = family.sort_values("p_approx")
    family.to_csv(destination / "full_family_fdr_correction.csv", index=False)

    for name in ["metrics", "comparisons", "event_leads"]:
        tables[name].to_csv(destination / f"orthogonal_addendum_{name}.csv", index=False)

    n_sig = int(family.significant_fdr_5pct.sum())
    lines = [
        "# Orthogonal-topology addendum (post-hoc, not pre-declared)",
        "",
        "Added after inspecting phase 1's results, so this is retrospective, not part of the declared improvement protocol's "
        "pre-declared protocol. It never touches phase 1's evaluate_groups() call, so phase 1's byte-identical "
        "reproduction of the published baseline is unaffected.",
        "", f"Reproduction check: {reproduction_note}", "",
        "## What orthogonalization tests", "",
        "Each of the three topology features is replaced by its residual after a trailing 60-session linear "
        "regression on the market group, refit at every date using only the 60 prior sessions -- no future "
        "information. The residual is close to uncorrelated with market by construction, so "
        "`combined_orthogonal - market` is a cleaner test of genuinely separate information than the raw "
        "`combined - market` comparison, where a real but small effect can be masked by collinearity.",
        "", "## Results", "",
    ]
    for target, loss in TARGETS.items():
        m = tables["metrics"][(tables["metrics"].target == target) & (tables["metrics"].budget == 0.05)]
        c = tables["comparisons"][tables["comparisons"].target == target]
        lines += [f"### Target: SPY loss of at least {loss:.1%} within 20 sessions", "",
                  md(m[["model", "auroc", "average_precision", "within_year_auroc", "within_year_ap"]]), "",
                  md(c[["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]), ""]
    lines += [
        "## Multiple-testing correction across the whole improvement study", "",
        f"Benjamini-Hochberg FDR control at 5%, applied jointly to every 60-session-block comparison in phase 1 "
        f"and this addendum ({len(family)} tests). {n_sig} survive; see full_family_fdr_correction.csv for every "
        "p-value, q-value and pass/fail flag. Approximate two-sided p-values come from the share of the 500 "
        "paired bootstrap draws on each side of zero.", "",
        md(family[family.significant_fdr_5pct][["model", "reference", "block_sessions", "ap_difference", "p_approx", "q_value_bh", "source"]])
        if n_sig else "No comparison in the full family survives 5% FDR control.", "",
    ]
    (destination / "ORTHOGONAL_ADDENDUM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ORTHOGONAL ADDENDUM COMPLETE", flush=True)


def run_xgb_extension():
    """Phase 3 (post-hoc): does a nonlinear learner extract more from topology?

    Motivated by the published classic ladder, where XGB_topology already clearly
    beats LR_topology on the same three columns -- but never under this study's
    rigorous purged/calibrated protocol until now. `market_xgb` is a control: if
    XGBoost helps market just as much as topology, the gain is "XGBoost is a better
    learner here," not "topology specifically benefits from nonlinearity."
    """
    import sqlite3
    from feature_builder import MARKET, TOPOLOGY, TARGETS, PRIMARY_FAMILY, horizon_label
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    groups = dict(market=MARKET, topology=TOPOLOGY, combined=MARKET + TOPOLOGY,
                  market_xgb=MARKET, topology_xgb=TOPOLOGY, combined_xgb=MARKET + TOPOLOGY)
    kinds = dict(market_xgb="xgb", topology_xgb="xgb", combined_xgb="xgb")
    comparisons_wanted = [
        ("topology_xgb", "topology"), ("market_xgb", "market"),
        ("combined_xgb", "market_xgb"), ("combined_xgb", "combined"),
    ]

    def run_target(df, spy, events, target, loss):
        """Evaluate one target definition with identical test dates."""
        frame = df.copy()
        label = horizon_label(spy, loss, 20)[0].reindex(frame.index)
        if target == "loss10":
            assert np.array_equal(label.to_numpy(), frame.label.to_numpy())
        frame["label"] = label
        result = evaluate_groups(frame, groups, kinds=kinds)
        result.pop("models")
        predictions = result["predictions"]
        reference = None
        for _, g in predictions.groupby(["model", "budget"]):
            if reference is None:
                reference = g.date.tolist()
            assert g.date.tolist() == reference, "Models must share test dates"
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        primary = predictions[predictions.budget == 0.05]
        comparisons = paired_intervals(primary, comparisons_wanted, primary_family=PRIMARY_FAMILY)
        leads = event_leads(predictions, events, spy.index)
        tuning = result["tuning"]
        tuning["columns"] = tuning["columns"].map(json.dumps)
        for table in [scores, comparisons, leads, tuning, result["selected"]]:
            table.insert(0, "target", target)
        print("XGB EXTENSION TARGET COMPLETE", target, flush=True)
        return dict(metrics=scores, comparisons=comparisons, event_leads=leads, tuning=tuning, selected=result["selected"])

    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        df, _ = load_panel(connection)
    connection.close()
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    events = drawdown_events(spy)
    outputs = [run_target(df, spy, events, target, loss) for target, loss in TARGETS.items()]
    tables = {name: pd.concat([o[name] for o in outputs], ignore_index=True) for name in outputs[0]}
    for name in ["metrics", "comparisons", "event_leads", "tuning", "selected"]:
        tables[name].to_csv(destination / f"xgb_extension_{name}.csv", index=False)

    phase1 = pd.read_csv(destination / "comparisons.csv")
    phase1 = phase1[phase1.block_sessions == 60].copy(); phase1["source"] = "phase1_improvement_study"
    phase2 = pd.read_csv(destination / "orthogonal_addendum_comparisons.csv")
    phase2 = phase2[phase2.block_sessions == 60].copy(); phase2["source"] = "phase2_orthogonal_addendum"
    phase3 = tables["comparisons"][tables["comparisons"].block_sessions == 60].copy()
    phase3["source"] = "phase3_xgb_extension"
    family = pd.concat([phase1, phase2, phase3], ignore_index=True)
    family["p_approx"] = (2 * np.minimum(family.share_draws_favouring_model, 1 - family.share_draws_favouring_model)).clip(1.0 / 501, 1.0)
    family["q_value_bh"], family["significant_fdr_5pct"] = benjamini_hochberg(family.p_approx.to_numpy())
    family = family.sort_values("p_approx")
    family.to_csv(destination / "full_family_fdr_correction.csv", index=False)

    n_sig = int(family.significant_fdr_5pct.sum())
    lines = [
        "# XGBoost topology extension (post-hoc, phase 3)", "",
        "Added after phases 1-2, motivated by the published classic ladder (XGB_topology already beats "
        "LR_topology there). This runs that comparison under the study's rigorous purged protocol, with "
        "its own evaluate_groups() call so phase 1's baseline reproduction stays untouched. `market_xgb` "
        "is a control.", "", "## Results", "",
    ]
    for target, loss in TARGETS.items():
        m = tables["metrics"][(tables["metrics"].target == target) & (tables["metrics"].budget == 0.05)]
        c = tables["comparisons"][tables["comparisons"].target == target]
        lines += [f"### Target: SPY loss of at least {loss:.1%} within 20 sessions", "",
                  md(m.sort_values("average_precision", ascending=False)[
                      ["model", "auroc", "average_precision", "within_year_auroc", "within_year_ap", "false_alarm_episodes"]]),
                  "", md(c[["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]), ""]
    lines += [
        "## Multiple-testing correction across all three phases", "",
        f"Benjamini-Hochberg 5% FDR control across phases 1-3's 60-session-block comparisons ({len(family)} total). "
        f"{n_sig} survive.", "",
        md(family[family.significant_fdr_5pct][["model", "reference", "ap_difference", "p_approx", "q_value_bh", "source"]])
        if n_sig else "No comparison in the full three-phase family survives 5% FDR control.", "",
    ]
    (destination / "XGB_EXTENSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("XGB EXTENSION COMPLETE", flush=True)


def run_persistence_image_extension():
    """Phase 5 (post-hoc): full persistence images (400 pixels/day) instead of 3 summaries.

    Adams et al. (2017) motivate persistence images precisely because reducing a
    diagram to a few scalars can discard information. Same crash-day target
    throughout -- only the topology representation changes. Needs
    cache/tables/persistence_images.npz, produced by topology_analyser.run_topology()
    (the `data` and `tda` stages of run_all.py).
    """
    import sqlite3
    from topology_analyser import load_persistence_images
    from feature_builder import MARKET, TOPOLOGY, PRIMARY_FAMILY
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    pi = load_persistence_images()
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        df = pd.read_sql_query("SELECT * FROM features_and_labels", connection)
    connection.close()
    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date").sort_index()
    df["label_end"] = pd.to_datetime(df["label_end"])
    df["log_vix"] = np.log(df.vix)

    pi_cols = list(pi.columns)
    df = df.join(pi, how="inner")
    missing = df[pi_cols].isna().sum().sum()
    assert missing == 0, f"{missing} missing persistence-image values on shared dates"
    print("Shared dates with persistence images:", len(df), "of", len(pi), "PI rows", flush=True)

    groups = dict(market=MARKET, topology=TOPOLOGY, combined=MARKET + TOPOLOGY,
                  topology_pi_xgb=pi_cols, combined_pi_xgb=MARKET + pi_cols)
    kinds = dict(topology_pi_xgb="xgb", combined_pi_xgb="xgb")
    result = evaluate_groups(df, groups, kinds=kinds)
    result.pop("models")
    predictions = result["predictions"]
    reference = None
    for _, g in predictions.groupby(["model", "budget"]):
        if reference is None:
            reference = g.date.tolist()
        assert g.date.tolist() == reference, "Models must share test dates"
    scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
    primary = predictions[predictions.budget == 0.05]
    comparisons = paired_intervals(primary, [
        ("topology_pi_xgb", "topology"), ("topology_pi_xgb", "market"),
        ("combined_pi_xgb", "market"), ("combined_pi_xgb", "combined"),
    ], primary_family=PRIMARY_FAMILY)
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    events = drawdown_events(spy)
    leads = event_leads(predictions, events, spy.index)

    scores.to_csv(destination / "persistence_image_metrics.csv", index=False)
    comparisons.to_csv(destination / "persistence_image_comparisons.csv", index=False)
    leads.to_csv(destination / "persistence_image_event_leads.csv", index=False)

    primary_scores = scores[scores.budget == 0.05].sort_values("average_precision", ascending=False)
    primary_events = leads[leads.budget == 0.05][leads.model.isin(groups)]
    event_summary = (primary_events[primary_events.fully_observed]
                      .groupby("model").agg(events=("event", "size"), detected=("lead_sessions", "count"),
                                             median_lead=("lead_sessions", "median")))
    lines = [
        "# Full persistence images as the topology representation (post-hoc, phase 5)", "",
        "Motivated by Adams et al. (2017): persistence images keep more diagram information than the three "
        "hand-picked summaries the primary study uses. Phase 3 already showed XGBoost extracts real value "
        "from the compact set; this asks whether the full 400-value image does more, on the same crash-day "
        "target.", "", "## Primary result (5% calibration budget, 10% loss target)", "",
        md(primary_scores[["model", "auroc", "average_precision", "within_year_auroc", "within_year_ap", "false_alarm_episodes"]]),
        "", "## Paired AP differences (60-session blocks)", "",
        md(comparisons[comparisons.block_sessions == 60][["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]),
        "", "## Mechanical drawdown event recall (5% budget)", "",
        md(event_summary.reset_index()), "",
    ]
    (destination / "PERSISTENCE_IMAGE_EXTENSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(primary_scores.to_string(index=False), flush=True)
    print("PERSISTENCE IMAGE EXTENSION COMPLETE", flush=True)


def run_regime_persistence_60day():
    """Phase 6 (post-hoc): regime-normalized persistence images at a 60-session horizon.

    Combines the three things that each showed something on their own: persistence
    images (phase 5) beat the compact topology set; regime-normalization (phase 1's
    topology_z) beat raw topology at secondary thresholds; and this study has never
    tested a horizon longer than the locked 20-session primary target. The 20-session
    protocol itself is untouched -- this is a disclosed exploration at 60 sessions,
    evaluated the same purged way, reported completely either way.
    """
    import sqlite3
    from topology_analyser import load_persistence_images
    from feature_builder import MARKET, TOPOLOGY, PRIMARY_FAMILY, horizon_label, regime_normalize
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    horizon = 60
    targets = {"loss10": 0.10, "loss075": 0.075, "loss05": 0.05}

    pi = load_persistence_images()
    pi_cols = list(pi.columns)
    regime = regime_normalize(pi, pi_cols)
    pi = pi.join(regime)

    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        df = pd.read_sql_query("SELECT * FROM features_and_labels", connection)
    connection.close()
    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date").sort_index()
    df["log_vix"] = np.log(df.vix)
    df = df[MARKET + TOPOLOGY].join(pi, how="inner")
    missing = df.isna().sum().sum()
    assert missing == 0, f"{missing} missing values on shared dates"
    print("Shared dates:", len(df), flush=True)

    def run_target(df, spy, events, target, loss):
        """Evaluate one target definition with identical test dates."""
        frame = df.copy()
        label, label_end = horizon_label(spy, loss, horizon)
        frame["label"] = label.reindex(frame.index)
        frame["label_end"] = label_end.reindex(frame.index)
        frame = frame.dropna(subset=["label", "label_end"])
        pi_raw = [c for c in df.columns if c.startswith("pi_") and not c.endswith("_z")]
        pi_regime = [c for c in df.columns if c.endswith("_z")]
        groups = dict(market=MARKET, combined=MARKET + TOPOLOGY, topology_pi_xgb=pi_raw,
                      topology_pi_xgb_regime=pi_regime, combined_pi_xgb_regime=MARKET + pi_regime)
        kinds = dict(topology_pi_xgb="xgb", topology_pi_xgb_regime="xgb", combined_pi_xgb_regime="xgb")
        result = evaluate_groups(frame, groups, kinds=kinds)
        result.pop("models")
        predictions = result["predictions"]
        reference = None
        for _, g in predictions.groupby(["model", "budget"]):
            if reference is None:
                reference = g.date.tolist()
            assert g.date.tolist() == reference, "Models must share test dates"
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        primary = predictions[predictions.budget == 0.05]
        comparisons = paired_intervals(primary, [
            ("topology_pi_xgb_regime", "topology_pi_xgb"),   # does regime-normalization help, on its own?
            ("combined_pi_xgb_regime", "market"),              # the actual RQ3 question, at this horizon
            ("combined_pi_xgb_regime", "combined"),
        ], primary_family=PRIMARY_FAMILY)
        leads = event_leads(predictions, events, spy.index)
        for table in [scores, comparisons, leads]:
            table.insert(0, "target", target)
        print("REGIME-PI-60D TARGET COMPLETE", target, flush=True)
        return dict(metrics=scores, comparisons=comparisons, event_leads=leads)

    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    events = drawdown_events(spy)
    outputs = [run_target(df, spy, events, t, loss) for t, loss in targets.items()]
    tables = {k: pd.concat([o[k] for o in outputs], ignore_index=True) for k in outputs[0]}
    for k in tables:
        tables[k].to_csv(destination / f"regime_pi_60d_{k}.csv", index=False)

    family = tables["comparisons"][tables["comparisons"].block_sessions == 60].copy()
    family["source"] = "phase6_regime_pi_60day"
    family["p_approx"] = (2 * np.minimum(family.share_draws_favouring_model, 1 - family.share_draws_favouring_model)).clip(1 / 501, 1.0)
    family["q_value_bh"], family["significant_fdr_5pct"] = benjamini_hochberg(family.p_approx.to_numpy())

    lines = [
        "# Regime-normalized persistence images at a 60-session horizon (post-hoc, phase 6)", "",
        "Combines persistence images (phase 5), regime-normalization (phase 1), and a horizon longer than "
        "the locked 20-session primary target -- tested together here for the first time. The 20-session "
        "protocol is untouched; this is a disclosed, additional exploration at 60 sessions.", "",
    ]
    for target, loss in targets.items():
        m = tables["metrics"][(tables["metrics"].target == target) & (tables["metrics"].budget == 0.05)]
        c = tables["comparisons"][(tables["comparisons"].target == target) & (tables["comparisons"].block_sessions == 60)]
        lines += [f"## Target: SPY loss of at least {loss:.1%} within {horizon} sessions", "",
                  md(m.sort_values("average_precision", ascending=False)[
                      ["model", "n", "positives", "auroc", "average_precision", "recall", "false_alarm_episodes"]]),
                  "", md(c[["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]), ""]
    n_sig = int(family.significant_fdr_5pct.sum())
    lines += ["## Multiple-testing check (this phase's own 9 tests, 60-session blocks)", "",
              f"{n_sig} of {len(family)} survive 5% FDR control.", "",
              md(family[family.significant_fdr_5pct][["target", "model", "reference", "ap_difference", "q_value_bh"]]) if n_sig else "None survive.", ""]
    (destination / "REGIME_PI_60DAY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    family.to_csv(destination / "regime_pi_60d_fdr.csv", index=False)
    print(tables["metrics"][tables["metrics"].budget == 0.05].to_string(index=False), flush=True)
    print("REGIME-PI-60D COMPLETE", flush=True)


def run_alternative_learners():
    """Phase 7 (post-hoc): Random Forest and Lasso alongside the existing LR/XGBoost results.

    Two questions, both about *why* XGBoost helped topology in phases 3 and 5, not
    about chasing a better score: (a) is the gain about tree ensembles generally, or
    boosting specifically -- Random Forest answers this; (b) do the sparse 400-pixel
    persistence images just need feature selection rather than a nonlinear model --
    L1 (Lasso) logistic regression answers this, since most pixels are exactly zero
    most days (phase 6's finding). market_rf/market_lasso are controls, same logic as
    phase 3's market_xgb. Same purged/calibrated protocol, all three loss thresholds,
    reported completely either way.
    """
    import sqlite3
    from topology_analyser import load_persistence_images
    from feature_builder import MARKET, TOPOLOGY, TARGETS, PRIMARY_FAMILY, horizon_label
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    groups = dict(
        market=MARKET, topology=TOPOLOGY, combined=MARKET + TOPOLOGY,
        market_rf=MARKET, topology_rf=TOPOLOGY, combined_rf=MARKET + TOPOLOGY,
        market_lasso=MARKET, topology_lasso=TOPOLOGY, combined_lasso=MARKET + TOPOLOGY,
    )
    kinds = dict(
        market_rf="rf", topology_rf="rf", combined_rf="rf",
        market_lasso="lasso", topology_lasso="lasso", combined_lasso="lasso",
    )
    comparisons_wanted = [
        ("topology_rf", "topology"), ("market_rf", "market"), ("combined_rf", "market_rf"),
        ("topology_lasso", "topology"), ("market_lasso", "market"), ("combined_lasso", "market_lasso"),
    ]

    def run_target(df, pi, pi_cols, spy, events, target, loss):
        """Evaluate persistence-image features for one target definition."""
        frame = df.copy()
        label = horizon_label(spy, loss, 20)[0].reindex(frame.index)
        if target == "loss10":
            assert np.array_equal(label.to_numpy(), frame.label.to_numpy())
        frame["label"] = label

        # Compact topology set: LR/RF/Lasso side by side.
        result = evaluate_groups(frame, groups, kinds=kinds)
        result.pop("models")
        predictions = result["predictions"]
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        primary = predictions[predictions.budget == 0.05]
        comparisons = paired_intervals(primary, comparisons_wanted, primary_family=PRIMARY_FAMILY)
        leads = event_leads(predictions, events, spy.index)

        # Persistence images: RF/Lasso only (XGBoost's own run is phase 5, reused for reference).
        pi_frame = frame[["label", "label_end"] + MARKET + TOPOLOGY].join(pi, how="inner")
        pi_groups = dict(
            market=MARKET, combined=MARKET + TOPOLOGY, topology_pi_rf=pi_cols, combined_pi_rf=MARKET + pi_cols,
            topology_pi_lasso=pi_cols, combined_pi_lasso=MARKET + pi_cols,
        )
        pi_kinds = dict(topology_pi_rf="rf", combined_pi_rf="rf", topology_pi_lasso="lasso", combined_pi_lasso="lasso")
        pi_result = evaluate_groups(pi_frame, pi_groups, kinds=pi_kinds)
        pi_result.pop("models")
        pi_predictions = pi_result["predictions"]
        pi_scores = pi_result["metrics"].merge(stratified(pi_predictions), on=["model", "budget"])
        pi_primary = pi_predictions[pi_predictions.budget == 0.05]
        pi_comparisons = paired_intervals(pi_primary, [
            ("topology_pi_rf", "market"), ("combined_pi_rf", "market"), ("combined_pi_rf", "combined"),
            ("topology_pi_lasso", "market"), ("combined_pi_lasso", "market"), ("combined_pi_lasso", "combined"),
            ("topology_pi_lasso", "topology_pi_rf"),
        ], primary_family=PRIMARY_FAMILY)
        pi_leads = event_leads(pi_predictions, events, spy.index)

        scores = pd.concat([scores, pi_scores[~pi_scores.model.isin(scores.model)]], ignore_index=True)
        comparisons = pd.concat([comparisons, pi_comparisons], ignore_index=True)
        leads = pd.concat([leads, pi_leads[~pi_leads.model.isin(leads.model.unique())]], ignore_index=True)
        for table in [scores, comparisons, leads]:
            table.insert(0, "target", target)
        print("ALTERNATIVE LEARNERS TARGET COMPLETE", target, flush=True)
        return dict(metrics=scores, comparisons=comparisons, event_leads=leads)

    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        df, _ = load_panel(connection)
    connection.close()
    pi = load_persistence_images()
    pi_cols = list(pi.columns)
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    events = drawdown_events(spy)
    outputs = [run_target(df, pi, pi_cols, spy, events, target, loss) for target, loss in TARGETS.items()]
    tables = {name: pd.concat([o[name] for o in outputs], ignore_index=True) for name in outputs[0]}
    for name in ["metrics", "comparisons", "event_leads"]:
        tables[name].to_csv(destination / f"alternative_learners_{name}.csv", index=False)

    phase1 = pd.read_csv(destination / "comparisons.csv")
    phase1 = phase1[phase1.block_sessions == 60].copy(); phase1["source"] = "phase1_improvement_study"
    phase2 = pd.read_csv(destination / "orthogonal_addendum_comparisons.csv")
    phase2 = phase2[phase2.block_sessions == 60].copy(); phase2["source"] = "phase2_orthogonal_addendum"
    phase3 = pd.read_csv(destination / "xgb_extension_comparisons.csv")
    phase3 = phase3[phase3.block_sessions == 60].copy(); phase3["source"] = "phase3_xgb_extension"
    phase4 = pd.read_csv(destination / "transfer_trained_comparisons.csv")
    phase4 = phase4[phase4.block_sessions == 60].copy(); phase4["source"] = "phase4_transfer_trained"
    phase7 = tables["comparisons"][tables["comparisons"].block_sessions == 60].copy()
    phase7["source"] = "phase7_alternative_learners"
    family = pd.concat([phase1, phase2, phase3, phase4, phase7], ignore_index=True)
    family["p_approx"] = (2 * np.minimum(family.share_draws_favouring_model, 1 - family.share_draws_favouring_model)).clip(1.0 / 501, 1.0)
    family["q_value_bh"], family["significant_fdr_5pct"] = benjamini_hochberg(family.p_approx.to_numpy())
    family = family.sort_values("p_approx")
    family.to_csv(destination / "full_family_fdr_correction.csv", index=False)

    # Phase 5's own topology_pi_xgb/combined_pi_xgb, for reference only (different
    # evaluate_groups() call, so not paired against these results -- descriptive only).
    phase5 = pd.read_csv(destination / "persistence_image_metrics.csv")
    phase5 = phase5[phase5.model.isin(["topology_pi_xgb", "combined_pi_xgb"]) & (phase5.budget == 0.05)]

    n_sig = int(family.significant_fdr_5pct.sum())
    lines = [
        "# Random Forest and Lasso: a third and fourth learner family (post-hoc, phase 7)",
        "",
        "Motivated by phases 3 and 5, where XGBoost beat linear regression on topology but still "
        "trailed market overall: is that gain about tree ensembles generally (Random Forest, a "
        "bagging family distinct from XGBoost's boosting), or about feature selection on sparse "
        "columns (Lasso, since most persistence-image pixels are exactly zero most days -- phase "
        "6's finding)? `market_rf`/`market_lasso` are controls, same logic as phase 3's "
        "`market_xgb`. Same purged three-candidate protocol as every other phase.",
        "",
        "## Reference: phase 5's XGBoost result on persistence images (different run, not paired here)",
        "",
        md(phase5[["model", "auroc", "average_precision", "false_alarm_episodes"]]),
        "",
        "## Results",
        "",
    ]
    for target, loss in TARGETS.items():
        m = tables["metrics"][(tables["metrics"].target == target) & (tables["metrics"].budget == 0.05)]
        c = tables["comparisons"][tables["comparisons"].target == target]
        lines += [f"### Target: SPY loss of at least {loss:.1%} within 20 sessions", "",
                  md(m.sort_values("average_precision", ascending=False)[
                      ["model", "auroc", "average_precision", "within_year_auroc", "within_year_ap", "false_alarm_episodes"]]),
                  "", md(c[["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]), ""]
    lines += [
        "## Multiple-testing correction across phases 1-4 and 7", "",
        f"Benjamini-Hochberg 5% FDR control across these phases' 60-session-block comparisons "
        f"({len(family)} tests total; phases 5 and 6 run their own separate corrections). {n_sig} survive.", "",
        md(family[family.significant_fdr_5pct][["model", "reference", "ap_difference", "p_approx", "q_value_bh", "source"]])
        if n_sig else "No comparison in this family survives 5% FDR control.", "",
    ]
    (destination / "ALTERNATIVE_LEARNERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ALTERNATIVE LEARNERS COMPLETE", flush=True)


def run_delay_embedding_topology():
    """Phase 8 (post-hoc): delay-embedded SPY dynamics as a topology source.

    Instead of a cross-sectional correlation network across hundreds of stocks,
    this embeds SPY's own return series (60-session window, 5-lag delay embedding,
    the same construction already used for the RQ5 sector proxies) and runs
    persistent homology on that single series. Tests whether a cleaner, lower-noise
    topology input beats the full-panel network. Same purged protocol, all three
    thresholds, folded into the running multiple-testing family.
    """
    import sqlite3
    from sector_builder import delay_summary
    from feature_builder import MARKET, TOPOLOGY, TARGETS, PRIMARY_FAMILY, horizon_label
    from report_builder import md

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()
    r = np.log(spy).diff()
    values = r.to_numpy()
    rows = []
    for t in range(60, len(r)):
        total, entropy = delay_summary(values[t - 59:t + 1])
        rows.append(dict(date=r.index[t], delay_total=total, delay_entropy=entropy))
    delay = pd.DataFrame(rows).set_index("date")

    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        df, _ = load_panel(connection)
    connection.close()
    df = df.join(delay, how="inner")
    missing = df[["delay_total", "delay_entropy"]].isna().sum().sum()
    assert missing == 0, f"{missing} missing delay values on shared dates"

    groups = dict(market=MARKET, topology=TOPOLOGY, combined=MARKET + TOPOLOGY,
                  delay_topology=["delay_total", "delay_entropy"],
                  combined_delay=MARKET + ["delay_total", "delay_entropy"])
    comparisons_wanted = [
        ("delay_topology", "topology"), ("delay_topology", "market"),
        ("combined_delay", "market"), ("combined_delay", "combined"),
    ]

    def run_target(frame, spy, events, target, loss):
        """Evaluate delay-embedding features for one target definition."""
        frame = frame.copy()
        label = horizon_label(spy, loss, 20)[0].reindex(frame.index)
        if target == "loss10":
            assert np.array_equal(label.to_numpy(), frame.label.to_numpy())
        frame["label"] = label
        result = evaluate_groups(frame, groups)
        result.pop("models")
        predictions = result["predictions"]
        reference = None
        for _, g in predictions.groupby(["model", "budget"]):
            if reference is None:
                reference = g.date.tolist()
            assert g.date.tolist() == reference, "Models must share test dates"
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        primary = predictions[predictions.budget == 0.05]
        comparisons = paired_intervals(primary, comparisons_wanted, primary_family=PRIMARY_FAMILY)
        leads = event_leads(predictions, events, spy.index)
        for table in [scores, comparisons, leads]:
            table.insert(0, "target", target)
        print("DELAY EMBEDDING TARGET COMPLETE", target, flush=True)
        return dict(metrics=scores, comparisons=comparisons, event_leads=leads)

    events = drawdown_events(spy)
    outputs = [run_target(df, spy, events, target, loss) for target, loss in TARGETS.items()]
    tables = {name: pd.concat([o[name] for o in outputs], ignore_index=True) for name in outputs[0]}
    for name in ["metrics", "comparisons", "event_leads"]:
        tables[name].to_csv(destination / f"delay_embedding_{name}.csv", index=False)

    prior = pd.read_csv(destination / "full_family_fdr_correction.csv")
    prior = prior[prior.source != "phase8_delay_embedding"]
    phase8 = tables["comparisons"][tables["comparisons"].block_sessions == 60].copy()
    phase8["source"] = "phase8_delay_embedding"
    family = pd.concat([prior, phase8], ignore_index=True)
    family["p_approx"] = (2 * np.minimum(family.share_draws_favouring_model, 1 - family.share_draws_favouring_model)).clip(1.0 / 501, 1.0)
    family["q_value_bh"], family["significant_fdr_5pct"] = benjamini_hochberg(family.p_approx.to_numpy())
    family = family.sort_values("p_approx")
    family.to_csv(destination / "full_family_fdr_correction.csv", index=False)

    n_sig = int(family.significant_fdr_5pct.sum())
    lines = [
        "# Delay-embedded market dynamics as a topology source (post-hoc, phase 8)", "",
        "Instead of a cross-sectional correlation network across hundreds of stocks, this embeds "
        "SPY's own return series (same 60-session/5-lag construction already used for the RQ5 "
        "sector proxies) and runs persistent homology on that single series.", "",
        "## Results", "",
    ]
    for target, loss in TARGETS.items():
        m = tables["metrics"][(tables["metrics"].target == target) & (tables["metrics"].budget == 0.05)]
        c = tables["comparisons"][tables["comparisons"].target == target]
        lines += [f"### Target: SPY loss of at least {loss:.1%} within 20 sessions", "",
                  md(m.sort_values("average_precision", ascending=False)[
                      ["model", "auroc", "average_precision", "within_year_auroc", "false_alarm_episodes"]]),
                  "", md(c[["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]), ""]
    lines += [
        "## Multiple-testing correction across phases 1-4, 7 and 8", "",
        f"Benjamini-Hochberg 5% FDR control across these phases' 60-session-block comparisons "
        f"({len(family)} total). {n_sig} survive.", "",
        md(family[family.significant_fdr_5pct][["model", "reference", "ap_difference", "p_approx", "q_value_bh", "source"]])
        if n_sig else "No comparison in this family survives 5% FDR control.", "",
    ]
    (destination / "DELAY_EMBEDDING_TOPOLOGY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("DELAY EMBEDDING TOPOLOGY COMPLETE", flush=True)


def run_improvements():
    """Run every improvement-study phase in sequence, then the dissertation figures.

    Replaces running the 8 old standalone phase scripts by hand. Each phase writes
    only to results/improved and is independent of the others' evaluate_groups()
    calls, so an earlier phase's verified reproduction is never put at risk by a
    later one.
    """
    from international_builder import run_transfer_trained
    from feature_visualiser import build_figures

    run_improvement_study()      # phase 1: declared plan
    run_orthogonal_addendum()    # phase 2: is topology just a market proxy?
    run_xgb_extension()          # phase 3: does nonlinearity help topology specifically?
    run_transfer_trained()       # phase 4: fair, in-basket-trained transfer
    run_persistence_image_extension()  # phase 5: full persistence images + XGBoost
    run_regime_persistence_60day()     # phase 6: regime images at a 60-session horizon
    run_alternative_learners()         # phase 7: Random Forest and Lasso
    run_delay_embedding_topology()     # phase 8: delay-embedded SPY dynamics as topology
    build_figures()
    print("ALL IMPROVEMENT-STUDY PHASES COMPLETE", flush=True)

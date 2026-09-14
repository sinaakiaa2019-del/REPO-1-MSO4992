"""Exploratory frozen-model transfer to real credit and international baskets."""

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd
from matrix_builder import correlation
from topology_analyser import features, w2


def build_transfer(prices):
    """Build transfer-test features from a target-market price panel."""
    prices = prices.dropna(how="all")
    returns = np.log(prices).diff()
    previous = None
    rows = []
    for t in range(60, len(returns)):
        block = returns.iloc[t - 59 : t + 1]
        good = block.notna().all() & (block.std() > 1e-12)
        a = block.loc[:, good].to_numpy()
        if a.shape[1] < 3:
            continue
        h0, h1, b, s, pi = features(correlation(a))
        s["date"] = returns.index[t]
        s["w2_per_sqrt_asset"] = (
            w2(previous[0], h1) / np.sqrt((previous[1] + a.shape[1]) / 2)
            if previous is not None
            else np.nan
        )
        previous = (h1, a.shape[1])
        rows.append(s)
    if not rows:
        raise ValueError("No eligible transfer windows")
    return pd.DataFrame(rows).set_index("date")


def run_transfer():
    """Run the international and credit transfer experiments."""
    from model_training import metrics
    import pickle

    with (HERE / "models.pkl").open("rb") as f:
        models = pickle.load(f)
    markets = pd.read_pickle(ROOT / "cache/markets.pkl")
    rows = []
    allpred = []
    status = []
    baskets = {
        "credit_bond_ETFs": (["LQD", "HYG", "JNK", "VCIT", "VCSH", "USIG", "BND"], "HYG"),
        "international_indices": (
            ["^FTSE", "^N225", "^STOXX50E", "^GDAXI", "^FCHI", "^HSI", "^AXJO"],
            "^FTSE",
        ),
    }
    for name, (tickers, target) in baskets.items():
        if any((t not in markets for t in tickers)):
            status.append(dict(universe=name, status="unavailable: input series missing"))
            continue
        panel = markets[tickers].dropna(how="all")
        f = build_transfer(panel)
        price = markets[target].dropna()
        forward = pd.concat([price.shift(-i) for i in range(1, 21)], axis=1)
        y = (forward.min(axis=1) / price - 1 <= -0.1).astype(float)
        y.iloc[-20:] = np.nan
        f = f.join(y.rename("label")).dropna()
        f.to_csv(OUT / f"{name}_transfer_features.csv")
        for year in range(2007, 2025):
            use = (f.index >= f"{year}-01-01") & (f.index < f"{year + 1}-01-01")
            if not use.any():
                continue
            for modelname in ["LR_topology", "XGB_topology"]:
                m, cols, cut = models[year, modelname]
                p = m.predict_proba(f.loc[use, cols])[:, 1]
                a = p > cut
                rows.append(
                    dict(
                        universe=name,
                        model=modelname,
                        year=year,
                        target=target,
                        **metrics(f.loc[use, "label"], p, a),
                    )
                )
                allpred.append(
                    pd.DataFrame(
                        dict(
                            date=f.index[use],
                            universe=name,
                            model=modelname,
                            year=year,
                            label=f.loc[use, "label"].to_numpy(),
                            score=p,
                            threshold=cut,
                            alarm=a,
                        )
                    )
                )
        status.append(
            dict(
                universe=name,
                status="exploratory real basket; no target adaptation; source thresholds frozen",
            )
        )
        print("TRANSFER COMPLETE", name, len(f), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rq4_transfer_annual.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "rq4_transfer_status.csv", index=False)
    if allpred:
        p = pd.concat(allpred)
        p.to_csv(OUT / "rq4_transfer_predictions.csv", index=False)
        result = [
            dict(universe=u, model=m, **metrics(g.label, g.score, g.alarm))
            for (u, m), g in p.groupby(["universe", "model"])
        ]
        pd.DataFrame(result).to_csv(OUT / "rq4_transfer_aggregate.csv", index=False)


def transfer_controls(source_models):
    """Frozen equal-budget US topology versus observable target-market scores.

    Baseline thresholds use up to 252 target observations with labels already
    resolved at the forecast date. At least 40 negative calibration days are
    required. The frozen topology threshold remains source-calibrated; this is
    a comparison against target-informed baselines, not identical adaptation.
    """
    from model_training import alarm_threshold, metrics
    markets=pd.read_pickle(HERE/'markets.pkl')
    rows=[]
    for universe,target in [('credit_bond_ETFs','HYG'),('international_indices','^FTSE')]:
        features=pd.read_csv(OUT/f'{universe}_transfer_features.csv',index_col=0,parse_dates=True)
        price=markets[target].dropna()
        ret=np.log(price).diff()
        base=pd.DataFrame(dict(target_volatility=ret.rolling(20).std()*np.sqrt(252),
                               target_momentum=-np.log(price/price.shift(20))))
        fwd=pd.concat([price.shift(-i) for i in range(1,21)],axis=1)
        base['label']=(fwd.min(axis=1)/price-1<=-.1).astype(float)
        base['label_end']=pd.Series(price.index,index=price.index).shift(-20)
        base=base.dropna()
        for year in range(2007,2025):
            if (year,'topology') not in source_models:continue
            model,cols,constant,cuts=source_models[year,'topology']
            use=features[(features.index.year==year) & features.index.isin(base.index)].copy()
            if use.empty:continue
            scores=model.predict_proba(use[cols])[:,1] if model else np.full(len(use),constant)
            for (day,row),score in zip(use.iterrows(),scores):
                known=base[base.label_end<day].tail(252)
                if (known.label==0).sum()<40:continue
                truth=int(base.loc[day,'label'])
                rows.append(dict(date=day,year=year,universe=universe,model='frozen_US_topology',
                    label=truth,score=score,threshold=cuts[.05],alarm=score>cuts[.05]))
                for name in ['target_volatility','target_momentum']:
                    value=float(base.loc[day,name])
                    cut=alarm_threshold(known.label,known[name],.05)
                    rows.append(dict(date=day,year=year,universe=universe,model=name,
                                     label=truth,score=value,threshold=cut,alarm=value>cut))
    pred=pd.DataFrame(rows).sort_values(['universe','model','date'])
    pred.to_csv(OUT/'defence_transfer_predictions.csv',index=False)
    aggregate=pd.DataFrame([dict(universe=u,model=m,**metrics(g.label,g.score,g.alarm))
        for (u,m),g in pred.groupby(['universe','model'])])
    aggregate.to_csv(OUT/'defence_transfer_metrics.csv',index=False)
    return pred


def run_transfer_trained():
    """Phase 4 (post-hoc): a fair, in-basket-trained version of the transfer test.

    transfer_controls() above freezes a US-trained topology model and applies it
    zero-shot, while its target_volatility/target_momentum comparators train on the
    target market's own history -- not a fair fight, and likely why frozen topology
    scored so poorly there. This trains and calibrates every group, topology included,
    inside each basket under the same purged protocol used everywhere else. Reuses the
    topology features already computed for the frozen test; no new TDA or data.
    """
    from model_training import evaluate_groups, stratified, paired_intervals, benjamini_hochberg
    from robustness_builder import drawdown_events, event_leads
    from feature_builder import PRIMARY_FAMILY
    from report_builder import md
    import sqlite3

    destination = ROOT / "results/improved"
    destination.mkdir(parents=True, exist_ok=True)
    topology = ["h1_entropy_normalized", "h1_total_per_asset", "w2_per_sqrt_asset"]
    baskets = {
        "credit_bond_ETFs": dict(table="credit_bond_ETFs_transfer_features", target="HYG"),
        "international_indices": dict(table="international_indices_transfer_features", target="^FTSE"),
    }
    markets = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True)

    def build_frame(connection, spec):
        """Load and align one transfer basket from saved evidence."""
        features = pd.read_sql_query(f'SELECT * FROM "{spec["table"]}"', connection, parse_dates=["date"]).set_index("date")
        price = markets[spec["target"]].dropna()
        ret = np.log(price).diff()
        frame = pd.DataFrame(index=features.index)
        frame["target_volatility"] = (ret.rolling(20).std() * np.sqrt(252)).reindex(features.index)
        frame["target_momentum"] = (-np.log(price / price.shift(20))).reindex(features.index)
        for col in topology:
            frame[col] = features[col]
        frame["label"] = features["label"]
        frame["label_end"] = pd.Series(price.index, index=price.index).shift(-20).reindex(features.index)
        return frame.replace([np.inf, -np.inf], np.nan).dropna()

    def run_basket(name, frame):
        """Fit and evaluate the declared models for one basket."""
        groups = dict(
            target_market=["target_volatility", "target_momentum"],
            target_topology=topology,
            target_combined=["target_volatility", "target_momentum"] + topology,
            target_topology_xgb=topology,
        )
        result = evaluate_groups(frame, groups, kinds=dict(target_topology_xgb="xgb"))
        result.pop("models")
        predictions = result["predictions"]
        scores = result["metrics"].merge(stratified(predictions), on=["model", "budget"])
        primary = predictions[predictions.budget == 0.05]
        comparisons = paired_intervals(primary, [
            ("target_combined", "target_market"),
            ("target_topology", "target_market"),
            ("target_topology_xgb", "target_topology"),
            ("target_combined", "target_topology_xgb"),
        ], primary_family=PRIMARY_FAMILY)
        spy = markets.SPY.dropna()
        events = drawdown_events(spy)
        leads = event_leads(predictions, events, spy.index)
        tuning = result["tuning"]
        tuning["columns"] = tuning["columns"].map(json.dumps)
        for table in [scores, comparisons, leads, tuning, result["selected"]]:
            table.insert(0, "basket", name)
        print("TRANSFER MARKET-TRAINED BASKET COMPLETE", name, flush=True)
        return dict(metrics=scores, comparisons=comparisons, event_leads=leads, tuning=tuning, selected=result["selected"])

    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        frames = {name: build_frame(connection, spec) for name, spec in baskets.items()}
    connection.close()
    outputs = [run_basket(name, frame) for name, frame in frames.items()]
    tables = {key: pd.concat([o[key] for o in outputs], ignore_index=True) for key in outputs[0]}
    for key in ["metrics", "comparisons", "event_leads", "tuning", "selected"]:
        tables[key].to_csv(destination / f"transfer_trained_{key}.csv", index=False)

    published = pd.read_csv(ROOT / "results/defence_transfer_metrics.csv")
    published = published[published.model.isin(["frozen_US_topology", "target_volatility", "target_momentum"])]

    # Fold this phase's block-60 tests into the running multiple-testing family. Plain
    # concat keeps every phase's own identifying column ('target' for phases 1-3, 'basket' here)
    # instead of silently dropping whichever column the other phases don't share.
    prior = pd.read_csv(destination / "full_family_fdr_correction.csv")
    prior = prior[prior.source != "phase4_transfer_trained"]
    phase4 = tables["comparisons"][tables["comparisons"].block_sessions == 60].copy()
    phase4["source"] = "phase4_transfer_trained"
    family = pd.concat([prior, phase4], ignore_index=True)
    family["p_approx"] = (2 * np.minimum(family.share_draws_favouring_model, 1 - family.share_draws_favouring_model)).clip(1.0 / 501, 1.0)
    family["q_value_bh"], family["significant_fdr_5pct"] = benjamini_hochberg(family.p_approx.to_numpy())
    family = family.sort_values("p_approx")
    family.to_csv(destination / "full_family_fdr_correction.csv", index=False)
    n_sig = int(family.significant_fdr_5pct.sum())

    lines = [
        "# Fair, in-market-trained transfer test (post-hoc, phase 4)",
        "",
        "The published transfer test above freezes a US-trained topology model and applies it "
        "zero-shot, while target_volatility/target_momentum train on the target market's own "
        "history. Here every group, including topology, is trained and calibrated within each "
        "basket under the same purged protocol used everywhere else. Same topology features "
        "(already published), same target definitions, no new data.", "",
        "## Published, frozen, zero-shot result (for comparison; untouched)", "",
        md(published[["universe", "model", "auroc", "average_precision", "false_alarm_episodes"]]), "",
        "## This phase: trained and calibrated within each basket", "",
    ]
    for name in baskets:
        m = tables["metrics"][(tables["metrics"].basket == name) & (tables["metrics"].budget == 0.05)]
        c = tables["comparisons"][tables["comparisons"].basket == name]
        lines += [f"### {name}", "",
                  md(m.sort_values("average_precision", ascending=False)[
                      ["model", "n", "positives", "auroc", "average_precision", "within_year_auroc", "false_alarm_episodes"]]),
                  "", md(c[["model", "reference", "ap_difference", "lower95", "upper95", "share_draws_favouring_model"]]), ""]
    lines += [
        "## Multiple-testing correction across all four phases", "",
        f"Benjamini-Hochberg 5% FDR control across phases 1-4's 60-session-block comparisons "
        f"({len(family)} tests total). {n_sig} survive.", "",
        md(family[family.significant_fdr_5pct][["model", "reference", "ap_difference", "p_approx", "q_value_bh", "source"]])
        if n_sig else "No comparison in the full four-phase family survives 5% FDR control.", "",
        "## Reading this honestly", "",
        "Only 3 (credit) and 4 (international) calendar years in these baskets have any positive "
        "labels at all -- less independent crisis evidence than even the primary 18-year US study "
        "had. Wide intervals reflect that directly; a favourable point estimate on this little data "
        "is not the same as a validated effect.", "",
    ]
    (destination / "TRANSFER_TRAINED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("TRANSFER MARKET-TRAINED COMPLETE", flush=True)

"""One ML runner replaces the overlapping weekly ML scripts."""

import sys
from pathlib import Path

if __name__ == "__main__":
    # Running this file directly (e.g. `python ml/run_ml.py --research`)
    # only puts this folder on sys.path, not the project root, so config.py
    # would not be found without this line.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd
import pickle
from feature_builder import data
from model_training import select, fit, alarm_threshold, metrics, stratified
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def run_models():
    """Run the classic machine-learning comparison and save its evidence."""
    df, groups = data()
    y = df.label.astype(int)
    end = df.label_end
    specs = {
        "VIX": ("raw", ["vix"]),
        "constant": ("constant", ["vix"]),
        "LR_base": ("lr", groups["base"]),
        "LR_topology": ("lr", groups["compact"]),
        "LR_combined": ("lr", groups["combined"]),
        "LR_dynamic_combined": ("lr", groups["dynamic_combined"]),
        "XGB_base": ("xgb", groups["base"]),
        "XGB_topology": ("xgb", groups["compact"]),
        "XGB_combined": ("xgb", groups["combined"]),
        "XGB_full_topology": ("xgb", groups["full"]),
    }
    for prefix, kind in [("LR", "lr"), ("XGB", "xgb")]:
        for group in ["enhanced_base", "enhanced_combined"]:
            specs[f"{prefix}_{group}"] = (kind, groups[group])
    rows = []
    preds = []
    splitlog = []
    tuning = []
    saved = {}
    scale = []
    for year in range(2007, 2025):
        start = pd.Timestamp(f"{year}-01-01")
        stop = pd.Timestamp(f"{year + 1}-01-01")
        test = (df.index >= start) & (df.index < stop)
        known = np.flatnonzero((end < start).to_numpy())
        calidx = known[-252:]
        calstart = df.index[calidx[0]]
        train = end < calstart
        assert end.loc[train].max() < calstart and end.iloc[calidx].max() < start
        splitlog.append(
            dict(
                year=year,
                fit_start=str(df.index[train].min()),
                fit_end=str(df.index[train].max()),
                fit_label_end=str(end.loc[train].max()),
                calibration_start=str(calstart),
                calibration_end=str(df.index[calidx[-1]]),
                calibration_label_end=str(end.iloc[calidx].max()),
                test_start=str(df.index[test].min()),
                test_end=str(df.index[test].max()),
                fit_n=int(train.sum()),
                calibration_n=len(calidx),
                test_n=int(test.sum()),
            )
        )
        for name, (kind, columns) in specs.items():
            X = df[columns]
            param = None
            model = None
            if kind == "raw":
                pc = X.iloc[calidx, 0].to_numpy()
                pt = X.loc[test].iloc[:, 0].to_numpy()
            elif kind == "constant":
                pc = np.full(len(calidx), y.loc[train].mean())
                pt = np.full(int(test.sum()), y.loc[train].mean())
            else:
                param, records = select(
                    X.loc[train], y.loc[train], end.loc[train], kind
                )
                tuning.extend((dict(year=year, model=name, **r) for r in records))
                model = fit(X.loc[train], y.loc[train], kind, param)
                pc = model.predict_proba(X.iloc[calidx])[:, 1]
                pt = model.predict_proba(X.loc[test])[:, 1]
            cut = alarm_threshold(y.iloc[calidx], pc)
            alarm = pt > cut
            row = dict(
                year=year,
                model=name,
                parameter=param,
                threshold=cut,
                calibration_fpr=float(
                    (pc[y.iloc[calidx].to_numpy() == 0] > cut).mean()
                ),
                **metrics(y.loc[test], pt, alarm),
            )
            rows.append(row)
            if kind not in ["raw"]:
                row["brier"] = float(brier_score_loss(y.loc[test], pt))
            preds.append(
                pd.DataFrame(
                    dict(
                        date=df.index[test],
                        year=year,
                        model=name,
                        label=y.loc[test].values,
                        score=pt,
                        threshold=cut,
                        alarm=alarm,
                    )
                )
            )
            saved[year, name] = (model, columns, cut)
        best = None
        for j in range(200):
            c = f"betti_{j:03}"
            auc = roc_auc_score(y.loc[train], df.loc[train, c])
            direction = 1 if auc >= 0.5 else -1
            score = max(auc, 1 - auc)
            if best is None or score > best[0]:
                best = (score, j, direction)
        _, j, direction = best
        p = direction * df.loc[test, f"betti_{j:03}"]
        scale.append(
            dict(
                year=year,
                epsilon=2 * j / 199,
                correlation=1 - (2 * j / 199) ** 2 / 2,
                direction=direction,
                selection_train_auroc=best[0],
                test_auroc=(
                    roc_auc_score(y.loc[test], p)
                    if y.loc[test].nunique() == 2
                    else None
                ),
                test_ap=(
                    average_precision_score(y.loc[test], p)
                    if y.loc[test].sum()
                    else None
                ),
            )
        )
        pd.DataFrame(rows).to_csv(OUT / "annual_metrics.csv", index=False)
        print("Year complete", year, flush=True)
    pred = pd.concat(preds)
    pred.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(splitlog).to_csv(OUT / "splits.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "tuning.csv", index=False)
    pd.DataFrame(scale).to_csv(OUT / "filtration_selected.csv", index=False)
    summary = []
    for name, g in pred.groupby("model"):
        g = g.sort_values("date")
        summary.append(dict(model=name, **metrics(g.label, g.score, g.alarm)))
    pd.DataFrame(summary).to_csv(OUT / "aggregate_metrics.csv", index=False)
    with open(HERE / "models.pkl", "wb") as f:
        pickle.dump(saved, f)
    print(
        pd.DataFrame(summary)[
            ["model", "auroc", "average_precision", "false_positive_rate", "recall"]
        ].to_string(index=False),
        flush=True,
    )
    print("EVALUATION COMPLETE", flush=True)


def classic_correction(df, registry, published):
    """Post-hoc bug fix: remove two coverage time trends from the classic enhanced ladder.

    "eligible_assets"/"eligible_fraction" (enhanced_features(), feature_builder.py) turn
    out to track calendar date almost exactly (universe size grows through the sample),
    so the classic enhanced models were partly learning "what year is it," not stress.
    Reruns the classic two-candidate protocol for the published models and their
    coverage-variable-free counterparts, side by side.
    """
    base, compact, enhanced = registry["base"], registry["compact"], registry["enhanced_base"]
    clean = [name for name in enhanced if name not in ["eligible_assets", "eligible_fraction"]]
    specs = {
        "LR_base": ("lr", base),
        "LR_enhanced_base": ("lr", enhanced),
        "LR_enhanced_combined": ("lr", enhanced + compact),
        "XGB_enhanced_base": ("xgb", enhanced),
        "XGB_enhanced_combined": ("xgb", enhanced + compact),
        "LR_enhanced_base_clean": ("lr", clean),
        "LR_enhanced_combined_clean": ("lr", clean + compact),
        "XGB_enhanced_base_clean": ("xgb", clean),
        "XGB_enhanced_combined_clean": ("xgb", clean + compact),
    }
    y = df.label.astype(int)
    end = df.label_end
    rows = []
    for year in range(2007, 2025):
        start = pd.Timestamp(f"{year}-01-01")
        test = (df.index >= start) & (df.index < pd.Timestamp(f"{year + 1}-01-01"))
        known = np.flatnonzero((end < start).to_numpy())
        calidx = known[-252:]
        train = end < df.index[calidx[0]]
        for name, (kind, columns) in specs.items():
            X = df[columns]
            param, _ = select(X.loc[train], y.loc[train], end.loc[train], kind)
            model = fit(X.loc[train], y.loc[train], kind, param)
            pc = model.predict_proba(X.iloc[calidx])[:, 1]
            pt = model.predict_proba(X.loc[test])[:, 1]
            cut = alarm_threshold(y.iloc[calidx], pc)
            rows.append(pd.DataFrame(dict(
                date=df.index[test], year=year, model=name, label=y.loc[test].to_numpy(),
                score=pt, threshold=cut, alarm=pt > cut, parameter=param,
            )))
        print("Classic correction year", year, flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    aggregate = pd.DataFrame(
        [dict(model=name, **metrics(g.label, g.score, g.alarm)) for name, g in predictions.groupby("model")]
    )
    aggregate = aggregate.merge(stratified(predictions.assign(budget=0.05)).drop(columns="budget"), on="model")
    aggregate["coverage_trend_variables"] = aggregate.model.map(
        lambda name: "none" if name == "LR_base" else ("removed" if name.endswith("_clean") else "included")
    )
    # Reproduction check: do the "included" models still match the published classic ladder?
    reproduction = {}
    for name, g in predictions.groupby("model"):
        old = published[published.model == name]
        if len(old):
            joined = g.merge(old, on="date", suffixes=("", "_published"))
            reproduction[name] = dict(
                max_score_difference=float((joined.score - joined.score_published).abs().max()),
                alarm_mismatches=int((joined.alarm != joined.alarm_published.astype(bool)).sum()),
            )
    return predictions, aggregate, reproduction


def run_research():
    """A bounded retrospective search; outer outcomes never choose settings."""
    import sqlite3
    from itertools import product
    from robustness_builder import research_inputs, research_comparisons
    from model_training import research_select, research_train_index

    target = ROOT / "results/research"
    target.mkdir(exist_ok=True)
    (target / "manifest.json").unlink(missing_ok=True)
    protocol = dict(
        status="Retrospective development; all 2000-2024 outcomes previously inspected",
        selection="maximum pooled inner average precision; two chronological purged blocks",
        C=[0.01, 0.1, 1.0],
        history_sessions=[0, 756, 1260],
        history_zero="expanding",
        target="next 20 sessions SPY loss at least 10 percent",
        calibration_sessions=252,
        calibration_negative_fpr=0.05,
        tda_grid=[
            "default:60/.94/.1",
            "short:40/.94/.1",
            "long:120/.98/.1",
            "shrink:60/.94/.2",
        ],
        quality="separate fixed sensitivity: mask abs(log return)>0.5, never call it corrected truth",
        seed=42,
        test_years=[2007, 2024],
        primary_metric="average_precision",
        multiple_testing="No untouched holdout; bootstrap intervals do not correct research selection",
    )
    (target / "protocol.json").write_text(json.dumps(protocol, indent=2))
    df, topology, topology_landscape, topology_network, specs, input_hashes = research_inputs()
    market = {
        "vix_level": ["log_vix"],
        "vix_small": ["log_vix", "vix_change5", "vix_ratio60", "vol20"],
        "market_small": [
            "log_vix",
            "vol20",
            "momentum_20",
            "drawdown_60",
            "yield_curve",
        ],
    }
    sets = {}
    for group, columns in market.items():
        sets["market/" + group] = columns
    for name, columns in topology.items():
        sets["topology/" + name + "/compact"] = columns[:3]
        for col in columns[3:]:
            sets["topology/" + name + "/" + col.rsplit("_", 1)[1]] = [col]
        sets["combined/" + name] = market["vix_small"] + columns[:3]
    for name, columns in topology_landscape.items():
        sets["topology/" + name + "/landscape"] = columns
    for name, columns in topology_network.items():
        sets["topology/" + name + "/network"] = columns
    candidates = [
        dict(
            id=f"{group}/C={c}/history={history}",
            group=group,
            columns=columns,
            C=c,
            history=history,
        )
        for group, columns in sets.items()
        for c, history in product(protocol["C"], protocol["history_sessions"])
    ]
    (target / "protocol.json").write_text(
        json.dumps(
            dict(protocol, candidates=candidates, input_hashes=input_hashes), indent=2
        )
    )
    predictions, annual, tuning, selected, splits = [], [], [], [], []
    for year in range(2007, 2025):
        start = pd.Timestamp(f"{year}-01-01")
        test = (df.index >= start) & (df.index < pd.Timestamp(f"{year+1}-01-01"))
        known = np.flatnonzero((df.label_end < start).to_numpy())
        cal = known[-252:]
        train = np.flatnonzero((df.label_end < df.index[cal[0]]).to_numpy())
        assert df.label_end.iloc[train].max() < df.index[cal[0]]
        assert df.label_end.iloc[cal].max() < df.index[test].min()
        _, records = research_select(df.iloc[train], candidates)
        tuning.extend(dict(year=year, **r) for r in records)
        families = {
            "selected_market": lambda r: r["group"].startswith("market/"),
            "market_expanding": lambda r: r["group"].startswith("market/")
            and r["history"] == 0,
            "market_rolling": lambda r: r["group"].startswith("market/")
            and r["history"] > 0,
            "selected_topology": lambda r: r["group"].startswith("topology/")
            and "/quality/" not in r["group"]
            and "/landscape" not in r["group"]
            and "/network" not in r["group"],
            "selected_combined": lambda r: r["group"].startswith("combined/")
            and "quality" not in r["group"],
            "selected_quality": lambda r: r["group"] == "combined/quality",
            "selected_quality_topology": lambda r: r["group"].startswith("topology/quality/")
            and "/landscape" not in r["group"]
            and "/network" not in r["group"],
            "selected_landscape": lambda r: "/landscape" in r["group"],
            "selected_network": lambda r: "/network" in r["group"],
        }
        family_scores = {}
        for model_name, predicate in families.items():
            best = max(
                [r for r in records if predicate(r)],
                key=lambda r: -1 if r["inner_ap"] is None else r["inner_ap"],
            )
            fitidx = research_train_index(
                df.label_end, df.index[cal[0]], best["history"]
            )
            yfit = df.label.iloc[fitidx].astype(int)
            if yfit.nunique() < 2:
                pc, pt = np.full(len(cal), yfit.mean()), np.full(
                    test.sum(), yfit.mean()
                )
                fallback = True
            else:
                m = fit(df.iloc[fitidx][best["columns"]], yfit, "lr", best["C"])
                pc = m.predict_proba(df.iloc[cal][best["columns"]])[:, 1]
                pt = m.predict_proba(df.loc[test, best["columns"]])[:, 1]
                fallback = False
            cut = alarm_threshold(df.label.iloc[cal], pc)
            family_scores[model_name] = (pc, pt)
            selected.append(
                dict(year=year, model=model_name, constant_fallback=fallback, **best)
            )
            splits.append(
                dict(
                    year=year,
                    model=model_name,
                    fit_n=len(fitidx),
                    fit_start=str(df.index[fitidx[0]]),
                    fit_end=str(df.index[fitidx[-1]]),
                    fit_label_end=str(df.label_end.iloc[fitidx].max()),
                    calibration_start=str(df.index[cal[0]]),
                    calibration_label_end=str(df.label_end.iloc[cal].max()),
                    test_start=str(df.index[test].min()),
                )
            )
            annual.append(
                dict(
                    year=year,
                    model=model_name,
                    calibration_fpr=float(
                        (pc[df.label.iloc[cal].to_numpy() == 0] > cut).mean()
                    ),
                    **metrics(df.label.loc[test], pt, pt > cut),
                )
            )
            predictions.append(
                pd.DataFrame(
                    dict(
                        date=df.index[test],
                        year=year,
                        model=model_name,
                        label=df.label.loc[test].to_numpy(),
                        score=pt,
                        threshold=cut,
                        alarm=pt > cut,
                    )
                )
            )
        # Fixed 50/50 blend of the pure market-only and pure topology-only models.
        # The weights are fixed in advance, not fit on any data, so averaging two
        # fitted model scores adds no extra tuning of its own. Threshold
        # calibration does not imply probability calibration.
        pc_market, pt_market = family_scores["selected_market"]
        pc_topology, pt_topology = family_scores["selected_topology"]
        pc, pt = 0.5 * pc_market + 0.5 * pc_topology, 0.5 * pt_market + 0.5 * pt_topology
        cut = alarm_threshold(df.label.iloc[cal], pc)
        annual.append(
            dict(
                year=year,
                model="selected_ensemble",
                calibration_fpr=float(
                    (pc[df.label.iloc[cal].to_numpy() == 0] > cut).mean()
                ),
                **metrics(df.label.loc[test], pt, pt > cut),
            )
        )
        predictions.append(
            pd.DataFrame(
                dict(
                    date=df.index[test],
                    year=year,
                    model="selected_ensemble",
                    label=df.label.loc[test].to_numpy(),
                    score=pt,
                    threshold=cut,
                    alarm=pt > cut,
                )
            )
        )
        pc, pt = df.vix.iloc[cal].to_numpy(), df.vix.loc[test].to_numpy()
        cut = alarm_threshold(df.label.iloc[cal], pc)
        predictions.append(
            pd.DataFrame(
                dict(
                    date=df.index[test],
                    year=year,
                    model="VIX",
                    label=df.label.loc[test].to_numpy(),
                    score=pt,
                    threshold=cut,
                    alarm=pt > cut,
                )
            )
        )
        annual.append(
            dict(
                year=year,
                model="VIX",
                calibration_fpr=float(
                    (pc[df.label.iloc[cal].to_numpy() == 0] > cut).mean()
                ),
                **metrics(df.label.loc[test], pt, pt > cut),
            )
        )
        print("Nested research year", year, "candidates", len(candidates), flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    # Original models on exactly the same dates provide a fair before/after reference.
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        old = pd.read_sql_query("SELECT * FROM predictions", connection)
    old.date = pd.to_datetime(old.date)
    old = old[
        old.date.isin(pd.to_datetime(pred.date.unique()))
        & old.model.isin(["LR_base", "LR_topology", "LR_combined", "XGB_full_topology"])
    ]
    old.model = "original_" + old.model
    pred = pd.concat([pred, old], ignore_index=True).sort_values(["model", "date"])
    aggregate = pd.DataFrame(
        [
            dict(model=name, **metrics(g.label, g.score, g.alarm))
            for name, g in pred.groupby("model")
        ]
    )
    comparisons = research_comparisons(pred)
    aggregate.to_csv(target / "metrics.csv", index=False)
    comparisons.to_csv(target / "comparisons.csv", index=False)
    selected_frame = pd.DataFrame(selected)
    for frame in [selected_frame, pd.DataFrame(tuning)]:
        if "columns" in frame:
            frame["columns"] = frame["columns"].map(json.dumps)
    tuning_frame = pd.DataFrame(tuning)
    tuning_frame["columns"] = tuning_frame["columns"].map(json.dumps)
    with sqlite3.connect(target / "evidence.sqlite") as connection:
        for name, frame in dict(
            predictions=pred,
            annual=pd.DataFrame(annual),
            tuning=tuning_frame,
            selected=selected_frame,
            splits=pd.DataFrame(splits),
            metrics=aggregate,
            comparisons=comparisons,
        ).items():
            frame.to_sql(name, connection, if_exists="replace", index=False)
        coverage = pd.read_csv(ROOT / "results/asset_coverage.csv")
        coverage[coverage.status != "used"].to_sql(
            "unresolved_assets", connection, if_exists="replace", index=False
        )
        pd.read_csv(OUT / "extreme_returns_membership_review.csv").to_sql(
            "extreme_return_review", connection, if_exists="replace", index=False
        )
        from data_processing import audit_reference_prices

        audit_reference_prices().to_sql(
            "reference_price_checks", connection, if_exists="replace", index=False
        )
        pd.read_csv(HERE / "research/tda_coverage.csv").to_sql(
            "tda_coverage", connection, if_exists="replace", index=False
        )
    selected_frame.to_csv(target / "selected_settings.csv", index=False)
    from feature_visualiser import build_chart

    build_chart(research=True)
    from report_builder import make_research_report

    make_research_report()
    print(
        aggregate[
            ["model", "auroc", "average_precision", "false_positive_rate"]
        ].to_string(index=False),
        flush=True,
    )
    verify_research()


def verify_research():
    """Recheck metrics, split purges and training-only parameter selection."""
    import sqlite3
    import hashlib

    manifest_path = ROOT / "results/research/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for filename, expected in {
            **manifest["files"],
            **manifest["source_hashes"],
        }.items():
            assert (
                hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
            ), filename
    with sqlite3.connect(ROOT / "results/research/evidence.sqlite") as connection:
        pred = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY model,date", connection
        )
        saved = pd.read_sql_query("SELECT * FROM metrics", connection).set_index(
            "model"
        )
        expected_models = {
            "VIX", "selected_market", "market_expanding", "market_rolling",
            "selected_topology", "selected_combined", "selected_quality",
            "selected_quality_topology", "original_LR_base", "original_LR_topology",
            "original_LR_combined", "original_XGB_full_topology",
            "selected_landscape", "selected_network", "selected_ensemble",
        }
        assert set(saved.index) == expected_models, "Incomplete research model set"
        assert set(pred.model) == expected_models
        dates = None
        for name, g in pred.groupby("model"):
            assert not g.date.duplicated().any()
            if dates is None:
                dates = list(g.date)
            assert list(g.date) == dates
            assert ((g.score > g.threshold) == g.alarm.astype(bool)).all()
            for key, value in metrics(g.label, g.score, g.alarm).items():
                if value is not None:
                    assert abs(saved.loc[name, key] - value) < 1e-10, (name, key)
        splits = pd.read_sql_query("SELECT * FROM splits", connection)
        assert (splits.fit_label_end < splits.calibration_start).all()
        assert (splits.calibration_label_end < splits.test_start).all()
        annual = pd.read_sql_query("SELECT * FROM annual", connection)
        assert annual.calibration_fpr.max() <= 0.05 + 1e-12
        selections = pd.read_sql_query("SELECT * FROM selected", connection)
        tuning = pd.read_sql_query("SELECT * FROM tuning", connection)
        assert len(selections) == 18 * 9
        assert len(tuning) == 18 * 387
        assert set(tuning.year) == set(range(2007, 2025))
        assert not tuning.duplicated(['year', 'id']).any()
        for row in selections.itertuples():
            candidates = tuning[tuning.year == row.year]
            if row.model.startswith("market_") or row.model == "selected_market":
                candidates = candidates[candidates.group.str.startswith("market/")]
                if row.model == "market_expanding":
                    candidates = candidates[candidates.history == 0]
                if row.model == "market_rolling":
                    candidates = candidates[candidates.history > 0]
            elif row.model == "selected_quality_topology":
                candidates = candidates[
                    candidates.group.str.startswith("topology/quality/")
                    & ~candidates.group.str.contains("landscape")
                    & ~candidates.group.str.contains("network")
                ]
            elif row.model == "selected_landscape":
                candidates = candidates[candidates.group.str.contains("/landscape")]
            elif row.model == "selected_network":
                candidates = candidates[candidates.group.str.contains("/network")]
            elif row.model == "selected_topology":
                candidates = candidates[
                    candidates.group.str.startswith("topology/")
                    & ~candidates.group.str.contains("quality")
                    & ~candidates.group.str.contains("landscape")
                    & ~candidates.group.str.contains("network")
                ]
            elif row.model == "selected_combined":
                candidates = candidates[
                    candidates.group.str.startswith("combined/")
                    & ~candidates.group.str.contains("quality")
                ]
            else:
                candidates = candidates[candidates.group == "combined/quality"]
            assert (
                row.id
                == candidates.iloc[
                    candidates.inner_ap.fillna(-1).to_numpy().argmax()
                ].id
            )
    print(
        "Research verified: saved scores, dates, thresholds, purges, inner-only selections.",
        flush=True,
    )


if __name__ == "__main__":
    if "--research" in sys.argv:
        import contextlib

        class ResearchLog:
            """Keep the progress visible and save the same text for review."""

            def __init__(self, console, handle):
                """Store the console and log destinations."""
                self.console, self.handle = console, handle

            def write(self, text):
                """Write the same message to the console and project log."""
                self.console.write(text)
                self.handle.write(text)
                self.flush()

            def flush(self):
                """Flush both output destinations."""
                self.console.flush()
                self.handle.flush()

        folder = ROOT / "results/research"
        folder.mkdir(exist_ok=True)
        with (folder / "run.log").open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(
                ResearchLog(sys.stdout, handle)
            ), contextlib.redirect_stderr(ResearchLog(sys.stderr, handle)):
                run_research()
        from report_builder import publish_research

        publish_research()
    elif "--verify-research" in sys.argv:
        verify_research()
    else:
        run_models()

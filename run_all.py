"""One entry point for execution, result verification and table export."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
# run_all.py doesn't import config.py, so it needs its own copy of the same
# path setup: data_pipeline/ and ml/ hold Octavian's and Sina's modules, and
# the "data" stage below runs before anything else would have added them.
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "ml"))
STAGES = [
    "data",
    "math",
    "tda",
    "coverage",
    "ml",
    "sensitivity",
    "transfer",
    "sectors",
    "defence",
    "report",
    "checks",
    "publish",
    "improvements",
    "figures",
]


def verify_results():
    """Recalculate ranking scores and false alarms without numerical packages."""
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        metrics = connection.execute("SELECT * FROM aggregate_metrics").fetchall()
        reference_dates = None
        for metric in metrics:
            rows = connection.execute(
                "SELECT * FROM predictions WHERE model=? ORDER BY date", (metric["model"],)
            ).fetchall()
            dates = [row["date"] for row in rows]
            if reference_dates is None:
                reference_dates = dates
            assert dates == reference_dates and len(set(dates)) == len(dates)
            positives = sum(row["label"] for row in rows)
            ordered = sorted(rows, key=lambda row: row["score"])
            rank_sum = 0
            start = 0
            while start < len(ordered):
                stop = start + 1
                while stop < len(ordered) and ordered[stop]["score"] == ordered[start]["score"]:
                    stop += 1
                rank_sum += (start + 1 + stop) / 2 * sum(r["label"] for r in ordered[start:stop])
                start = stop
            auc = (rank_sum - positives * (positives + 1) / 2) / (
                positives * (len(rows) - positives)
            )
            ordered.reverse()
            start = 0
            true_positives = 0
            average_precision = 0
            while start < len(ordered):
                stop = start + 1
                while stop < len(ordered) and ordered[stop]["score"] == ordered[start]["score"]:
                    stop += 1
                added = sum(r["label"] for r in ordered[start:stop])
                true_positives += added
                average_precision += added / positives * true_positives / stop
                start = stop
            assert abs(auc - metric["auroc"]) < 1e-10
            assert abs(average_precision - metric["average_precision"]) < 1e-10
            assert (
                sum(r["label"] == 0 and bool(r["alarm"]) for r in rows)
                == metric["false_positive_days"]
            )
            assert all(bool(r["alarm"]) == (r["score"] > r["threshold"]) for r in rows)
        for row in connection.execute("SELECT * FROM splits"):
            assert row["fit_label_end"] < row["calibration_start"]
            assert row["calibration_label_end"] < row["test_start"]
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    print(f"Verified {len(metrics)} models, saved alarms, shared dates, splits and file hashes.")


def export_table(name):
    """Export an audit table when needed, without filling the folder with files."""
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        names = [
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        if name not in names:
            raise ValueError("Choose a table from: " + ", ".join(names))
        cursor = connection.execute('SELECT * FROM "' + name + '"')
        destination = ROOT / "results" / (name + ".csv")
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([column[0] for column in cursor.description])
            writer.writerows(cursor)
    print(destination)


def stage(name, args):
    """Import numerical code only when running the corresponding stage."""
    if name == "data":
        from data_loader import prepare
        from data_processing import audit_inputs

        prepare(args.input_dir or ROOT / "inputs", args.prices)
        audit_inputs()
    elif name == "coverage":
        from data_processing import audit_universe

        audit_universe()
    elif name == "math":
        sys.path.insert(0, str(ROOT / "tests"))
        from test_math import check_math

        check_math()
    elif name == "tda":
        from topology_analyser import run_topology

        run_topology()
    elif name == "ml":
        from run_ml import run_models

        run_models()
    elif name == "sensitivity":
        from robustness_builder import run_sensitivity

        run_sensitivity()
    elif name == "transfer":
        from international_builder import run_transfer

        run_transfer()
    elif name == "sectors":
        from sector_builder import run_sectors

        run_sectors()
    elif name == "report":
        from report_builder import make_report

        make_report()
    elif name == "defence":
        from robustness_builder import run_defence

        run_defence()
    elif name == "checks":
        sys.path.insert(0, str(ROOT / "tests"))
        from test_run import check_run

        check_run()
    elif name == "publish":
        from report_builder import publish

        publish()
    elif name == "improvements":
        # Post-hoc improvement study (the declared improvement protocol and later phases).
        # Needs results/evidence.sqlite (the "publish" stage) already written.
        from robustness_builder import run_improvements

        run_improvements()
    elif name == "figures":
        from feature_visualiser import build_figures

        build_figures()


def validate_protocol():
    """Reject settings that the fixed experiment does not implement."""
    config = json.loads((ROOT / "config.json").read_text())
    fixed = {
        "price_field": "adjusted_close",
        "horizon_sessions": 20,
        "forward_loss_threshold": -0.1,
        "step_size": 1,
        "window_sessions": 60,
        "ewma_lambda": 0.94,
        "minimum_observed_return_fraction": 1.0,
        "liquidity_cap": None,
        "filtration": [0, 2, 200],
        "homology_dimensions": [0, 1],
        "outer_test_years": [2007, 2024],
        "calibration_sessions": 252,
        "calibration_negative_day_false_alarm_budget": 0.05,
        "seed": 42,
    }
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError(
            "The numerical protocol is locked; update the implementation and checks before changing it"
        )


def main():
    """Parse command-line options and run the requested project stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices", type=Path, help="Path to the original adjusted-close equity CSV"
    )
    parser.add_argument(
        "--input-dir", type=Path, help="Optional replacement folder for the documented inputs"
    )
    parser.add_argument("--verify-results", action="store_true")
    parser.add_argument("--export-table", help="Export one table from results/evidence.sqlite")
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_results:
        verify_results()
        return
    if args.export_table:
        export_table(args.export_table)
        return
    if args.stage:
        stage(args.stage, args)
        return
    if not args.prices and not args.input_dir:
        parser.error(
            "Supply --prices, or --input-dir containing prices.csv and the supporting inputs"
        )
    validate_protocol()
    (ROOT / "cache/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    # A failed rerun must not leave an old success marker looking current.
    (ROOT / "results/manifest.json").unlink(missing_ok=True)
    status = []
    with (ROOT / "results/run.log").open("w", encoding="utf-8") as log:
        for name in STAGES:
            print("Running " + name, flush=True)
            log.write("\nStage: " + name + "\n")
            log.flush()
            command = [sys.executable, "-u", str(ROOT / "run_all.py"), "--stage", name]
            if args.prices:
                command += ["--prices", str(args.prices.resolve())]
            if args.input_dir:
                command += ["--input-dir", str(args.input_dir.resolve())]
            started = time.time()
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status.append(
                dict(
                    stage=name,
                    exit_code=process.returncode,
                    seconds=round(time.time() - started, 2),
                )
            )
            (ROOT / "cache/tables/stage_status.json").write_text(json.dumps(status, indent=2))
            if process.returncode:
                raise SystemExit("Stopped at " + name + ". See results/run.log.")
    # Publish the complete run status, including publication itself.
    from report_builder import publish

    publish()
    verify_results()
    print("Finished. Read results/REPORT.md.", flush=True)


if __name__ == "__main__":
    main()

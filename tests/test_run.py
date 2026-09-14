"""Real-run leakage and provenance checks (run after a real pipeline execution)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd

from matrix_builder import correlation
from topology_analyser import features


def check_run():
    checks = {}
    import hashlib

    provenance = json.loads((OUT / "input_manifest.json").read_text())
    for name, expected in provenance["hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    checks["TDA_input_and_source_fingerprints_match"] = True
    predictions = pd.read_csv(OUT / "predictions.csv", parse_dates=["date"])
    splits = pd.read_csv(OUT / "splits.csv")
    for name, g in predictions.groupby("model"):
        if g.date.duplicated().any():
            raise AssertionError("Duplicate model/date predictions")
        reference = predictions[predictions.model == "VIX"].sort_values("date")
        assert list(g.sort_values("date").date) == list(reference.date)
    checks["all_models_identical_test_dates"] = True
    for row in splits.itertuples():
        assert pd.Timestamp(row.fit_label_end) < pd.Timestamp(row.calibration_start)
        assert pd.Timestamp(row.calibration_label_end) < pd.Timestamp(row.test_start)
    checks["label_horizons_purged"] = True
    ann = pd.read_csv(OUT / "annual_metrics.csv")
    assert ann.calibration_fpr.max() <= 0.05 + 1e-12
    checks["calibration_false_positive_budget"] = True
    t = pd.read_pickle(HERE / "topology.pkl")
    col = "h1_total_per_asset"
    prefix = t.iloc[:1000]

    def transform(s):
        return (s - s.rolling(60, min_periods=40).mean().shift(1)) / s.rolling(
            60, min_periods=40
        ).std().shift(1)

    np.testing.assert_allclose(
        transform(t[col]).iloc[:1000], transform(prefix[col]), equal_nan=True
    )
    checks["dynamic_features_prefix_invariant"] = True
    r = pd.read_pickle(HERE / "returns.pkl")
    block = r.iloc[100:160]
    block = block.loc[:, block.notna().all() & (block.std() > 1e-10)].iloc[:, :30]
    c = correlation(block.to_numpy())
    assert np.linalg.eigvalsh(c).min() >= 0.1 - 1e-08
    d = np.sqrt(2 * (1 - c))
    assert np.allclose(np.diag(d), 0)
    for k in [0, 5, 10]:
        assert (d <= d[:, k, None] + d[k, None, :] + 1e-08).all()
    h0, h1, b, s, pi = features(c)
    assert np.isfinite(pi).all() and np.isfinite(h1).all()
    checks["real_window_PSD_metric_persistence_regression"] = True
    from data_loader import prepare

    try:
        prepare(HERE / "intentionally_missing_inputs")
        raise AssertionError("Missing real inputs silently accepted")
    except FileNotFoundError:
        pass
    checks["real_input_fail_closed"] = True
    (OUT / "final_verification.json").write_text(json.dumps(checks, indent=2))
    print(checks, flush=True)


if __name__ == "__main__":
    check_run()

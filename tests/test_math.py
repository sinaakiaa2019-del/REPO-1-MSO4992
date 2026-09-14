"""Focused mathematics checks: correlation, topology and ML-side math."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd

from matrix_builder import correlation
from topology_analyser import w2, features, GRID
from model_training import alarm_threshold, metrics


def check_math():
    checks = {}
    from robustness_builder import drawdown_events
    from sector_builder import delay_summary
    test_price=pd.Series([100.,99.,89.,80.,101.,90.,102.],index=pd.bdate_range('2000-01-03',periods=7))
    events=drawdown_events(test_price)
    assert len(events)==2
    assert events.iloc[0].onset==test_price.index[2] and events.iloc[0].recovery==test_price.index[4]
    assert not events.censored.any()
    sample=np.random.default_rng(7).normal(size=60)
    np.testing.assert_allclose(delay_summary(sample),delay_summary(sample*3+2),rtol=1e-5,atol=1e-7)
    assert all(np.isnan(delay_summary(np.r_[sample[:-1],np.nan])))
    checks['mechanical_events_and_delay_embedding_invariance']=True
    from topology_analyser import landscape_top1, network_summary
    grid = np.array([0., 0.5, 1., 1.5, 2.])
    np.testing.assert_allclose(landscape_top1(np.array([[0., 2.]]), grid), [0., .5, 1., .5, 0.])
    np.testing.assert_allclose(landscape_top1(np.empty((0, 2)), grid), np.zeros(5))
    np.testing.assert_allclose(network_summary(np.eye(3)), [0., 0.], atol=1e-12)
    np.testing.assert_allclose(network_summary(np.ones((3, 3))), [1., 1.5], atol=1e-12)
    checks['landscape_and_network_known_examples'] = True
    from feature_builder import enhanced_features
    from model_training import research_train_index, research_select

    research_dates = pd.bdate_range("2000-01-03", periods=300)
    research_end = pd.Series(research_dates, index=research_dates).shift(-20)
    indices = research_train_index(research_end, research_dates[200], 50)
    assert len(indices) == 50 and indices[-1] == 179
    assert research_end.iloc[indices].max() < research_dates[200]
    research_frame = pd.DataFrame(
        {
            "x": np.arange(300) / 300,
            "label": np.r_[np.zeros(220), np.ones(40), np.zeros(40)],
            "label_end": research_end,
        },
        index=research_dates,
    ).iloc[:280]
    candidates = [
        dict(id=str(c), group="test", columns=["x"], C=c, history=50)
        for c in [0.1, 1.0]
    ]
    best, records = research_select(research_frame, candidates)
    assert all(r["inner_n"] == 168 and r["inner_positives"] == 40 for r in records)
    assert best["inner_ap"] == max(r["inner_ap"] for r in records)
    checks["research_purge_rolling_history_and_single_class_validation"] = True

    dates = pd.bdate_range("2000-01-03", periods=350)
    price = pd.Series(100 + np.arange(350) * 0.1, index=dates)
    vix = pd.Series(20 + np.sin(np.arange(350)), index=dates)
    coverage = pd.DataFrame({"eligible": 450, "expected_members": 500}, index=dates)
    full = enhanced_features(price, vix, coverage)
    prefix = enhanced_features(price.iloc[:300], vix.iloc[:300], coverage.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], prefix)
    checks["enhanced_features_prefix_invariant"] = True
    broken = price.copy()
    broken.iloc[100] = np.nan
    returns = np.log(broken).diff()
    assert pd.isna(returns.iloc[100]) and pd.isna(returns.iloc[101])
    checks["missing_price_never_bridged"] = True
    a = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [4.0, 1.0, 2.0], [3.0, 5.0, 6.0]])
    c = correlation(a)
    assert np.allclose(c, c.T) and np.allclose(np.diag(c), 1)
    assert np.linalg.eigvalsh(c).min() >= 0.1 - 1e-10
    assert np.allclose(correlation(a + 100), c)
    checks["centered_PSD_correlation"] = True
    try:
        correlation(np.array([[1.0, np.nan], [2.0, 3.0]]))
        raise RuntimeError("Missing data accepted")
    except AssertionError:
        pass
    checks["missing_returns_rejected"] = True
    assert w2(np.empty((0, 2)), np.empty((0, 2))) == 0
    assert abs(w2(np.array([[0.0, 2.0]]), np.empty((0, 2))) - np.sqrt(2)) < 1e-10
    checks["W2_diagonal_and_identity"] = True
    h0, h1, betti, s, image = features(c)
    assert betti.shape == (200, 2) and image.shape == (20, 20)
    assert (
        abs(
            s["h1_integrated_l2_per_asset"]
            - np.sqrt(np.trapezoid(betti[:, 1] ** 2, GRID)) / 3
        )
        < 1e-10
    )
    assert image.min() >= 0 and np.isfinite(image).all()
    checks["topology_grid_and_integrated_norm"] = True
    y = np.array([0] * 100 + [1] * 10)
    p = np.linspace(0, 1, 110)
    t = alarm_threshold(y, p)
    assert (p[y == 0] > t).mean() <= 0.05
    assert not (np.ones(20) > alarm_threshold(np.zeros(20), np.ones(20))).any()
    checks["false_alarm_budget_and_ties"] = True
    m = metrics([0, 0, 1, 0, 0], [0.9, 0.9, 0.9, 0.1, 0.9], [1, 1, 1, 0, 1])
    assert m["alarm_episodes"] == 2 and m["false_alarm_episodes"] == 1
    checks["alarm_episode_accounting"] = True
    (OUT / "mathematical_tests.json").write_text(json.dumps(checks, indent=2))
    print(checks)


def test_mathematics():
    check_math()


if __name__ == "__main__":
    check_math()

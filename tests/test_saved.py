"""Verify the shipped balanced/sector/transfer evidence without TDA caches."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd

from model_training import alarm_threshold, metrics


def check_saved():
    """Verify the shipped balanced/sector/transfer evidence without TDA caches."""
    import sqlite3
    from robustness_builder import drawdown_events, event_leads

    checks = {}
    with sqlite3.connect(ROOT / 'results/evidence.sqlite') as con:
        def read(name):
            return pd.read_sql_query('SELECT * FROM "' + name + '"', con)

        for prefix, extra in [('defence', []), ('rq5', ['symbol'])]:
            pred = read(prefix + '_predictions')
            cal = read(prefix + '_calibration')
            saved = read(prefix + '_metrics').set_index(extra + ['model', 'budget'])
            pred.date = pd.to_datetime(pred.date)
            reference = None
            for key, group in pred.groupby(extra + ['model', 'budget']):
                group = group.sort_values('date')
                dates = group.date.tolist()
                if reference is None:
                    reference = dates
                assert dates == reference and not group.date.duplicated().any()
                assert set(group.year) == set(range(2007, 2025))
                for field, value in metrics(group.label, group.score, group.alarm).items():
                    assert value is None or abs(saved.loc[key, field] - value) < 1e-10
                info = dict(zip(extra + ['model', 'budget'], key))
                for year, annual in group.groupby('year'):
                    c = cal[(cal.model == info['model']) & (cal.year == year)]
                    for field in extra:
                        c = c[c[field] == info[field]]
                    assert len(c) == 252
                    cut = alarm_threshold(c.label, c.score, info['budget'])
                    assert np.allclose(annual.threshold, cut, atol=1e-12)
                    assert ((annual.score > annual.threshold) == annual.alarm.astype(bool)).all()
                    assert (c.loc[c.label == 0, 'score'] > cut).mean() <= info['budget'] + 1e-12
            splits = read(prefix + '_splits')
            assert (splits.fit_label_end < splits.calibration_start).all()
            assert (splits.calibration_label_end < splits.test_start).all()
            tuning, selected = read(prefix + '_tuning'), read(prefix + '_selected')
            expected = 18 * (9 if prefix == 'defence' else 9 * 3)
            assert len(selected) == expected and len(tuning) == 3 * expected
            for row in selected.itertuples():
                candidates = tuning[(tuning.year == row.year) & (tuning['group'] == row.model)]
                for field in extra:
                    candidates = candidates[candidates[field] == getattr(row, field)]
                assert len(candidates) == 3 and set(candidates.C) == {.01, .1, 1.0}
                best = candidates.iloc[candidates.inner_ap.fillna(-1).to_numpy().argmax()]
                assert row.C == best.C
            # The calendar and saved predictions independently account for every
            # event warning; missing detections must remain in the output.
            spy = pd.read_csv(ROOT / 'inputs/markets.csv', index_col=0, parse_dates=True).SPY.dropna()
            events = drawdown_events(spy)
            columns = ('model', 'budget') if not extra else ('symbol', 'sector_proxy', 'model', 'budget')
            actual = event_leads(pred, events, spy.index, columns)
            previous = read(prefix + '_event_leads')
            keys = list(columns) + ['event']
            actual = actual.set_index(keys).sort_index()
            previous = previous.set_index(keys).sort_index()
            assert actual.index.equals(previous.index)
            np.testing.assert_allclose(actual.lead_sessions, previous.lead_sessions, equal_nan=True)
            assert np.array_equal(actual.alarm_days, previous.alarm_days)
            checks[prefix + '_metrics_dates_calibration_purges_selection_events'] = True

        pred = read('defence_transfer_predictions')
        saved = read('defence_transfer_metrics').set_index(['universe', 'model'])
        for universe, universe_rows in pred.groupby('universe'):
            reference = None
            for name, group in universe_rows.groupby('model'):
                group = group.sort_values('date')
                dates = group.date.tolist()
                if reference is None:
                    reference = dates
                assert dates == reference and not group.date.duplicated().any()
                assert ((group.score > group.threshold) == group.alarm.astype(bool)).all()
                for field, value in metrics(group.label, group.score, group.alarm).items():
                    assert value is None or abs(saved.loc[(universe, name), field] - value) < 1e-10
        checks['transfer_metrics_shared_dates_and_alarms'] = True
        for table, metric_table, extra in [
            ('defence_comparisons', 'defence_metrics', []),
            ('rq5_comparisons', 'rq5_metrics', ['symbol']),
            ('defence_transfer_comparisons', 'defence_transfer_metrics', ['universe']),
        ]:
            scores = read(metric_table)
            if 'budget' in scores:
                scores = scores[scores.budget == .05]
            scores = scores.set_index(extra + ['model'])
            for row in read(table).itertuples():
                a = tuple(getattr(row, field) for field in extra) + (row.model,)
                b = tuple(getattr(row, field) for field in extra) + (row.reference,)
                if not extra:
                    a, b = a[0], b[0]
                assert abs(row.ap_difference - (scores.loc[a, 'average_precision'] - scores.loc[b, 'average_precision'])) < 1e-10
                assert 0 < row.draws <= 500 and row.lower95 <= row.upper95
        checks['paired_comparison_point_estimates_and_interval_contracts'] = True
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    check_saved()

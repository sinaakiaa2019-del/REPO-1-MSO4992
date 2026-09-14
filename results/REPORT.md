# Revised real-data run

Daily topology windows: 6,229; eligible assets: 295–501. No liquidity cap.

## Matched-date model results

| model | auroc | average_precision | false_positive_rate | recall | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| LR_base | 0.6292 | 0.2049 | 0.1180 | 0.2995 | 97 |
| LR_combined | 0.6050 | 0.2043 | 0.0923 | 0.2538 | 98 |
| LR_dynamic_combined | 0.5908 | 0.1689 | 0.0869 | 0.2335 | 102 |
| LR_enhanced_base | 0.3868 | 0.0370 | 0.1060 | 0.1675 | 80 |
| LR_enhanced_combined | 0.3980 | 0.0375 | 0.0835 | 0.1574 | 91 |
| LR_topology | 0.4410 | 0.0428 | 0.1036 | 0.1218 | 103 |
| VIX | 0.7440 | 0.2198 | 0.1595 | 0.4010 | 73 |
| XGB_base | 0.6703 | 0.0779 | 0.1493 | 0.2183 | 56 |
| XGB_combined | 0.6869 | 0.0794 | 0.1530 | 0.2437 | 75 |
| XGB_enhanced_base | 0.6528 | 0.0768 | 0.1384 | 0.4822 | 42 |
| XGB_enhanced_combined | 0.6840 | 0.0776 | 0.1449 | 0.4416 | 38 |
| XGB_full_topology | 0.6964 | 0.0752 | 0.0846 | 0.1066 | 103 |
| XGB_topology | 0.6741 | 0.0679 | 0.0992 | 0.1117 | 101 |
| constant | 0.4249 | 0.0445 | 0.0000 | 0.0000 | 0 |

## Incremental topology value: paired 60-session block bootstrap

| model | reference | delta_average_precision | lower95 | upper95 | bootstrap_samples | block_sessions |
| --- | --- | --- | --- | --- | --- | --- |
| LR_combined | LR_base | -0.0007 | -0.0198 | 0.0189 | 500 | 60 |
| XGB_combined | XGB_base | 0.0015 | -0.0156 | 0.0204 | 500 | 60 |
| LR_dynamic_combined | LR_base | -0.0360 | -0.0901 | 0.0012 | 500 | 60 |
| LR_combined | VIX | -0.0156 | -0.0644 | 0.0264 | 500 | 60 |
| LR_enhanced_combined | LR_enhanced_base | 0.0005 | -0.0177 | 0.0132 | 500 | 60 |
| XGB_enhanced_combined | XGB_enhanced_base | 0.0008 | -0.0375 | 0.0111 | 500 | 60 |
| LR_enhanced_base | LR_base | -0.1679 | -0.3331 | -0.0051 | 500 | 60 |
| XGB_enhanced_base | XGB_base | -0.0011 | -0.0449 | 0.0449 | 500 | 60 |

## Event lead times

| event | peak_date | model | first_alarm | lead_sessions | alarm_days | covered_sessions |
| --- | --- | --- | --- | --- | --- | --- |
| GFC | 2007-10-09 | LR_base | 2007-08-10 00:00:00 | 41.0000 | 8 | 60 |
| GFC | 2007-10-09 | LR_combined | 2007-09-10 00:00:00 | 21.0000 | 1 | 60 |
| GFC | 2007-10-09 | LR_dynamic_combined | 2007-09-10 00:00:00 | 21.0000 | 3 | 60 |
| GFC | 2007-10-09 | LR_enhanced_base | NaT | nan | 0 | 60 |
| GFC | 2007-10-09 | LR_enhanced_combined | NaT | nan | 0 | 60 |
| GFC | 2007-10-09 | LR_topology | NaT | nan | 0 | 60 |
| GFC | 2007-10-09 | VIX | 2007-07-25 00:00:00 | 53.0000 | 50 | 60 |
| GFC | 2007-10-09 | XGB_base | 2007-08-06 00:00:00 | 45.0000 | 20 | 60 |
| GFC | 2007-10-09 | XGB_combined | 2007-08-06 00:00:00 | 45.0000 | 20 | 60 |
| GFC | 2007-10-09 | XGB_enhanced_base | 2007-08-14 00:00:00 | 39.0000 | 4 | 60 |
| GFC | 2007-10-09 | XGB_enhanced_combined | 2007-08-14 00:00:00 | 39.0000 | 3 | 60 |
| GFC | 2007-10-09 | XGB_full_topology | NaT | nan | 0 | 60 |
| GFC | 2007-10-09 | XGB_topology | NaT | nan | 0 | 60 |
| GFC | 2007-10-09 | constant | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | LR_base | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | LR_combined | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | LR_dynamic_combined | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | LR_enhanced_base | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | LR_enhanced_combined | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | LR_topology | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | VIX | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | XGB_base | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | XGB_combined | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | XGB_enhanced_base | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | XGB_enhanced_combined | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | XGB_full_topology | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | XGB_topology | NaT | nan | 0 | 60 |
| COVID | 2020-02-19 | constant | NaT | nan | 0 | 60 |

## Interpretation and limitations

These are retrospective development results on an already inspected dataset, not an untouched final test. The prior release is retained separately. Compare the matched models within this run; changes from older releases can also reflect different training histories.

No liquidity cap is used. A complete 60-SPY-session return window is required per included asset. Weighted centering and 10% identity shrinkage replace zero-filled uncentered covariance. This removes missing-return imputation but excludes more incomplete histories. Window length, decay and shrinkage were fixed before results; they were not optimised on test outcomes.

The 5% budget constrains negative-day false positives in the separate pre-test calibration year. It does not guarantee 5% on future data. Daily false alarms, alarm episodes and time under warning are distinct quantities. Thresholds are not optimised against outer test outcomes. Consecutive positive-label episodes are not independent economic crises.

Two hyperparameter candidates per learner are selected with two purged chronological inner validation blocks; final fitting ends before calibration begins. Calibration has 252 observations with known 20-session outcomes. Scaling is fitted inside training only. No oversampling or synthetic training observations are used. VIX and yield slope are lagged one session; market-price features are available after the close.

PR-AUC here means average precision. The headline scores pool saved annual predictions on identical dates; annual and two-period tables are also supplied. Block-bootstrap intervals are approximate and do not remove the dependence on a small number of crises. Searching several feature/model families adds selection uncertainty.

Permanent security IDs, historical sector classifications, full foreign/credit constituent panels, and historical macro vintages remain unavailable. These public-data limits are documented. No current-survivor sector result is promoted to historical sector leadership. RQ4 baskets, if reported, are exploratory. RQ5 remains unsupported. Current CIK references are not asserted to be historical security identifiers. No ticker renaming is guessed.

Do not use the legacy results as the new run's outputs. Detailed tables are stored in evidence.sqlite. Export any table with python run_all.py --export-table TABLE_NAME. Window caches are keyed by code, protocol and input hashes.

Data-quality review flagged 755 absolute daily log returns above 0.5 across the historical price panel; 123 occurred while the symbol was an index member. These are review flags, not proven errors. They remain uncorrected without reliable corporate-action/security-identity evidence. See extreme_returns_membership_review.csv. This is a material limit on empirical conclusions.


## Remaining research questions

RQ2: training-selected epsilon ranges from 0.9347 to 0.9749; annual selections are in filtration_selected.csv.

RQ4: exploratory frozen-model transfer on small real baskets; these are not full constituent-panel findings.

| universe | model | n | auroc | average_precision | recall |
| --- | --- | --- | --- | --- | --- |
| credit_bond_ETFs | LR_topology | 4383 | 0.7119 | 0.0733 | 0.7473 |
| credit_bond_ETFs | XGB_topology | 4383 | 0.5946 | 0.0414 | 0.4725 |
| international_indices | LR_topology | 1394 | 0.8935 | 0.1344 | 1.0000 |
| international_indices | XGB_topology | 1394 | 0.5615 | 0.0379 | 0.0000 |

RQ5: unavailable without dated historical sector classifications.

## Interpretation

The highest pooled average precision is 0.2198 (VIX). This is a descriptive ranking of retrospective experiments, not an independently selected winner.

Check the paired differences above to judge whether additional features helped the same learner. An interval containing zero does not establish an improvement. The intervals are not adjusted for testing multiple models. The calibration false-alarm budget is not a guarantee on later years.

## Historical asset coverage

All 915 supplied symbols were audited; 794 entered at least one eligible index window. 294 used symbols were no longer members at the end of 2024. 209 membership symbols had no punctuation-equivalent price symbol. The asset_coverage.csv table gives the reason for every unused supplied symbol. These figures describe ticker coverage, not verified unique economic securities. Membership handling reduces survivorship bias but cannot eliminate missing-history, identity or delisting-return bias.

See ../METHODOLOGY.md for definitions and sources. The database table_index lists every detailed table.

# Full persistence images as the topology representation (post-hoc, phase 5)

Motivated by Adams et al. (2017): persistence images keep more diagram information than the three hand-picked summaries the primary study uses. Phase 3 already showed XGBoost extracts real value from the compact set; this asks whether the full 400-value image does more, on the same crash-day target.

## Primary result (5% calibration budget, 10% loss target)

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| VIX | 0.7440 | 0.2198 | 0.5958 | 0.2471 | 73 |
| combined | 0.7538 | 0.1805 | 0.5932 | 0.2095 | 67 |
| market | 0.7592 | 0.1781 | 0.5860 | 0.2089 | 47 |
| combined_pi_xgb | 0.7196 | 0.1045 | 0.6014 | 0.2014 | 73 |
| topology_pi_xgb | 0.7101 | 0.0889 | 0.5897 | 0.2001 | 98 |
| topology | 0.5825 | 0.0513 | 0.5680 | 0.1819 | 122 |

## Paired AP differences (60-session blocks)

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_pi_xgb | topology | 0.0376 | -0.0134 | 0.1252 | 0.9140 |
| topology_pi_xgb | market | -0.0892 | -0.1773 | -0.0068 | 0.0140 |
| combined_pi_xgb | market | -0.0736 | -0.1504 | -0.0040 | 0.0100 |
| combined_pi_xgb | combined | -0.0760 | -0.1475 | -0.0027 | 0.0080 |

## Mechanical drawdown event recall (5% budget)

| model | events | detected | median_lead |
| --- | --- | --- | --- |
| combined | 6 | 5 | 14.0000 |
| combined_pi_xgb | 6 | 5 | 3.0000 |
| market | 6 | 6 | 12.0000 |
| topology | 6 | 3 | 13.0000 |
| topology_pi_xgb | 6 | 2 | 11.0000 |


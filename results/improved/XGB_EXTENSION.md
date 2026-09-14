# XGBoost topology extension (post-hoc, phase 3)

Added after phases 1-2, motivated by the published classic ladder (XGB_topology already beats LR_topology there). This runs that comparison under the study's rigorous purged protocol, with its own evaluate_groups() call so phase 1's baseline reproduction stays untouched. `market_xgb` is a control.

## Results

### Target: SPY loss of at least 10.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| VIX | 0.7440 | 0.2198 | 0.5958 | 0.2471 | 73 |
| combined | 0.7538 | 0.1805 | 0.5932 | 0.2095 | 67 |
| market | 0.7592 | 0.1781 | 0.5860 | 0.2089 | 47 |
| market_xgb | 0.6622 | 0.0762 | 0.5006 | 0.1791 | 45 |
| combined_xgb | 0.6604 | 0.0718 | 0.4908 | 0.1618 | 66 |
| topology_xgb | 0.6374 | 0.0688 | 0.5578 | 0.1870 | 145 |
| topology | 0.5825 | 0.0513 | 0.5680 | 0.1819 | 122 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_xgb | topology | 0.0175 | -0.0067 | 0.0444 | 0.9380 |
| market_xgb | market | -0.1019 | -0.2392 | -0.0175 | 0.0000 |
| combined_xgb | market_xgb | -0.0044 | -0.0237 | 0.0073 | 0.2280 |
| combined_xgb | combined | -0.1087 | -0.2548 | -0.0233 | 0.0000 |
| topology_xgb | topology | 0.0175 | -0.0095 | 0.0567 | 0.8880 |
| market_xgb | market | -0.1019 | -0.2201 | -0.0026 | 0.0160 |
| combined_xgb | market_xgb | -0.0044 | -0.0279 | 0.0102 | 0.2260 |
| combined_xgb | combined | -0.1087 | -0.2214 | -0.0101 | 0.0000 |
| topology_xgb | topology | 0.0175 | -0.0108 | 0.0684 | 0.8480 |
| market_xgb | market | -0.1019 | -0.2069 | 0.0045 | 0.0360 |
| combined_xgb | market_xgb | -0.0044 | -0.0283 | 0.0109 | 0.2480 |
| combined_xgb | combined | -0.1087 | -0.2061 | -0.0061 | 0.0040 |

### Target: SPY loss of at least 7.5% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| market | 0.7056 | 0.2303 | 0.4999 | 0.2391 | 47 |
| VIX | 0.6927 | 0.2277 | 0.5011 | 0.2524 | 62 |
| combined | 0.6942 | 0.2267 | 0.4961 | 0.2395 | 53 |
| combined_xgb | 0.6738 | 0.1402 | 0.4523 | 0.2057 | 47 |
| market_xgb | 0.6621 | 0.1305 | 0.4479 | 0.2197 | 40 |
| topology_xgb | 0.6251 | 0.1296 | 0.5345 | 0.2504 | 125 |
| topology | 0.6048 | 0.1084 | 0.5399 | 0.2618 | 84 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_xgb | topology | 0.0212 | -0.0043 | 0.0537 | 0.9520 |
| market_xgb | market | -0.0998 | -0.1985 | -0.0187 | 0.0040 |
| combined_xgb | market_xgb | 0.0097 | -0.0101 | 0.0355 | 0.8500 |
| combined_xgb | combined | -0.0865 | -0.1845 | -0.0060 | 0.0140 |
| topology_xgb | topology | 0.0212 | -0.0058 | 0.0695 | 0.9040 |
| market_xgb | market | -0.0998 | -0.1873 | -0.0030 | 0.0180 |
| combined_xgb | market_xgb | 0.0097 | -0.0092 | 0.0403 | 0.8340 |
| combined_xgb | combined | -0.0865 | -0.1610 | 0.0022 | 0.0340 |
| topology_xgb | topology | 0.0212 | -0.0090 | 0.0817 | 0.8620 |
| market_xgb | market | -0.0998 | -0.1820 | 0.0025 | 0.0320 |
| combined_xgb | market_xgb | 0.0097 | -0.0147 | 0.0437 | 0.7720 |
| combined_xgb | combined | -0.0865 | -0.1561 | 0.0039 | 0.0500 |

### Target: SPY loss of at least 5.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| market | 0.6599 | 0.3282 | 0.4834 | 0.3244 | 36 |
| combined | 0.6533 | 0.3260 | 0.5025 | 0.3340 | 42 |
| VIX | 0.6582 | 0.3259 | 0.4919 | 0.3343 | 46 |
| combined_xgb | 0.6476 | 0.2726 | 0.4889 | 0.3241 | 47 |
| market_xgb | 0.6284 | 0.2539 | 0.4575 | 0.3126 | 41 |
| topology_xgb | 0.5803 | 0.2524 | 0.5122 | 0.3335 | 112 |
| topology | 0.5408 | 0.2124 | 0.4931 | 0.3323 | 79 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_xgb | topology | 0.0400 | -0.0102 | 0.0984 | 0.9240 |
| market_xgb | market | -0.0743 | -0.1398 | -0.0106 | 0.0080 |
| combined_xgb | market_xgb | 0.0186 | -0.0070 | 0.0480 | 0.9080 |
| combined_xgb | combined | -0.0534 | -0.1104 | 0.0029 | 0.0400 |
| topology_xgb | topology | 0.0400 | -0.0279 | 0.1249 | 0.8520 |
| market_xgb | market | -0.0743 | -0.1421 | 0.0053 | 0.0460 |
| combined_xgb | market_xgb | 0.0186 | -0.0099 | 0.0569 | 0.8880 |
| combined_xgb | combined | -0.0534 | -0.1088 | 0.0117 | 0.0620 |
| topology_xgb | topology | 0.0400 | -0.0358 | 0.1399 | 0.7740 |
| market_xgb | market | -0.0743 | -0.1460 | 0.0101 | 0.0540 |
| combined_xgb | market_xgb | 0.0186 | -0.0137 | 0.0575 | 0.8180 |
| combined_xgb | combined | -0.0534 | -0.1015 | 0.0082 | 0.0640 |

## Multiple-testing correction across all three phases

Benjamini-Hochberg 5% FDR control across phases 1-3's 60-session-block comparisons (57 total). 0 survive.

No comparison in the full three-phase family survives 5% FDR control.


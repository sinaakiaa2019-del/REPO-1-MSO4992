# Random Forest and Lasso: a third and fourth learner family (post-hoc, phase 7)

Motivated by phases 3 and 5, where XGBoost beat linear regression on topology but still trailed market overall: is that gain about tree ensembles generally (Random Forest, a bagging family distinct from XGBoost's boosting), or about feature selection on sparse columns (Lasso, since most persistence-image pixels are exactly zero most days -- phase 6's finding)? `market_rf`/`market_lasso` are controls, same logic as phase 3's `market_xgb`. Same purged three-candidate protocol as every other phase.

## Reference: phase 5's XGBoost result on persistence images (different run, not paired here)

| model | auroc | average_precision | false_alarm_episodes |
| --- | --- | --- | --- |
| combined_pi_xgb | 0.7196 | 0.1045 | 73 |
| topology_pi_xgb | 0.7101 | 0.0889 | 98 |

## Results

### Target: SPY loss of at least 10.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| VIX | 0.7440 | 0.2198 | 0.5958 | 0.2471 | 73 |
| combined | 0.7538 | 0.1805 | 0.5932 | 0.2095 | 67 |
| market | 0.7592 | 0.1781 | 0.5860 | 0.2089 | 47 |
| topology_rf | 0.6752 | 0.1028 | 0.5905 | 0.2355 | 134 |
| combined_rf | 0.7172 | 0.0973 | 0.5162 | 0.1491 | 62 |
| combined_pi_lasso | 0.7332 | 0.0922 | 0.5813 | 0.1944 | 35 |
| combined_pi_rf | 0.7189 | 0.0911 | 0.5889 | 0.1850 | 54 |
| market_lasso | 0.6900 | 0.0843 | 0.5526 | 0.1966 | 51 |
| topology_pi_lasso | 0.6996 | 0.0816 | 0.4803 | 0.1242 | 30 |
| combined_lasso | 0.6844 | 0.0795 | 0.5583 | 0.1924 | 70 |
| topology_pi_rf | 0.6892 | 0.0781 | 0.5825 | 0.1818 | 57 |
| topology_lasso | 0.6303 | 0.0757 | 0.5061 | 0.1245 | 31 |
| market_rf | 0.6727 | 0.0746 | 0.4622 | 0.1657 | 50 |
| topology | 0.5825 | 0.0513 | 0.5680 | 0.1819 | 122 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_rf | topology | 0.0514 | -0.0006 | 0.1613 | 0.9720 |
| market_rf | market | -0.1035 | -0.2479 | -0.0205 | 0.0000 |
| combined_rf | market_rf | 0.0227 | -0.0057 | 0.0651 | 0.9500 |
| topology_lasso | topology | 0.0244 | -0.0136 | 0.0799 | 0.8980 |
| market_lasso | market | -0.0939 | -0.2312 | -0.0081 | 0.0100 |
| combined_lasso | market_lasso | -0.0047 | -0.0173 | 0.0043 | 0.2260 |
| topology_rf | topology | 0.0514 | -0.0072 | 0.1826 | 0.9480 |
| market_rf | market | -0.1035 | -0.2296 | -0.0090 | 0.0020 |
| combined_rf | market_rf | 0.0227 | -0.0070 | 0.0778 | 0.9020 |
| topology_lasso | topology | 0.0244 | -0.0190 | 0.1130 | 0.8460 |
| market_lasso | market | -0.0939 | -0.2098 | 0.0101 | 0.0600 |
| combined_lasso | market_lasso | -0.0047 | -0.0221 | 0.0088 | 0.3240 |
| topology_rf | topology | 0.0514 | -0.0088 | 0.1974 | 0.9040 |
| market_rf | market | -0.1035 | -0.2093 | -0.0027 | 0.0160 |
| combined_rf | market_rf | 0.0227 | -0.0115 | 0.0826 | 0.8400 |
| topology_lasso | topology | 0.0244 | -0.0194 | 0.1384 | 0.8100 |
| market_lasso | market | -0.0939 | -0.1783 | 0.0130 | 0.1020 |
| combined_lasso | market_lasso | -0.0047 | -0.0194 | 0.0095 | 0.2880 |
| topology_pi_rf | market | -0.1001 | -0.2405 | -0.0153 | 0.0060 |
| combined_pi_rf | market | -0.0870 | -0.2196 | -0.0113 | 0.0080 |
| combined_pi_rf | combined | -0.0894 | -0.2318 | -0.0091 | 0.0080 |
| topology_pi_lasso | market | -0.0965 | -0.2340 | -0.0126 | 0.0160 |
| combined_pi_lasso | market | -0.0860 | -0.2139 | -0.0134 | 0.0040 |
| combined_pi_lasso | combined | -0.0883 | -0.2231 | -0.0111 | 0.0040 |
| topology_pi_lasso | topology_pi_rf | 0.0035 | -0.0155 | 0.0281 | 0.6580 |
| topology_pi_rf | market | -0.1001 | -0.2044 | -0.0047 | 0.0160 |
| combined_pi_rf | market | -0.0870 | -0.1827 | -0.0025 | 0.0180 |
| combined_pi_rf | combined | -0.0894 | -0.1829 | -0.0001 | 0.0220 |
| topology_pi_lasso | market | -0.0965 | -0.2010 | -0.0018 | 0.0220 |
| combined_pi_lasso | market | -0.0860 | -0.1826 | -0.0064 | 0.0060 |
| combined_pi_lasso | combined | -0.0883 | -0.1837 | -0.0018 | 0.0180 |
| topology_pi_lasso | topology_pi_rf | 0.0035 | -0.0178 | 0.0307 | 0.7160 |
| topology_pi_rf | market | -0.1001 | -0.1949 | 0.0021 | 0.0320 |
| combined_pi_rf | market | -0.0870 | -0.1758 | 0.0036 | 0.0460 |
| combined_pi_rf | combined | -0.0894 | -0.1718 | 0.0036 | 0.0520 |
| topology_pi_lasso | market | -0.0965 | -0.1828 | 0.0044 | 0.0500 |
| combined_pi_lasso | market | -0.0860 | -0.1692 | -0.0030 | 0.0100 |
| combined_pi_lasso | combined | -0.0883 | -0.1672 | -0.0002 | 0.0220 |
| topology_pi_lasso | topology_pi_rf | 0.0035 | -0.0212 | 0.0330 | 0.6600 |

### Target: SPY loss of at least 7.5% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| market | 0.7056 | 0.2303 | 0.4999 | 0.2391 | 47 |
| VIX | 0.6927 | 0.2277 | 0.5011 | 0.2524 | 62 |
| combined | 0.6942 | 0.2267 | 0.4961 | 0.2395 | 53 |
| market_lasso | 0.6940 | 0.1931 | 0.4817 | 0.2292 | 56 |
| combined_lasso | 0.6916 | 0.1915 | 0.4822 | 0.2285 | 60 |
| combined_pi_lasso | 0.6389 | 0.1640 | 0.4796 | 0.2480 | 37 |
| topology_rf | 0.6489 | 0.1622 | 0.5601 | 0.2589 | 119 |
| combined_pi_rf | 0.6645 | 0.1492 | 0.5253 | 0.2465 | 61 |
| combined_rf | 0.6747 | 0.1459 | 0.4512 | 0.1982 | 53 |
| topology_pi_rf | 0.6418 | 0.1411 | 0.5281 | 0.2332 | 55 |
| topology_pi_lasso | 0.6528 | 0.1330 | 0.5559 | 0.2541 | 43 |
| market_rf | 0.6389 | 0.1222 | 0.4058 | 0.1872 | 39 |
| topology_lasso | 0.6102 | 0.1147 | 0.4691 | 0.2045 | 54 |
| topology | 0.6048 | 0.1084 | 0.5399 | 0.2618 | 84 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_rf | topology | 0.0538 | 0.0046 | 0.1229 | 0.9860 |
| market_rf | market | -0.1080 | -0.2076 | -0.0281 | 0.0000 |
| combined_rf | market_rf | 0.0237 | -0.0010 | 0.0526 | 0.9700 |
| topology_lasso | topology | 0.0063 | -0.0100 | 0.0290 | 0.7700 |
| market_lasso | market | -0.0372 | -0.0993 | 0.0103 | 0.0760 |
| combined_lasso | market_lasso | -0.0016 | -0.0164 | 0.0146 | 0.3480 |
| topology_rf | topology | 0.0538 | -0.0018 | 0.1450 | 0.9680 |
| market_rf | market | -0.1080 | -0.1943 | -0.0112 | 0.0020 |
| combined_rf | market_rf | 0.0237 | -0.0031 | 0.0693 | 0.9540 |
| topology_lasso | topology | 0.0063 | -0.0098 | 0.0414 | 0.7780 |
| market_lasso | market | -0.0372 | -0.0799 | 0.0118 | 0.1480 |
| combined_lasso | market_lasso | -0.0016 | -0.0153 | 0.0129 | 0.3720 |
| topology_rf | topology | 0.0538 | -0.0038 | 0.1551 | 0.9300 |
| market_rf | market | -0.1080 | -0.1922 | -0.0068 | 0.0060 |
| combined_rf | market_rf | 0.0237 | -0.0068 | 0.0712 | 0.9080 |
| topology_lasso | topology | 0.0063 | -0.0108 | 0.0504 | 0.8020 |
| market_lasso | market | -0.0372 | -0.0605 | 0.0104 | 0.1700 |
| combined_lasso | market_lasso | -0.0016 | -0.0145 | 0.0104 | 0.3360 |
| topology_pi_rf | market | -0.0891 | -0.1884 | -0.0083 | 0.0120 |
| combined_pi_rf | market | -0.0811 | -0.1723 | -0.0106 | 0.0080 |
| combined_pi_rf | combined | -0.0775 | -0.1716 | -0.0083 | 0.0140 |
| topology_pi_lasso | market | -0.0973 | -0.2053 | -0.0054 | 0.0180 |
| combined_pi_lasso | market | -0.0663 | -0.1391 | 0.0009 | 0.0260 |
| combined_pi_lasso | combined | -0.0627 | -0.1383 | 0.0077 | 0.0460 |
| topology_pi_lasso | topology_pi_rf | -0.0082 | -0.0410 | 0.0188 | 0.2800 |
| topology_pi_rf | market | -0.0891 | -0.1726 | 0.0146 | 0.0440 |
| combined_pi_rf | market | -0.0811 | -0.1529 | 0.0025 | 0.0300 |
| combined_pi_rf | combined | -0.0775 | -0.1474 | 0.0087 | 0.0420 |
| topology_pi_lasso | market | -0.0973 | -0.2026 | 0.0180 | 0.0680 |
| combined_pi_lasso | market | -0.0663 | -0.1374 | 0.0049 | 0.0560 |
| combined_pi_lasso | combined | -0.0627 | -0.1372 | 0.0108 | 0.0760 |
| topology_pi_lasso | topology_pi_rf | -0.0082 | -0.0483 | 0.0231 | 0.2980 |
| topology_pi_rf | market | -0.0891 | -0.1673 | 0.0221 | 0.0700 |
| combined_pi_rf | market | -0.0811 | -0.1462 | 0.0106 | 0.0520 |
| combined_pi_rf | combined | -0.0775 | -0.1382 | 0.0109 | 0.0620 |
| topology_pi_lasso | market | -0.0973 | -0.2071 | 0.0260 | 0.1120 |
| combined_pi_lasso | market | -0.0663 | -0.1262 | 0.0103 | 0.0760 |
| combined_pi_lasso | combined | -0.0627 | -0.1254 | 0.0130 | 0.1120 |
| topology_pi_lasso | topology_pi_rf | -0.0082 | -0.0548 | 0.0239 | 0.3180 |

### Target: SPY loss of at least 5.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| market | 0.6599 | 0.3282 | 0.4834 | 0.3244 | 36 |
| combined | 0.6533 | 0.3260 | 0.5025 | 0.3340 | 42 |
| VIX | 0.6582 | 0.3259 | 0.4919 | 0.3343 | 46 |
| market_lasso | 0.6400 | 0.3252 | 0.4852 | 0.3270 | 35 |
| combined_lasso | 0.6151 | 0.3198 | 0.4877 | 0.3346 | 37 |
| combined_rf | 0.6441 | 0.2923 | 0.4743 | 0.3131 | 34 |
| topology_rf | 0.5881 | 0.2780 | 0.5158 | 0.3382 | 109 |
| combined_pi_rf | 0.6209 | 0.2737 | 0.5199 | 0.3572 | 37 |
| market_rf | 0.6221 | 0.2584 | 0.4326 | 0.2919 | 35 |
| topology_pi_rf | 0.5902 | 0.2553 | 0.5103 | 0.3330 | 42 |
| combined_pi_lasso | 0.5406 | 0.2470 | 0.4893 | 0.3446 | 23 |
| topology | 0.5408 | 0.2124 | 0.4931 | 0.3323 | 79 |
| topology_lasso | 0.5621 | 0.2112 | 0.4891 | 0.3272 | 47 |
| topology_pi_lasso | 0.4951 | 0.2091 | 0.4657 | 0.3112 | 38 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_rf | topology | 0.0656 | -0.0043 | 0.1467 | 0.9620 |
| market_rf | market | -0.0698 | -0.1374 | -0.0024 | 0.0240 |
| combined_rf | market_rf | 0.0339 | -0.0088 | 0.0794 | 0.9420 |
| topology_lasso | topology | -0.0013 | -0.0185 | 0.0247 | 0.5340 |
| market_lasso | market | -0.0030 | -0.0156 | 0.0127 | 0.3620 |
| combined_lasso | market_lasso | -0.0054 | -0.0227 | 0.0077 | 0.2200 |
| topology_rf | topology | 0.0656 | -0.0291 | 0.1741 | 0.9000 |
| market_rf | market | -0.0698 | -0.1375 | 0.0052 | 0.0520 |
| combined_rf | market_rf | 0.0339 | -0.0151 | 0.0916 | 0.8800 |
| topology_lasso | topology | -0.0013 | -0.0174 | 0.0229 | 0.4980 |
| market_lasso | market | -0.0030 | -0.0188 | 0.0104 | 0.3200 |
| combined_lasso | market_lasso | -0.0054 | -0.0221 | 0.0077 | 0.2680 |
| topology_rf | topology | 0.0656 | -0.0338 | 0.1826 | 0.8320 |
| market_rf | market | -0.0698 | -0.1450 | 0.0128 | 0.0720 |
| combined_rf | market_rf | 0.0339 | -0.0186 | 0.0993 | 0.8100 |
| topology_lasso | topology | -0.0013 | -0.0165 | 0.0208 | 0.4820 |
| market_lasso | market | -0.0030 | -0.0179 | 0.0101 | 0.3080 |
| combined_lasso | market_lasso | -0.0054 | -0.0220 | 0.0063 | 0.2740 |
| topology_pi_rf | market | -0.0729 | -0.1485 | 0.0015 | 0.0280 |
| combined_pi_rf | market | -0.0545 | -0.1193 | 0.0070 | 0.0480 |
| combined_pi_rf | combined | -0.0523 | -0.1202 | 0.0115 | 0.0500 |
| topology_pi_lasso | market | -0.1191 | -0.2106 | -0.0342 | 0.0020 |
| combined_pi_lasso | market | -0.0812 | -0.1702 | -0.0037 | 0.0200 |
| combined_pi_lasso | combined | -0.0790 | -0.1640 | -0.0027 | 0.0200 |
| topology_pi_lasso | topology_pi_rf | -0.0462 | -0.0936 | -0.0094 | 0.0040 |
| topology_pi_rf | market | -0.0729 | -0.1512 | 0.0148 | 0.0540 |
| combined_pi_rf | market | -0.0545 | -0.1196 | 0.0189 | 0.0660 |
| combined_pi_rf | combined | -0.0523 | -0.1167 | 0.0167 | 0.0740 |
| topology_pi_lasso | market | -0.1191 | -0.2299 | -0.0109 | 0.0140 |
| combined_pi_lasso | market | -0.0812 | -0.1825 | 0.0051 | 0.0360 |
| combined_pi_lasso | combined | -0.0790 | -0.1768 | 0.0098 | 0.0500 |
| topology_pi_lasso | topology_pi_rf | -0.0462 | -0.1086 | -0.0016 | 0.0180 |
| topology_pi_rf | market | -0.0729 | -0.1501 | 0.0207 | 0.0760 |
| combined_pi_rf | market | -0.0545 | -0.1127 | 0.0195 | 0.0740 |
| combined_pi_rf | combined | -0.0523 | -0.1081 | 0.0203 | 0.0920 |
| topology_pi_lasso | market | -0.1191 | -0.2515 | 0.0078 | 0.0400 |
| combined_pi_lasso | market | -0.0812 | -0.1801 | 0.0156 | 0.0880 |
| combined_pi_lasso | combined | -0.0790 | -0.1732 | 0.0144 | 0.0920 |
| topology_pi_lasso | topology_pi_rf | -0.0462 | -0.1047 | 0.0028 | 0.0380 |

## Multiple-testing correction across phases 1-4 and 7

Benjamini-Hochberg 5% FDR control across these phases' 60-session-block comparisons (104 tests total; phases 5 and 6 run their own separate corrections). 9 survive.

| model | reference | ap_difference | p_approx | q_value_bh | source |
| --- | --- | --- | --- | --- | --- |
| combined_xgb | combined | -0.1087 | 0.0020 | 0.0346 | phase3_xgb_extension |
| target_combined | target_market | 0.0000 | 0.0020 | 0.0346 | phase4_transfer_trained |
| target_topology_xgb | target_topology | -0.0001 | 0.0020 | 0.0346 | phase4_transfer_trained |
| target_topology | target_market | -0.1481 | 0.0020 | 0.0346 | phase4_transfer_trained |
| topology_orthogonal | topology | -0.0384 | 0.0020 | 0.0346 | phase2_orthogonal_addendum |
| target_combined | target_topology_xgb | 0.1482 | 0.0020 | 0.0346 | phase4_transfer_trained |
| market_rf | market | -0.1035 | 0.0040 | 0.0462 | phase7_alternative_learners |
| topology_orthogonal | topology | -0.0658 | 0.0040 | 0.0462 | phase2_orthogonal_addendum |
| market_rf | market | -0.1080 | 0.0040 | 0.0462 | phase7_alternative_learners |


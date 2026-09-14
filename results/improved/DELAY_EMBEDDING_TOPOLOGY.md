# Delay-embedded market dynamics as a topology source (post-hoc, phase 8)

Instead of a cross-sectional correlation network across hundreds of stocks, this embeds SPY's own return series (same 60-session/5-lag construction already used for the RQ5 sector proxies) and runs persistent homology on that single series.

## Results

### Target: SPY loss of at least 10.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | false_alarm_episodes |
| --- | --- | --- | --- | --- |
| VIX | 0.7440 | 0.2198 | 0.5958 | 73 |
| combined_delay | 0.7387 | 0.1937 | 0.5546 | 55 |
| combined | 0.7538 | 0.1805 | 0.5932 | 67 |
| market | 0.7592 | 0.1781 | 0.5860 | 47 |
| topology | 0.5825 | 0.0513 | 0.5680 | 122 |
| delay_topology | 0.4738 | 0.0436 | 0.5439 | 82 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| delay_topology | topology | -0.0077 | -0.0545 | 0.0222 | 0.2880 |
| delay_topology | market | -0.1345 | -0.2837 | -0.0374 | 0.0000 |
| combined_delay | market | 0.0156 | -0.0135 | 0.0559 | 0.7100 |
| combined_delay | combined | 0.0132 | -0.0146 | 0.0511 | 0.7280 |
| delay_topology | topology | -0.0077 | -0.0542 | 0.0244 | 0.3060 |
| delay_topology | market | -0.1345 | -0.2748 | -0.0168 | 0.0060 |
| combined_delay | market | 0.0156 | -0.0108 | 0.0618 | 0.6920 |
| combined_delay | combined | 0.0132 | -0.0125 | 0.0584 | 0.7480 |
| delay_topology | topology | -0.0077 | -0.0489 | 0.0199 | 0.3000 |
| delay_topology | market | -0.1345 | -0.2608 | -0.0044 | 0.0100 |
| combined_delay | market | 0.0156 | -0.0100 | 0.0480 | 0.7260 |
| combined_delay | combined | 0.0132 | -0.0104 | 0.0479 | 0.7880 |

### Target: SPY loss of at least 7.5% within 20 sessions

| model | auroc | average_precision | within_year_auroc | false_alarm_episodes |
| --- | --- | --- | --- | --- |
| combined_delay | 0.6825 | 0.2472 | 0.4844 | 48 |
| market | 0.7056 | 0.2303 | 0.4999 | 47 |
| VIX | 0.6927 | 0.2277 | 0.5011 | 62 |
| combined | 0.6942 | 0.2267 | 0.4961 | 53 |
| topology | 0.6048 | 0.1084 | 0.5399 | 84 |
| delay_topology | 0.4699 | 0.0914 | 0.5366 | 58 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| delay_topology | topology | -0.0170 | -0.0600 | 0.0384 | 0.2240 |
| delay_topology | market | -0.1389 | -0.2436 | -0.0388 | 0.0020 |
| combined_delay | market | 0.0169 | -0.0117 | 0.0459 | 0.7940 |
| combined_delay | combined | 0.0205 | -0.0107 | 0.0520 | 0.8580 |
| delay_topology | topology | -0.0170 | -0.0593 | 0.0357 | 0.2280 |
| delay_topology | market | -0.1389 | -0.2478 | -0.0042 | 0.0200 |
| combined_delay | market | 0.0169 | -0.0116 | 0.0533 | 0.7760 |
| combined_delay | combined | 0.0205 | -0.0105 | 0.0578 | 0.8520 |
| delay_topology | topology | -0.0170 | -0.0580 | 0.0322 | 0.2260 |
| delay_topology | market | -0.1389 | -0.2533 | -0.0004 | 0.0260 |
| combined_delay | market | 0.0169 | -0.0117 | 0.0456 | 0.7880 |
| combined_delay | combined | 0.0205 | -0.0135 | 0.0504 | 0.8480 |

### Target: SPY loss of at least 5.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | false_alarm_episodes |
| --- | --- | --- | --- | --- |
| market | 0.6599 | 0.3282 | 0.4834 | 36 |
| combined | 0.6533 | 0.3260 | 0.5025 | 42 |
| VIX | 0.6582 | 0.3259 | 0.4919 | 46 |
| combined_delay | 0.6344 | 0.3085 | 0.4631 | 36 |
| topology | 0.5408 | 0.2124 | 0.4931 | 79 |
| delay_topology | 0.4371 | 0.1727 | 0.4764 | 47 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| delay_topology | topology | -0.0397 | -0.1003 | 0.0275 | 0.1220 |
| delay_topology | market | -0.1554 | -0.2412 | -0.0656 | 0.0000 |
| combined_delay | market | -0.0197 | -0.0523 | 0.0170 | 0.1660 |
| combined_delay | combined | -0.0175 | -0.0499 | 0.0218 | 0.2080 |
| delay_topology | topology | -0.0397 | -0.1131 | 0.0364 | 0.1600 |
| delay_topology | market | -0.1554 | -0.2520 | -0.0311 | 0.0100 |
| combined_delay | market | -0.0197 | -0.0501 | 0.0213 | 0.2220 |
| combined_delay | combined | -0.0175 | -0.0458 | 0.0236 | 0.2540 |
| delay_topology | topology | -0.0397 | -0.1210 | 0.0354 | 0.1680 |
| delay_topology | market | -0.1554 | -0.2698 | -0.0212 | 0.0120 |
| combined_delay | market | -0.0197 | -0.0462 | 0.0218 | 0.2060 |
| combined_delay | combined | -0.0175 | -0.0454 | 0.0243 | 0.2740 |

## Multiple-testing correction across phases 1-4, 7 and 8

Benjamini-Hochberg 5% FDR control across these phases' 60-session-block comparisons (116 total). 6 survive.

| model | reference | ap_difference | p_approx | q_value_bh | source |
| --- | --- | --- | --- | --- | --- |
| combined_xgb | combined | -0.1087 | 0.0020 | 0.0386 | phase3_xgb_extension |
| target_combined | target_market | 0.0000 | 0.0020 | 0.0386 | phase4_transfer_trained |
| target_topology_xgb | target_topology | -0.0001 | 0.0020 | 0.0386 | phase4_transfer_trained |
| target_topology | target_market | -0.1481 | 0.0020 | 0.0386 | phase4_transfer_trained |
| topology_orthogonal | topology | -0.0384 | 0.0020 | 0.0386 | phase2_orthogonal_addendum |
| target_combined | target_topology_xgb | 0.1482 | 0.0020 | 0.0386 | phase4_transfer_trained |


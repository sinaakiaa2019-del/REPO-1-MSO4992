# Fair, in-market-trained transfer test (post-hoc, phase 4)

The published transfer test above freezes a US-trained topology model and applies it zero-shot, while target_volatility/target_momentum train on the target market's own history. Here every group, including topology, is trained and calibrated within each basket under the same purged protocol used everywhere else. Same topology features (already published), same target definitions, no new data.

## Published, frozen, zero-shot result (for comparison; untouched)

| universe | model | auroc | average_precision | false_alarm_episodes |
| --- | --- | --- | --- | --- |
| credit_bond_ETFs | frozen_US_topology | 0.6595 | 0.0528 | 2 |
| credit_bond_ETFs | target_momentum | 0.8117 | 0.1405 | 68 |
| credit_bond_ETFs | target_volatility | 0.7997 | 0.2627 | 18 |
| international_indices | frozen_US_topology | 0.8302 | 0.1441 | 0 |
| international_indices | target_momentum | 0.7205 | 0.1685 | 17 |
| international_indices | target_volatility | 0.9480 | 0.3985 | 10 |

## This phase: trained and calibrated within each basket

### credit_bond_ETFs

| model | n | positives | auroc | average_precision | within_year_auroc | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| target_topology_xgb | 3754 | 23 | 0.2273 | 0.0076 | 0.5000 | 5 |
| target_market | 3754 | 23 | 0.4315 | 0.0066 | 0.6136 | 26 |
| target_combined | 3754 | 23 | 0.4316 | 0.0064 | 0.6219 | 28 |
| target_topology | 3754 | 23 | 0.1677 | 0.0062 | 0.5507 | 53 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| target_combined | target_market | -0.0002 | -0.0023 | 0.0002 | 0.2540 |
| target_topology | target_market | -0.0004 | -0.0151 | 0.0039 | 0.4898 |
| target_topology_xgb | target_topology | 0.0014 | 0.0002 | 0.0042 | 0.9909 |
| target_combined | target_topology_xgb | -0.0013 | -0.0072 | 0.0138 | 0.3741 |
| target_combined | target_market | -0.0002 | -0.0015 | 0.0005 | 0.2900 |
| target_topology | target_market | -0.0004 | -0.0086 | 0.0023 | 0.3984 |
| target_topology_xgb | target_topology | 0.0014 | 0.0003 | 0.0047 | 0.9837 |
| target_combined | target_topology_xgb | -0.0013 | -0.0061 | 0.0066 | 0.2683 |
| target_combined | target_market | -0.0002 | -0.0013 | 0.0011 | 0.2802 |
| target_topology | target_market | -0.0004 | -0.0133 | 0.0018 | 0.3392 |
| target_topology_xgb | target_topology | 0.0014 | 0.0003 | 0.0054 | 0.9971 |
| target_combined | target_topology_xgb | -0.0013 | -0.0065 | 0.0129 | 0.3274 |

### international_indices

| model | n | positives | auroc | average_precision | within_year_auroc | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| target_combined | 1394 | 46 | 0.9295 | 0.1906 | 0.4221 | 7 |
| target_market | 1394 | 46 | 0.9295 | 0.1906 | 0.4221 | 7 |
| target_topology | 1394 | 46 | 0.3372 | 0.0426 | 0.5000 | 0 |
| target_topology_xgb | 1394 | 46 | 0.3318 | 0.0425 | 0.5000 | 0 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| target_combined | target_market | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| target_topology | target_market | -0.1481 | -0.4277 | -0.0373 | 0.0000 |
| target_topology_xgb | target_topology | -0.0001 | -0.0012 | 0.0000 | 0.0000 |
| target_combined | target_topology_xgb | 0.1482 | 0.0384 | 0.4277 | 1.0000 |
| target_combined | target_market | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| target_topology | target_market | -0.1481 | -0.5421 | -0.0259 | 0.0000 |
| target_topology_xgb | target_topology | -0.0001 | -0.0013 | 0.0000 | 0.0000 |
| target_combined | target_topology_xgb | 0.1482 | 0.0261 | 0.5423 | 1.0000 |
| target_combined | target_market | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| target_topology | target_market | -0.1481 | -0.4773 | -0.0579 | 0.0000 |
| target_topology_xgb | target_topology | -0.0001 | -0.0009 | 0.0000 | 0.0000 |
| target_combined | target_topology_xgb | 0.1482 | 0.0579 | 0.4774 | 1.0000 |

## Multiple-testing correction across all four phases

Benjamini-Hochberg 5% FDR control across phases 1-4's 60-session-block comparisons (65 tests total). 7 survive.

| model | reference | ap_difference | p_approx | q_value_bh | source |
| --- | --- | --- | --- | --- | --- |
| combined_xgb | combined | -0.1087 | 0.0020 | 0.0216 | phase3_xgb_extension |
| topology_orthogonal | topology | -0.0384 | 0.0020 | 0.0216 | phase2_orthogonal_addendum |
| target_topology | target_market | -0.1481 | 0.0020 | 0.0216 | phase4_transfer_trained |
| target_combined | target_market | 0.0000 | 0.0020 | 0.0216 | phase4_transfer_trained |
| target_topology_xgb | target_topology | -0.0001 | 0.0020 | 0.0216 | phase4_transfer_trained |
| target_combined | target_topology_xgb | 0.1482 | 0.0020 | 0.0216 | phase4_transfer_trained |
| topology_orthogonal | topology | -0.0658 | 0.0040 | 0.0371 | phase2_orthogonal_addendum |

## Reading this honestly

Only 3 (credit) and 4 (international) calendar years in these baskets have any positive labels at all -- less independent crisis evidence than even the primary 18-year US study had. Wide intervals reflect that directly; a favourable point estimate on this little data is not the same as a validated effect.


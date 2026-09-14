# Regime-normalized persistence images at a 60-session horizon (post-hoc, phase 6)

Combines persistence images (phase 5), regime-normalization (phase 1), and a horizon longer than the locked 20-session primary target -- tested together here for the first time. The 20-session protocol is untouched; this is a disclosed, additional exploration at 60 sessions.

## Target: SPY loss of at least 10.0% within 60 sessions

| model | n | positives | auroc | average_precision | recall | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| market | 4470 | 653 | 0.6469 | 0.2976 | 0.3675 | 32 |
| combined | 4470 | 653 | 0.6387 | 0.2975 | 0.3155 | 52 |
| topology_pi_xgb | 4470 | 653 | 0.6247 | 0.2092 | 0.1853 | 78 |
| combined_pi_xgb_regime | 4470 | 653 | 0.3169 | 0.1030 | 0.1623 | 108 |
| topology_pi_xgb_regime | 4470 | 653 | 0.2551 | 0.0961 | 0.0505 | 96 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_pi_xgb_regime | topology_pi_xgb | -0.1131 | -0.2305 | -0.0237 | 0.0020 |
| combined_pi_xgb_regime | market | -0.1946 | -0.3543 | -0.0244 | 0.0040 |
| combined_pi_xgb_regime | combined | -0.1945 | -0.3527 | -0.0254 | 0.0060 |

## Target: SPY loss of at least 7.5% within 60 sessions

| model | n | positives | auroc | average_precision | recall | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| market | 4470 | 943 | 0.6217 | 0.3510 | 0.3245 | 28 |
| combined | 4470 | 943 | 0.6135 | 0.3436 | 0.2715 | 47 |
| topology_pi_xgb | 4470 | 943 | 0.5509 | 0.2471 | 0.1474 | 71 |
| combined_pi_xgb_regime | 4470 | 943 | 0.3425 | 0.1546 | 0.1304 | 75 |
| topology_pi_xgb_regime | 4470 | 943 | 0.3153 | 0.1509 | 0.1368 | 73 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_pi_xgb_regime | topology_pi_xgb | -0.0963 | -0.1790 | -0.0238 | 0.0000 |
| combined_pi_xgb_regime | market | -0.1964 | -0.3357 | -0.0455 | 0.0020 |
| combined_pi_xgb_regime | combined | -0.1890 | -0.3298 | -0.0405 | 0.0000 |

## Target: SPY loss of at least 5.0% within 60 sessions

| model | n | positives | auroc | average_precision | recall | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| combined | 4470 | 1485 | 0.6235 | 0.4502 | 0.2162 | 37 |
| market | 4470 | 1485 | 0.5995 | 0.4399 | 0.2640 | 18 |
| topology_pi_xgb | 4470 | 1485 | 0.4987 | 0.3472 | 0.0923 | 47 |
| topology_pi_xgb_regime | 4470 | 1485 | 0.3771 | 0.2741 | 0.2215 | 63 |
| combined_pi_xgb_regime | 4470 | 1485 | 0.3719 | 0.2723 | 0.2492 | 77 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| topology_pi_xgb_regime | topology_pi_xgb | -0.0731 | -0.1508 | 0.0097 | 0.0440 |
| combined_pi_xgb_regime | market | -0.1677 | -0.2775 | -0.0515 | 0.0020 |
| combined_pi_xgb_regime | combined | -0.1779 | -0.2865 | -0.0609 | 0.0020 |

## Multiple-testing check (this phase's own 9 tests, 60-session blocks)

8 of 9 survive 5% FDR control.

| target | model | reference | ap_difference | q_value_bh |
| --- | --- | --- | --- | --- |
| loss10 | topology_pi_xgb_regime | topology_pi_xgb | -0.1131 | 0.0060 |
| loss10 | combined_pi_xgb_regime | market | -0.1946 | 0.0103 |
| loss10 | combined_pi_xgb_regime | combined | -0.1945 | 0.0135 |
| loss075 | topology_pi_xgb_regime | topology_pi_xgb | -0.0963 | 0.0060 |
| loss075 | combined_pi_xgb_regime | market | -0.1964 | 0.0060 |
| loss075 | combined_pi_xgb_regime | combined | -0.1890 | 0.0060 |
| loss05 | combined_pi_xgb_regime | market | -0.1677 | 0.0060 |
| loss05 | combined_pi_xgb_regime | combined | -0.1779 | 0.0060 |


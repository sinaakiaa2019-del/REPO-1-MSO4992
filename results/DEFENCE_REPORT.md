# Critical evaluation and strengthened real-data results

All numbers below come from the executed public-data experiment. Earlier releases are retained separately, not overwritten or relabelled. A positive result was not a completion requirement.

## Data quality and survivorship

The public refresh supplied 703 equity histories. Dated membership and complete windows admit 794 histories, including 294 former members. 209 historical membership symbols have no matching price history. Every requested symbol and failure is in inputs/public_download_audit.csv.

There is no liquidity cap, current-survivor filter, future-neighbour deletion or substitution from the suspect original CSV. Only observed finite positive adjusted prices enter log returns; missing prices break both adjacent returns. Corporate actions and provider metadata are archived. This improves provenance and removes the supplied retrospective cleaning rule, but missing delisted histories, ticker reuse and historical adjustment vintages remain unresolved. Public quotes are not a permanent-ID database. Coverage loss can itself introduce selection bias.

## Shared-date prediction results

The primary equal-budget comparison has 4,510 test sessions and 197 positive target days. The target is a 10% SPY loss during the next 20 observed sessions. AP is average precision, used here as PR-AUC.

| model | auroc | average_precision | false_positive_rate | recall | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- |
| VIX | 0.7440 | 0.2198 | 0.1595 | 0.4010 | 73 |
| combined | 0.7538 | 0.1805 | 0.1342 | 0.3553 | 67 |
| combined_correlation | 0.7425 | 0.1533 | 0.1245 | 0.3299 | 57 |
| market | 0.7592 | 0.1781 | 0.1546 | 0.4112 | 47 |
| market_correlation | 0.7407 | 0.1478 | 0.1238 | 0.3350 | 58 |
| scale_080 | 0.7638 | 0.1825 | 0.1551 | 0.4365 | 54 |
| scale_090 | 0.7494 | 0.1817 | 0.1484 | 0.4518 | 47 |
| scale_100 | 0.7656 | 0.1747 | 0.1584 | 0.4162 | 53 |
| scale_110 | 0.7014 | 0.1769 | 0.1145 | 0.2944 | 66 |
| topology | 0.5825 | 0.0513 | 0.1261 | 0.3655 | 122 |

Each learned group has the same three regularisation candidates, the same two purged inner blocks and the same separate 252-session calibration. Raw VIX is a fixed benchmark. Equal search budgets do not imply equal model complexity. All tested scales and weaker models remain visible.

## RQ1: Does topology give earlier, useful warnings?

The evaluation now includes every mechanically detected 10% SPY high-water drawdown in the scored period. Each event lasts until recovery of its preceding peak. Warnings are measured in the 20 sessions strictly before the first 10% crossing. These event annotations are not predictors. The event definition differs from the supervised forward-loss label, so event recall and label-based false alarms must both be reported.

| model | budget | events | detected | median_lead_when_detected | event_recall |
| --- | --- | --- | --- | --- | --- |
| VIX | 0.0500 | 6 | 6 | 13.0000 | 1.0000 |
| combined | 0.0500 | 6 | 5 | 14.0000 | 0.8333 |
| market | 0.0500 | 6 | 6 | 12.0000 | 1.0000 |
| topology | 0.0500 | 6 | 3 | 13.0000 | 0.5000 |

A missed event has no lead time; it is not assigned zero or removed from recall. Median lead time is conditional on detection. Read defence_event_leads and the 1%, 5%, 10% budget tables for every event and threshold. Few independent episodes and retrospective data prevent a general early-warning guarantee.

## RQ2: Which filtration scale is useful?

The new scale comparison holds window length (60), EWMA decay (.94), shrinkage (.1), model family and training history fixed. It compares four Betti coordinates at grid indices 80, 90, 100 and 110. Epsilon maps to shrunk correlation through rho=1-epsilon²/2. The annual scale choice uses earlier validation only; all outer scores are retained.

| year | scale_index | epsilon | shrunk_correlation_threshold | C | inner_ap |
| --- | --- | --- | --- | --- | --- |
| 2007 | 90 | 0.9045 | 0.5909 | 0.0100 | 0.0230 |
| 2008 | 80 | 0.8040 | 0.6768 | 0.0100 | nan |
| 2009 | 80 | 0.8040 | 0.6768 | 0.0100 | nan |
| 2010 | 90 | 0.9045 | 0.5909 | 0.0100 | 0.4044 |
| 2011 | 110 | 1.1055 | 0.3889 | 1.0000 | 0.3232 |
| 2012 | 110 | 1.1055 | 0.3889 | 1.0000 | 0.3048 |
| 2013 | 110 | 1.1055 | 0.3889 | 1.0000 | 0.2680 |
| 2014 | 80 | 0.8040 | 0.6768 | 0.0100 | 0.2403 |
| 2015 | 80 | 0.8040 | 0.6768 | 0.0100 | 0.2713 |
| 2016 | 110 | 1.1055 | 0.3889 | 0.0100 | 0.2838 |
| 2017 | 110 | 1.1055 | 0.3889 | 0.0100 | 0.2632 |
| 2018 | 110 | 1.1055 | 0.3889 | 0.0100 | 0.2587 |
| 2019 | 100 | 1.0050 | 0.4950 | 0.0100 | 0.2558 |
| 2020 | 100 | 1.0050 | 0.4950 | 0.0100 | 0.2639 |
| 2021 | 100 | 1.0050 | 0.4950 | 0.0100 | 0.2599 |
| 2022 | 100 | 1.0050 | 0.4950 | 1.0000 | 0.1137 |
| 2023 | 100 | 1.0050 | 0.4950 | 1.0000 | 0.0622 |
| 2024 | 100 | 1.0050 | 0.4950 | 1.0000 | 0.0645 |

This answers relative scale performance within the declared grid. It does not establish a universal optimum or prove a correlation threshold transfers to other universes. The earlier broader joint search is a separate development exercise.

## RQ3: Does topology add information beyond market and correlation controls?

Combined-minus-market AP is +0.0023, with paired 60-session-block interval [-0.0115, +0.0104]. The interval includes zero; a reliable increment is not established.

After adding ordinary correlation controls to both models, the topology increment is +0.0054, interval [-0.0015, +0.0131]. This is the more demanding attribution comparison.

| model | reference | block_sessions | draws | ap_difference | lower95 | upper95 |
| --- | --- | --- | --- | --- | --- | --- |
| combined | market | 20 | 500 | 0.0023 | -0.0119 | 0.0177 |
| combined_correlation | market_correlation | 20 | 500 | 0.0054 | -0.0015 | 0.0153 |
| combined | VIX | 20 | 500 | -0.0394 | -0.0732 | -0.0037 |
| combined | market | 60 | 500 | 0.0023 | -0.0115 | 0.0104 |
| combined_correlation | market_correlation | 60 | 500 | 0.0054 | -0.0015 | 0.0131 |
| combined | VIX | 60 | 500 | -0.0394 | -0.0684 | 0.0015 |
| combined | market | 120 | 500 | 0.0023 | -0.0089 | 0.0101 |
| combined_correlation | market_correlation | 120 | 500 | 0.0054 | -0.0019 | 0.0122 |
| combined | VIX | 120 | 500 | -0.0394 | -0.0614 | 0.0050 |

Intervals use 500 paired draws for each of 20-, 60- and 120-session block lengths. These are sensitivity checks for dependence, not corrections for all repeated research or multiple comparisons. This history has no untouched holdout. A zero-crossing interval does not prove equivalence.

## RQ4: Does the model transfer?

The equal-budget US topology model is frozen for each annual target test. Target volatility and negative momentum are explicit baselines on exactly matching dates. Their thresholds use only already-resolved target labels; the topology threshold remains source-calibrated. This compares zero-shot transfer with target-informed conventional scores, not identical adaptation.

| universe | model | n | positives | auroc | average_precision | false_alarm_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| credit_bond_ETFs | frozen_US_topology | 4363 | 91 | 0.6595 | 0.0528 | 2 |
| credit_bond_ETFs | target_momentum | 4363 | 91 | 0.8117 | 0.1405 | 68 |
| credit_bond_ETFs | target_volatility | 4363 | 91 | 0.7997 | 0.2627 | 18 |
| international_indices | frozen_US_topology | 1394 | 46 | 0.8302 | 0.1441 | 0 |
| international_indices | target_momentum | 1394 | 46 | 0.7205 | 0.1685 | 17 |
| international_indices | target_volatility | 1394 | 46 | 0.9480 | 0.3985 | 10 |

Read defence_transfer_comparisons for paired uncertainty and defence_transfer_predictions for all scores. Small ETF/index baskets and correlated target/source crises limit generalisation. A high AUROC from one warning episode is not evidence of repeatedly forecasting independent crises. The original frozen-model transfer remains available as a separate baseline.

## RQ5: Which sector proxies show earlier topology-related stress?

This is a disclosed proxy version of RQ5. It uses the nine actual Select Sector funds launched before 2000, rather than assigning current company sectors to the past. Each fund uses a trailing 60-return window and five-lag delay embedding, standardised within that window. Its topology measures the shape of return dynamics, not the within-sector stock correlation network. Fund exposures and sector classifications can change through time.

| symbol | sector_proxy | model | auroc | average_precision | false_positive_rate | recall |
| --- | --- | --- | --- | --- | --- | --- |
| XLB | Materials | sector_combined | 0.6131 | 0.0735 | 0.1101 | 0.3147 |
| XLB | Materials | sector_market | 0.7260 | 0.0980 | 0.1296 | 0.3807 |
| XLB | Materials | sector_topology | 0.4146 | 0.0343 | 0.0934 | 0.1269 |
| XLE | Energy | sector_combined | 0.6100 | 0.0604 | 0.1317 | 0.2183 |
| XLE | Energy | sector_market | 0.6492 | 0.0639 | 0.1312 | 0.2183 |
| XLE | Energy | sector_topology | 0.4224 | 0.0350 | 0.1018 | 0.1980 |
| XLF | Financials | sector_combined | 0.7266 | 0.2079 | 0.1048 | 0.3147 |
| XLF | Financials | sector_market | 0.7410 | 0.2128 | 0.1282 | 0.4010 |
| XLF | Financials | sector_topology | 0.4630 | 0.0397 | 0.0800 | 0.0863 |
| XLI | Industrials | sector_combined | 0.7616 | 0.1598 | 0.1083 | 0.4670 |
| XLI | Industrials | sector_market | 0.7613 | 0.1676 | 0.1189 | 0.4315 |
| XLI | Industrials | sector_topology | 0.5179 | 0.0443 | 0.0744 | 0.0152 |
| XLK | Technology | sector_combined | 0.7007 | 0.1479 | 0.1039 | 0.3959 |
| XLK | Technology | sector_market | 0.7373 | 0.1614 | 0.1125 | 0.4213 |
| XLK | Technology | sector_topology | 0.4331 | 0.0358 | 0.1104 | 0.1726 |
| XLP | Consumer staples | sector_combined | 0.6435 | 0.1143 | 0.1046 | 0.3299 |
| XLP | Consumer staples | sector_market | 0.6717 | 0.1482 | 0.1115 | 0.3858 |
| XLP | Consumer staples | sector_topology | 0.4911 | 0.0389 | 0.0907 | 0.1523 |
| XLU | Utilities | sector_combined | 0.6121 | 0.1089 | 0.0839 | 0.2843 |
| XLU | Utilities | sector_market | 0.5968 | 0.1182 | 0.1189 | 0.2843 |
| XLU | Utilities | sector_topology | 0.4144 | 0.0340 | 0.0763 | 0.0508 |
| XLV | Health care | sector_combined | 0.7337 | 0.1191 | 0.1199 | 0.3503 |
| XLV | Health care | sector_market | 0.7504 | 0.1248 | 0.1271 | 0.3756 |
| XLV | Health care | sector_topology | 0.4399 | 0.0358 | 0.1530 | 0.1117 |
| XLY | Consumer discretionary | sector_combined | 0.7432 | 0.1626 | 0.1354 | 0.4670 |
| XLY | Consumer discretionary | sector_market | 0.7389 | 0.1712 | 0.1449 | 0.4467 |
| XLY | Consumer discretionary | sector_topology | 0.3849 | 0.0347 | 0.1131 | 0.0914 |

| symbol | sector_proxy | model | events | detected | median_lead_when_detected |
| --- | --- | --- | --- | --- | --- |
| XLB | Materials | sector_topology | 6 | 3 | 5.0000 |
| XLE | Energy | sector_topology | 6 | 2 | 20.0000 |
| XLF | Financials | sector_topology | 6 | 0 | nan |
| XLI | Industrials | sector_topology | 6 | 3 | 16.0000 |
| XLK | Technology | sector_topology | 6 | 3 | 20.0000 |
| XLP | Consumer staples | sector_topology | 6 | 1 | 20.0000 |
| XLU | Utilities | sector_topology | 6 | 3 | 18.0000 |
| XLV | Health care | sector_topology | 6 | 1 | 18.0000 |
| XLY | Consumer discretionary | sector_topology | 6 | 3 | 5.0000 |

The sector tables answer comparative performance and event warnings for these fund proxies. They do not recover historical firm-sector leadership. All nine funds and all three model groups are reported; rankings after viewing test outcomes are descriptive, not a validated sector-selection strategy. Paired AP intervals versus each fund’s conventional baseline are in rq5_comparisons.

## Overall academic assessment

RQ1–RQ4 now have broader or better-controlled empirical tests. RQ5 has an executed and explicitly narrower proxy study. The original company-sector question still requires dated classifications. The strongest contribution is an auditable test of incremental topology information, with negative and inconclusive findings retained. Public-data coverage, limited independent crises and repeated use of the same history remain material weaknesses. Neither AUROC nor this software package can guarantee a dissertation grade.

## Reproduce and inspect

Run python run_all.py --prices inputs/public_prices.csv after installing requirements.txt. That command performs data audit, TDA, classic models, sector proxies, equal-budget comparisons, transfer controls, reports and checks. Run python -m streamlit run streamlit_app.py for the dashboard. Detailed outputs are consolidated in results/evidence.sqlite; python run_all.py --export-table TABLE exports any table. Source responses are in the separate provenance archive supplied with this release.

Sources: [nested evaluation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [sector fund definitions](https://www.ssga.com/us/en/intermediary/capabilities/equities/sector-investing/select-sector-etfs), [issuer inception-date table](https://www.ssga.com/library-content/pdfs/etf/us/target-a-lower-cost-of-ownership-with-sector-etfs.pdf).

# Methodology and limits

## Scope and forecast timing

This is retrospective development on previously examined 2000–2024 history, not an untouched confirmatory backtest. The main question is whether topology adds useful information to an otherwise matching conventional model.

A forecast is made after the current close. Its label is one if SPY falls at least 10% below that close during any of the next 20 observed SPY sessions. The final 20 dates have unresolved targets and are excluded. Early history supplies warm-up/training; annual tests cover 2007–2024.

Returns are log(adjusted_close[t]/adjusted_close[t-1]) on consecutive SPY sessions. Missing, infinite or nonpositive prices become missing observations. Both adjacent returns are broken; there is no bridging, zero fill or guessed terminal loss.

## Primary data and survivorship

The fresh public snapshot contains 3,658,880 rows and 703 equity ticker histories. Requests span 1,209 equity names from the original supplied panel and membership reconstruction. The original price values are not fallback data. Every request, failure, metadata record and successful response hash is recorded; compressed raw responses are in the separate source archive. Corporate actions are recorded as provider evidence, not independently certified adjustments.

The supplied retrospective cleaner remains only for old-experiment reproduction. Its next-observation glitch rule and completed stale-run rule do not enter this primary run. This removes one known lookahead path but does not make public adjustment histories point-in-time vintages.

On each date, the latest dated membership snapshot identifies expected members. A stock must be a member that day, have 60 consecutive valid returns and nonzero variation. Pre-membership returns may supply its already-observed lookback. There is no liquidity cap or requirement to survive to 2024. An end-of-2024 membership flag is used only in the audit.

The new run admits 794 histories, including 294 former members. There are 209 unmatched membership symbols. Matched member-sessions are 2,884,226 of 3,141,040 expected (91.8%). After warm-up, complete eligible member-sessions are 2,651,335 of 3,111,521 (85.2%). Traceability must not be equated with absence of selection bias.

Only punctuation-equivalent ticker matching is available. Reused tickers, mergers, name changes and terminal delisting returns are not fully resolved. Former members are not necessarily delisted firms. Complete-window selection can exclude interrupted/distressed histories. No estimated delisting loss is inserted.

Membership is a public reconstruction, not a verified permanent-ID security master. Snapshot dates are treated as effective by that close; announcement timing is unverified. See the [membership source](https://github.com/fja05680/sp500). Missing delisting outcomes can be economically important; see [Shumway, The Delisting Bias in CRSP Data](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x).

## Supporting series

SPY and transfer ETF/index observations are real saved market data. VIX uses the [CBOE history](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv); the Treasury slope is actual [FRED T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y), not a yield-level proxy. Input hashes and acquisition references are retained in source_manifest and the evidence documents.

VIX is lagged one SPY session. T10Y2Y is reindexed to SPY dates, carried forward at most five sessions, then lagged one session. This is a timing precaution, not a historical release-vintage reconstruction. Price-based predictors use the current close. Different international calendars, closing times and currencies remain transfer limitations.

## Correlation and TDA

The trailing window contains 60 daily returns. Exponential weights decay by .94; the weighted mean is removed before covariance estimation. Correlation receives 10% shrinkage toward identity, providing a positive-definite matrix. Training-only shrinkage diagnostics are retained separately and do not change the primary matrix after test scores are seen.

Distance is sqrt(2*(1-correlation)). Vietoris–Rips persistence measures components (H0) and loops (H1); Betti curves are sampled at 200 equally spaced points from 0 to 2. Finite H0 lifetimes omit the infinite component; the H0 Betti curve retains the surviving component.

The primary topology-only group contains normalised H1 entropy, H1 total lifetime per asset, and diagram movement per square root of asset count. Diagram movement is 2-Wasserstein with Euclidean ground cost and diagonal matching, divided by the square root of mean asset count across adjacent windows. Membership changes can still cause movement; normalisation does not prove size invariance.

The classic ladder also uses lifetime/count summaries, integrated Betti norms, full curves and trailing changes. Integrated L2 norms use the square root of a trapezoidal integral of the squared curve. Persistence images are diagnostic 20-by-20 Gaussian pixel integrals in birth/lifetime coordinates with .05 standard deviation and lifetime weighting; they are not a headline learner. Landscape and network summaries remain available in the legacy research extension; ordinary graph statistics are not persistent homology.

## Primary equal-budget models

| Group | Inputs |
|---|---|
| market | log(lagged VIX), SPY volatility20, drawdown60, lagged T10Y2Y |
| topology | Three compact persistent-homology features above |
| combined | Market plus topology |
| market_correlation | Market plus average correlation and largest eigenvalue/asset count |
| combined_correlation | Matching market/correlation controls plus topology |
| scale_080 / 090 / 100 / 110 | Market plus one normalised H1 Betti coordinate |
| VIX | Fixed raw lagged-VIX score, no fitted learner |

Each of the nine learned groups receives C={.01,.1,1}, expanding training history and the same two purged chronological inner blocks. This is 27 candidates per year and 486 candidate records across 18 years. Each candidate is fitted/scored in both earlier blocks. Average precision selects C within each family; ties use declared candidate order. Scaling is fitted inside each logistic training set.

Single-class training predicts training prevalence instead of dropping the fold. Inner validation without positives has undefined AP and uses the deterministic first candidate. Missing inner AP in 2008/2009 is therefore not evidence that the reported scale was empirically optimal.

Before each annual test, the latest 252 observations with resolved targets form calibration. Training labels must end strictly before calibration starts, and calibration labels strictly before the test. Inner tuning also purges label overlap. All primary groups use matching dates. Shared dates inherit the complete-feature intersection of the classic feature panel, a conservative restriction that can discard otherwise usable rows.

The classic 14-model logistic/XGBoost ladder is rerun separately with its original two-candidate protocol. Its feature definitions and scores must not be conflated with the primary equal-budget groups. The older optional broad joint search is not executed as the new primary analysis.

## Thresholds, metrics and uncertainty

Thresholds come from earlier negative calibration scores at budgets 1%, 5% and 10%, using the higher order statistic. Alarm means score strictly greater than threshold. The constraint holds in calibration, including ties; it does not guarantee the same future FPR. There is no cooldown or persistence requirement.

An alarm episode is a consecutive run of warning days. It is false only if none of its days has a positive forward-loss label. Report negative-day FPR, recall, precision, time under warning, episode counts and ranking scores together.

AP means average precision, the PR-AUC convention used here; it is not trapezoidal integration of the precision–recall curve. The main matched test has 4,510 sessions and 197 positives (4.368% prevalence). Repeated positive days are not independent crises. Pooled probabilities come from annual refits and may have different scales across years.

Paired moving-block intervals use 500 attempted resamples at each of 20, 60 and 120 sessions. Resamples without both label classes are omitted; actual accepted draws are saved. Sector comparisons use 60-session blocks. These are conditional uncertainty checks, not a refit of the complete research process or correction for repeated model/sector searches. Intervals crossing zero establish neither superiority nor equivalence.

## RQ1: Mechanical event warnings

An event begins at the first 10% fall from the preceding running SPY high and ends at recovery of that high. An unrecovered final event is censored. Peak/recovery dates are evaluation annotations, never predictors.

Warnings are sought in the 20 sessions strictly before the first 10% crossing. Save every event, coverage flag, first warning, lead and miss. Six event onsets fall in the scored period. Median lead is conditional on detection; a missed event is not assigned zero or removed from event recall.

This event target differs from the supervised next-20-session loss target. For example, early warnings after a prolonged decline can precede the peak-drawdown crossing without matching a new 10% forward loss. Report both event recall and label-based false alarms. Older two-peak examples remain supplementary.

## RQ2: Scale

Keep window60, decay .94, shrinkage .1, model family and history fixed. Grid indices 80/90/100/110 correspond to epsilon about .8040/.9045/1.0050/1.1055 and shrunk-correlation thresholds about .6768/.5909/.4950/.3889, through rho=1-epsilon²/2.

Each scale gets three C values. Annual scale selection considers all 12 earlier-validation candidates; that cross-scale selection has a larger budget than any single family and is reported as a selection diagnostic. Every fixed-scale outer score is retained. No pooled-test winner is presented as a validated universal optimum.

## RQ3: Attribution

Compare combined with the same market model, then combined_correlation with market_correlation. Compare VIX separately. This avoids attributing ordinary correlation information to persistent homology. Equal candidate counts do not imply equal model complexity. Retrospective development and sparse crises remain qualifications even with correct chronological splits.

## RQ4: Transfer

Freeze each year's primary US topology model and source-calibrated 5% threshold. Apply matching topology definitions to observed small credit/bond ETF and international-index baskets. Targets are forward 20-session 10% falls in HYG and FTSE, respectively.

Target volatility and negative 20-session momentum are conventional controls on exactly matching dates. Their rolling thresholds use up to 252 target observations whose targets have already resolved, requiring at least 40 negative observations. They do not fit target-market classifiers. Their target-informed threshold calibration differs from the zero-shot topology threshold; disclose that asymmetry. AUROC/AP themselves do not depend on threshold choice.

All scores, baselines, uncertainty and episode counts are saved. Small baskets, asynchronous closes, common crises and only one long topology warning episode in the international test limit generalisation.

## RQ5: Explicit fund-proxy revision

Use nine actual Select Sector funds launched in 1998: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY. Do not backfill later-launched real-estate or communication funds. These are evolving fund exposures, not fixed historical company-sector labels. See the [issuer sector definitions](https://www.ssga.com/us/en/intermediary/capabilities/equities/sector-investing/select-sector-etfs) and [inception table](https://www.ssga.com/library-content/pdfs/etf/us/target-a-lower-cost-of-ownership-with-sector-etfs.pdf).

For each fund, a trailing 60-return window forms 56 five-lag points. Centre/scale within that window and divide by sqrt(5). Euclidean H1 persistence produces total lifetime per point and normalised entropy. This describes the shape of a fund's return dynamics, not a within-sector stock-correlation network. Its filtration units differ from RQ2.

Compare delay topology, conventional fund volatility/momentum, and their combination with the same three-C chronological protocol. Targets remain SPY crashes, so this tests warning of broad-market stress. Save all nine funds, three groups, event warnings and paired AP intervals. Post-test rankings are descriptive, not a validated fund-selection rule.

The proxy study answers the revised question. The original historical company-sector ranking still needs dated classifications and validated identities.

## Academic interpretation

A negative or inconclusive empirical answer is a completed test, not a software failure. No result here proves all topology models ineffective or guarantees a grade. Explain public-data selection, repeated historical development and the revised sector scope prominently. Genuine confirmation needs a frozen design and new, independently validated observations.

For the selection principle, see [scikit-learn's nested evaluation explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

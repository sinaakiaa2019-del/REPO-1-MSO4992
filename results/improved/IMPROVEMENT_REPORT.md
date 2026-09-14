# Improvement study — results and interpretation

This follows the improvement protocol specified before this run. The published `results/`
release is unchanged: `run_all.py --verify-results` and `tests/test_saved.py`
still pass against it, and `results/improved/verification.json` confirms the primary
10%-loss label, the shared test dates and the published baseline scores (VIX, market,
combined, combined_correlation, topology, scale_080/090/100/110) reproduce here to
machine precision (max |Δscore| ≈ 8e-15) before any new group is added. All tables
below are in `results/improved/`; generated figures are in `figures/`.

## Finding 1 (bug fix, not a modelling choice): the classic "enhanced" ladder was scored on a data-coverage artefact

`LR_enhanced_base` was published with pooled AUROC **0.379** — worse than the constant
model (0.425). The cause is two of its inputs: `eligible_assets` and
`eligible_fraction`, the daily count/share of index members with valid prices. Both
have Spearman ρ = **1.000** with calendar date over 2000–2024, because the audited
public universe simply grows every year (230 → 487 eligible assets). A logistic model
trained on an expanding window partly learns "later years are less risky," which
inverts on the years it has not yet seen. This is an audit variable standing in for
calendar time, not market information, and it should not have been offered to the
classifier.

Removing only those two columns and re-running the identical published protocol
(same two-candidate LR search, same purged splits, same calibration):

| model | coverage variables | AUROC | AP | within-year AUROC |
|---|---|---|---|---|
| LR_enhanced_base (published) | included | 0.379 | 0.036 | 0.567 |
| LR_enhanced_base (corrected) | removed | **0.674** | **0.124** | 0.625 |
| LR_enhanced_combined (published) | included | 0.380 | 0.035 | 0.605 |
| LR_enhanced_combined (corrected) | removed | **0.653** | **0.128** | 0.647 |
| XGB_enhanced_base (published) | included | 0.647 | 0.069 | 0.505 |
| XGB_enhanced_base (corrected) | removed | **0.712** | **0.094** | 0.514 |
| XGB_enhanced_combined (published) | included | 0.676 | 0.069 | 0.525 |
| XGB_enhanced_combined (corrected) | removed | **0.727** | **0.091** | 0.530 |

This is a genuine, disclosable fix: the "enhanced" logistic models were never as bad
as published, and the two coverage columns should be dropped from that feature group
in any future release (`results/improved/classic_correction.csv`,
`figures/coverage_artifact_fix.png`). It does not change the primary
`defence_*` results, which never used these two columns.

## Finding 2: pooled AUROC overstates day-level timing skill

Pooled AUROC mixes two different kinds of discrimination: telling a calm year from a
crisis year, and telling a risky day from a calm day *within* the same year. Because
positive days cluster tightly in a few crisis years, a model can score well pooled
just by knowing which years were bad. A positive-weighted **within-year** AUROC (each
year's own ranking, averaged by that year's positive count) strips out the
between-year part:

| model | pooled AUROC | within-year AUROC | pooled AP | within-year AP |
|---|---|---|---|---|
| scale_080 | 0.772 | 0.554 | 0.184 | — |
| market | 0.759 | 0.586 | 0.178 | 0.209 |
| combined_ews | 0.765 | **0.639** | 0.154 | 0.214 |
| combined | 0.748 | 0.588 | 0.176 | 0.206 |
| VIX | 0.744 | 0.596 | 0.220 | 0.247 |
| topology | 0.534 | 0.515 | 0.046 | 0.193 |

Every model's within-year AUROC is much closer to 0.5–0.65 than its pooled figure
suggests. `combined_ews` (the persistence-landscape early-warning statistics, defined
below) has the highest within-year AUROC of any primary group despite a lower pooled
AP than market — evidence that its ranking is comparatively better *inside* a given
year, even though it is not a better overall PR-AUC score. Both numbers are reported
together from here on; neither alone is the right summary
(`results/improved/metrics.csv`, `figures/pooled_vs_within_year_auroc.png`).

## Finding 3: a regime-relative topology representation, tested as declared

Two new representations of the same three primary topology series were pre-declared
and computed causally (only past observations, ≥126 prior sessions required):

- **`topology_z`** — each of the three primary topology features expressed as a
  z-score against its own trailing 252-session mean/SD, clipped to ±10.
- **`topology_ews`** — the coefficient of variation and Kendall-τ trend of the
  persistence-landscape L1 norm over the trailing 60 sessions, following the
  early-warning statistics used for financial crashes by Gidea & Katz (2018,
  *Physica A* 491).

Neither changes the conclusion of RQ3: at the primary 10%-loss target,
`combined_z − market` is −0.0079 [−0.0220, +0.0049] and `combined_ews − market` is
−0.0240 [−0.0692, +0.0218] (60-session paired blocks; both Bonferroni-adjusted
intervals also cross zero). Topology, regime-relative or not, still does not add AP
beyond the market baseline in the primary comparison.

But isolated against **raw topology levels** (not market), the regime-relative
representation is a real improvement, and at the two higher-power secondary targets
it clears even the Bonferroni-adjusted bound:

| target | positives | topology_z − topology | 95% CI | Bonferroni CI |
|---|---|---|---|---|
| 10% loss (primary) | 197 | +0.0174 | [−0.0108, +0.0401] | crosses zero |
| 7.5% loss | 389 | **+0.0339** | **[+0.0090, +0.0606]** | **[+0.0043, +0.0640]** |
| 5% loss | 805 | **+0.0404** | **[+0.0081, +0.0736]** | **[+0.0023, +0.0764]** |

`topology_ews` is not distinguishable from raw topology at any target (all intervals
cross zero). This is a modest, honestly bounded finding: normalising the same three
topology quantities against their own recent history measures something the raw
levels do not, at least for topology taken alone — but it still does not make
topology add value once ordinary market features are already in the model
(`results/improved/comparisons.csv`, `figures/topology_representation_comparison.png`).

## Finding 4: event recall is mixed, not uniformly better or worse

At the primary 5% calibration budget, `combined_ews` and `combined_z` detect 4 of 6
mechanical drawdown events (vs. 5/6 for the published `combined` and 6/6 for
`market`/VIX) at the 10% target, but `combined_ews` matches `market`/VIX at 6/6 at
the 5%-loss target and ties `combined` at 6/6 for the 7.5%-loss target. There is no
target at which any topology variant beats `market` on event recall. This is reported
in full in `results/improved/event_summary.csv`; no single number should be quoted
without the target it belongs to.

## What this changes and does not change

- The primary RQ1–RQ5 answers in `results/DEFENCE_REPORT.md` are unchanged and still
  the numbers to defend: topology does not demonstrate an information advantage over
  market and correlation controls in the primary 10%-loss, 5%-budget comparison.
- Two concrete, disclosable improvements exist and are recommended for the write-up:
  (1) the enhanced classic ladder should drop `eligible_assets`/`eligible_fraction` —
  a genuine coverage-artefact bug, not a modelling choice; (2) a regime-relative
  topology representation measurably outperforms raw topology levels when topology is
  evaluated alone, significant at the two higher-power secondary thresholds, though
  it still does not beat the market baseline or change the primary conclusion.
- Positive-weighted within-year AUROC is a useful companion statistic throughout;
  `combined_ews` in particular separates risky from calm days better within a year
  than any other primary group despite a lower pooled AP than market.
- No parameter, clip, window or threshold above was chosen after seeing its test
  score; all were fixed before this run. This remains
  retrospective development on already-inspected 2000–2024 history — the same caveat
  that applies to every number in the original release.

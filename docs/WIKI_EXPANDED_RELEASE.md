# WIKI-expanded data release — comparison against the public-data release

Following this project's own established practice (the filtered-panel release and the
public-refresh release were both retained, never overwritten), this release is also
retained separately: `results_public_data_release/` is the untouched, previously-published
release; `results/` (this file's sibling) is the new run described here.

## What changed

212 of the 421 equity histories missing from the public-data release were recovered from
Nasdaq Data Link's WIKI/PRICES dataset (free, community-maintained, frozen at 2018-03-27 --
covers 2000-2018 only, not 2019-2024) and merged into `inputs/public_prices_expanded.csv`
(915 symbols total, zero symbol overlap with the original 703). The full pipeline was
rerun on this expanded panel with the numerical protocol completely unchanged
(`config.json` untouched; `run_all.py`'s own `validate_protocol()` check passed).

| Coverage measure | Public-data release | WIKI-expanded release |
|---|---|---|
| Eligible equity histories | 600 | **794** (+194) |
| Former members included | 113 | **294** (+181) |
| Missing membership symbols | 421 | **209** (-212) |
| Matched member-sessions | 73.1% | **91.8%** |
| Eligible member-sessions after warm-up | 70.0% | **85.2%** |

This is a real, substantial reduction in the coverage gap that drove most of this
project's disclosed survivorship-bias caveat. It is not a full fix (209 symbols remain
unrecoverable for free, and the 212 recovered ones have zero 2019-2024 coverage), but it
is the single largest legitimate improvement made in this whole study.

## Primary equal-budget results (5% budget), old vs. new

| Model | Old AUROC | New AUROC | Old AP | New AP |
|---|---|---|---|---|
| VIX | 0.744 | 0.744 (unchanged -- external series) | 0.220 | 0.220 |
| market | 0.759 | 0.759 (unchanged -- SPY-only features) | 0.178 | 0.178 |
| combined | 0.748 | 0.754 | 0.176 | 0.181 |
| topology | 0.534 | **0.583** | 0.046 | 0.051 |

**RQ3 verdict, precisely**: `combined - market` AP difference moved from -0.0022 (95% CI
[-0.0195, +0.0085]) to **+0.0023** (95% CI [-0.0115, +0.0104]). The point estimate crossed
from negative to positive, but the confidence interval still comfortably includes zero in
both releases -- this is not a statistically established reversal, it is noise-level
movement on top of a still-inconclusive test. Reporting it as "topology now wins" would be
overclaiming; the honest statement is "no reliable increment was established before, and
none is established now -- but the point estimate is no longer negative."

## RQ1: event detection, old vs. new

| Model | Old (events detected / lead) | New (events detected / lead) |
|---|---|---|
| topology | 2/6, 9-session median lead | **3/6, 13-session median lead** |
| market, VIX | 6/6 | 6/6 (unchanged) |

A genuine, measurable improvement in topology's own event-detection quality. Still well
below market/VIX's complete recall.

## RQ4: transfer, old vs. new (the largest shift in the whole study)

| Universe | Old frozen-topology AUROC/AP | New frozen-topology AUROC/AP |
|---|---|---|
| credit_bond_ETFs | 0.464 / 0.022 | **0.660 / 0.053** |
| international_indices | 0.557 / 0.065 | **0.830 / 0.144** |

Paired significance (vs. target_momentum / target_volatility, 60-session blocks): 3 of 4
comparisons moved from a confidence interval entirely below zero (topology significantly
worse) to a confidence interval that now crosses zero (not statistically distinguishable).
The fourth (credit vs. target_volatility) remains significantly negative. Read precisely:
**transfer improved substantially and is no longer clearly worse than the target-informed
baselines in most comparisons -- but it still does not clearly beat them.** Point estimates
still favour the conventional, target-trained baselines in every comparison.

## RQ5: unchanged, as expected

Sector-fund results are identical to the public-data release -- RQ5 uses each sector
ETF's own price/topology history, which the equity-universe expansion does not touch.
This is a consistency check that passed, not an oversight.

## What this does and doesn't establish

This is real, verified, legitimate improvement from fixing a real, disclosed data
limitation -- not from search, tuning, or selective reporting. The headline verdict on
RQ3 (does topology add value beyond market) is **unchanged**: no reliable increment is
established, before or after. What changed is the *quality of the evidence* the negative
finding rests on, and a genuine narrowing of the transfer gap. Both releases remain
available, hash-verifiable, and neither is silently preferred over the other -- exactly
the project's own standing practice for its earlier release comparisons.

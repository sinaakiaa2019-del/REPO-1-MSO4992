# Orthogonal-topology addendum (post-hoc, not pre-declared)

Added after inspecting phase 1's results, so this is retrospective, not part of the declared improvement protocol's pre-declared protocol. It never touches phase 1's evaluate_groups() call, so phase 1's byte-identical reproduction of the published baseline is unaffected.

Reproduction check: Small differences from phase 1 (see orthogonal_reproduction_check.csv): this addendum's shared dates start ~40 sessions later because the orthogonalization regression needs that much prior history, which can shift the earliest purged split boundaries.

## What orthogonalization tests

Each of the three topology features is replaced by its residual after a trailing 60-session linear regression on the market group, refit at every date using only the 60 prior sessions -- no future information. The residual is close to uncorrelated with market by construction, so `combined_orthogonal - market` is a cleaner test of genuinely separate information than the raw `combined - market` comparison, where a real but small effect can be masked by collinearity.

## Results

### Target: SPY loss of at least 10.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap |
| --- | --- | --- | --- | --- |
| VIX | 0.7440 | 0.2198 | 0.5958 | 0.2471 |
| combined | 0.7599 | 0.1902 | 0.5956 | 0.2147 |
| combined_orthogonal | 0.7615 | 0.1831 | 0.5936 | 0.2144 |
| market | 0.7643 | 0.1949 | 0.5918 | 0.2164 |
| topology | 0.6135 | 0.0546 | 0.5848 | 0.1780 |
| topology_orthogonal | 0.4741 | 0.0389 | 0.5255 | 0.1764 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| combined_orthogonal | market | -0.0118 | -0.0344 | 0.0045 | 0.1460 |
| topology_orthogonal | topology | -0.0157 | -0.0425 | -0.0040 | 0.0120 |
| combined_orthogonal | combined | -0.0071 | -0.0361 | 0.0175 | 0.3980 |
| combined_orthogonal | market | -0.0118 | -0.0251 | 0.0034 | 0.1180 |
| topology_orthogonal | topology | -0.0157 | -0.0400 | -0.0017 | 0.0100 |
| combined_orthogonal | combined | -0.0071 | -0.0262 | 0.0157 | 0.3540 |
| combined_orthogonal | market | -0.0118 | -0.0216 | 0.0034 | 0.1160 |
| topology_orthogonal | topology | -0.0157 | -0.0431 | -0.0004 | 0.0220 |
| combined_orthogonal | combined | -0.0071 | -0.0228 | 0.0114 | 0.3620 |

### Target: SPY loss of at least 7.5% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap |
| --- | --- | --- | --- | --- |
| VIX | 0.6927 | 0.2277 | 0.5011 | 0.2524 |
| combined | 0.6939 | 0.2189 | 0.4973 | 0.2345 |
| combined_orthogonal | 0.6920 | 0.2227 | 0.5093 | 0.2403 |
| market | 0.7036 | 0.2287 | 0.5016 | 0.2391 |
| topology | 0.6223 | 0.1144 | 0.5491 | 0.2601 |
| topology_orthogonal | 0.4621 | 0.0760 | 0.5120 | 0.2406 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| combined_orthogonal | market | -0.0060 | -0.0172 | 0.0049 | 0.1320 |
| topology_orthogonal | topology | -0.0384 | -0.0709 | -0.0177 | 0.0000 |
| combined_orthogonal | combined | 0.0038 | -0.0103 | 0.0187 | 0.7140 |
| combined_orthogonal | market | -0.0060 | -0.0164 | 0.0057 | 0.1580 |
| topology_orthogonal | topology | -0.0384 | -0.0686 | -0.0124 | 0.0000 |
| combined_orthogonal | combined | 0.0038 | -0.0080 | 0.0184 | 0.6680 |
| combined_orthogonal | market | -0.0060 | -0.0166 | 0.0058 | 0.1940 |
| topology_orthogonal | topology | -0.0384 | -0.0685 | -0.0111 | 0.0040 |
| combined_orthogonal | combined | 0.0038 | -0.0082 | 0.0163 | 0.7000 |

### Target: SPY loss of at least 5.0% within 20 sessions

| model | auroc | average_precision | within_year_auroc | within_year_ap |
| --- | --- | --- | --- | --- |
| VIX | 0.6582 | 0.3259 | 0.4919 | 0.3343 |
| combined | 0.6459 | 0.3245 | 0.4987 | 0.3303 |
| combined_orthogonal | 0.6454 | 0.3161 | 0.4940 | 0.3251 |
| market | 0.6529 | 0.3185 | 0.4865 | 0.3205 |
| topology | 0.5543 | 0.2222 | 0.5027 | 0.3432 |
| topology_orthogonal | 0.4434 | 0.1564 | 0.5049 | 0.3391 |

| model | reference | ap_difference | lower95 | upper95 | share_draws_favouring_model |
| --- | --- | --- | --- | --- | --- |
| combined_orthogonal | market | -0.0024 | -0.0116 | 0.0079 | 0.3240 |
| topology_orthogonal | topology | -0.0658 | -0.1079 | -0.0262 | 0.0000 |
| combined_orthogonal | combined | -0.0084 | -0.0304 | 0.0144 | 0.2920 |
| combined_orthogonal | market | -0.0024 | -0.0112 | 0.0092 | 0.3680 |
| topology_orthogonal | topology | -0.0658 | -0.1168 | -0.0197 | 0.0020 |
| combined_orthogonal | combined | -0.0084 | -0.0258 | 0.0132 | 0.2880 |
| combined_orthogonal | market | -0.0024 | -0.0110 | 0.0080 | 0.3400 |
| topology_orthogonal | topology | -0.0658 | -0.1176 | -0.0206 | 0.0000 |
| combined_orthogonal | combined | -0.0084 | -0.0225 | 0.0109 | 0.2680 |

## Multiple-testing correction across the whole improvement study

Benjamini-Hochberg FDR control at 5%, applied jointly to every 60-session-block comparison in phase 1 and this addendum (45 tests). 0 survive; see full_family_fdr_correction.csv for every p-value, q-value and pass/fail flag. Approximate two-sided p-values come from the share of the 500 paired bootstrap draws on each side of zero.

No comparison in the full family survives 5% FDR control.


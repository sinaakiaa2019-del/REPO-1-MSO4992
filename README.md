# TDA crash prediction — strengthened real-data study

This project tests whether the shape of return correlations helps forecast large SPY falls. Extracting the ZIP gives you the observed input prices, code and results directly, with no extra wrapping folder to open first. Data cover 2000–2024; annual tests cover 2007–2024.

## How to run it

### 1. Install

Use Python 3.12 or later. Extract the ZIP and open a terminal inside the extracted folder.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

From here on, "python" means that environment's Python — on Windows, substitute `.\.venv\Scripts\python.exe` if you have not activated the virtual environment. Installing needs internet access; running the pipeline on the included data does not.

### 2. Run the full pipeline

```text
python run_all.py --prices inputs/public_prices_expanded.csv
```

Use `public_prices_expanded.csv`, not the older `public_prices.csv`. The expanded file folds in recovered price histories for delisted names and is what the results shipped with this release were actually built from — the older file still runs, but fingerprints differently and will not reproduce these numbers.

This runs all fourteen stages in order — input validation, mathematical checks, daily TDA, coverage, classic ML, sensitivity, transfer, sectors, balanced comparisons, reporting, final checks, publish, the improvement study, and figures — and finishes with `Finished. Read results/REPORT.md.` Missing real inputs cause a clean failure, never a silent synthetic fallback. On ordinary hardware a full run takes around 40 minutes, almost all of it the improvement-study stage; the core daily-TDA stage alone finishes in under two minutes.

Reruns replace local results, so copy the folder first if you want to keep this release's numbers untouched. A failed run removes the old success manifest. The numerical protocol is locked: changing `config.json` alone is unsupported.

### 3. Run one stage at a time (optional)

Useful when checking a single change rather than the whole pipeline:

```text
python run_all.py --stage tda --input-dir inputs --prices inputs/public_prices_expanded.csv
```

Stages must run in this order, since later ones read files earlier ones write:

```text
data, math, tda, coverage, ml, sensitivity, transfer, sectors, defence, report, checks, publish, improvements, figures
```

Running one out of order — for example `coverage` before `tda` — fails with a missing-file error rather than silently using stale data.

### 4. Verify results

```text
python run_all.py --verify-results
python tests/test_saved.py
python tests/test_math.py
```

`--verify-results` checks classic scores, saved alarms, split dates and file hashes using the standard library. `tests/test_saved.py` additionally verifies strengthened and sector metrics, calibration, selections and transfer results straight from the delivered database — no TDA cache needed, so it works on a downloaded copy with no run at all. `tests/test_math.py` runs small, self-contained mathematical and timing checks; it is also the one `pytest tests/` discovers and runs by default. A fourth file, `tests/test_run.py`, checks a real completed run for data leakage and provenance, but needs a TDA cache already on disk — run it after your own real run, not on the delivered snapshot alone.

### 5. Open the dashboard

```text
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open the URL printed in the terminal, normally http://localhost:8501. The dashboard reads saved results and has five tabs: main results, research questions, warnings and extensions, data quality, and method and files. Change the calibration budget between 1%, 5% and 10% to inspect warning trade-offs. This is a local saved-results dashboard, not a public deployment or a live signal. If the port is occupied, choose `--server.port 8502`. Stop it with Ctrl+C in the server terminal.

---

## Extra information

### Headline result

**Start with results/DEFENCE_REPORT.md.** Combined AUROC is 0.7538 and average precision is 0.1805; VIX scores 0.7440 and 0.2198. The small AUROC advantage does not establish better crash prediction. All weaker results are retained.

This release removes the supplied future-dependent price filter and uses a traceable public-price snapshot. Historical coverage is poorer, however. Survivorship bias is not eliminated. RQ5 now has a disclosed sector-fund proxy study; historical company-sector leadership remains unresolved. No dissertation grade is guaranteed — the contribution is a controlled, auditable test with explicit negative findings and public-data limitations.

### Read the package

| File | Purpose |
|---|---|
| results/DEFENCE_REPORT.md | Main results and RQ1–RQ5 answers |
| METHODOLOGY.md | Definitions, timing, equations and limitations |
| results/defence_metrics.csv | Ten primary models at three warning budgets |
| results/rq5_metrics.csv | Nine funds and three sector model groups |
| results/defence_transfer_metrics.csv | Transfer results and target controls |
| results/evidence.sqlite | All detailed predictions, tuning, splits and audits |
| results/REPORT.md | Separate classic 14-model comparison on the refreshed data |
| results/manifest.json | Versions, stage status and delivery hashes |
| results/improved/IMPROVEMENT_REPORT.md | Findings from the improvement study: a coverage-artefact fix and a regime-relative topology test |
| figures/README.md | Index of the dissertation PNG figures and what each supports |

Markdown and CSV files can be opened without running Python. Detailed results are consolidated in SQLite to avoid hundreds of loose files.

### Project layout

The code is organised into three folders, one per role, plus a shared top level.

- `data_pipeline/` — Octavian Gurau-Golban's part. Loads and validates the raw inputs, builds the correlation matrices, assembles the feature table, and runs the sector-fund analysis.
- `topology_analyser/` — Kunal Jha's part. The actual topological data analysis: Vietoris-Rips persistence, Betti curves, persistence images and the Wasserstein distance between consecutive days. Split into one small file per concern rather than one large one.
- `ml/` — Sina Kia's part. Model fitting, purged cross-validation, calibration, the robustness and improvement-study extensions, and the cross-asset transfer tests.
- The project root holds what all three depend on: `config.py` / `config.json` (shared paths and the locked protocol), the entry points (`run_all.py`, `run_pipeline.py`), reporting (`report_builder.py`, `feature_visualiser.py`), and the dashboard (`streamlit_app.py`).
- `tests/` holds three checks rather than one combined file: `test_math.py` (small, self-contained mathematical checks — the one `pytest` runs by default), `test_run.py` (checks a completed real run for leakage and provenance), and `test_saved.py` (checks the delivered database matches what it claims).

Every file still imports its neighbours the same way it always did (`from matrix_builder import correlation`, and so on); `config.py` adds `data_pipeline/` and `ml/` to the import path so those imports keep resolving no matter which folder the importing file lives in.

### Module responsibilities

| Module | Responsibility |
|---|---|
| config.py / config.json | Paths, runtime and fixed protocol |
| data_pipeline/data_loader.py | Download, archive and input validation |
| data_pipeline/data_processing.py | Input audit and historical coverage |
| data_pipeline/matrix_builder.py | Weighted centred correlation and shrinkage |
| data_pipeline/feature_builder.py | Trailing features and forward labels |
| data_pipeline/sector_builder.py | Sector-fund delay topology and matched controls |
| topology_analyser/ | Daily persistence and checkpoints, split into tda_config.py (the epsilon grid), wasserstein.py (diagram distance), betti_curves.py (landscape and network summaries), persistence_images.py (saved-image loading) and persistence.py (the ripser pipeline itself) |
| ml/model_training.py | Fit, purged selection, calibration and metrics |
| ml/run_ml.py | Classic models and optional older research extension |
| ml/robustness_builder.py | Balanced comparisons, events, scale and uncertainty |
| ml/international_builder.py | Transfer and target controls |
| report_builder.py / feature_visualiser.py | Reports, evidence database and classic plots |
| run_all.py / run_pipeline.py | Main entry point and retained wrapper |
| streamlit_app.py | Saved-results dashboard |
| tests/test_math.py | Mathematical checks; what pytest runs by default |
| tests/test_run.py | Real-run leakage and provenance checks |
| tests/test_saved.py | Delivered-database checks |

The optional `python ml/run_ml.py --research` runs the older broad joint search after a real-data run. It is not this release's primary analysis; old filtered-panel search scores are not included as fresh results.

Original project roles: Octavian Gurau-Golban owns `data_pipeline/`, Kunal Jha owns `topology_analyser/`, Sina Kia owns `ml/`. Everything at the project root and in `tests/` is shared between the three. Implementation/evaluation revisions in this release should not be attributed to the original authors without their review.

### Inputs and provenance

The expanded snapshot (`public_prices_expanded.csv`) contains 4,461,004 observations across 915 equity ticker histories, spanning 6,289 SPY trading sessions. Of the 915 supplied symbols, 794 are actually used inside eligible dated-membership windows, including 294 former members; 121 supplied symbols never get used and 209 membership symbols have no matching price history at all. The older `public_prices.csv` (kept for comparison, see `docs/WIKI_EXPANDED_RELEASE.md`) covers fewer names and years.

| Input | Contract |
|---|---|
| public_prices_expanded.csv | date,symbol,adjusted_close; unique date/symbol keys; the file actually used to build this release |
| public_prices.csv | Same contract; an earlier, smaller snapshot kept for comparison, not what these results were built from |
| membership.csv | date,tickers; dated comma-separated snapshots |
| markets.csv | Date index, SPY and supporting ETF/index prices |
| vix.csv | DATE,CLOSE; actual VIX |
| yield_curve.csv | Date index and actual T10Y2Y |
| sector_etfs.csv | Date index and XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY |
| source_manifest.json | Real-data declaration, sources and hashes |
| public_download_audit.csv | Every request, status, metadata and response hash |
| corporate_actions.csv | Provider-reported split/distribution records |

The original panel supplies requested symbol names, not replacement values. No liquidity cap, current-survivor filter, future-neighbour deletion or guessed delisting loss enters the primary study. Invalid/missing prices break returns. A public adjusted price is not independent verification of security identity or corporate actions.

The optional separate provenance ZIP contains `public_source.sqlite`, the audit and manifest. In its `responses` table, `raw` is a zlib-compressed JSON response. The provenance archive is unnecessary for an included-price rerun.

### Replace data later

For replacement equity prices within the same study period:

```text
python run_all.py --prices "C:\Data\replacement_prices.csv"
```

For replacement supporting inputs:

```text
python run_all.py --input-dir "C:\Data\bundle" --prices "C:\Data\bundle\prices.csv"
```

Provide the required files above and update their source manifest. The sector stage currently reads the project's `inputs/sector_etfs.csv`, so also replace that file when using a different input bundle. Check membership, calendars, identities and column definitions before comparing scores.

To repeat the public acquisition for the same symbol universe:

```text
python data_pipeline/data_loader.py --refresh-public inputs/public_download_audit.csv
python run_all.py --prices inputs/public_prices_expanded.csv
```

The downloader reads the audit's symbol column before replacing the audit. It reuses successful responses in `cache/public_source.sqlite` and retries missing requests. Move that cache archive to a backup location first if an entirely new snapshot is wanted. Public availability and adjustments can change, so a new download need not match this one.

The experiment is locked to 2000–2024. Extending to new years requires updating calendar validation, outer years, protocol and tests together, before examining new outcomes. Permanent identifiers and dated sectors require a richer input contract; renaming a ticker column does not implement them.

### Export every detailed table

```text
python run_all.py --export-table table_index
python run_all.py --export-table defence_predictions
python run_all.py --export-table defence_event_leads
python run_all.py --export-table defence_tuning
python run_all.py --export-table rq5_event_leads
```

Exports appear in results/TABLE.csv. The table_index lists every table and row count. The documents table contains JSON audits, protocol and verification records.

| Table family | Evidence |
|---|---|
| defence_* | Balanced models, calibration, scale choices, events and uncertainty |
| defence_transfer_* | Frozen topology and target controls |
| rq5_* | All sector models, selections, comparisons and event warnings |
| predictions / annual_metrics / aggregate_metrics | Classic 14-model comparison |
| rq2_* / rq4_* | Retained earlier scale/transfer diagnostics |
| asset_coverage / coverage | Per-symbol and daily eligibility |
| documents | Input contracts, protocol, stage status and checks |

The classic models.pkl stores fitted classic models. The strengthened experiments deliver all scores and selections; rerunning refits their models. Load only trusted pickle files. Streamlit reads SQLite/CSV instead.

### Improvement study and dissertation figures

A follow-on study reproduces every published `defence_*` score to machine precision from `results/evidence.sqlite` and then tests two pre-registered additions: a regime-relative topology representation and a fix for a data-coverage time trend that had corrupted the classic "enhanced" ladder. Reproduce with:

```text
python run_all.py --stage improvements
python run_all.py --stage figures
```

The first writes `results/improved/` (tables, an evidence database and `IMPROVEMENT_REPORT.md`); the published `results/` folder and its hashes are untouched. The second renders the ten retained PNG figures in `figures/` — see `figures/README.md` for what each one shows and which result it supports. Neither script refits a model outside its own declared protocol or changes the primary RQ1–RQ5 answers in `results/DEFENCE_REPORT.md`.

### Troubleshooting and interpretation

- Missing dependency: install with the same Python used to execute.
- Missing input: correct the paths and real-data bundle.
- Hash mismatch: restore the exact release or complete a fresh run; do not hide changed numerical sources/results by editing hashes.
- Interrupted run: rerun the command to reuse valid checkpoints.
- Blank metric: a subset can lack positives; blank is not zero.
- Blank event lead: no warning, or incomplete coverage; inspect event recall.
- A 5% calibration budget can yield a higher future FPR.
- High AUROC with poor AP or one long alarm episode does not establish useful forecasting.

"""Reporting: paired uncertainty, alarm episodes and consolidated evidence."""

from config import ROOT, HERE, OUT
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from model_training import metrics


def make_research_report():
    """Explain the saved experiment and its research-question limits."""
    import sqlite3

    target = ROOT / "results/research"
    with sqlite3.connect(target / "evidence.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        metrics = [
            dict(r)
            for r in connection.execute(
                "SELECT * FROM metrics ORDER BY average_precision DESC"
            )
        ]
        comparisons = [dict(r) for r in connection.execute("SELECT * FROM comparisons")]
        choices = [
            dict(r)
            for r in connection.execute("SELECT * FROM selected WHERE year=2024")
        ]
        frequencies = list(
            connection.execute(
                "SELECT model, `group`, C, history, count(*) AS years FROM selected GROUP BY model, `group`, C, history ORDER BY model, years DESC"
            )
        )
        fallbacks = connection.execute(
            "SELECT count(*) FROM selected WHERE constant_fallback=1"
        ).fetchone()[0]
        reference = dict(
            connection.execute(
                "SELECT * FROM reference_price_checks LIMIT 1"
            ).fetchone()
        )
        extremes = list(
            connection.execute(
                "SELECT symbol,count(*) AS n FROM extreme_return_review WHERE member_on_observation_date=1 GROUP BY symbol ORDER BY n DESC"
            )
        )
        coverage = list(connection.execute("SELECT * FROM tda_coverage"))
        lead_rows = []
        connection.execute("DROP TABLE IF EXISTS crisis_lead_times")
        connection.execute(
            "CREATE TABLE crisis_lead_times (event TEXT, peak_date TEXT, model TEXT, first_alarm TEXT, lead_sessions INTEGER, alarm_days INTEGER, covered_sessions INTEGER)"
        )
        for event, peak in [("GFC", "2007-10-09"), ("COVID", "2020-02-19")]:
            for model in [
                "VIX",
                "selected_market",
                "selected_topology",
                "selected_combined",
                "selected_quality",
                "selected_landscape",
                "selected_network",
                "selected_ensemble",
            ]:
                days = list(
                    connection.execute(
                        "SELECT date,alarm FROM predictions WHERE model=? AND date<? ORDER BY date DESC LIMIT 60",
                        (model, peak),
                    )
                )
                days.reverse()
                alarms = [i for i, r in enumerate(days) if r["alarm"]]
                first = days[alarms[0]]["date"][:10] if alarms else None
                lead = len(days) - alarms[0] if alarms else None
                row = (event, peak, model, first, lead, len(alarms), len(days))
                lead_rows.append(row)
                connection.execute(
                    "INSERT INTO crisis_lead_times VALUES (?,?,?,?,?,?,?)", row
                )
        connection.commit()
    report = [
        "# Real-data settings experiment",
        "",
        "This is a retrospective development experiment on the same 2000–2024 history already examined. Settings were selected within earlier chronological validation; none is a confirmed universal or live-trading optimum.",
        "",
        "The original experiment is retained in the parent results folder. These additional results use the same crash target: a loss of at least 10% in one of the next 20 SPY trading sessions. Average precision (AP) is the primary metric; it is the step-weighted area under the precision–recall curve.",
        "",
        "## Results on shared test dates",
        "",
        "| Model | AUROC | AP | False-positive rate | Recall | False alarm episodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        report.append(
            f"| {row['model']} | {row['auroc']:.4f} | {row['average_precision']:.4f} | {row['false_positive_rate']:.2%} | {row['recall']:.2%} | {row['false_alarm_episodes']} |"
        )
    report += [
        "",
        f"All models use {metrics[0]['n']:,} test sessions and {metrics[0]['positives']} positive target days. Raw VIX remains an explicit benchmark; old models are rescored on exactly these dates.",
        "",
        "## What changed",
        "",
        "- Small regularised logistic models compare log VIX alone, VIX with changes/relative level/volatility, and a small market group with momentum, drawdown and yield curve.",
        "- Training is expanding, the latest 756 sessions, or the latest 1,260 sessions. Regularisation C is 0.01, 0.1 or 1. Scaling is fitted within each training fold.",
        "- Topology is reduced to entropy, total persistence per asset and Wasserstein change per square-root asset, or one normalised Betti value at epsilon 0.8040, 0.9045, 1.0050 or 1.1055.",
        "- TDA configurations are window/lambda/shrinkage 60/.94/.1 (original), 40/.94/.1, 120/.98/.1 and 60/.94/.2. The long-window configuration changes two settings jointly, so it cannot identify their separate effects.",
        "- A separate quality sensitivity masks absolute log returns above 0.5 before complete-window eligibility. This is not verified error correction: genuine crash returns can also be removed. It is excluded from the main topology and combined selection families.",
        "- All configurations use daily windows, dated members, adjusted-close log returns and no liquidity cap. Longer windows require more observed history; no missing prices are invented.",
        "- Additional representations across all five TDA configurations: top persistence-landscape L1/L2 summaries divided by asset count, plus a custom weighted closed-walk ratio and normalised Laplacian connectivity. The graph statistics are network baselines, not persistent homology; the ratio is not Barrat clustering. Landscape and network families may select the separately flagged return-masking configuration.",
        "- A fixed 50/50 average of the selected market and topology scores. The weight is fixed before this rerun; the score average is not a probability-calibration procedure. It remains retrospective development on previously examined outcomes.",
        "",
        "## Selection and uncertainty",
        "",
        "387 candidates are declared in protocol.json. Two purged chronological inner validation blocks select pooled AP. Candidate validation dates are identical. The original 252-session threshold calibration remains separate, and all training labels end before calibration starts. Calibration labels end before the annual test starts. Ties use the first declared candidate.",
        "",
        "The candidate budgets differ by family: 27 market, 180 main topology, 36 main combined, 54 separate quality (45 topology plus 9 combined), 45 landscape and 45 network candidates (nine each across all five TDA configurations). The quality topology and combined winners are evaluated separately. The ensemble model is a fixed combination of two already-selected models, not a separate candidate search. This is a search, not an equal-budget proof that topology helps.",
        "",
        f"Single-class outer training used a constant training-prevalence prediction in {fallbacks} model-years. The same rule applies inside validation. No difficult validation period is silently discarded.",
        "",
        "| Comparison | AP difference | Paired 95% interval |",
        "|---|---:|---:|",
    ]
    for row in comparisons:
        report.append(
            f"| {row['model']} minus {row['reference']} | {row['ap_difference']:+.4f} | [{row['lower_95']:+.4f}, {row['upper_95']:+.4f}] |"
        )
    report += [
        "",
        "Intervals use 500 paired moving-block draws of 60 sessions. They describe sampling uncertainty conditional on these predictions. They do not correct for multiple comparisons, repeated research on this history, model selection or overlapping crisis episodes. An interval excluding zero is not untouched-holdout confirmation.",
        "",
        "## Most recent annual choices",
        "",
        "These settings were selected before the 2024 test year, using only data available under that fold. They are not a recommendation for 2026: no post-2024 data was supplied.",
        "",
        "| Family | Selected group | C | Training sessions (0 = expanding) | Inner AP |",
        "|---|---|---:|---:|---:|",
    ]
    for row in choices:
        report.append(
            f"| {row['model']} | {row['group']} | {row['C']} | {row['history']} | {row['inner_ap']:.4f} |"
        )
    report += [
        "",
        "## Stability across years",
        "",
        "The most frequently selected configuration in each family is shown below. Frequency is descriptive; it must not be used to reconstruct earlier predictions with hindsight.",
        "",
        "| Family | Most frequent group | C | History | Years selected |",
        "|---|---|---:|---:|---:|",
    ]
    seen = set()
    for row in frequencies:
        if row["model"] not in seen:
            seen.add(row["model"])
            report.append(
                f"| {row['model']} | {row['group']} | {row['C']} | {row['history']} | {row['years']} / 18 |"
            )
    report += [
        "",
        "## Data quality and survivorship limits",
        "",
        "A spot-check identified a concrete raw-close mismatch: the supplied CFC close on 14 November 2007 is 129, whereas Countrywide's SEC prospectus reports 13.37 for that date. This is not explained by the choice between close and adjusted_close because the comparison uses raw close on both sides. [Countrywide prospectus](https://www.sec.gov/Archives/edgar/data/25191/000095012407005885/v35644orsv3asr.htm). The check is preserved in reference_price_checks. It flags the history for replacement from a verified source; it does not establish the identity of every other row.",
        "",
        "CBE, ACS and CFC account for 440, 113 and 57 of the member-date extreme returns respectively: 610 of 860 observations. CFC alternates between adjusted prices near 127 and 0.028 in late 2007. Such records make data contamination a substantive limitation of the empirical findings, not merely a theoretical caveat. The quality sensitivity only masks large jumps; it cannot establish that remaining smooth observations belong to the intended issuer.",
        "",
        "The supplied panel contains 1,017 ticker histories; 906 contribute to the original 60-session analysis, including 423 former members. The public membership history has 46 unmatched price symbols, and 98 supplied histories have no valid prices during their membership periods. These records remain in the unresolved_assets table. No tickers were renamed or prices backfilled without security-level evidence.",
        "",
        "860 extreme daily log returns occur while the corresponding ticker is an index member. They are retained in the original experiment and individually listed in extreme_return_review. Masking them is only a sensitivity analysis. Complete dated security identities, corporate-action verification, historical sector classifications and terminal delisting returns remain unavailable. Dated membership reduces survivor selection but cannot eliminate these remaining biases.",
        "",
        "The public SEC interfaces provide filings and company information; they do not supply the missing verified adjusted-close panel. Historical identifier tables are available through services such as WRDS, but the user has no such dataset. Sources: [SEC developer resources](https://www.sec.gov/about/developer-resources), [WRDS historical identifiers](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/using-compustat-historical-identifier-notebook/), [Shumway, The Delisting Bias in CRSP Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03818.x).",
        "",
        "## Reproduce and inspect",
        "",
        "From final, first run the standard real-data pipeline with your equity CSV, then `python run_ml.py --research`. Check the additional saved results with `python run_ml.py --verify-research`. The research command uses genuine cached real-data features; it has no synthetic fallback. It creates resumable hashed daily TDA checkpoints under cache/research. The supplied ZIP excludes the raw equity CSV and temporary caches.",
        "",
        "The research evidence.sqlite database holds predictions, annual metrics, tuning, selected settings, splits, comparisons and data-quality records. metrics.csv, comparisons.csv and selected_settings.csv are convenient exports. Original RQ1–RQ5 results remain in results/REPORT.md; this extension focuses on predictive settings and data sensitivity, and does not supply missing historical-sector or full transfer-universe data.",
    ]
    report += [
        "",
        "## Answers to the research questions",
        "",
        "### RQ1: Does topology warn earlier than VIX?",
        "",
        "The following is the same predefined 60-session pre-peak event study, now applied to the selected models. It is conditional on two named peaks. It is not a test across all possible crises, and an earlier alarm is not necessarily more useful if false warnings are frequent.",
        "",
        "| Event | Model | First pre-peak alarm | Lead sessions | Alarm days in 60-session window |",
        "|---|---|---|---:|---:|",
    ]
    for event, peak, model, first, lead, alarms, covered in lead_rows:
        report.append(
            f"| {event} | {model} | {first or 'None'} | {lead if lead is not None else '—'} | {alarms} |"
        )
    report += [
        "",
        "Report these lead times alongside false-positive rates and false alarm episodes in the main table. The main supervised target concerns a future loss from the forecast date; this pre-peak event study asks a different question and should be presented separately.",
        "",
        "### RQ2: Is there a useful filtration or correlation setting?",
        "",
        "The annual selected-settings table and all candidate validation scores answer this within the tested grid. An epsilon-selected Betti feature represents one filtration scale; entropy, total persistence and Wasserstein change summarise diagrams across scales, so they do not identify one epsilon. Under d=sqrt(2(1-rho)), epsilon maps to a threshold on the shrunk correlation. This is not the original unshrunk sample correlation. A setting chosen in some years is evidence of a conditional choice, not a universal optimum.",
        "",
        "### RQ3: Does topology improve prediction over conventional information?",
        "",
        "Compare selected_combined against selected_market and VIX on exactly the same test sessions, using AP first and AUROC second. The paired AP interval is supplied above. Compare selected_topology separately to distinguish information in topology alone from information contributed by VIX. A higher combined score alone does not establish that topology caused the improvement. Search budgets differ and the whole history has already influenced research choices.",
        "",
        "### RQ4: Does the signal transfer to credit and international markets?",
        "",
        "The retained original transfer experiment reports credit-basket LR topology AUROC 0.7347/AP 0.0714 and international-index LR topology AUROC 0.9047/AP 0.1429. These are exploratory small-basket results, not validation of this newly selected parameter grid. The international LR result has one long alarm episode, so its high AUROC should not be described as successful prediction of many independent crises. Full constituent-level transfer remains untested without target panels and suitable dated identities.",
        "",
        "### RQ5: Which sectors lead systemic stress?",
        "",
        "The supplied data cannot support a defensible historical sector ranking because dated classifications are unavailable. The correct conclusion is that this research question is unresolved with the available inputs. Current sector labels would introduce historical classification and survivor-selection problems. Present this as a data limitation and a concrete extension, rather than supplying an unsupported ranking.",
        "",
        "## How to make the dissertation stronger",
        "",
        "State the contribution as a reproducible test of whether changing correlation topology adds predictive information beyond conventional stress measures. Explain the correlation matrix, loops and their changes in plain language before introducing equations. Show one diagram/Betti example, one chronological split diagram, the shared-date model table and the paired comparison interval.",
        "",
        "Explain why AP is primary for a target with few positive days; why a calibrated 5% false-alarm budget can drift out of sample; why many positive days can belong to the same crash; and why a high AUROC from a small transfer basket is limited evidence. Discuss negative results as findings, not defects to hide.",
        "",
        "Keep the synthetic demonstration in the methods-validation section, explicitly labelled as intentionally easy. It demonstrates that the code can recover a planted signal, not that it predicts actual crashes with similar accuracy. List public-data gaps, all attempted configurations and unresolved RQs. These practices support a credible submission, but no numerical score or implementation can guarantee a dissertation grade.",
    ]

    report += [
        "",
        "## TDA coverage under each setting",
        "",
        "These are observed eligible member counts, not liquidity caps. The 40-session setting can admit a history earlier; the 120-session setting requires a longer complete history. All scored models still share the same test dates.",
        "",
        "| Configuration | Daily windows | Minimum assets | Mean assets | Maximum assets |",
        "|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {r['configuration']} | {r['daily_windows']} | {r['minimum_assets']} | {r['mean_assets']:.1f} | {r['maximum_assets']} |"
        for r in coverage
    )
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        audit = json.loads(
            connection.execute(
                "SELECT content FROM documents WHERE name='survivorship_audit.json'"
            ).fetchone()[0]
        )
        transfer = list(
            connection.execute(
                "SELECT * FROM rq4_transfer_aggregate WHERE model='LR_topology'"
            )
        )
    for i, paragraph in enumerate(report):
        if paragraph.startswith("Compare selected_combined against"):
            scores = {r["model"]: r for r in metrics}
            difference = next(
                r
                for r in comparisons
                if r["model"] == "selected_combined"
                and r["reference"] == "selected_market"
            )
            conclusion = (
                "The conditional interval supports a positive increment within this experiment."
                if difference["lower_95"] > 0
                else (
                    "The conditional interval indicates worse combined performance within this experiment."
                    if difference["upper_95"] < 0
                    else "The interval crosses zero, so this experiment does not establish a reliable topology increment."
                )
            )
            report[i] = (
                f"Selected topology alone achieves AUROC {scores['selected_topology']['auroc']:.4f} and AP {scores['selected_topology']['average_precision']:.4f}. "
                f"The selected combined model achieves AP {scores['selected_combined']['average_precision']:.4f}, versus "
                f"{scores['selected_market']['average_precision']:.4f} for the selected market model and {scores['VIX']['average_precision']:.4f} for VIX. "
                f"The combined-minus-market AP difference is {difference['ap_difference']:+.4f}, with paired interval "
                f"[{difference['lower_95']:+.4f}, {difference['upper_95']:+.4f}]. {conclusion} "
                "Unequal search budgets, repeated examination of this history and the documented data discrepancy limit stronger conclusions."
            )
        elif paragraph.startswith("The annual selected-settings table and"):
            frequent = next(r for r in frequencies if r["model"] == "selected_topology")
            report[i] = (
                f"The most frequent topology-only choice was {frequent['group']}, C={frequent['C']}, "
                f"history={frequent['history']} (zero means expanding), selected in {frequent['years']} of 18 years. "
                "The annual settings and validation records give the other choices. Compact persistence summaries integrate over filtration scales; "
                "they do not identify one optimum epsilon. A chosen Betti coordinate identifies one tested epsilon, whose corresponding "
                "shrunk-correlation threshold is 1-epsilon²/2. This coarse joint grid does not establish a universal filtration or correlation optimum."
            )
        elif paragraph.startswith("Report these lead times alongside"):
            earlier = sum(
                1
                for event in ["GFC", "COVID"]
                if next(
                    r[4]
                    for r in lead_rows
                    if r[0] == event and r[2] == "selected_topology"
                )
                is not None
                and (
                    next(r[4] for r in lead_rows if r[0] == event and r[2] == "VIX")
                    is None
                    or next(
                        r[4]
                        for r in lead_rows
                        if r[0] == event and r[2] == "selected_topology"
                    )
                    > next(r[4] for r in lead_rows if r[0] == event and r[2] == "VIX")
                )
            )
            report[i] = (
                f"Topology alone gives an earlier pre-peak warning than VIX, or a warning where VIX gives none, in {earlier} of these two windows. "
                "This conditional event count is insufficient to establish generally earlier warning. Read it with the false-positive rates "
                "and false alarm episodes above. The supervised future-loss target and the pre-peak event study answer different questions."
            )
        elif paragraph.startswith("A spot-check identified"):
            report[i] = (
                f"Historical raw-close reference check: {reference['symbol']} on {reference['date']}; "
                f"supplied close {reference['supplied_close']}, SEC reference {reference['reference_close']}; "
                f"status: {reference['status']}. A mismatch flags the history for investigation and verified replacement. "
                "One checked quote cannot validate or repair an entire adjusted-price series. "
                f"[Countrywide prospectus]({reference['source']}). Full evidence is in reference_price_checks."
            )
        elif paragraph.startswith("CBE, ACS and CFC"):
            report[i] = (
                "The most frequent member-date extreme-return flags are "
                + ", ".join(f"{r['symbol']}: {r['n']}" for r in extremes[:3])
                + ". The quality sensitivity masks large jumps; it cannot establish that remaining smooth observations belong to the intended issuer."
            )
        elif paragraph.startswith("The supplied panel contains"):
            report[i] = (
                f"The supplied panel contains {audit['supplied_symbols']:,} ticker histories; "
                f"{audit['supplied_symbols_used']} contribute to the original analysis, including "
                f"{audit['former_member_symbols_used']} former members. There are "
                f"{audit['missing_historical_price_symbols']} unmatched price symbols and "
                f"{audit['supplied_symbols_with_no_membership_price_overlap']} supplied histories with no membership-price overlap. "
                "These gaps remain documented in unresolved_assets. No unverified prices or ticker mappings were invented."
            )
        elif paragraph.startswith("860 extreme daily"):
            report[i] = (
                f"{sum(r['n'] for r in extremes)} extreme daily log returns occur while the ticker is an index member. "
                "The original analysis retains them. Masking is only a sensitivity analysis. Historical security identities, "
                "sector classifications and terminal delisting returns remain unresolved, so survivorship bias is not eliminated."
            )
        elif paragraph.startswith("The retained original transfer experiment reports"):
            report[i] = (
                "Retained original LR topology transfer results: "
                + "; ".join(
                    f"{r['universe']}: AUROC {r['auroc']:.4f}, AP {r['average_precision']:.4f}, {r['alarm_episodes']} alarm episodes"
                    for r in transfer
                )
                + ". These small-basket results do not validate the new parameter grid. A long warning episode is not evidence of repeatedly predicting independent crises. Full constituent-level transfer remains untested."
            )
    if (ROOT / "results/integration/data_audit.json").exists():
        report[0:0] = ["# Integrated rerun: user-filtered real prices", "",
            "This rerun uses the supplied prices_cleaned.csv. Cleaning uses later observations and full-history ticker exclusions. Model fitting and threshold calibration are temporally purged, but the cleaning is retrospective, so this is not a fully point-in-time backtest. Exclusion evidence is incomplete and some price-level arguments may confuse adjusted and raw prices. See ../integration/REPORT.md and data_audit.json. No unverified replacement price was inserted.", ""]
    (target / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def publish_research():
    """Mark a completed settings run after its log has been closed."""
    import hashlib

    target = ROOT / "results/research"
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(
        status="completed",
        source_hashes={p.name: digest(p) for p in ROOT.glob("*.py")},
        files={
            str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for p in target.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    )
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))


def md(df):
    """Convert a result table to compact Markdown."""

    def f(v):
        """Format a table value for readable academic reporting."""
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)

    return "\n".join(
        [
            "| " + " | ".join(df.columns) + " |",
            "| " + " | ".join(["---"] * len(df.columns)) + " |",
        ]
        + [
            "| " + " | ".join((f(x) for x in row)) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
    )


def make_defence_report():
    """Answer each RQ from the new saved experiments, with its scope explicit."""
    destination=ROOT/'results'
    m=pd.read_csv(OUT/'defence_metrics.csv')
    primary=m[m.budget==.05].set_index('model')
    comp=pd.read_csv(OUT/'defence_comparisons.csv')
    leads=pd.read_csv(OUT/'defence_event_leads.csv')
    sector_leads=pd.read_csv(OUT/'rq5_event_leads.csv')
    sector=pd.read_csv(OUT/'rq5_metrics.csv')
    transfer=pd.read_csv(OUT/'defence_transfer_metrics.csv')
    audit=json.loads((OUT/'survivorship_audit.json').read_text())
    source=json.loads((ROOT/'inputs/source_manifest.json').read_text())
    event_summary=leads[leads.fully_observed].groupby(['model','budget']).agg(
        events=('event','size'),detected=('lead_sessions','count'),median_lead_when_detected=('lead_sessions','median')).reset_index()
    event_summary['event_recall']=event_summary.detected/event_summary.events
    event_summary.to_csv(OUT/'defence_event_summary.csv',index=False)
    sector_summary=sector_leads[sector_leads.fully_observed].groupby(['symbol','sector_proxy','model']).agg(
        events=('event','size'),detected=('lead_sessions','count'),median_lead_when_detected=('lead_sessions','median')).reset_index()
    sector_summary.to_csv(OUT/'rq5_event_summary.csv',index=False)
    increment=comp[(comp.model=='combined')&(comp.reference=='market')&(comp.block_sessions==60)].iloc[0]
    corr_increment=comp[(comp.model=='combined_correlation')&(comp.reference=='market_correlation')&(comp.block_sessions==60)].iloc[0]
    conclusion=('The conditional interval is positive.' if increment.lower95>0 else
                'The interval includes zero; a reliable increment is not established.' if increment.upper95>=0 else
                'The conditional interval favours the market baseline.')
    lines=['# Critical evaluation and strengthened real-data results','',
        'All numbers below come from the executed public-data experiment. Earlier releases are retained separately, not overwritten or relabelled. A positive result was not a completion requirement.','',
        '## Data quality and survivorship','',
        f"The public refresh supplied {source['available_equity_symbols']} equity histories. Dated membership and complete windows admit {audit['supplied_symbols_used']} histories, including {audit['former_member_symbols_used']} former members. {audit['missing_historical_price_symbols']} historical membership symbols have no matching price history. Every requested symbol and failure is in inputs/public_download_audit.csv.",'',
        'There is no liquidity cap, current-survivor filter, future-neighbour deletion or substitution from the suspect original CSV. Only observed finite positive adjusted prices enter log returns; missing prices break both adjacent returns. Corporate actions and provider metadata are archived. This improves provenance and removes the supplied retrospective cleaning rule, but missing delisted histories, ticker reuse and historical adjustment vintages remain unresolved. Public quotes are not a permanent-ID database. Coverage loss can itself introduce selection bias.','',
        '## Shared-date prediction results','',
        f"The primary equal-budget comparison has {int(primary.loc['VIX','n']):,} test sessions and {int(primary.loc['VIX','positives'])} positive target days. The target is a 10% SPY loss during the next 20 observed sessions. AP is average precision, used here as PR-AUC.",'',
        md(primary.reset_index()[['model','auroc','average_precision','false_positive_rate','recall','false_alarm_episodes']]),'',
        'Each learned group has the same three regularisation candidates, the same two purged inner blocks and the same separate 252-session calibration. Raw VIX is a fixed benchmark. Equal search budgets do not imply equal model complexity. All tested scales and weaker models remain visible.','',
        '## RQ1: Does topology give earlier, useful warnings?','',
        'The evaluation now includes every mechanically detected 10% SPY high-water drawdown in the scored period. Each event lasts until recovery of its preceding peak. Warnings are measured in the 20 sessions strictly before the first 10% crossing. These event annotations are not predictors. The event definition differs from the supervised forward-loss label, so event recall and label-based false alarms must both be reported.','',
        md(event_summary[(event_summary.budget==.05)&event_summary.model.isin(['market','topology','combined','VIX'])]),'',
        'A missed event has no lead time; it is not assigned zero or removed from recall. Median lead time is conditional on detection. Read defence_event_leads and the 1%, 5%, 10% budget tables for every event and threshold. Few independent episodes and retrospective data prevent a general early-warning guarantee.','',
        '## RQ2: Which filtration scale is useful?','',
        'The new scale comparison holds window length (60), EWMA decay (.94), shrinkage (.1), model family and training history fixed. It compares four Betti coordinates at grid indices 80, 90, 100 and 110. Epsilon maps to shrunk correlation through rho=1-epsilon²/2. The annual scale choice uses earlier validation only; all outer scores are retained.','',
        md(pd.read_csv(OUT/'defence_scale_choices.csv')),'',
        'This answers relative scale performance within the declared grid. It does not establish a universal optimum or prove a correlation threshold transfers to other universes. The earlier broader joint search is a separate development exercise.','',
        '## RQ3: Does topology add information beyond market and correlation controls?','',
        f"Combined-minus-market AP is {increment.ap_difference:+.4f}, with paired 60-session-block interval [{increment.lower95:+.4f}, {increment.upper95:+.4f}]. {conclusion}",'',
        f"After adding ordinary correlation controls to both models, the topology increment is {corr_increment.ap_difference:+.4f}, interval [{corr_increment.lower95:+.4f}, {corr_increment.upper95:+.4f}]. This is the more demanding attribution comparison.",'',
        md(comp[(comp.model.isin(['combined','combined_correlation']))]),'',
        'Intervals use 500 paired draws for each of 20-, 60- and 120-session block lengths. These are sensitivity checks for dependence, not corrections for all repeated research or multiple comparisons. This history has no untouched holdout. A zero-crossing interval does not prove equivalence.','',
        '## RQ4: Does the model transfer?','',
        'The equal-budget US topology model is frozen for each annual target test. Target volatility and negative momentum are explicit baselines on exactly matching dates. Their thresholds use only already-resolved target labels; the topology threshold remains source-calibrated. This compares zero-shot transfer with target-informed conventional scores, not identical adaptation.','',
        md(transfer[['universe','model','n','positives','auroc','average_precision','false_alarm_episodes']]),'',
        'Read defence_transfer_comparisons for paired uncertainty and defence_transfer_predictions for all scores. Small ETF/index baskets and correlated target/source crises limit generalisation. A high AUROC from one warning episode is not evidence of repeatedly forecasting independent crises. The original frozen-model transfer remains available as a separate baseline.','',
        '## RQ5: Which sector proxies show earlier topology-related stress?','',
        'This is a disclosed proxy version of RQ5. It uses the nine actual Select Sector funds launched before 2000, rather than assigning current company sectors to the past. Each fund uses a trailing 60-return window and five-lag delay embedding, standardised within that window. Its topology measures the shape of return dynamics, not the within-sector stock correlation network. Fund exposures and sector classifications can change through time.','',
        md(sector[['symbol','sector_proxy','model','auroc','average_precision','false_positive_rate','recall']]),'',
        md(sector_summary[sector_summary.model=='sector_topology']),'',
        'The sector tables answer comparative performance and event warnings for these fund proxies. They do not recover historical firm-sector leadership. All nine funds and all three model groups are reported; rankings after viewing test outcomes are descriptive, not a validated sector-selection strategy. Paired AP intervals versus each fund’s conventional baseline are in rq5_comparisons.','',
        '## Overall academic assessment','',
        'RQ1–RQ4 now have broader or better-controlled empirical tests. RQ5 has an executed and explicitly narrower proxy study. The original company-sector question still requires dated classifications. The strongest contribution is an auditable test of incremental topology information, with negative and inconclusive findings retained. Public-data coverage, limited independent crises and repeated use of the same history remain material weaknesses. Neither AUROC nor this software package can guarantee a dissertation grade.','',
        '## Reproduce and inspect','',
        'Run python run_all.py --prices inputs/public_prices.csv after installing requirements.txt. That command performs data audit, TDA, classic models, sector proxies, equal-budget comparisons, transfer controls, reports and checks. Run python -m streamlit run streamlit_app.py for the dashboard. Detailed outputs are consolidated in results/evidence.sqlite; python run_all.py --export-table TABLE exports any table. Source responses are in the separate provenance archive supplied with this release.','',
        'Sources: [nested evaluation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [sector fund definitions](https://www.ssga.com/us/en/intermediary/capabilities/equities/sector-investing/select-sector-etfs), [issuer inception-date table](https://www.ssga.com/library-content/pdfs/etf/us/target-a-lower-cost-of-ownership-with-sector-etfs.pdf).'
    ]
    (destination/'DEFENCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name in ['defence_metrics','defence_comparisons','defence_event_summary','defence_scale_choices',
                 'defence_transfer_metrics','rq5_metrics','rq5_event_summary','rq5_comparisons']:
        import shutil
        shutil.copy2(OUT/f'{name}.csv',destination/f'{name}.csv')


def make_report():
    """Build the main written report from saved result tables."""
    pred = pd.read_csv(OUT / "predictions.csv", parse_dates=["date"])
    summary = pd.read_csv(OUT / "aggregate_metrics.csv")
    splits = pd.read_csv(OUT / "splits.csv")
    for r in splits.itertuples():
        assert pd.Timestamp(r.fit_label_end) < pd.Timestamp(r.calibration_start)
        assert pd.Timestamp(r.calibration_label_end) < pd.Timestamp(r.test_start)
    pivot = pred.pivot(index="date", columns="model", values="score")
    assert not pivot.isna().any().any()
    truth = (
        pred[pred.model == "VIX"]
        .set_index("date")
        .label.reindex(pivot.index)
        .to_numpy()
    )
    rng = np.random.default_rng(42)
    pairs = [
        ("LR_combined", "LR_base"),
        ("XGB_combined", "XGB_base"),
        ("LR_dynamic_combined", "LR_base"),
        ("LR_combined", "VIX"),
    ]
    pairs += [
        ("LR_enhanced_combined", "LR_enhanced_base"),
        ("XGB_enhanced_combined", "XGB_enhanced_base"),
        ("LR_enhanced_base", "LR_base"),
        ("XGB_enhanced_base", "XGB_base"),
    ]
    draws = {pair: [] for pair in pairs}
    n = len(pivot)
    for _ in range(500):
        starts = rng.integers(0, n - 60 + 1, size=int(np.ceil(n / 60)))
        ix = np.concatenate([np.arange(s, s + 60) for s in starts])[:n]
        if len(np.unique(truth[ix])) < 2:
            continue
        scores = {
            name: average_precision_score(truth[ix], pivot[name].to_numpy()[ix])
            for name in set(sum(([a, b] for a, b in pairs), []))
        }
        for pair in pairs:
            draws[pair].append(scores[pair[0]] - scores[pair[1]])
    cis = []
    for (a, b), d in draws.items():
        cis.append(
            dict(
                model=a,
                reference=b,
                delta_average_precision=float(
                    average_precision_score(truth, pivot[a])
                    - average_precision_score(truth, pivot[b])
                ),
                lower95=float(np.quantile(d, 0.025)),
                upper95=float(np.quantile(d, 0.975)),
                bootstrap_samples=len(d),
                block_sessions=60,
            )
        )
    pd.DataFrame(cis).to_csv(OUT / "paired_block_bootstrap.csv", index=False)
    eventrows = []
    idx = pivot.index
    for event, date in [("GFC", "2007-10-09"), ("COVID", "2020-02-19")]:
        before = idx[idx < pd.Timestamp(date)][-60:]
        for name, g in pred.groupby("model"):
            a = g[g.date.isin(before) & g.alarm]
            first = a.date.min() if len(a) else pd.NaT
            eventrows.append(
                dict(
                    event=event,
                    peak_date=date,
                    model=name,
                    first_alarm=first,
                    lead_sessions=(
                        int(((idx >= first) & (idx < pd.Timestamp(date))).sum())
                        if pd.notna(first)
                        else None
                    ),
                    alarm_days=len(a),
                    covered_sessions=int(g.date.isin(before).sum()),
                )
            )
    pd.DataFrame(eventrows).to_csv(OUT / "crisis_lead_times.csv", index=False)
    pos = truth == 1
    starts = np.flatnonzero(pos & ~np.r_[False, pos[:-1]])
    episodes = []
    for start in starts:
        stop = start + 1
        while stop < n and pos[stop]:
            stop += 1
        for name, g in pred.groupby("model"):
            a = g.set_index("date").reindex(idx).alarm.iloc[start:stop].to_numpy()
            episodes.append(
                dict(
                    model=name,
                    episode_start=idx[start],
                    episode_end=idx[stop - 1],
                    positive_days=stop - start,
                    detected=bool(a.any()),
                    first_alarm=(
                        idx[start + np.flatnonzero(a)[0]] if a.any() else pd.NaT
                    ),
                )
            )
    ep = pd.DataFrame(episodes)
    ep.to_csv(OUT / "positive_label_episode_detection.csv", index=False)
    ep.groupby("model").detected.agg(["count", "sum", "mean"]).to_csv(
        OUT / "episode_detection_summary.csv"
    )
    periodrows = []
    for label, start, stop in [
        ("2007-2015", "2007", "2016"),
        ("2016-2024", "2016", "2025"),
    ]:
        for name, g in pred.groupby("model"):
            g = g[(g.date >= start) & (g.date < stop)].sort_values("date")
            periodrows.append(
                dict(period=label, model=name, **metrics(g.label, g.score, g.alarm))
            )
    pd.DataFrame(periodrows).to_csv(OUT / "period_metrics.csv", index=False)
    annual = pd.read_csv(OUT / "annual_metrics.csv")
    assert annual.calibration_fpr.max() <= 0.05 + 1e-10
    for row in summary.itertuples():
        g = pred[pred.model == row.model]
        assert abs(roc_auc_score(g.label, g.score) - row.auroc) < 1e-10
    verification = dict(
        shared_test_dates=True,
        fit_calibration_test_horizons_disjoint=True,
        calibration_negative_day_budget_verified=True,
        saved_predictions_reproduce_auroc=True,
        outer_years=len(splits),
        models=len(summary),
        retrospective_only=True,
    )
    (OUT / "evaluation_verification.json").write_text(
        json.dumps(verification, indent=2)
    )
    coverage = pd.read_csv(OUT / "coverage.csv")
    chosen = summary
    notes = "These are retrospective development results on an already inspected dataset, not an untouched final test. The prior release is retained separately. Compare the matched models within this run; changes from older releases can also reflect different training histories.\n\nNo liquidity cap is used. A complete 60-SPY-session return window is required per included asset. Weighted centering and 10% identity shrinkage replace zero-filled uncentered covariance. This removes missing-return imputation but excludes more incomplete histories. Window length, decay and shrinkage were fixed before results; they were not optimised on test outcomes.\n\nThe 5% budget constrains negative-day false positives in the separate pre-test calibration year. It does not guarantee 5% on future data. Daily false alarms, alarm episodes and time under warning are distinct quantities. Thresholds are not optimised against outer test outcomes. Consecutive positive-label episodes are not independent economic crises.\n\nTwo hyperparameter candidates per learner are selected with two purged chronological inner validation blocks; final fitting ends before calibration begins. Calibration has 252 observations with known 20-session outcomes. Scaling is fitted inside training only. No oversampling or synthetic training observations are used. VIX and yield slope are lagged one session; market-price features are available after the close.\n\nPR-AUC here means average precision. The headline scores pool saved annual predictions on identical dates; annual and two-period tables are also supplied. Block-bootstrap intervals are approximate and do not remove the dependence on a small number of crises. Searching several feature/model families adds selection uncertainty.\n\nPermanent security IDs, historical sector classifications, full foreign/credit constituent panels, and historical macro vintages remain unavailable. These public-data limits are documented. No current-survivor sector result is promoted to historical sector leadership. RQ4 baskets, if reported, are exploratory. RQ5 remains unsupported. Current CIK references are not asserted to be historical security identifiers. No ticker renaming is guessed.\n\nDo not use the legacy results as the new run's outputs. Every V2 output lives under outputs/v2 and window caches are keyed by code, protocol and input hashes.\n"
    audit_file = OUT / "data_audit.json"
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        notes += f"\nData-quality review flagged {audit.get('extreme_returns_flagged', 'unknown')} absolute daily log returns above 0.5 across the historical price panel; {audit.get('extreme_returns_while_index_member', 'unknown')} occurred while the symbol was an index member. These are review flags, not proven errors. They remain uncorrected without reliable corporate-action/security-identity evidence. See extreme_returns_membership_review.csv. This is a material limit on empirical conclusions.\n"
    (OUT / "LIMITATIONS.md").write_text(notes, encoding="utf-8")
    text = [
        "# Revised real-data run",
        "",
        f"Daily topology windows: {len(coverage):,}; eligible assets: {coverage.eligible.min()}–{coverage.eligible.max()}. No liquidity cap.",
        "",
        "## Matched-date model results",
        "",
        md(
            chosen[
                [
                    "model",
                    "auroc",
                    "average_precision",
                    "false_positive_rate",
                    "recall",
                    "false_alarm_episodes",
                ]
            ]
        ),
        "",
        "## Incremental topology value: paired 60-session block bootstrap",
        "",
        md(pd.DataFrame(cis)),
        "",
        "## Event lead times",
        "",
        md(pd.DataFrame(eventrows)),
        "",
        "## Interpretation and limitations",
        "",
        notes,
    ]
    scales = pd.read_csv(OUT / "filtration_selected.csv")
    text.extend(
        [
            "",
            "## Remaining research questions",
            "",
            f"RQ2: training-selected epsilon ranges from {scales.epsilon.min():.4f} to {scales.epsilon.max():.4f}; annual selections are in filtration_selected.csv.",
        ]
    )
    transfer_file = OUT / "rq4_transfer_aggregate.csv"
    if transfer_file.exists():
        text.extend(
            [
                "",
                "RQ4: exploratory frozen-model transfer on small real baskets; these are not full constituent-panel findings.",
                "",
                md(
                    pd.read_csv(transfer_file)[
                        [
                            "universe",
                            "model",
                            "n",
                            "auroc",
                            "average_precision",
                            "recall",
                        ]
                    ]
                ),
            ]
        )
    else:
        text.extend(["", "RQ4: unavailable; no transfer output was generated."])
    best = summary.loc[summary.average_precision.idxmax()]
    text.extend(
        [
            "",
            "RQ5: unavailable without dated historical sector classifications.",
            "",
            "## Interpretation",
            "",
            f"The highest pooled average precision is {best.average_precision:.4f} ({best.model}). This is a descriptive ranking of retrospective experiments, not an independently selected winner.",
            "",
            "Check the paired differences above to judge whether additional features helped the same learner. An interval containing zero does not establish an improvement. The intervals are not adjusted for testing multiple models. The calibration false-alarm budget is not a guarantee on later years.",
        ]
    )
    reference_file = OUT / "reference_price_checks.csv"
    if reference_file.exists():
        reference_checks = pd.read_csv(reference_file)
        if reference_checks.status.eq("mismatch").any():
            text[1:1] = [
                "",
                "**Data-quality qualification:** at least one supplied historical raw close disagrees with its dated reference. Empirical conclusions remain provisional until affected histories are verified. See the reference_price_checks evidence table.",
                "",
            ]
    (OUT / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    print("REPORT COMPLETE", flush=True)


def publish():
    """Collect detailed evidence in one database and expose a few useful files."""
    import sqlite3
    import shutil
    import hashlib
    import platform
    from importlib.metadata import version

    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    temporary = destination / "evidence.build.sqlite"
    temporary.unlink(missing_ok=True)
    table_index = []
    with sqlite3.connect(temporary) as connection:
        for path in sorted(OUT.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            frame.to_sql(path.stem, connection, index=False, if_exists="replace")
            table_index.append(dict(table=path.stem, rows=len(frame)))
        documents = [
            {"name": p.name, "content": p.read_text(encoding="utf-8")}
            for p in sorted(OUT.glob("*.json"))
        ]
        pd.DataFrame(documents, columns=["name", "content"]).to_sql(
            "documents", connection, index=False, if_exists="replace"
        )
        pd.DataFrame(table_index).to_sql(
            "table_index", connection, index=False, if_exists="replace"
        )
        connection.commit()
    # sqlite3's context manager commits but does not close the connection.
    connection.close()
    temporary.replace(destination / "evidence.sqlite")
    from feature_visualiser import build_chart

    build_chart()
    for source, target in [
        ("aggregate_metrics.csv", "model_results.csv"),
        ("paired_block_bootstrap.csv", "model_comparisons.csv"),
        ("asset_coverage.csv", "asset_coverage.csv"),
    ]:
        shutil.copy2(OUT / source, destination / target)
    report = (OUT / "RESULTS.md").read_text(encoding="utf-8")
    report = report.replace(
        "Every V2 output lives under outputs/v2 and window caches are keyed by code, protocol and input hashes.",
        "Detailed tables are stored in evidence.sqlite. Export any table with python run_all.py --export-table TABLE_NAME. Window caches are keyed by code, protocol and input hashes.",
    )
    audit = json.loads((OUT / "survivorship_audit.json").read_text())
    report += "\n\n## Historical asset coverage\n\n"
    report += (
        f"All {audit['supplied_symbols']:,} supplied symbols were audited; "
        f"{audit['supplied_symbols_used']:,} entered at least one eligible index window. "
        f"{audit['former_member_symbols_used']:,} used symbols were no longer members at the end of 2024. "
        f"{audit['missing_historical_price_symbols']:,} membership symbols had no punctuation-equivalent price symbol. "
        "The asset_coverage.csv table gives the reason for every unused supplied symbol. "
        "These figures describe ticker coverage, not verified unique economic securities. "
        "Membership handling reduces survivorship bias but cannot eliminate missing-history, identity or delisting-return bias.\n"
    )
    report += "\nSee ../METHODOLOGY.md for definitions and sources. The database table_index lists every detailed table.\n"
    (destination / "REPORT.md").write_text(report, encoding="utf-8")
    shutil.copy2(HERE / "models.pkl", destination / "models.pkl")
    packages = {
        name: version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "ripser"]
    }
    paths = (
        list(ROOT.glob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
        + list(ROOT.glob("*.md"))
        + [ROOT / "config.json", ROOT / "requirements.txt"]
    )
    paths += [p for p in (ROOT / "inputs").iterdir() if p.is_file()]
    paths += [
        p
        for p in destination.iterdir()
        if p.is_file() and p.name not in ["manifest.json", "run.log"]
    ]
    manifest = dict(
        python=platform.python_version(),
        packages=packages,
        evidence="retrospective real-data evaluation",
        raw_equity_csv_included=(ROOT/'inputs/public_prices.csv').exists(),
        stages=json.loads((OUT / "stage_status.json").read_text()),
        files={
            str(p.relative_to(ROOT))
            .replace("\\", "/"): hashlib.sha256(p.read_bytes())
            .hexdigest()
            for p in paths
        },
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))

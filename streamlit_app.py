"""Read-only dashboard for the final project results.

Run with: python -m streamlit run streamlit_app.py
"""

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


@st.cache_data
def load_results():
    """Load the final saved tables used by the dashboard."""
    with sqlite3.connect(RESULTS / "evidence.sqlite") as connection:
        metrics = pd.read_sql_query("SELECT * FROM defence_metrics", connection)
        events = pd.read_sql_query("SELECT * FROM defence_event_summary", connection)
        transfer = pd.read_sql_query("SELECT * FROM defence_transfer_metrics", connection)
        sectors = pd.read_sql_query("SELECT * FROM rq5_metrics", connection)
        coverage = pd.read_sql_query("SELECT * FROM coverage", connection, parse_dates=["date"])
    delay = pd.read_csv(RESULTS / "improved/delay_embedding_metrics.csv")
    classic = pd.read_csv(RESULTS / "model_results.csv")
    return metrics, events, transfer, sectors, coverage, delay, classic


@st.cache_data
def load_improvement_results():
    """Load the four pre-registered improvement-study extensions."""
    folder = RESULTS / "improved"
    return dict(
        alternative=pd.read_csv(folder / "alternative_learners_metrics.csv"),
        persistence_image=pd.read_csv(folder / "persistence_image_metrics.csv"),
        regime_pi=pd.read_csv(folder / "regime_pi_60d_metrics.csv"),
        xgb=pd.read_csv(folder / "xgb_extension_metrics.csv"),
    )


def show_figure(filename, caption):
    """Display a retained dissertation figure with a short explanation underneath."""
    path = FIGURES / filename
    if path.exists():
        st.image(str(path), width="stretch")
        st.caption(caption)


def formatted_metrics(frame, extra_columns=()):
    """Return the main measures needed to compare models, plainly labelled."""
    labels_by_name = {
        "model": "Model", "auroc": "AUROC", "average_precision": "Average precision",
        "recall": "Recall", "precision": "Precision",
        "false_positive_rate": "False-positive rate", "false_alarm_episodes": "False-alarm episodes",
        "within_year_auroc": "Within-year AUROC", "within_year_ap": "Within-year AP",
    }
    columns = ["model", "auroc", "average_precision", "recall", "precision",
               "false_positive_rate", "false_alarm_episodes", *extra_columns]
    result = frame[columns].copy()
    result.columns = [labels_by_name.get(c, c) for c in columns]
    return result


def family_snapshot(frame, models, target=None, budget=0.05):
    """A small, ordered comparison table for one improvement-study extension."""
    subset = frame[frame.budget == budget]
    if target is not None and "target" in subset.columns:
        subset = subset[subset.target == target]
    subset = subset[subset.model.isin(models)].copy()
    subset["model"] = pd.Categorical(subset["model"], categories=models, ordered=True)
    subset = subset.sort_values("model").reset_index(drop=True)
    return formatted_metrics(subset, extra_columns=["within_year_auroc"])


def apply_style():
    """Times New Roman, one consistent heading scale, no colour decoration."""
    st.markdown("""
    <style>
    html, body, [class*="st-"], .stApp, button, input, select, textarea {
        font-family: "Times New Roman", Times, serif !important;
    }
    h1 {font-size: 25pt !important; font-weight: 700 !important; margin-bottom: 0.2rem !important;}
    h2 {font-size: 16pt !important; font-weight: 700 !important;}
    h3, h4 {font-size: 13pt !important; font-weight: 700 !important;}
    p, div, span, td, th, li, label {font-size: 12pt !important;}
    .block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 20pt !important;}
    [data-testid="stMetricLabel"] {font-size: 10.5pt !important; color: #444 !important;}
    [data-testid="stCaptionContainer"] {color: #555 !important;}
    [data-testid*="Icon"], [class*="material-symbols"], [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)


FIGURE_CAPTIONS = {
    "primary_model_comparison.png": "Main AUROC and average-precision comparison.",
    "rq3_attribution_forest.png": "Uncertainty in the additional contribution of topology.",
    "pooled_vs_within_year_auroc.png": "Pooled and within-year AUROC comparison.",
    "universe_coverage_over_time.png": "Historical coverage of the equity universe.",
    "event_timeline.png": "Drawdown events and first warning dates.",
    "rq2_scale_grid.png": "Results across filtration scales.",
    "transfer_trained_diagnosis.png": "Cross-market transfer results.",
    "persistence_image_extension.png": "Persistence-image model results.",
    "delay_embedding_topology.png": "Delay-embedding extension results.",
    "topology_math_illustration.png": "Correlation networks and persistent loops.",
}


def main():
    """Build the project dashboard from saved evidence."""
    st.set_page_config(page_title="TDA Market Stress Study", page_icon="📈", layout="wide")
    apply_style()

    metrics, events, transfer, sectors, coverage, delay, classic = load_results()
    improvements = load_improvement_results()
    primary = metrics.query("budget == 0.05").set_index("model")

    st.title("Topological early-warning signals of equity-market stress")
    st.write("Persistent homology and machine learning are tested against conventional market indicators using data from 2000 to 2024. Out-of-sample evaluation covers 2007 to 2024.")

    overview, questions, warnings, evidence, improvement_tab, method, all_results, gallery = st.tabs(
        ["Main results", "Research questions", "Warnings and extensions", "Data quality",
         "Improvement study", "Method and files", "All results", "All figures"])

    with overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market AUROC", f"{primary.loc['market','auroc']:.3f}")
        c2.metric("Combined AUROC", f"{primary.loc['combined','auroc']:.3f}")
        c3.metric("VIX average precision", f"{primary.loc['VIX','average_precision']:.3f}")
        c4.metric("Combined average precision", f"{primary.loc['combined','average_precision']:.3f}")
        st.write("The combined model has a slightly higher AUROC than VIX. VIX has higher average precision. The evidence does not show that topology is generally better than conventional indicators.")
        st.divider()

        st.subheader("The two most important charts")
        left, right = st.columns(2)
        with left:
            show_figure("primary_model_comparison.png", FIGURE_CAPTIONS["primary_model_comparison.png"])
        with right:
            show_figure("topology_math_illustration.png", FIGURE_CAPTIONS["topology_math_illustration.png"])
        st.divider()

        st.subheader("Ten primary models, one calibration budget")
        order = ["VIX", "market", "combined", "combined_correlation", "scale_080", "topology"]
        st.dataframe(formatted_metrics(primary.loc[order].reset_index()), hide_index=True, width="stretch")
        st.divider()

        with st.expander("Classic 14-model comparison (results/REPORT.md)"):
            st.write("A separate, earlier comparison of logistic regression and XGBoost across four feature groups, run on the same refreshed data.")
            st.dataframe(formatted_metrics(classic), hide_index=True, width="stretch")

    with questions:
        st.subheader("Answers to the research questions")
        rows = [
            ("RQ1", "Early warning", "Topology alone detected 3 of 6 drawdown events. VIX and the market model detected all 6. The combined model detected 5 and gave a median lead of 14 sessions when it detected an event."),
            ("RQ2", "Filtration scale", "Performance changes across the tested filtration scales. The results do not establish one universal best scale."),
            ("RQ3", "Additional information", "The combined-minus-market gain in average precision is small and its confidence interval includes zero. Additional value is not established."),
            ("RQ4", "Transfer", "Topology transfers to some degree, especially for international ranking, but target-market volatility is stronger."),
            ("RQ5", "Sector evidence", "The sector-fund analysis is a proxy study. Conventional sector features generally perform better than sector topology."),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Question", "Topic", "Finding"]), hide_index=True, width="stretch")
        st.divider()

        st.subheader("Where the uncertainty comes from")
        show_figure("rq3_attribution_forest.png", "The intervals show uncertainty around the additional contribution of topology.")
        st.divider()

        st.subheader("Filtration-scale sensitivity")
        show_figure("rq2_scale_grid.png", "The chart compares the tested filtration scales.")

    with warnings:
        st.subheader("Warning behaviour at each calibration budget")
        budget = st.selectbox("Calibration warning budget", [0.01, 0.05, 0.10], index=1,
                              format_func=lambda value: f"{value:.0%}")
        st.write("The budget is set using earlier calibration observations. The later false-positive rate can be higher.")
        st.dataframe(events.query("budget == @budget"), hide_index=True, width="stretch")
        show_figure("event_timeline.png", "The timeline shows the six drawdown events and the first warnings from the main models.")
        st.divider()

        st.subheader("Delay-embedding extension")
        selected = delay.query("target == 'loss10' and budget == 0.05 and model in ['VIX','market','combined','combined_delay','delay_topology']")
        st.dataframe(formatted_metrics(selected), hide_index=True, width="stretch")
        st.write("Delay topology performs poorly alone. When combined with market variables, it reaches average precision of 0.194 and detects 5 of 6 events. The improvement is exploratory because its confidence interval includes zero.")
        show_figure("delay_embedding_topology.png", "The chart compares delay-embedding topology with the cross-sectional topology model.")

    with evidence:
        st.subheader("Data coverage")
        eligible = coverage.eligible.dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Price observations", "4,461,004")
        c2.metric("Price histories", "915")
        c3.metric("Eligible assets per day", f"{int(eligible.min())}–{int(eligible.max())}")
        st.write("The expanded price file covers 3 January 2000 to 31 December 2024. Nasdaq Data Link WIKI observations end in March 2018; Yahoo Finance supplies later observations. Historical ticker identity and delisting coverage remain incomplete.")
        show_figure("universe_coverage_over_time.png", "The chart shows how the eligible stock universe changes through time.")
        st.divider()

        st.subheader("Cross-market transfer")
        st.dataframe(transfer[["universe", "model", "auroc", "average_precision", "recall"]], hide_index=True, width="stretch")
        show_figure("transfer_trained_diagnosis.png", "The frozen equity-trained model applied without retraining to each target market.")
        st.divider()

        st.subheader("Sector-fund results")
        sector = sectors.query("budget == 0.05")
        symbol = st.selectbox("Sector fund", sorted(sector.symbol.unique()))
        st.dataframe(sector.query("symbol == @symbol")[["sector_proxy", "model", "auroc", "average_precision", "recall"]], hide_index=True, width="stretch")

    with improvement_tab:
        st.subheader("Four pre-registered extensions")
        st.write("A follow-on study reproduces every published defence_* score to machine precision from results/evidence.sqlite, then tests four additions declared before they were run: alternative learners, persistence images through XGBoost, a regime-normalised persistence image, and XGBoost on the raw topology features. None of them changes the primary RQ1–RQ5 answers.")
        st.divider()

        st.subheader("Pooled vs within-year AUROC")
        st.write("Pooled AUROC mixes telling a crisis year from a calm year with telling a risky day from a calm day inside the same year. The within-year figure isolates the harder, second task, and is reported alongside the pooled figure throughout this tab.")
        show_figure("pooled_vs_within_year_auroc.png", "Every model's within-year AUROC sits well below its pooled figure.")
        st.divider()

        st.subheader("Alternative learners: Random Forest and Lasso")
        st.dataframe(
            family_snapshot(improvements["alternative"],
                            ["VIX", "market", "combined", "market_rf", "combined_rf", "topology", "topology_rf", "topology_lasso"],
                            target="loss10"),
            hide_index=True, width="stretch")
        st.divider()

        st.subheader("Persistence images through XGBoost")
        st.dataframe(
            family_snapshot(improvements["persistence_image"],
                            ["VIX", "market", "combined", "topology", "topology_pi_xgb", "combined_pi_xgb"]),
            hide_index=True, width="stretch")
        show_figure("persistence_image_extension.png", "Persistence images add little over the three summary statistics already in the primary topology group.")
        st.divider()

        st.subheader("Regime-normalised persistence image")
        st.dataframe(
            family_snapshot(improvements["regime_pi"],
                            ["market", "topology_pi_xgb", "topology_pi_xgb_regime", "combined_pi_xgb_regime"],
                            target="loss10"),
            hide_index=True, width="stretch")
        st.divider()

        st.subheader("XGBoost on raw topology")
        st.dataframe(
            family_snapshot(improvements["xgb"],
                            ["VIX", "market", "combined", "topology", "topology_xgb", "combined_xgb"],
                            target="loss10"),
            hide_index=True, width="stretch")
        st.divider()

        with st.expander("What this changes and does not change"):
            st.write(
                "The primary RQ1–RQ5 answers in results/DEFENCE_REPORT.md are unchanged: topology "
                "does not demonstrate an information advantage over market and correlation controls "
                "in the primary 10%-loss, 5%-budget comparison. Two disclosable improvements are "
                "recommended for the write-up: the classic enhanced ladder should drop two "
                "coverage-artefact columns that were standing in for calendar time, and a "
                "regime-relative topology representation measurably outperforms raw topology levels "
                "when topology is evaluated alone — though it still does not beat the market baseline."
            )
        st.download_button("Download improvement-study tables (Markdown)",
                           (RESULTS / "improved/SUMMARY_TABLES.md").read_text(encoding="utf-8"),
                           "SUMMARY_TABLES.md", "text/markdown")

    with method:
        st.subheader("Method")
        st.write("The study uses 60-session rolling correlation networks, persistent homology, yearly out-of-sample tests, purged training labels and separate 252-session calibration periods. The primary target is a fall of at least 10% in SPY during the next 20 trading sessions.")
        show_figure("topology_math_illustration.png", "The figure illustrates the correlation network and persistent H1 loops on a calm date and a crisis date.")
        st.divider()

        st.subheader("Project files")
        st.markdown("- `results/DEFENCE_REPORT.md`: full results and answers\n- `METHODOLOGY.md`: definitions and limitations\n- `results/evidence.sqlite`: detailed predictions and validation records\n- `results/improved/IMPROVEMENT_REPORT.md`: the four extensions in full\n- `figures/README.md`: figure index\n- `README.md`: installation and reproduction instructions")
        st.download_button("Download primary metrics", metrics.to_csv(index=False), "defence_metrics.csv", "text/csv")

    with all_results:
        st.subheader("Every results table, complete and unfiltered")
        st.write("Every other tab shows a curated subset — one calibration budget, one loss target, a handful of models chosen for comparison. This tab has the complete table behind each of them, every row, every model, every budget.")
        tables = {
            "Primary defence metrics — all 10 models, all 3 budgets (30 rows)": metrics,
            "Classic 14-model comparison (14 rows)": classic,
            "Sector-fund metrics — all 9 funds, all models, all budgets (27 rows)": sectors,
            "Cross-market transfer metrics (6 rows)": transfer,
            "Delay-embedding extension — all targets and budgets (54 rows)": delay,
            "Alternative learners: Random Forest and Lasso (126 rows)": improvements["alternative"],
            "Persistence images through XGBoost (18 rows)": improvements["persistence_image"],
            "Regime-normalised persistence image (45 rows)": improvements["regime_pi"],
            "XGBoost on raw topology (63 rows)": improvements["xgb"],
        }
        choice = st.selectbox("Table", list(tables.keys()))
        chosen = tables[choice]
        st.dataframe(chosen, hide_index=True, width="stretch")
        st.download_button("Download this table as CSV", chosen.to_csv(index=False),
                           choice.split(" — ")[0].split(" (")[0].replace(" ", "_").lower() + ".csv", "text/csv")

    with gallery:
        st.subheader("Every retained figure, in one place")
        st.write("All ten dissertation figures, generated from saved results. Each one also appears next to the section it supports elsewhere in this dashboard.")
        for filename, caption in FIGURE_CAPTIONS.items():
            show_figure(filename, caption)
            st.divider()


if __name__ == "__main__":
    main()

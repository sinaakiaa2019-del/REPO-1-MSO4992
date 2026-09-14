"""Charts: a small SVG model-comparison chart, plus the full dissertation figure set."""

from pathlib import Path
import html
import sqlite3

ROOT = Path(__file__).resolve().parent


def build_figures():
    """Render the ten retained dissertation figures from saved results.

    Reads results/evidence.sqlite, results/improved/ and inputs/markets.csv only.
    Writes ten PNGs to figures/. Colours follow the project's light-mode palette.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    retained = {
        "primary_model_comparison.png", "rq3_attribution_forest.png",
        "pooled_vs_within_year_auroc.png", "universe_coverage_over_time.png",
        "event_timeline.png", "rq2_scale_grid.png",
        "transfer_trained_diagnosis.png", "persistence_image_extension.png",
        "delay_embedding_topology.png", "topology_math_illustration.png",
    }
    dpi = 200
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    )
    INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 12,
    })

    def style_axes(ax, x_grid=False, y_grid=True):
        """Apply the shared dissertation chart style to one axis."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        # matplotlib turns the grid ON if any style kwarg is passed, even with b=False --
        # so the off case must be its own bare call, not one call with a boolean flag.
        if y_grid:
            ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        else:
            ax.grid(False, axis="y")
        if x_grid:
            ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        else:
            ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    def savefig(fig, name, note=None):
        """Save a clean dissertation figure with consistent typography."""
        if name not in retained:
            plt.close(fig)
            return
        # Supporting detail belongs in the report caption, not below the chart.
        # Apply the required typeface and size even where an older chart supplied
        # a local font setting.
        for item in fig.findobj(match=matplotlib.text.Text):
            item.set_fontfamily("Nimbus Roman")
            item.set_fontsize(12)
        fig.savefig(figs / name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)

    PRIMARY_ORDER = ["VIX", "market", "combined", "combined_correlation", "scale_080",
                      "scale_090", "scale_100", "scale_110", "topology"]
    PALETTE_9 = [MUTED, BLUE, VIOLET, MAGENTA, AQUA, AQUA, AQUA, AQUA, ORANGE]

    with sqlite3.connect(ROOT / "results/evidence.sqlite") as con:
        defence_metrics = pd.read_sql_query("SELECT * FROM defence_metrics WHERE budget=0.05", con)
        scale_choices = pd.read_sql_query("SELECT * FROM defence_scale_choices", con)
        coverage = pd.read_sql_query("SELECT * FROM coverage", con, parse_dates=["date"])
        defence_events = pd.read_sql_query("SELECT * FROM defence_events", con, parse_dates=["onset", "recovery", "peak"])
        defence_predictions = pd.read_sql_query(
            "SELECT * FROM defence_predictions WHERE budget=0.05 AND model IN ('combined','VIX','market')",
            con, parse_dates=["date"],
        )

    improved_dir = ROOT / "results/improved"
    metrics = pd.read_csv(improved_dir / "metrics.csv")
    comparisons = pd.read_csv(improved_dir / "comparisons.csv")
    classic = pd.read_csv(improved_dir / "classic_correction.csv")
    spy = pd.read_csv(ROOT / "inputs/markets.csv", index_col=0, parse_dates=True).SPY.dropna()

    NOTE_PUBLISHED = "Retrospective 2007-2024 evaluation. Source: results/defence_metrics.csv (published)."
    NOTE_IMPROVED = "Retrospective 2007-2024 evaluation. Source: results/improved/ (declared in the declared improvement protocol)."

    # Figure 1: primary equal-budget model comparison, published.
    d = defence_metrics.set_index("model").loc[PRIMARY_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, col, title, xmax in [(axes[0], "auroc", "AUROC", 1.0), (axes[1], "average_precision", "Average precision (PR-AUC)", 0.28)]:
        ax.barh(y, d[col], color=PALETTE_9, height=0.62)
        for yi, v in zip(y, d[col]):
            ax.text(v + xmax * 0.012, yi, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        style_axes(ax, y_grid=False, x_grid=True)
        ax.set_yticks(y)
    axes[0].set_yticklabels([n.replace("_", " ") for n in d.index], fontsize=10.5)
    axes[0].axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    fig.suptitle("Primary equal-budget models: shared 4,510-session test, 5% calibration budget", fontsize=13.5, y=1.02)
    fig.tight_layout()
    savefig(fig, "primary_model_comparison.png", NOTE_PUBLISHED)

    # Figure 2: RQ3 attribution, paired AP-difference forest plot.
    rows = [
        ("combined - market", "combined", "market", "published"),
        ("combined_corr - market_corr", "combined_correlation", "market_correlation", "published"),
        ("combined_z - market", "combined_z", "market", "declared, primary family"),
        ("combined_ews - market", "combined_ews", "market", "declared, primary family"),
    ]
    primary_comp = comparisons[(comparisons.target == "loss10") & (comparisons.block_sessions == 60)]
    forest = []
    for label, a, b, kind in rows:
        r = primary_comp[(primary_comp.model == a) & (primary_comp.reference == b)]
        if len(r) == 0:
            r = pd.read_csv(ROOT / "results/defence_comparisons.csv")
            r = r[(r.model == a) & (r.reference == b) & (r.block_sessions == 60)]
        r = r.iloc[0]
        forest.append(dict(label=label, mid=r.ap_difference, lo=r.lower95, hi=r.upper95, kind=kind))
    forest = pd.DataFrame(forest)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(forest))[::-1]
    colors = [BLUE if k == "published" else VIOLET for k in forest.kind]
    for yi, row, c in zip(y, forest.itertuples(), colors):
        ax.plot([row.lo, row.hi], [yi, yi], color=c, linewidth=2.2, solid_capstyle="round")
        ax.plot(row.mid, yi, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(row.hi + 0.003, yi, f"{row.mid:+.4f} [{row.lo:+.4f}, {row.hi:+.4f}]", va="center", fontsize=9, color=INK2)
    ax.axvline(0, color=MUTED, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(forest.label, fontsize=11)
    ax.set_xlabel("Average-precision difference (paired, 60-session moving blocks, 95% CI)")
    ax.set_xlim(-0.11, 0.075)
    style_axes(ax, y_grid=False)
    ax.set_title("RQ3: does topology add information beyond market controls?", fontsize=13, loc="left")
    handles = [plt.Line2D([0], [0], color=BLUE, lw=2.2, marker="o", label="Published comparison"),
               plt.Line2D([0], [0], color=VIOLET, lw=2.2, marker="o", label="Declared regime-relative topology")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    savefig(fig, "rq3_attribution_forest.png", NOTE_PUBLISHED + " New rows: the declared improvement protocol.")

    # Figure 3: pooled vs. within-year AUROC.
    order = ["market_correlation", "market", "combined_ews", "combined", "VIX", "topology"]
    sel = metrics[(metrics.target == "loss10") & (metrics.budget == 0.05) & metrics.model.isin(order)]
    sel = sel.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(sel))
    w = 0.34
    ax.bar(x - w / 2, sel.auroc, width=w, color=BLUE, label="Pooled AUROC")
    ax.bar(x + w / 2, sel.within_year_auroc, width=w, color=ORANGE, label="Within-year AUROC (positive-weighted)")
    for xi, (pooled, within) in enumerate(zip(sel.auroc, sel.within_year_auroc)):
        ax.text(xi - w / 2, pooled + 0.012, f"{pooled:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, within + 0.012, f"{within:.3f}", ha="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in sel.model], fontsize=10.5)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("Pooled AUROC mixes year selection with day-level timing", fontsize=13, loc="left")
    fig.tight_layout()
    savefig(fig, "pooled_vs_within_year_auroc.png", NOTE_IMPROVED)

    # Figure 4: coverage-artefact fix, before/after.
    names = ["LR_enhanced_base", "LR_enhanced_combined", "XGB_enhanced_base", "XGB_enhanced_combined"]
    before = classic[classic.model.isin(names)].set_index("model").loc[names]
    after = classic[classic.model.isin([n + "_clean" for n in names])].set_index("model").loc[[n + "_clean" for n in names]]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(names))
    w = 0.34
    ax.bar(x - w / 2, before.auroc.to_numpy(), width=w, color=RED, label="Published (coverage variables included)")
    ax.bar(x + w / 2, after.auroc.to_numpy(), width=w, color=GREEN, label="Corrected (coverage variables removed)")
    for xi, (b, a) in enumerate(zip(before.auroc, after.auroc)):
        ax.text(xi - w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(len(names) - 0.5, 0.51, "constant-score line (AUROC 0.5)", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=10.5)
    ax.set_ylim(0.3, 0.85)
    ax.set_ylabel("Pooled AUROC")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("A coverage-count time trend, not model failure, explained the published AUROC below 0.5", fontsize=12.3, loc="left")
    fig.tight_layout()
    savefig(fig, "coverage_artifact_fix.png", NOTE_IMPROVED + " Removed: eligible_assets, eligible_fraction (Spearman correlation with calendar date = 1.000).")

    # Figure 5: eligible universe size over time.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(coverage.date, coverage.eligible, color=BLUE, linewidth=1.6)
    ax.fill_between(coverage.date, coverage.eligible, color=BLUE, alpha=0.10)
    ax.plot(coverage.date, coverage.expected_members, color=MUTED, linewidth=1.2, linestyle=(0, (4, 2)))
    ax.text(coverage.date.iloc[-1], coverage.eligible.iloc[-1], f"  eligible ({coverage.eligible.iloc[-1]})",
            color=BLUE, fontsize=9.5, va="center")
    ax.text(coverage.date.iloc[-1], coverage.expected_members.iloc[-1], f"  expected members ({coverage.expected_members.iloc[-1]})",
            color=MUTED, fontsize=9.5, va="center")
    ax.set_ylabel("Number of S&P 500 constituents")
    style_axes(ax)
    ax.set_title("Eligible modelling universe grows through the sample (survivorship-coverage context)", fontsize=12.5, loc="left")
    fig.tight_layout()
    savefig(fig, "universe_coverage_over_time.png",
            "Source: results/coverage.csv (published, exported via evidence.sqlite). No liquidity cap or current-survivor filter.")

    # Figure 6: topology representation comparison across targets.
    targets = [("loss10", "10% loss (primary)", 197), ("loss075", "7.5% loss", 389), ("loss05", "5% loss", 805)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (target, label, n) in zip(axes, targets):
        m = metrics[(metrics.target == target) & (metrics.budget == 0.05) &
                    metrics.model.isin(["topology", "topology_z", "topology_ews"])]
        m = m.set_index("model").loc[["topology", "topology_z", "topology_ews"]]
        ax.bar(range(3), m.average_precision, color=[MUTED, VIOLET, AQUA], width=0.58)
        for xi, v in enumerate(m.average_precision):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
        c = comparisons[(comparisons.target == target) & (comparisons.block_sessions == 60) &
                         (comparisons.model == "topology_z") & (comparisons.reference == "topology")].iloc[0]
        sig = "significant, 95% CI" if c.lower95 > 0 else "CI crosses zero"
        ax.set_title(f"{label}\n({n} positive days) -- z-score vs raw: {sig}", fontsize=10, loc="left")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["raw levels", "regime z-score", "early-warning\n(CV + trend)"], fontsize=9.5)
        style_axes(ax)
    axes[0].set_ylabel("Average precision")
    fig.suptitle("Topology-only models: raw levels vs. declared regime-relative representations", fontsize=13, y=1.05)
    fig.tight_layout()
    savefig(fig, "topology_representation_comparison.png", NOTE_IMPROVED)

    # Figure 7: event timeline, SPY price, drawdown events and first alarms.
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(spy.index, spy.to_numpy(), color=INK2, linewidth=1.0)
    for row in defence_events.itertuples():
        ax.axvspan(row.onset, row.recovery, color=RED, alpha=0.08)
    alarm_colors = {"VIX": ORANGE, "market": BLUE, "combined": VIOLET}
    markers = {"VIX": "^", "market": "s", "combined": "o"}
    for name, color in alarm_colors.items():
        g = defence_predictions[(defence_predictions.model == name) & (defence_predictions.alarm.astype(bool))]
        first_per_episode = g.sort_values("date").assign(gap=lambda d: d.date.diff().dt.days.fillna(999) > 3)
        first_per_episode["episode"] = first_per_episode.gap.cumsum()
        starts = first_per_episode.groupby("episode").first()
        ax.scatter(starts.date, spy.reindex(starts.date), color=color, marker=markers[name], s=34,
                   zorder=5, edgecolor=SURFACE, linewidth=0.8, label=f"{name} -- first alarm of each episode")
    ax.set_yscale("log")
    yticks = [80, 100, 150, 200, 300, 450, 650]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("SPY adjusted close (log scale)")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=3)
    ax.set_title("Mechanical 10% drawdown events (shaded) and alarm episodes at the 5% budget", fontsize=13, loc="left")
    fig.tight_layout()
    savefig(fig, "event_timeline.png",
            "Shaded bands: onset of a 10% high-water drawdown to recovery of the preceding peak (results/defence_events.csv). "
            "Markers: first alarm of each alarm episode, 5% calibration budget (results/defence_predictions.csv).")

    # Figure 8: RQ2 filtration-scale grid.
    scales = defence_metrics[defence_metrics.model.str.startswith("scale_")].copy()
    scales["index"] = scales.model.str.extract(r"(\d+)").astype(int)
    scales = scales.sort_values("index")
    freq = scale_choices.scale_index.value_counts().reindex(scales["index"]).fillna(0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(scales["index"].astype(str), scales.auroc, color=AQUA, width=0.58)
    for xi, v in enumerate(scales.auroc):
        axes[0].text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
    axes[0].set_ylim(0.6, 0.82)
    axes[0].set_xlabel("Filtration grid index")
    axes[0].set_ylabel("Pooled AUROC")
    axes[0].set_title("Pooled test AUROC by fixed scale", fontsize=11.5, loc="left")
    style_axes(axes[0], y_grid=True)
    axes[1].bar(freq.index.astype(str), freq.to_numpy(), color=VIOLET, width=0.58)
    for xi, v in enumerate(freq.to_numpy()):
        axes[1].text(xi, v + 0.15, str(v), ha="center", fontsize=9.5, color=INK)
    axes[1].set_ylim(0, 18)
    axes[1].set_xlabel("Filtration grid index")
    axes[1].set_ylabel("Years selected (of 18) by earlier-validation only")
    axes[1].set_title("How often each scale wins cross-scale inner selection", fontsize=11.5, loc="left")
    style_axes(axes[1], y_grid=True)
    fig.suptitle("RQ2: filtration scale -- descriptive after testing, not a validated universal optimum", fontsize=12.5, y=1.03)
    fig.tight_layout()
    savefig(fig, "rq2_scale_grid.png", NOTE_PUBLISHED + " Right panel: results/defence_scale_choices.csv.")

    # Figure 9: orthogonal-topology addendum.
    orth_comp = pd.read_csv(improved_dir / "orthogonal_addendum_comparisons.csv")
    orth_comp = orth_comp[orth_comp.block_sessions == 60]
    fdr = pd.read_csv(improved_dir / "full_family_fdr_correction.csv")
    fdr = fdr[fdr.source == "phase2_orthogonal_addendum"]
    target_labels = {"loss10": "10% loss (primary)", "loss075": "7.5% loss", "loss05": "5% loss"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.5))
    for ax, (a, b), title in zip(
        axes,
        [("topology_orthogonal", "topology"), ("combined_orthogonal", "market")],
        ["topology_orthogonal - topology\n(is topology's own signal just a market proxy?)",
         "combined_orthogonal - market\n(orthogonalized topology added to market)"],
    ):
        rows = orth_comp[(orth_comp.model == a) & (orth_comp.reference == b)].merge(
            fdr[(fdr.model == a) & (fdr.reference == b)][["target", "q_value_bh", "significant_fdr_5pct"]], on="target",
        )
        rows["target"] = pd.Categorical(rows.target, categories=list(target_labels), ordered=True)
        rows = rows.sort_values("target")
        y = np.arange(len(rows))[::-1]
        colors = [RED if sig else MUTED for sig in rows.significant_fdr_5pct]
        for yi, row, c in zip(y, rows.itertuples(), colors):
            ax.plot([row.lower95, row.upper95], [yi, yi], color=c, linewidth=2.4, solid_capstyle="round")
            ax.plot(row.ap_difference, yi, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
            note = f"  q={row.q_value_bh:.3f}" + (" *" if row.significant_fdr_5pct else " (n.s. after FDR)")
            ax.text(row.upper95 + 0.003, yi, f"{row.ap_difference:+.4f} [{row.lower95:+.4f}, {row.upper95:+.4f}]{note}",
                    va="center", fontsize=8.7, color=INK2)
        ax.axvline(0, color=MUTED, linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([target_labels[t] for t in rows.target], fontsize=10.5)
        ax.set_xlabel("Average-precision difference (60-session blocks, 95% CI)")
        ax.set_title(title, fontsize=11, loc="left")
        style_axes(ax, y_grid=False)
        ax.set_xlim(-0.11, 0.075)
    fig.suptitle("Post-hoc addendum: orthogonalizing topology against the market group", fontsize=13, y=1.06)
    fig.tight_layout()
    savefig(fig, "orthogonal_topology_addendum.png",
            "Post-hoc, not pre-declared (results/improved/ORTHOGONAL_ADDENDUM.md). Red = survives Benjamini-Hochberg "
            "5% FDR control jointly across all 45 block-60 comparisons in the improvement study; grey = does not.")

    # Figure 10: XGBoost topology extension.
    xgb_comp = pd.read_csv(improved_dir / "xgb_extension_comparisons.csv")
    xgb_comp = xgb_comp[xgb_comp.block_sessions == 60]
    fdr3 = pd.read_csv(improved_dir / "full_family_fdr_correction.csv")
    fdr3 = fdr3[fdr3.source == "phase3_xgb_extension"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.5))
    for ax, (a, b), title, xlim in zip(
        axes,
        [("topology_xgb", "topology"), ("market_xgb", "market")],
        ["topology_xgb - topology\n(does nonlinearity help extract topology's signal?)",
         "market_xgb - market (control)\n(is XGBoost just better at everything here?)"],
        [(-0.045, 0.115), (-0.30, 0.08)],
    ):
        rows = xgb_comp[(xgb_comp.model == a) & (xgb_comp.reference == b)].merge(
            fdr3[(fdr3.model == a) & (fdr3.reference == b)][["target", "q_value_bh", "significant_fdr_5pct"]], on="target",
        )
        rows["target"] = pd.Categorical(rows.target, categories=list(target_labels), ordered=True)
        rows = rows.sort_values("target")
        y = np.arange(len(rows))[::-1]
        colors = [RED if sig else (GREEN if a == "topology_xgb" else ORANGE) for sig in rows.significant_fdr_5pct]
        for yi, row, c in zip(y, rows.itertuples(), colors):
            ax.plot([row.lower95, row.upper95], [yi, yi], color=c, linewidth=2.4, solid_capstyle="round")
            ax.plot(row.ap_difference, yi, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(row.upper95 + (xlim[1] - xlim[0]) * 0.02, yi,
                    f"{row.ap_difference:+.4f} [{row.lower95:+.4f}, {row.upper95:+.4f}]",
                    va="center", fontsize=8.7, color=INK2)
        ax.axvline(0, color=MUTED, linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([target_labels[t] for t in rows.target], fontsize=10.5)
        ax.set_xlabel("Average-precision difference (60-session blocks, 95% CI)")
        ax.set_title(title, fontsize=11, loc="left")
        style_axes(ax, y_grid=False)
        ax.set_xlim(*xlim)
    fig.suptitle("Post-hoc phase 3: does a nonlinear learner extract more from topology than from market?", fontsize=13, y=1.06)
    fig.tight_layout()
    q_note = "; ".join(
        f"{r.target}: q={r.q_value_bh:.2f}" for r in pd.concat(
            [fdr3[(fdr3.model == "topology_xgb")], fdr3[(fdr3.model == "market_xgb")]]
        ).sort_values(["model", "target"]).itertuples()
    )
    savefig(fig, "xgb_topology_extension.png",
            "Post-hoc, motivated by the published classic ladder (XGB_topology already beat LR_topology there). None of these "
            "comparisons survives Benjamini-Hochberg 5% FDR control across all three improvement-study phases (57 tests): " + q_note + ".")

    # Figure 11: why fair in-basket-trained transfer underperforms.
    with sqlite3.connect(ROOT / "results/evidence.sqlite") as con4:
        credit_feat = pd.read_sql_query('SELECT * FROM "credit_bond_ETFs_transfer_features"', con4, parse_dates=["date"]).set_index("date")
        intl_feat = pd.read_sql_query('SELECT * FROM "international_indices_transfer_features"', con4, parse_dates=["date"]).set_index("date")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    by_year = credit_feat.groupby(credit_feat.index.year).label.sum()
    known_before = {y: int((credit_feat.index < f"{y}-01-01").sum()) for y in by_year.index}
    tested = pd.Series({y: known_before[y] >= 525 for y in by_year.index})
    colors_by_year = [GREEN if tested[y] else RED for y in by_year.index]
    axes[0].bar(by_year.index.astype(str), by_year.to_numpy(), color=colors_by_year, width=0.6)
    for i, (y, v) in enumerate(by_year.items()):
        if v > 0:
            label = f"{int(v)}\n" + ("tested" if tested[y] else "skipped:\n<525d history")
            axes[0].text(i, v + 2, label, ha="center", va="bottom", fontsize=8, color=INK)
    axes[0].set_title("Credit basket: positive days by year\n(green = enough in-basket history to test; red = skipped)", fontsize=10.5, loc="left")
    axes[0].set_ylabel("Positive-label days")
    axes[0].set_ylim(0, by_year.max() * 1.35)
    style_axes(axes[0])
    axes[0].set_xticks(range(0, len(by_year), 3))
    axes[0].set_xticklabels([str(y) for y in by_year.index[::3]], fontsize=9)

    counts = intl_feat.n_assets.value_counts().sort_index()
    h1_by_n = intl_feat.groupby("n_assets").h1_entropy_normalized.mean()
    axes[1].bar(counts.index.astype(str), counts.to_numpy(), color=[RED if h1_by_n.get(n, 1) == 0 else AQUA for n in counts.index], width=0.55)
    for i, (n, v) in enumerate(counts.items()):
        axes[1].text(i, v + counts.max() * 0.03, f"{v}\nH1 entropy={h1_by_n.get(n,0):.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    axes[1].set_title("International basket: eligible assets per day\n(red = too few points for H1 to carry any information)", fontsize=10.5, loc="left")
    axes[1].set_xlabel("Assets eligible that day (of 7 possible)")
    axes[1].set_ylabel("Days")
    axes[1].set_ylim(0, counts.max() * 1.25)
    style_axes(axes[1])

    fig.suptitle("Cross-market transfer: data availability and test coverage", fontsize=12.5, y=1.03)
    fig.tight_layout()
    savefig(fig, "transfer_trained_diagnosis.png",
            "Left: credit basket's topology-eligible history starts mid-2007, so the 2008-2009 GFC (68 of 91 positive days) never "
            "accumulates the 525 prior sessions this study's protocol requires before testing a year -- only COVID-2020 gets tested. "
            "Right: on 55% of days the international basket has only 3 assets, for which Vietoris-Rips H1 is structurally trivial "
            "(entropy is exactly 0.0 for every such day, not approximately -- confirmed from the raw published feature table).")

    # Figure 12: persistence images + XGBoost.
    pi_metrics = pd.read_csv(improved_dir / "persistence_image_metrics.csv")
    pi_metrics = pi_metrics[pi_metrics.budget == 0.05]
    order12 = ["VIX", "market", "combined", "combined_pi_xgb", "topology_pi_xgb", "topology"]
    pi_metrics = pi_metrics.set_index("model").loc[order12].reset_index()
    pi_leads = pd.read_csv(improved_dir / "persistence_image_event_leads.csv")
    pi_leads = pi_leads[(pi_leads.budget == 0.05) & pi_leads.fully_observed]
    lead_summary = (pi_leads.groupby("model").agg(events=("event", "size"), detected=("lead_sessions", "count"),
                                                   median_lead=("lead_sessions", "median")).reindex(order12))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors12 = [MUTED, BLUE, VIOLET, AQUA, GREEN, ORANGE]
    axes[0].barh(range(len(pi_metrics)), pi_metrics.auroc, color=colors12)
    for i, v in enumerate(pi_metrics.auroc):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    axes[0].set_yticks(range(len(pi_metrics))); axes[0].set_yticklabels(pi_metrics.model, fontsize=10)
    axes[0].set_xlim(0, 0.85); axes[0].set_title("AUROC", fontsize=10.5, loc="left")
    axes[0].invert_yaxis(); style_axes(axes[0], y_grid=False)

    axes[1].barh(range(len(pi_metrics)), pi_metrics.average_precision, color=colors12)
    for i, v in enumerate(pi_metrics.average_precision):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    axes[1].set_yticks(range(len(pi_metrics))); axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 0.24); axes[1].set_title("Average precision", fontsize=10.5, loc="left")
    axes[1].invert_yaxis(); style_axes(axes[1], y_grid=False)

    valid = lead_summary.dropna(subset=["median_lead"])
    axes[2].barh(range(len(valid)), valid.median_lead, color=[colors12[order12.index(m)] for m in valid.index])
    for i, (m, row) in enumerate(valid.iterrows()):
        axes[2].text(row.median_lead + 0.3, i, f"{row.median_lead:.0f}d ({int(row.detected)}/{int(row.events)})", va="center", fontsize=9, color=INK)
    axes[2].set_yticks(range(len(valid))); axes[2].set_yticklabels(valid.index, fontsize=10)
    axes[2].set_xlim(0, 16); axes[2].set_title("Median warning lead when detected", fontsize=10.5, loc="left")
    axes[2].invert_yaxis(); style_axes(axes[2], y_grid=False)

    fig.suptitle("Persistence-image models and market benchmarks", fontsize=12.5, y=1.05)
    fig.tight_layout()
    savefig(fig, "persistence_image_extension.png",
            "Post-hoc, motivated by Adams et al. (2017), already cited in the literature review. Best topology-alone AUROC in "
            "the whole study, but combined_pi_xgb-market AP difference is -0.071 [-0.151,-0.003] (significant, 60-session "
            "blocks) and event lead time drops to 2-4 days vs 12-14 for the simpler models -- a better ranker, not an earlier "
            "or more precise warner.")

    # Figure 13: phase 6, regime-normalized persistence images at a 60-session horizon.
    p6 = pd.read_csv(improved_dir / "regime_pi_60d_metrics.csv")
    p6 = p6[p6.budget == 0.05]
    target_labels6 = {"loss10": "10% loss (primary)", "loss075": "7.5% loss", "loss05": "5% loss"}
    order13 = ["market", "combined", "topology_pi_xgb", "topology_pi_xgb_regime", "combined_pi_xgb_regime"]
    colors13 = {"market": BLUE, "combined": VIOLET, "topology_pi_xgb": GREEN, "topology_pi_xgb_regime": RED, "combined_pi_xgb_regime": ORANGE}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, target in zip(axes, target_labels6):
        m = p6[p6.target == target].set_index("model").reindex(order13)
        y = np.arange(len(order13))[::-1]
        ax.barh(y, m.average_precision, color=[colors13[k] for k in order13])
        for yi, (name, v) in zip(y, m.average_precision.items()):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8.7, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(order13 if target == "loss10" else [], fontsize=9.5)
        ax.set_xlim(0, m.average_precision.max() * 1.35)
        ax.set_title(target_labels6[target], fontsize=11, loc="left")
        ax.set_xlabel("Average precision")
        style_axes(ax, y_grid=False)
    fig.suptitle("Phase 6: regime-normalizing a sparse, 400-pixel representation backfires (60-session horizon)", fontsize=12.5, y=1.04)
    fig.tight_layout()
    savefig(fig, "regime_persistence_60day.png",
            "Post-hoc, the priority-ranked hypothesis after reviewing phases 1-5. Result: regime-normalization "
            "significantly HURTS here (topology_pi_xgb_regime vs. topology_pi_xgb: AP difference -0.11 to -0.07, "
            "CI excludes zero at the two higher thresholds) -- the opposite of phase 1's finding on the compact "
            "3-feature topology set. Likely cause: most of the 400 persistence-image pixels are exactly zero most "
            "days, so per-pixel z-scoring mostly amplifies noise rather than revealing a meaningful regime.")

    # Figure 14: phase 7, four learner families on the same two feature groups.
    al = pd.read_csv(improved_dir / "alternative_learners_metrics.csv")
    al = al[(al.target == "loss10") & (al.budget == 0.05)]
    xgbm = pd.read_csv(improved_dir / "xgb_extension_metrics.csv")
    xgbm = xgbm[(xgbm.target == "loss10") & (xgbm.budget == 0.05)]
    learners = ["LR", "XGBoost", "Random Forest", "Lasso"]
    topology_ap = [
        metrics[(metrics.target == "loss10") & (metrics.budget == 0.05) & (metrics.model == "topology")].average_precision.iloc[0],
        xgbm[xgbm.model == "topology_xgb"].average_precision.iloc[0],
        al[al.model == "topology_rf"].average_precision.iloc[0],
        al[al.model == "topology_lasso"].average_precision.iloc[0],
    ]
    market_ap = [
        metrics[(metrics.target == "loss10") & (metrics.budget == 0.05) & (metrics.model == "market")].average_precision.iloc[0],
        xgbm[xgbm.model == "market_xgb"].average_precision.iloc[0],
        al[al.model == "market_rf"].average_precision.iloc[0],
        al[al.model == "market_lasso"].average_precision.iloc[0],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colors14 = [MUTED, ORANGE, GREEN, AQUA]
    for ax, values, title in zip(axes, [topology_ap, market_ap], ["topology group", "market group"]):
        ax.bar(learners, values, color=colors14, width=0.6)
        for xi, v in enumerate(values):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=12, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("Average precision")
    fig.suptitle("Phase 7: four learner families on the same two feature groups", fontsize=13, y=1.03)
    fig.tight_layout()
    savefig(fig, "alternative_learners.png",
            NOTE_IMPROVED + " Nonlinear/regularized learners (XGBoost, Random Forest, Lasso) noticeably help the "
            "topology-only group over plain logistic regression, but all three noticeably HURT the market-only "
            "group compared to plain logistic regression -- the 4 market features are already close to linearly "
            "separable, and added model complexity does not help there.")

    # Figure 15: phase 8, delay-embedded SPY dynamics vs. the cross-sectional network.
    de_metrics = pd.read_csv(improved_dir / "delay_embedding_metrics.csv")
    de_metrics = de_metrics[(de_metrics.target == "loss10") & (de_metrics.budget == 0.05)]
    de_leads = pd.read_csv(improved_dir / "delay_embedding_event_leads.csv")
    de_leads = de_leads[(de_leads.target == "loss10") & (de_leads.budget == 0.05) & de_leads.fully_observed]
    de_lead_summary = de_leads.groupby("model").agg(
        events=("event", "size"), detected=("lead_sessions", "count"), median_lead=("lead_sessions", "median")
    )
    order15 = ["market", "topology", "combined", "delay_topology", "combined_delay"]
    colors15 = [BLUE, MUTED, VIOLET, RED, GREEN]
    m15 = de_metrics.set_index("model").reindex(order15)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(order15, m15.average_precision, color=colors15, width=0.6)
    for xi, v in enumerate(m15.average_precision):
        axes[0].text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
    axes[0].set_ylabel("Average precision")
    axes[0].set_title("Average precision", fontsize=12, loc="left")
    style_axes(axes[0])

    lead15 = de_lead_summary.reindex(order15)
    axes[1].bar(order15, lead15.median_lead, color=colors15, width=0.6)
    for xi, (name, row) in enumerate(lead15.iterrows()):
        axes[1].text(xi, row.median_lead + 0.3, f"{row.median_lead:.0f}d ({int(row.detected)}/{int(row.events)})",
                     ha="center", fontsize=9, color=INK)
    axes[1].set_ylabel("Median warning lead (sessions)")
    axes[1].set_title("Event lead time when detected", fontsize=12, loc="left")
    style_axes(axes[1])
    for ax in axes:
        ax.set_xticks(range(len(order15)))
        ax.set_xticklabels(order15, rotation=15, ha="right")

    fig.suptitle("Delay-embedding and correlation-network topology", fontsize=12.5, y=1.03)
    fig.tight_layout()
    savefig(fig, "delay_embedding_topology.png",
            NOTE_IMPROVED + " delay_topology (SPY's own return series, 60-session/5-lag embedding) alone is "
            "worse than everything else, but combined_delay (added to market) reaches the best AP and longest "
            "lead time of any topology-informed model in the study -- though both comparisons against market "
            "and combined still cross zero (not significant at 95%).")

    # Figure 16: what the topology actually looks like -- a calm day vs. a crisis day.
    # Recomputes correlation/persistence for two single days from cached returns
    # (same functions as the real pipeline, not a new computation path). The tree is
    # a minimum spanning tree (Mantegna 1999 -- the source of this project's own
    # correlation-distance formula): strongest links only, so it has no loops by
    # construction. The loops below come from the full network, weaker links included.
    import networkx as nx
    from scipy.sparse.csgraph import minimum_spanning_tree
    from config import HERE
    from matrix_builder import correlation as build_correlation
    from topology_analyser import features as tda_features

    CALM_DATE, CRISIS_DATE = "2017-06-01", "2020-03-23"
    returns = pd.read_pickle(HERE / "returns.pkl")
    member16 = (pd.read_csv(HERE / "membership_fja.csv", parse_dates=["date"])
                .set_index("date").tickers.sort_index().reindex(returns.index, method="ffill"))
    colmap16 = {s.replace(".", "-"): i for i, s in enumerate(returns.columns)}
    values16 = returns.to_numpy()

    def day_snapshot(date_str):
        """Calculate the correlation and persistence data for one date."""
        t = returns.index.get_indexer([pd.Timestamp(date_str)], method="nearest")[0]
        names = sorted(set(member16.iloc[t].split(",")))
        mapped = [colmap16[s.replace(".", "-")] for s in names if s.replace(".", "-") in colmap16]
        block = values16[t - 59:t + 1]
        valid = np.isfinite(block).all(axis=0) & (np.std(block, axis=0) > 1e-12)
        eligible = [i for i in mapped if valid[i]]
        c = build_correlation(block[:, eligible])
        _, h1, _, _, _ = tda_features(c)
        return returns.index[t], c, h1

    def mst_graph(c):
        """Create a minimum-spanning-tree view of a correlation matrix."""
        d = np.sqrt(np.clip(2 * (1 - c), 0, None))
        np.fill_diagonal(d, 0)
        tree = minimum_spanning_tree(d).toarray()
        G = nx.Graph()
        G.add_nodes_from(range(len(c)))
        edges = np.argwhere(tree > 0)
        for i, j in edges:
            G.add_edge(int(i), int(j), weight=1.0 / tree[i, j])
        return G

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for col, (label, date_str, color) in enumerate([("Calm", CALM_DATE, BLUE), ("Crisis", CRISIS_DATE, RED)]):
        date, c, h1 = day_snapshot(date_str)
        G = mst_graph(c)
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.2 / np.sqrt(len(G)))
        ax = axes[0, col]
        degrees = dict(G.degree())
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=BASE, width=1.0, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[18 + 10 * degrees[n] for n in G.nodes()],
                                node_color=color, alpha=0.85, linewidths=0)
        ax.set_title(f"{label}: {date.date()}  ({len(G)} assets, minimum spanning tree)", fontsize=11.5, loc="left")
        ax.axis("off")

        ax2 = axes[1, col]
        if len(h1):
            lo, hi = 0, max(h1.max() * 1.1, 0.1)
            ax2.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
            ax2.scatter(h1[:, 0], h1[:, 1], color=color, s=22, alpha=0.75, edgecolor=SURFACE, linewidth=0.4)
            ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
        ax2.set_xlabel("Birth"); ax2.set_ylabel("Death")
        ax2.set_title(f"H1 persistence diagram ({len(h1)} loops)", fontsize=11.5, loc="left")
        style_axes(ax2, x_grid=True)
    fig.suptitle("Correlation network shape and persistent H1 loops: calm vs. crisis", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "topology_math_illustration.png",
            "Recomputed from cache/returns.pkl for two single days (not a search -- 2020-03-23 is the "
            "well-known COVID market bottom; 2017-06-01 is the only calm-day date tried). Tree: minimum "
            "spanning tree of the Mantegna correlation distance (strongest links only, no loops by "
            "construction). Diagram: H1 loops from the full network, including weaker links. Average "
            "correlation rises 0.18 -> 0.59 and total H1 persistence per asset falls during the crisis day "
            "-- the network collapses into a denser, more homogeneous shape, which is why the compact "
            "topology features tend to lose information exactly when it matters most.")

    print("All figures written to", figs)


def build_chart(research=False):
    """Compare AP and AUROC; the chart never refits a model."""
    folder = ROOT / ("results/research" if research else "results")
    table = "metrics" if research else "aggregate_metrics"
    with sqlite3.connect(folder / "evidence.sqlite") as connection:
        rows = connection.execute(
            "SELECT model, average_precision, auroc FROM "
            + table
            + " ORDER BY average_precision DESC"
        ).fetchall()
    height = 100 + len(rows) * 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Real-data model comparison</title>',
        '<desc id="desc">Average precision and AUROC for every model, ordered by average precision. Retrospective 2007–2024 evaluation.</desc>',
        f'<rect width="1000" height="{height}" fill="#ffffff"/>',
        '<g font-family="Times New Roman,Times,serif" font-size="12" fill="#172b4d">',
        '<text x="24" y="30" font-size="20">Model comparison · retrospective real-data evaluation</text>',
        '<text x="300" y="60">Average precision (0–1)</text>',
        '<text x="650" y="60">AUROC</text>',
    ]
    for i, (name, ap, auc) in enumerate(rows):
        y = 85 + i * 42
        parts.extend(
            [
                f'<text x="24" y="{y + 17}">{html.escape(name)}</text>',
                f'<rect x="300" y="{y}" width="{300 * ap:.2f}" height="23" fill="#157f82"/>',
                f'<text x="{307 + 300 * ap:.2f}" y="{y + 17}">{ap:.3f}</text>',
                f'<rect x="650" y="{y}" width="{270 * auc:.2f}" height="23" fill="#4865a3"/>',
                f'<text x="{657 + 270 * auc:.2f}" y="{y + 17}">{auc:.3f}</text>',
            ]
        )
    parts.append("</g></svg>")
    (folder / "model_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    build_chart()

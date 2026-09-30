#!/usr/bin/env python
# coding: utf-8
"""Compare downstream random-forest per-class accuracy across experiments.

gelos' comparison framework (``gelos.comparison``) has no ``COMP_METRICS``
entry that reads the ``*_random_forest_results.csv`` written by the analysis
stage, so this small gelos-lc CLI fills the gap: it collects the per-class
accuracy of several experiments into one long-form CSV and renders a grouped
bar chart (one bar per experiment per class, deltas vs. a baseline annotated).

Example::

    python src/compare_model_results.py \\
        --processed-data-dir /app/data/processed --data-version v0.50.1 \\
        --name 16_olmoearth_timestamps \\
        --experiment exp030_olmoearth_v1_2_base_s2:all_steps_of_middle_patch:layer_0:"S2 | Generic date" \\
        --experiment exp036_olmoearth_v1_2_base_s2_timestamps:all_steps_of_middle_patch:layer_0:"S2 | Real dates" \\
        --pair "S2 | Generic date":"S2 | Real dates"
"""

from dataclasses import dataclass
import os
from pathlib import Path

from gelos.analysis import build_prefix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

app = typer.Typer(add_completion=False)

# Same class-id -> name mapping as the `style.labels` block of the experiment
# configs; the `overall` row written by gelos.models._save_results_csv is kept.
CLASS_LABELS = {
    "1": "Water",
    "2": "Trees",
    "5": "Crops",
    "7": "Built Area",
    "8": "Bare Ground",
    "11": "Rangeland",
    "overall": "Overall",
}
CLASS_ORDER = list(CLASS_LABELS)

# Every MODELS entry writes {prefix}_{m_type}_{m_type}_results.csv: analysis.py
# builds run_name = f"{prefix}_{m_type}" and _save_results_csv appends
# f"_{model_type}_results.csv" again.
MODEL_TYPE = "random_forest"

# Paired (baseline, variant) hues; validated with the dataviz palette checker.
DEFAULT_COLORS = ["#7cb342", "#33691e", "#42a5f5", "#1565c0", "#eb6834", "#4a3aa7"]


@dataclass
class ExperimentSpec:
    config: str
    strategy: str
    layer: str
    label: str

    @classmethod
    def parse(cls, spec: str) -> "ExperimentSpec":
        parts = spec.split(":", 3)
        if len(parts) != 4 or not all(parts):
            raise typer.BadParameter(
                f"expected CONFIG:STRATEGY:LAYER:LABEL, got {spec!r}", param_hint="--experiment"
            )
        return cls(*parts)

    def results_csv(self, processed_data_dir: Path, data_version: str) -> Path:
        prefix = build_prefix(self.config, self.strategy, self.layer)
        return (
            processed_data_dir
            / data_version
            / self.config
            / self.layer
            / f"{prefix}_{MODEL_TYPE}_{MODEL_TYPE}_results.csv"
        )


def load_results(
    experiments: list[ExperimentSpec], processed_data_dir: Path, data_version: str
) -> pd.DataFrame:
    """Long-form frame: one row per (experiment, class) with its accuracy."""
    frames = []
    for exp in experiments:
        csv_path = exp.results_csv(processed_data_dir, data_version)
        if not csv_path.exists():
            raise FileNotFoundError(f"missing random forest results for {exp.label!r}: {csv_path}")
        df = pd.read_csv(csv_path, dtype={"class": str})
        df["class_label"] = df["class"].map(CLASS_LABELS).fillna(df["class"])
        df.insert(0, "layer", exp.layer)
        df.insert(0, "strategy", exp.strategy)
        df.insert(0, "config", exp.config)
        df.insert(0, "experiment", exp.label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def parse_pairs(pairs: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """``BASELINE:VARIANT`` strings -> tuples; default: everything vs the first label."""
    if not pairs:
        return [(labels[0], other) for other in labels[1:]]
    parsed = []
    for pair in pairs:
        base, sep, variant = pair.partition(":")
        if not sep or base not in labels or variant not in labels:
            raise typer.BadParameter(
                f"expected BASELINE_LABEL:VARIANT_LABEL using known labels {labels}, got {pair!r}",
                param_hint="--pair",
            )
        parsed.append((base, variant))
    return parsed


def plot_accuracy(
    results: pd.DataFrame,
    labels: list[str],
    pairs: list[tuple[str, str]],
    colors: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    """Grouped bars: x = class (incl. Overall), one bar per experiment.

    Every bar carries its accuracy as a direct label; variant bars also carry
    the delta (in percentage points) against their paired baseline.
    """
    classes = [c for c in CLASS_ORDER if c in set(results["class"])]
    wide = results.pivot(index="class", columns="experiment", values="accuracy").reindex(classes)
    n_exp = len(labels)
    group_width = 0.82
    bar_width = group_width / n_exp
    x = range(len(classes))
    baseline_of = {variant: base for base, variant in pairs}

    fig, ax = plt.subplots(figsize=(max(9, 1.9 * len(classes)), 5.5))
    for i, label in enumerate(labels):
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        values = wide[label].to_numpy()
        ax.bar(
            offsets,
            values,
            width=bar_width * 0.94,  # small gap between adjacent bars
            color=colors[label],
            label=label,
            linewidth=0,
        )
        for cls, xi, value in zip(classes, offsets, values):
            ax.annotate(
                f"{value:.3f}",
                (xi, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#0b0b0b",
            )
            if label in baseline_of:
                delta_pp = (value - wide.loc[cls, baseline_of[label]]) * 100
                ax.annotate(
                    f"{delta_pp:+.2f} pp",
                    (xi, value),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#52514e",
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in classes])
    ax.set_ylabel("Random forest accuracy (repeated stratified CV)")
    ymin = max(0.0, float(results["accuracy"].min()) - 0.05)
    ax.set_ylim(ymin, min(1.0, float(results["accuracy"].max()) + 0.04))
    ax.set_title(title)
    ax.yaxis.grid(True, color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    # legend below the axis so it never overlaps the (tall) bars
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        frameon=False,
        fontsize=8,
        ncol=2,
        title="Deltas: variant minus its paired baseline, in percentage points",
        title_fontsize=7.5,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@app.command()
def compare(
    experiment: list[str] = typer.Option(
        ..., "--experiment", help="Repeatable CONFIG:STRATEGY:LAYER:LABEL"
    ),
    name: str = typer.Option(..., help="Output stem, e.g. 16_olmoearth_timestamps"),
    processed_data_dir: Path = typer.Option(
        Path(os.environ.get("PROCESSED_PATH", "/app/data/processed")),
        help="Processed data directory (analysis outputs)",
    ),
    figures_dir: Path = typer.Option(Path("/app/reports/figures"), help="Figures directory"),
    data_version: str = typer.Option("v0.50.1", help="Data version string"),
    pair: list[str] = typer.Option(
        [],
        "--pair",
        help="Repeatable BASELINE_LABEL:VARIANT_LABEL for delta annotation "
        "(default: every experiment vs. the first)",
    ),
    color: list[str] = typer.Option(
        [], "--color", help="Repeatable hex color, one per --experiment in order"
    ),
    title: str = typer.Option("Random forest accuracy by class", help="Plot title"),
):
    """Collect random-forest per-class accuracies and plot them side by side."""
    experiments = [ExperimentSpec.parse(spec) for spec in experiment]
    labels = [exp.label for exp in experiments]
    if len(set(labels)) != len(labels):
        raise typer.BadParameter("experiment labels must be unique", param_hint="--experiment")
    if color and len(color) != len(experiments):
        raise typer.BadParameter("give one --color per --experiment", param_hint="--color")
    palette = color or DEFAULT_COLORS
    if len(palette) < len(experiments):
        raise typer.BadParameter(
            f"at most {len(DEFAULT_COLORS)} experiments without explicit --color",
            param_hint="--experiment",
        )
    colors = dict(zip(labels, palette))
    pairs = parse_pairs(pair, labels)

    results = load_results(experiments, processed_data_dir, data_version)

    # same layout as gelos.comparison: {processed}/comparisons/<stem>/
    csv_dir = processed_data_dir / "comparisons" / name
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{name}_{MODEL_TYPE}_comparison.csv"
    results.to_csv(csv_path, index=False)
    typer.echo(f"wrote {csv_path}")

    plot_path = figures_dir / "comparisons" / name / f"{name}_{MODEL_TYPE}_accuracy_plot.png"
    plot_accuracy(results, labels, pairs, colors, title, plot_path)
    typer.echo(f"wrote {plot_path}")


if __name__ == "__main__":
    app()

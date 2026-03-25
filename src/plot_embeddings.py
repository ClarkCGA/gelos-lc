from pathlib import Path
from typing import List

from matplotlib import patches, transforms
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_embeddings(
    final_layer_embedding,
    yaml_config,
    version,
    figures_dir: str | Path = None,
    vis_embed_depth: int = 128,
    n_per_grid: int = 36,
    date_ranges: List[str] = ["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"],
    has_cls: bool = True,
):
    """
    Plot embeddings, highlighting extraction strategies.
    Currently only works for Prithvi due to assumptions about embedding shape.
    """
    patch_tokens_flat = final_layer_embedding.reshape(-1, final_layer_embedding.shape[-1])
    vis_embed_depth = min(vis_embed_depth, patch_tokens_flat.shape[-1])
    first_feature = patch_tokens_flat[:, 0].detach().cpu().numpy()
    start_index = 1 if has_cls else 0

    neon_colors = ["#FF00FF", "#39FF14", "#FF0033"]
    cmap = ListedColormap(neon_colors, name="custom_neon")

    grid_side = int(np.ceil(np.sqrt(n_per_grid)))
    cells_per_grid = grid_side * grid_side
    start_index = 1 if has_cls else 0
    groups = [
        first_feature[i : i + n_per_grid]
        for i in range(start_index, first_feature.shape[0], n_per_grid)
    ]
    group_count = len(groups)
    cols = group_count + 1
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])
    mesh = None
    x_skew = 0
    y_skew = -25
    skew_base = transforms.Affine2D().skew_deg(x_skew, y_skew)

    highlight_dict = {}
    cls_first_patch_highlights = []
    # only works with prithvi configs, since this is capable of showing all extraction strategies including CLS
    for extraction_strategy, extraction_args in yaml_config[
        "embedding_extraction_strategies"
    ].items():
        slice_args = extraction_args["slice_args"]
        title = extraction_args["title"]
        highlight_spec = slice_args[0].copy()
        if highlight_spec["stop"] is None:
            highlight_spec["stop"] = n_per_grid * group_count
        if has_cls:
            highlight_spec["start"] -= start_index
            highlight_spec["stop"] -= start_index
        if (
            highlight_spec["stop"] - highlight_spec["start"] == n_per_grid
            and highlight_spec["step"] == 1
        ):
            highlight_spec["outline_plot"] = int(
                round((highlight_spec["stop"] - highlight_spec["start"]) / n_per_grid)
            )
        highlight_dict[title] = highlight_spec
        if highlight_spec["start"] < 0:
            cls_first_patch_highlights.append(title)
        print(title, highlight_spec)
    for idx, (name, spec) in enumerate(highlight_dict.items()):
        spec["color"] = cmap(idx)

    outline_map = {}
    highlight_map = {}
    for name, spec in highlight_dict.items():
        outline_idx = spec.get("outline_plot")
        if outline_idx is not None and 0 <= outline_idx < group_count:
            outline_map.setdefault(outline_idx, []).append(spec["color"])
            continue
        if spec.get("start", 0) < 0:
            continue
        abs_start = max(0, spec.get("start", 0))
        abs_stop = min(spec.get("stop", first_feature.shape[0]), first_feature.shape[0])
        abs_step = max(1, spec.get("step", 1))
        for token_idx in range(abs_start, abs_stop, abs_step):
            group_idx = token_idx // n_per_grid
            if group_idx >= group_count:
                break
            highlight_map.setdefault(group_idx, []).append(
                (token_idx % n_per_grid, spec["color"], name)
            )

    legend_patches = [
        patches.Patch(color=spec["color"], label=name) for name, spec in highlight_dict.items()
    ]

    # first patch (single square) on the left, sized like a normal patch cell and centered
    first_patch_value = first_feature[start_index]
    single_patch_grid = np.full((grid_side, grid_side), np.nan)
    center_idx = grid_side // 2
    single_patch_grid[center_idx, center_idx] = first_patch_value
    ax_first = axes_flat[0]
    mesh = ax_first.pcolormesh(
        single_patch_grid,
        cmap="coolwarm",
        edgecolors="none",
        linewidth=2,
        shading="auto",
        transform=skew_base + ax_first.transData,
        antialiased=True,
    )
    mesh.set_clip_on(False)
    for cls_name in cls_first_patch_highlights:
        color = highlight_dict[cls_name]["color"]
        rect = patches.Rectangle(
            (center_idx, center_idx),
            1,
            1,
            fill=False,
            edgecolor=color,
            linewidth=2,
            transform=skew_base + ax_first.transData,
            clip_on=False,
        )
        ax_first.add_patch(rect)
    ax_first.set_title("CLS Token (Prithvi)", y=0.7)
    ax_first.set_xticks([])
    ax_first.set_yticks([])
    ax_first.set_aspect("equal")
    for spine in ax_first.spines.values():
        spine.set_visible(False)

    for idx, group_values in enumerate(groups):
        chunk = group_values.copy()
        chunk.shape
        if chunk.shape[0] < n_per_grid:
            chunk = np.concatenate(
                [chunk, np.full(n_per_grid - chunk.shape[0], np.nan, dtype=chunk.dtype)]
            )
        if chunk.shape[0] < cells_per_grid:
            chunk = np.concatenate(
                [chunk, np.full(cells_per_grid - chunk.shape[0], np.nan, dtype=chunk.dtype)]
            )
        grid = chunk.reshape(grid_side, grid_side)
        ax = axes_flat[idx + 1]
        mesh = ax.pcolormesh(
            grid,
            cmap="coolwarm",
            edgecolors="white",
            linewidth=2,
            shading="auto",
            transform=skew_base + ax.transData,
            antialiased=True,
        )
        mesh.set_clip_on(False)
        for cell_idx, color, _name in highlight_map.get(idx, []):
            row = cell_idx // grid_side
            col = cell_idx % grid_side
            rect = patches.Rectangle(
                (col, row),
                1,
                1,
                fill=False,
                edgecolor=color,
                linewidth=2,
                transform=skew_base + ax.transData,
                clip_on=False,
            )
            ax.add_patch(rect)
        for outline_color in outline_map.get(idx, []):
            outline = patches.Rectangle(
                (0, 0),
                grid_side,
                grid_side,
                fill=False,
                edgecolor=outline_color,
                linewidth=2,
                transform=skew_base + ax.transData,
                clip_on=False,
            )
            ax.add_patch(outline)
        ax.set_title(f"{date_ranges[idx]}", y=0.95)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes_flat[group_count + 1 :]:
        ax.axis("off")

    fig.subplots_adjust(wspace=-0.60, hspace=-0.60)
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="center right",
            bbox_to_anchor=(1, 0.5),
            # ncol=max(len(legend_patches), 1),
            title="Embedding Extraction Strategies",
        )
    plt.suptitle("Example Embedding Outputs and Extraction Strategies")
    fig.canvas.draw()
    if figures_dir:
        plt.savefig(
            figures_dir / version / "example_embedding_outputs.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()

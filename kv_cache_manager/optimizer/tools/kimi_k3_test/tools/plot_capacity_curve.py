"""Extract and plot the capacity curve (axes[0] from analyze_results.py)."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GIB = 1 << 30


def load(path: Path):
    return json.loads(path.read_text())


def plot_capacity_curve(out: Path):
    """Plot the capacity curve showing input-token hit rate vs capacity."""
    stage1 = load(out / "stage1/summary.json")
    manifest = load(out / "trace_manifest.json")
    layout = load(out / "layout.json")

    # Extract data
    caps = stage1["capacity_gb"][:-1]
    rates = np.array(stage1["hit_rates_exact_from_integer_counts"][:-1]) * 100
    infinite_rate = stage1["hit_rates_exact_from_integer_counts"][-1] * 100
    knee = stage1["elbow_capacity_gib"]
    cap95 = stage1["capacity_at_95pct_infinite_hit_rate_gib"]
    plateau = next((c for c, r in zip(caps, rates) if abs(r - infinite_rate) < 1e-9), None)
    checkpoint = layout["checkpoint_tokens"]

    # Workload label
    source = manifest["source"]
    agentx = source.get("kind") == "agentx_observed_arrival_subset"
    workload_label = "AgentX-derived" if agentx else "Synthetic" if source.get("kind") == "synthetic_prefix_identity_trace" else "Reused-trace"

    # Set up plot style
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 130,
        "savefig.dpi": 180
    })
    colors = ["#5072A7", "#148A91", "#D37B12", "#7356A5", "#BD4A59", "#5A7A37"]

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1]}, constrained_layout=True)

    # Plot capacity curve (axes[0])
    axes[0].plot(caps, rates, color=colors[0], marker=".", linewidth=2,
                 label="LiteHit: immediate request write-back")
    axes[0].axhline(infinite_rate, color=colors[3], linestyle="--",
                    label=f"Infinite: {infinite_rate:.2f}%")

    # Add markers for knee, cap95, and plateau
    for cap, color, name in [(knee, colors[2], "Geometric elbow"),
                             (cap95, colors[1], "95% of infinite hit rate"),
                             (plateau, colors[5], "Sample plateau")]:
        if cap is None:
            continue
        rate = rates[caps.index(cap)]
        axes[0].scatter([cap], [rate], s=75, color=color, zorder=4)
        axes[0].axvline(cap, color=color, linestyle=":", alpha=.55)
        axes[0].annotate(f"{name}\n{cap:g} GiB / {rate:.2f}%", (cap, rate),
                         xytext=(cap - (85 if cap == knee else 15),
                                 rate - (19 if cap == knee else 31 if cap == cap95 else 46)),
                         arrowprops={"arrowstyle": "-", "color": color},
                         color=color, fontsize=10)

    axes[0].set(ylabel="Input-token hit rate (%)", ylim=(0, 100),
                title="Kimi-K3: capacity screening from one Facts replay")
    axes[0].legend(loc="lower right", fontsize=9)

    # Plot marginal gains (axes[1])
    slopes = np.diff(rates) / np.diff(caps)
    axes[1].bar(caps[1:], slopes, width=np.diff(caps)*.82, color=colors[0])
    axes[1].set(xlabel="External cache payload capacity (GiB, all TP8 ranks)",
                ylabel="Marginal gain\n(pp / GiB)", xlim=(-10, 1040))

    fig.suptitle(f"{manifest['requests']} {workload_label} requests | "
                 f"{checkpoint:,}-token atomic MLA + KDA bundles | "
                 f"heuristic elbow on 0-1024 GiB", fontsize=10)

    # Save figure
    output_dir = out / "report"
    output_dir.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"capacity_curve.{extension}", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved capacity curve to {output_dir}/capacity_curve.{{png,svg,pdf}}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_capacity_curve.py <output_directory>")
        sys.exit(1)

    out_path = Path(sys.argv[1])
    plot_capacity_curve(out_path)

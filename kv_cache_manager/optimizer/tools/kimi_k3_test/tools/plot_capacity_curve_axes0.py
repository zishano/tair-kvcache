#!/usr/bin/env python3
"""
Simplified plotting script to generate only the axes[0] capacity curve plot.
Extracts only the necessary plotting code from analyze_results.py's plot_and_report() function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    """Load JSON file."""
    return json.loads(path.read_text())


def plot_capacity_curve_axes0(
    capacity_data: dict,
    trace_manifest: dict,
    layout: dict,
    output_dir: Path
):
    """
    Plot only the axes[0] capacity curve.
    This is extracted from analyze_results.py's plot_and_report() function (lines 380-395).
    """
    # Extract data
    caps = capacity_data["capacity_gb"][:-1]  # Exclude infinite
    rates = np.array(capacity_data["hit_rates_percent"][:-1])
    infinite_rate = capacity_data["hit_rates_percent"][-1]
    knee = capacity_data["elbow_capacity_gib"]
    cap95 = capacity_data["capacity_at_95pct_infinite_gib"]
    checkpoint = layout["checkpoint_tokens"]

    # Find plateau (where rate equals infinite rate)
    plateau = next((c for c, r in zip(caps, rates) if abs(r - infinite_rate) < 1e-9), None)

    # Determine workload label
    source = trace_manifest["source"]
    agentx = source.get("kind") == "agentx_observed_arrival_subset"
    workload_label = "AgentX-derived" if agentx else "Synthetic" if source.get("kind") == "synthetic_prefix_identity_trace" else "Reused-trace"

    # Set up plot style (from line 362-364)
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

    # Color palette (from line 365)
    colors = ["#5072A7", "#148A91", "#D37B12", "#7356A5", "#BD4A59", "#5A7A37"]

    # Create figure with single subplot (axes[0] only)
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.0), constrained_layout=True)

    # Plot capacity curve (line 382)
    ax.plot(caps, rates, color=colors[0], marker=".", linewidth=2,
            label="LiteHit: immediate request write-back")

    # Plot infinite hit rate line (line 383)
    ax.axhline(infinite_rate, color=colors[3], linestyle="--",
               label=f"Infinite: {infinite_rate:.2f}%")

    # Add markers for key points (lines 384-393)
    for cap, color, name in [
        (knee, colors[2], "Geometric elbow"),
        (cap95, colors[1], "95% of infinite hit rate"),
        (plateau, colors[5], "Sample plateau")
    ]:
        if cap is None:
            continue
        rate = rates[caps.index(cap)]
        ax.scatter([cap], [rate], s=75, color=color, zorder=4)
        ax.axvline(cap, color=color, linestyle=":", alpha=.55)
        ax.annotate(
            f"{name}\n{cap:g} GiB / {rate:.2f}%",
            (cap, rate),
            xytext=(cap - (85 if cap == knee else 15),
                   rate - (19 if cap == knee else 31 if cap == cap95 else 46)),
            arrowprops={"arrowstyle": "-", "color": color},
            color=color,
            fontsize=10
        )

    # Set labels and title (line 394)
    ax.set(
        ylabel="Input-token hit rate (%)",
        ylim=(0, 100),
        title="Kimi-K3: capacity screening from one Facts replay"
    )

    # Add legend (line 395)
    ax.legend(loc="lower right", fontsize=9)

    # Set x-axis
    ax.set_xlabel("External cache payload capacity (GiB, all TP8 ranks)")
    ax.set_xlim((-10, 1040))

    # Add overall title (from line 400)
    fig.suptitle(
        f"{trace_manifest['requests']} {workload_label} requests | "
        f"{checkpoint:,}-token atomic MLA + KDA bundles | "
        f"heuristic elbow on 0-1024 GiB",
        fontsize=10
    )

    # Save figure in multiple formats
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        output_path = output_dir / f"capacity_curve_axes0.{extension}"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capacity-data", type=Path, required=True,
                   help="Path to capacity_data.json (from compute_capacity_data.py)")
    p.add_argument("--trace-manifest", type=Path, required=True,
                   help="Path to trace_manifest.json")
    p.add_argument("--layout", type=Path, required=True,
                   help="Path to layout.json")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for plots")

    args = p.parse_args()

    # Validate inputs
    if not args.capacity_data.exists():
        p.error(f"Capacity data not found: {args.capacity_data}")
    if not args.trace_manifest.exists():
        p.error(f"Trace manifest not found: {args.trace_manifest}")
    if not args.layout.exists():
        p.error(f"Layout file not found: {args.layout}")

    # Load data
    print("Loading data...")
    capacity_data = load_json(args.capacity_data)
    trace_manifest = load_json(args.trace_manifest)
    layout = load_json(args.layout)

    # Generate plot
    print("Generating capacity curve plot (axes[0] only)...")
    plot_capacity_curve_axes0(capacity_data, trace_manifest, layout, args.output_dir)

    print(f"\n✓ Plot saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

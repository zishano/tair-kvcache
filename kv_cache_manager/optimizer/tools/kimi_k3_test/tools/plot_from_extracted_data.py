#!/usr/bin/env python3
"""Plot capacity curve from extracted data.

This script reads the extracted data from extract_plot_data.py and generates
the capacity curve visualization.

Usage:
    python plot_from_extracted_data.py <plot_data_json>

Example:
    python plot_from_extracted_data.py output/plot_data/capacity_curve_data.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    """Load JSON file."""
    return json.loads(path.read_text())


def plot_capacity_curve(plot_data: dict, output_dir: Path):
    """Generate capacity curve plot from extracted data.

    Args:
        plot_data: Dictionary containing all plotting data
        output_dir: Directory to save output figures
    """
    # Extract data
    caps = plot_data["capacities_gib"]
    rates = plot_data["hit_rates_percent"]
    infinite_rate = plot_data["infinite_hit_rate_percent"]
    knee = plot_data["elbow_capacity_gib"]
    cap95 = plot_data["capacity_at_95pct_infinite_gib"]
    plateau = plot_data["plateau_capacity_gib"]
    checkpoint = plot_data["checkpoint_tokens"]
    num_requests = plot_data["num_requests"]
    workload_type = plot_data["workload_type"]

    # Convert to numpy array for easier manipulation
    rates_array = np.array(rates)

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

    # Color palette
    colors = ["#5072A7", "#148A91", "#D37B12", "#7356A5", "#BD4A59", "#5A7A37"]

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1]},
                             constrained_layout=True)

    # ========== Top subplot: Capacity curve ==========
    axes[0].plot(caps, rates, color=colors[0], marker=".", linewidth=2,
                 label="LiteHit: immediate request write-back")
    axes[0].axhline(infinite_rate, color=colors[3], linestyle="--",
                    label=f"Infinite: {infinite_rate:.2f}%")

    # Add markers for key points
    markers_data = [
        (knee, colors[2], "Geometric elbow", -85, -19),
        (cap95, colors[1], "95% of infinite hit rate", -15, -31),
        (plateau, colors[5], "Sample plateau", -15, -46)
    ]

    for cap, color, name, offset_x, offset_y in markers_data:
        if cap is None:
            continue
        # Find the hit rate at this capacity
        rate = rates[caps.index(cap)]
        axes[0].scatter([cap], [rate], s=75, color=color, zorder=4)
        axes[0].axvline(cap, color=color, linestyle=":", alpha=.55)
        axes[0].annotate(f"{name}\n{cap:g} GiB / {rate:.2f}%", (cap, rate),
                         xytext=(cap + offset_x, rate + offset_y),
                         arrowprops={"arrowstyle": "-", "color": color},
                         color=color, fontsize=10)

    axes[0].set(ylabel="Input-token hit rate (%)",
                ylim=(0, 100),
                title="Kimi-K3: capacity screening from one Facts replay")
    axes[0].legend(loc="lower right", fontsize=9)

    # ========== Bottom subplot: Marginal gains ==========
    slopes = np.diff(rates_array) / np.diff(caps)
    bar_widths = np.diff(caps) * 0.82
    axes[1].bar(caps[1:], slopes, width=bar_widths, color=colors[0])
    axes[1].set(xlabel="External cache payload capacity (GiB, all TP8 ranks)",
                ylabel="Marginal gain\n(pp / GiB)",
                xlim=(-10, 1040))

    # Overall title
    fig.suptitle(
        f"{num_requests} {workload_type} requests | "
        f"{checkpoint:,}-token atomic MLA + KDA bundles | "
        f"heuristic elbow on 0-1024 GiB",
        fontsize=10
    )

    # Save figure in multiple formats
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        output_path = output_dir / f"capacity_curve.{extension}"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nError: Missing required argument")
        print("Usage: python plot_from_extracted_data.py <plot_data_json>")
        sys.exit(1)

    data_path = Path(sys.argv[1])

    # Validate input file exists
    if not data_path.exists():
        print(f"Error: Data file does not exist: {data_path}")
        print("\nPlease run extract_plot_data.py first:")
        print("  python extract_plot_data.py <output_directory>")
        sys.exit(1)

    # Load data
    print(f"Loading plot data from: {data_path}")
    plot_data = load_json(data_path)

    # Determine output directory (same as data file location)
    output_dir = data_path.parent

    # Generate plot
    print(f"\nGenerating capacity curve plots...")
    plot_capacity_curve(plot_data, output_dir)

    print(f"\nPlot generation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

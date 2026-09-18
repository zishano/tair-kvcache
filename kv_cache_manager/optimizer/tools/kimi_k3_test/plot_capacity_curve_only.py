#!/usr/bin/env python3
"""Plot only the Kimi-K3 capacity curve from Stage 1 results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path):
    return json.loads(path.read_text())


def plot_capacity_curve(output_dir: Path) -> None:
    stage1 = load(output_dir / "stage1" / "summary.json")
    layout = load(output_dir / "layout.json")
    manifest = load(output_dir / "trace_manifest.json")

    # Stage 1 stores the final entry as the infinite-capacity result.
    capacities = stage1["capacity_gb"][:-1]
    rates = [rate * 100 for rate in stage1["hit_rates_exact_from_integer_counts"][:-1]]
    infinite_rate = stage1["hit_rates_exact_from_integer_counts"][-1] * 100
    knee = stage1["elbow_capacity_gib"]
    cap95 = stage1["capacity_at_95pct_infinite_hit_rate_gib"]
    plateau = next(
        (capacity for capacity, rate in zip(capacities, rates)
         if abs(rate - infinite_rate) < 1e-9),
        None,
    )

    source_kind = manifest.get("source", {}).get("kind")
    workload = {
        "agentx_observed_arrival_subset": "AgentX-derived",
        "synthetic_prefix_identity_trace": "Synthetic",
    }.get(source_kind, "Reused-trace")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 130,
        "savefig.dpi": 180,
    })
    colors = ["#5072A7", "#148A91", "#D37B12", "#7356A5", "#5A7A37"]

    fig, ax = plt.subplots(figsize=(10.8, 5.5), constrained_layout=True)
    ax.plot(capacities, rates, color=colors[0], marker=".", linewidth=2,
            label="LiteHit: immediate request write-back")
    ax.axhline(infinite_rate, color=colors[3], linestyle="--",
               label=f"Infinite: {infinite_rate:.2f}%")

    for capacity, color, name, x_offset, y_offset in (
        (knee, colors[2], "Geometric elbow", -85, -19),
        (cap95, colors[1], "95% of infinite hit rate", -15, -31),
        (plateau, colors[4], "Sample plateau", -15, -46),
    ):
        if capacity is None:
            continue
        index = capacities.index(capacity)
        rate = rates[index]
        ax.scatter([capacity], [rate], s=75, color=color, zorder=4)
        ax.axvline(capacity, color=color, linestyle=":", alpha=.55)
        ax.annotate(
            f"{name}\n{capacity:g} GiB / {rate:.2f}%",
            (capacity, rate),
            xytext=(capacity + x_offset, rate + y_offset),
            arrowprops={"arrowstyle": "-", "color": color},
            color=color,
            fontsize=10,
        )

    # Dynamically set x-axis range based on useful capacity data
    # Find where the curve reaches plateau (95% of infinite or actual plateau point)
    useful_max = cap95 if cap95 is not None else knee
    if plateau is not None:
        useful_max = max(useful_max or 0, plateau)

    # If no useful markers found, use the last significant change point
    if useful_max is None or useful_max == 0:
        # Find last point where rate increases by more than 0.1%
        for i in range(len(rates) - 1, 0, -1):
            if abs(rates[i] - rates[i-1]) > 0.1:
                useful_max = capacities[i]
                break
        if useful_max is None:
            useful_max = min(1024, max(capacities) if capacities else 1024)

    # Add 20% padding to the right of useful range, minimum 1024 GiB
    xlim_max = max(1024, useful_max * 1.2)

    # Get unbounded capacity for annotation
    unbounded_gib = manifest.get("unbounded_resident_payload_gib", 0)

    # Determine elbow range description for subtitle
    if xlim_max <= 1024:
        elbow_range = "0-1024 GiB"
    elif xlim_max <= 10000:
        elbow_range = f"0-{int(xlim_max):,} GiB"
    else:
        elbow_range = f"0-{xlim_max/1024:.1f} TiB"

    ax.set(
        xlabel="External cache payload capacity (GiB, all TP ranks)",
        ylabel="Input-token hit rate (%)",
        ylim=(0, 100),
        xlim=(-10, xlim_max),
        title="Kimi-K3: capacity screening from one Facts replay",
    )

    # Add unbounded capacity annotation as text in upper left
    if unbounded_gib > 0:
        unbounded_text = f"Unbounded (100% hit): {unbounded_gib:,.0f} GiB"
        if unbounded_gib >= 1024:
            unbounded_text = f"Unbounded (100% hit): {unbounded_gib/1024:.1f} TiB ({unbounded_gib:,.0f} GiB)"
        ax.text(0.02, 0.98, unbounded_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"{manifest['requests']} {workload} requests | "
        f"{layout['checkpoint_tokens']:,}-token atomic MLA + KDA bundles | "
        f"useful range: {elbow_range}",
        fontsize=10,
    )

    report_dir = output_dir / "capacity_curve_only"
    report_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(report_dir / f"capacity_curve.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved capacity curve to {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path,
                        help="Directory containing layout.json, trace_manifest.json, and stage1/summary.json")
    args = parser.parse_args()

    required = (
        args.output_dir / "layout.json",
        args.output_dir / "trace_manifest.json",
        args.output_dir / "stage1" / "summary.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing required files: " + ", ".join(missing))
    plot_capacity_curve(args.output_dir)


if __name__ == "__main__":
    main()

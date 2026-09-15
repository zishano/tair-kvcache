#!/usr/bin/env python3
"""Extract data for capacity curve plotting from run_workflow.py outputs.

This script extracts the data generated during the 'scan' phase of run_workflow.py
and saves it in a format suitable for plotting the capacity curve.

Usage:
    python extract_plot_data.py <output_directory>

The output_directory should be the same directory passed to run_workflow.py
(e.g., /workspace/dynosim_aic_sweep/reports/kimi_k3_iterations/agentx_rerun_01)

Required input files:
    <output_dir>/stage1/summary.json       - Stage1 capacity analysis results
    <output_dir>/trace_manifest.json       - Trace manifest
    <output_dir>/layout.json                - Layout configuration

Output:
    <output_dir>/plot_data/capacity_curve_data.json - Extracted plotting data
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def load_json(path: Path):
    """Load JSON file."""
    return json.loads(path.read_text())


def save_json(path: Path, value):
    """Save JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def extract_plot_data(output_dir: Path) -> dict:
    """Extract all data needed for plotting the capacity curve.

    Returns a dictionary containing:
        - capacities: list of capacity values in GiB
        - hit_rates: list of hit rates (0-100%)
        - infinite_hit_rate: hit rate at infinite capacity
        - elbow_capacity: geometric elbow point
        - cap95: capacity achieving 95% of infinite hit rate
        - plateau_capacity: capacity where curve plateaus (if reached)
        - checkpoint_tokens: checkpoint interval in tokens
        - num_requests: total number of requests
        - workload_type: type of workload (AgentX-derived, Synthetic, etc.)
    """
    # Load required files
    stage1 = load_json(output_dir / "stage1/summary.json")
    manifest = load_json(output_dir / "trace_manifest.json")
    layout = load_json(output_dir / "layout.json")

    # Extract capacity data (exclude infinite capacity point)
    capacities = stage1["capacity_gb"][:-1]  # All but last (infinite)
    hit_rates_raw = stage1["hit_rates_exact_from_integer_counts"]
    hit_rates = [rate * 100 for rate in hit_rates_raw[:-1]]  # Convert to percentage
    infinite_hit_rate = hit_rates_raw[-1] * 100

    # Extract key points
    elbow_capacity = stage1["elbow_capacity_gib"]
    cap95 = stage1["capacity_at_95pct_infinite_hit_rate_gib"]

    # Find plateau point (where hit rate equals infinite hit rate)
    plateau_capacity = None
    for cap, rate in zip(capacities, hit_rates):
        if abs(rate - infinite_hit_rate) < 1e-9:
            plateau_capacity = cap
            break

    # Extract workload information
    source = manifest["source"]
    kind = source.get("kind", "unknown")
    if kind == "agentx_observed_arrival_subset":
        workload_type = "AgentX-derived"
    elif kind == "synthetic_prefix_identity_trace":
        workload_type = "Synthetic"
    else:
        workload_type = "Reused-trace"

    # Compile all data
    plot_data = {
        "capacities_gib": capacities,
        "hit_rates_percent": hit_rates,
        "infinite_hit_rate_percent": infinite_hit_rate,
        "elbow_capacity_gib": elbow_capacity,
        "capacity_at_95pct_infinite_gib": cap95,
        "plateau_capacity_gib": plateau_capacity,
        "checkpoint_tokens": layout["checkpoint_tokens"],
        "num_requests": manifest["requests"],
        "workload_type": workload_type,
        "input_tokens": manifest["input_tokens"],
        "output_tokens": manifest["output_tokens"],
        "bundle_bytes": layout["bundle_bytes"],
        "attention_tp_size": layout["attention_tp_size"],
        "metadata": {
            "trace_sha256": manifest["trace_sha256"],
            "layout_sha256": manifest["layout_sha256"],
            "facts_sha256": stage1.get("facts_sha256"),
            "stage1_replay_wall_s": stage1.get("replay_wall_s"),
            "stage1_query_wall_s": stage1.get("query_wall_s"),
        }
    }

    return plot_data


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nError: Missing required argument")
        print("Usage: python extract_plot_data.py <output_directory>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    # Validate input directory exists
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    # Validate required files exist
    required_files = [
        output_dir / "stage1/summary.json",
        output_dir / "trace_manifest.json",
        output_dir / "layout.json"
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("Error: Missing required input files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run the 'prepare' and 'scan' phases first:")
        print("  python run_workflow.py --steps prepare,scan --output-dir <dir> ...")
        sys.exit(1)

    # Extract data
    print(f"Extracting plot data from: {output_dir}")
    plot_data = extract_plot_data(output_dir)

    # Save to output file
    output_file = output_dir / "plot_data" / "capacity_curve_data.json"
    save_json(output_file, plot_data)

    # Print summary
    print(f"\nSuccessfully extracted plot data:")
    print(f"  - Capacities: {len(plot_data['capacities_gib'])} points")
    print(f"  - Elbow capacity: {plot_data['elbow_capacity_gib']:.1f} GiB")
    print(f"  - 95% infinite capacity: {plot_data['capacity_at_95pct_infinite_gib']} GiB")
    print(f"  - Plateau capacity: {plot_data['plateau_capacity_gib']} GiB")
    print(f"  - Workload: {plot_data['workload_type']}, {plot_data['num_requests']} requests")
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

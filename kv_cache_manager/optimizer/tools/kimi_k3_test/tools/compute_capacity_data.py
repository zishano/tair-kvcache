#!/usr/bin/env python3
"""
Simplified workflow to compute and plot only the capacity curve (axes[0]).
Extracts only the necessary data computation from run_workflow.py's scan() function.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from kv_cache_manager.optimizer.tools.trace_converter.utils.prefix_hash import apply_prefix_hash

GIB = 1 << 30


def digest(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    """Save JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path):
    """Load JSON file."""
    return json.loads(path.read_text())


def run_command(command: list[str], output: Path, name: str, env=None) -> dict:
    """Run a command and log output."""
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {
        "command": command,
        "cwd": str(REPO),
        "returncode": result.returncode,
        "elapsed_wall_s": time.perf_counter() - start,
        "stdout_stderr": str(output / f"{name}.log"),
    }
    save_json(output / f"{name}.command.json", record)
    if result.returncode:
        raise RuntimeError(f"{name} failed: {(output / f'{name}.log').read_text()[-6000:]}")
    return record


def compute_capacity_curve_data(
    trace_path: Path,
    layout_path: Path,
    litehit_config_path: Path,
    output_dir: Path
) -> dict:
    """
    Compute capacity curve data using LiteHit.
    This is extracted from run_workflow.py's scan() function.

    Returns a dictionary with all data needed for plotting axes[0].
    """
    layout = load_json(layout_path)

    # Run LiteHit replay to generate Facts
    binary_dir = REPO / "bazel-bin/kv_cache_manager/optimizer"
    facts = output_dir / "litehit_facts.csv"

    print("Running LiteHit Facts replay...")
    replay = run_command(
        [str(binary_dir / "lite_hit_main"), str(litehit_config_path)],
        output_dir,
        "replay"
    )

    # Query multiple capacity points
    capacities = sorted(set([0.0, 50.0, 1024.0] + [float(x) for x in range(25, 1025, 25)])) + [-1.0]
    query_path = output_dir / "capacity_query.jsonl"

    print(f"Querying {len(capacities)} capacity points...")
    query = run_command(
        [str(binary_dir / "lite_hit_facts_query_main"), str(facts), str(query_path)]
        + [str(c) for c in capacities],
        output_dir,
        "query"
    )

    # Load query results
    result_rows = [json.loads(line) for line in query_path.read_text().splitlines()]
    summary = next(r for r in result_rows if r.get("summary") and "instance_id" not in r)
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    projected = [r for r in result_rows if not r.get("summary")]

    if len(projected) != len(trace) or summary["requests"] != len(trace):
        raise AssertionError("Facts/query request count differs from trace")

    # Validate with independent LRU cache simulation
    print("Validating with independent LRU cache...")
    checked = 0
    for slot, cap in enumerate(capacities):
        limit = None if cap < 0 else math.floor(cap * GIB) // layout["bundle_bytes"]
        cache = OrderedDict()
        for row, actual in zip(trace, projected):
            assert str(row["trace_id"]) == str(actual["trace_id"])
            hits = 0
            for key in row["bundle_keys"]:
                if key not in cache:
                    break
                hits += 1
            hits = min(hits, max(0, (row["input_len"] - 1) // layout["checkpoint_tokens"]))
            if actual["hit_blocks"][slot] != hits:
                raise AssertionError(f"Oracle mismatch: {cap=}, {row['trace_id']=}, {hits=}, {actual=}")
            checked += 1
            for key in reversed(row["bundle_keys"]):
                cache.pop(key, None)
                cache[key] = None
                if limit is not None:
                    while len(cache) > limit:
                        cache.popitem(last=False)

    # Calculate hit rates and key points
    counts = summary["total_hit_tokens"]
    rates = [v / summary["total_input_tokens"] for v in counts]
    assert all(a <= b for a, b in zip(counts, counts[1:]))

    finite = list(zip(capacities[:-1], rates[:-1]))
    gain = finite[-1][1] - finite[0][1]
    knee = max(finite, key=lambda p: ((p[1] - finite[0][1]) / gain if gain else 0) - p[0] / finite[-1][0])[0]
    cap95 = next((c for c, h in finite if h >= 0.95 * rates[-1]), None)

    # Save capacity curve CSV
    with (output_dir / "capacity_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["capacity_gib", "hit_tokens", "input_tokens", "hit_rate", "marginal_pp_per_gib"])
        writer.writeheader()
        previous = None
        for c, n, h in zip(capacities, counts, rates):
            slope = "" if previous is None or c < 0 else (h - previous[1]) * 100 / (c - previous[0])
            writer.writerow({
                "capacity_gib": "infinite" if c < 0 else c,
                "hit_tokens": n,
                "input_tokens": summary["total_input_tokens"],
                "hit_rate": h,
                "marginal_pp_per_gib": slope
            })
            previous = (c, h)

    result = {
        "capacity_gb": capacities,
        "hit_rates": rates,
        "hit_rates_percent": [r * 100 for r in rates],
        "elbow_capacity_gib": knee,
        "capacity_at_95pct_infinite_gib": cap95,
        "total_input_tokens": summary["total_input_tokens"],
        "total_hit_tokens": counts,
        "replay_wall_s": replay["elapsed_wall_s"],
        "query_wall_s": query["elapsed_wall_s"],
        "facts_sha256": digest(facts),
        "validation": {
            "independent_lru_checks": checked,
            "mismatches": 0
        }
    }

    save_json(output_dir / "capacity_data.json", result)
    print(f"✓ Facts replay: {replay['elapsed_wall_s']:.3f}s")
    print(f"✓ Query {len(capacities)} capacities: {query['elapsed_wall_s']:.3f}s")
    print(f"✓ Elbow capacity: {knee:g} GiB")
    print(f"✓ 95% infinite capacity: {cap95} GiB")
    print(f"✓ Independent LRU checks: {checked}")

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True, help="Path to trace.jsonl")
    p.add_argument("--layout", type=Path, required=True, help="Path to layout.json")
    p.add_argument("--litehit-config", type=Path, required=True, help="Path to litehit_config.json")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    args = p.parse_args()

    # Validate inputs
    if not args.trace.exists():
        p.error(f"Trace file not found: {args.trace}")
    if not args.layout.exists():
        p.error(f"Layout file not found: {args.layout}")
    if not args.litehit_config.exists():
        p.error(f"LiteHit config not found: {args.litehit_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Compute capacity curve data
    data = compute_capacity_curve_data(
        args.trace,
        args.layout,
        args.litehit_config,
        args.output_dir
    )

    print(f"\nCapacity curve data saved to: {args.output_dir / 'capacity_data.json'}")


if __name__ == "__main__":
    main()

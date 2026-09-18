#!/usr/bin/env python3
"""Curve-only Kimi-K3 capacity workflow.

This file intentionally keeps only the code used by run_capacity_curve_only.sh:
prepare -> scan. It does not import or run HiSim, AIC predictors, bench_serving,
or full report generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from kv_cache_manager.optimizer.tools.trace_converter.utils.prefix_hash import apply_prefix_hash

GIB = 1 << 30


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def run_command(command: list[str], output: Path, name: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log_path = output / f"{name}.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    record = {
        "command": command,
        "cwd": str(REPO),
        "returncode": result.returncode,
        "elapsed_wall_s": time.perf_counter() - start,
        "stdout_stderr": str(log_path),
    }
    save_json(output / f"{name}.command.json", record)
    if result.returncode:
        raise RuntimeError(f"{name} failed: {log_path.read_text()[-6000:]}")
    return record


def make_layout(args) -> dict:
    model_path = args.model_config.resolve()
    outer = load_json(model_path)
    if "KimiK3ForConditionalGeneration" not in outer.get("architectures", []):
        raise ValueError("This exporter requires an actual Kimi-K3 config, not a renamed dense model")
    cfg = outer.get("text_config", outer)
    linear = cfg["linear_attn_config"]
    kda = linear["kda_layers"]
    full = linear["full_attn_layers"]
    layers = cfg["num_hidden_layers"]
    if (len(set(kda)) != len(kda) or len(set(full)) != len(full)
            or set(kda) & set(full) or set(kda) | set(full) != set(range(1, layers + 1))):
        raise ValueError("KDA/MLA layer ids must be a disjoint complete 1-based partition")
    if args.tp_size <= 0 or linear["num_heads"] % args.tp_size:
        raise ValueError("KDA heads must be divisible by attention TP")
    if args.native_block_tokens <= 0 or args.checkpoint_tokens % args.native_block_tokens:
        raise ValueError("checkpoint interval must be a positive multiple of trace block size")
    if args.checkpoint_tokens <= 0:
        raise ValueError("checkpoint interval must be positive")

    heads = linear["num_heads"] // args.tp_size
    dim = linear["head_dim"]
    mla_token_bytes = len(full) * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * 2
    ssm_bytes = len(kda) * heads * dim * dim * 4
    conv_bytes = len(kda) * 3 * (linear["short_conv_kernel_size"] - 1) * heads * dim * 2
    state_bytes = ssm_bytes + conv_bytes
    rank_bundle = args.checkpoint_tokens * mla_token_bytes + state_bytes
    return {
        "schema_version": "hisim_hybrid_checkpoint_layout_v1",
        "model_name": "moonshotai/Kimi-K3",
        "model_config_path": str(model_path),
        "model_config_sha256": digest(model_path),
        "attention_tp_size": args.tp_size,
        "checkpoint_tokens": args.checkpoint_tokens,
        "native_block_tokens": args.native_block_tokens,
        "mla_layers": len(full),
        "kda_layers": len(kda),
        "mla_layer_ids_1based": full,
        "kda_layer_ids_1based": kda,
        "kv_dtype": "bfloat16",
        "conv_dtype": "bfloat16",
        "state_dtype": "float32",
        "mla_bytes_per_token_per_rank": mla_token_bytes,
        "kda_ssm_bytes_per_checkpoint_per_rank": ssm_bytes,
        "kda_conv_bytes_per_checkpoint_per_rank": conv_bytes,
        "kda_state_bytes_per_checkpoint_per_rank": state_bytes,
        "bundle_bytes_per_rank": rank_bundle,
        "bundle_bytes": rank_bundle * args.tp_size,
        "storage_scope": "all_tp_rank_payloads",
        "capacity_unit": "GiB_payload",
        "policy": "atomic_checkpoint_bundle_lru",
        "commit_order": "tail_to_head",
        "payload_accounting": "All TP rank tensor payloads retained; excludes storage index overhead",
    }


def flatten_agentx_requests(args, instance_id: str) -> tuple[list[dict], dict]:
    """Flatten AgentX session JSONL into the trace format needed by LiteHit."""
    result, sessions, filtered = [], [], Counter()
    total_seen = parent_checks = subagent_rows = 0

    def leaves(items, stream="main", parent_start=0.0):
        for index, request in enumerate(items):
            path = f"{stream}/{index}"
            if request.get("type") == "subagent":
                yield from leaves(request.get("requests", []), path + "/subagent", float(request["t"]))
            elif all(key in request for key in ("in", "out", "hash_ids", "t")):
                if float(request["t"]) + 1e-6 < parent_start:
                    raise ValueError("Child time is not session-relative")
                yield path, request
            else:
                filtered["not_a_complete_request"] += 1

    with args.trace.open() as stream:
        for session_index, line in enumerate(stream):
            if session_index >= args.agentx_sessions:
                break
            session = json.loads(line)
            if session.get("hash_id_scope") != "local" or session["block_size"] != args.native_block_tokens:
                raise ValueError("AgentX input must use one local hash namespace and one block size")
            flattened = sorted(leaves(session["requests"]), key=lambda item: (float(item[1]["t"]), item[0]))
            total_seen += len(flattened)
            kept = []
            for path, req in flattened:
                if req["in"] <= 0 or req["out"] <= 0:
                    filtered["nonpositive_input_or_output"] += 1
                elif req["in"] > args.max_input_tokens:
                    filtered["above_context_limit"] += 1
                elif req["t"] > args.agentx_window_seconds:
                    filtered["outside_source_time_window"] += 1
                elif len(kept) >= args.requests_per_session:
                    filtered["beyond_per_session_sample"] += 1
                else:
                    kept.append((path, req))

            parents = {}
            for path, req in kept:
                local_keys = req["hash_ids"]
                if len(local_keys) != req["in"] // args.native_block_tokens:
                    raise ValueError(f"AgentX length mismatch in {session['id']} {path}")
                previous = None
                for depth, key in enumerate(local_keys):
                    identity = (depth, previous)
                    if key in parents and parents[key] != identity:
                        raise ValueError("AgentX ids do not satisfy the prefix-chain contract")
                    parents[key] = identity
                    previous = key
                    parent_checks += 1
                keys = [((session_index + 1) << 32) | key for key in local_keys]
                is_subagent = "/subagent/" in path
                subagent_rows += is_subagent
                result.append({
                    "type": "request",
                    "trace_id": f"{session['id']}:{path}",
                    "instance_id": instance_id,
                    "timestamp_ns": 1_000_000_000 + round(float(req["t"]) / args.time_scale * 1e9),
                    "keys": keys,
                    "input_len": req["in"],
                    "output_len": req["out"],
                    "query_type": "prefix_match",
                    "block_mask": [],
                    "source_session_id": session["id"],
                    "source_stream_path": path,
                    "source_is_subagent": is_subagent,
                })
            sessions.append({
                "session_id": session["id"],
                "source_line_1based": session_index + 1,
                "available_leaf_requests": len(flattened),
                "selected_requests": len(kept),
            })

    result.sort(key=lambda row: (row["timestamp_ns"], row["trace_id"]))
    return result, {
        "kind": "agentx_observed_arrival_subset",
        "path": str(args.trace.resolve()),
        "sha256": digest(args.trace),
        "source_bytes": args.trace.stat().st_size,
        "sessions": sessions,
        "leaf_requests_examined": total_seen,
        "filtered_requests": dict(filtered),
        "selected_subagent_requests": subagent_rows,
        "prefix_parent_checks": parent_checks,
        "hash_globalization": "(source_session_line_1based << 32) | local_prefix_hash_id",
        "keys_are_prefix_chained": True,
        "model_mapping": "all selected source-model requests -> moonshotai/Kimi-K3",
        "arrival_policy": "observed session-relative timestamps / time_scale; fixed across capacities",
        "time_scale": args.time_scale,
        "source_time_window_s": args.agentx_window_seconds,
        "requests_per_session_limit": args.requests_per_session,
        "max_input_tokens": args.max_input_tokens,
        "official_agentx_benchmark": False,
    }


def prepare(args) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with args.trace.open() as stream:
        first = json.loads(next(stream))
    is_agentx = "requests" in first and "hash_id_scope" in first
    if is_agentx:
        args.native_block_tokens = int(first["block_size"])
        args.keys_are_prefix_chained = True

    layout = make_layout(args)
    save_json(out / "layout.json", layout)
    interval = args.checkpoint_tokens
    native = args.native_block_tokens
    stride = interval // native
    instance_id = f"kimi-k3.tp{args.tp_size}.checkpoint{interval}"

    if is_agentx:
        prepared, source = flatten_agentx_requests(args, instance_id)
    else:
        prepared = []
        previous = -1
        source_instances = set()
        for row in rows(args.trace):
            if row.get("type") == "write":
                continue
            if row.get("type") not in {"get", "request"}:
                raise ValueError("Trace must contain get/request rows")
            if row["timestamp_ns"] < previous:
                raise ValueError("Trace timestamps must be nondecreasing")
            previous = row["timestamp_ns"]
            if len(row["keys"]) != row["input_len"] // native:
                raise ValueError("Trace keys must cover all complete native blocks")
            row = dict(row)
            source_instances.add(row.get("instance_id", "default"))
            if len(source_instances) > 1:
                raise ValueError("Generic trace has multiple source instances; split it first")
            row["instance_id"] = instance_id
            row.setdefault("output_len", args.output_tokens)
            prepared.append(row)
        source = {"kind": "reused_trace", "path": str(args.trace.resolve()), "sha256": digest(args.trace)}

    if not prepared:
        raise ValueError("Empty trace")

    seen, unique_bundles = set(), set()
    for row in prepared:
        trace_id = str(row["trace_id"])
        if trace_id in seen:
            raise ValueError(f"Duplicate trace_id {trace_id}")
        seen.add(trace_id)
        chain = row["keys"] if args.keys_are_prefix_chained else apply_prefix_hash(row["keys"])
        row["bundle_keys"] = chain[stride - 1::stride]
        unique_bundles.update(row["bundle_keys"])
        row["max_cache_hit_tokens"] = max(0, (row["input_len"] - 1) // interval * interval)

    trace_path = out / "trace.jsonl"
    with trace_path.open("w") as stream:
        for row in prepared:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    spec_infos, kv_names, state_names = [], [], []
    for rank in range(args.tp_size):
        kv_name = f"mla.tp{rank}"
        state_name = f"kda.tp{rank}"
        kv_names.append(kv_name)
        state_names.append(state_name)
        spec_infos += [
            {"name": kv_name, "size": interval * layout["mla_bytes_per_token_per_rank"]},
            {"name": state_name, "size": layout["kda_state_bytes_per_checkpoint_per_rank"]},
        ]

    facts_dir = out / "stage1"
    facts_dir.mkdir(exist_ok=True)
    litehit_config = {
        "trace_file_path": str(trace_path),
        "output_result_path": str(facts_dir),
        "block_size": native,
        "pipeline_worker_count": 1,
        "reserve_last_token": True,
        "instance_groups": [{
            "name": "kimi-k3-bundles",
            "capacity_gb": [1],
            "eviction_policy": "lru",
            "ttl_seconds": 0,
            "enable_prefix_hash": not args.keys_are_prefix_chained,
        }],
        "instances": [{
            "instance_group_name": "kimi-k3-bundles",
            "instance_id": instance_id,
            "block_size": interval,
            "linear_step": 1,
            "location_spec_infos": spec_infos,
            "location_spec_groups": [
                {"name": "full", "spec_names": kv_names},
                {"name": "linear", "spec_names": state_names},
            ],
            "optimizer_state_info": {
                "full_location_spec_group_name": "full",
                "linear_location_spec_group_name": "linear",
            },
        }],
    }
    save_json(out / "litehit_config.json", litehit_config)
    save_json(out / "trace_manifest.json", {
        "source": source,
        "requests": len(prepared),
        "input_tokens": sum(row["input_len"] for row in prepared),
        "output_tokens": sum(row["output_len"] for row in prepared),
        "trace_sha256": digest(trace_path),
        "layout_sha256": digest(out / "layout.json"),
        "unique_bundles": len(unique_bundles),
        "unbounded_resident_payload_gib": len(unique_bundles) * layout["bundle_bytes"] / GIB,
    })
    unbounded_gib = len(unique_bundles) * layout["bundle_bytes"] / GIB
    print(f"Prepared {len(prepared)} requests; bundle={layout['bundle_bytes'] / (1 << 20):.3f} MiB over TP{args.tp_size}")
    print(f"Unique bundles: {len(unique_bundles):,}; unbounded capacity: {unbounded_gib:,.2f} GiB ({unbounded_gib/1024:.2f} TiB)")


def scan(args) -> None:
    out = args.output_dir
    phase = out / "stage1"
    layout = load_json(out / "layout.json")
    manifest = load_json(out / "trace_manifest.json")
    if manifest["trace_sha256"] != digest(out / "trace.jsonl") or manifest["layout_sha256"] != digest(out / "layout.json"):
        raise ValueError("Trace/layout changed; rerun prepare before scanning")

    binary_dir = REPO / "bazel-bin/kv_cache_manager/optimizer"
    facts = phase / "litehit_facts.csv"
    replay = run_command([str(binary_dir / "lite_hit_main"), str(out / "litehit_config.json")], phase, "replay")

    # Dynamically determine capacity range based on unbounded capacity
    unbounded_gib = manifest.get("unbounded_resident_payload_gib", 1024.0)
    max_capacity = max(1024.0, min(unbounded_gib * 1.1, 102400.0))  # Up to 100 TB max, 10% above unbounded

    # Generate capacity points: fine-grained up to 1024 GiB, then coarser for larger ranges
    if max_capacity <= 1024.0:
        # Original range for small datasets
        capacities = sorted(set([0.0, 50.0, 1024.0] + [float(value) for value in range(25, 1025, 25)]))
    else:
        # Extended range for large datasets
        capacities = sorted(set(
            [0.0, 50.0] +
            [float(value) for value in range(25, 1025, 25)] +  # 25-1024 GiB, step 25
            [float(value) for value in range(1100, int(min(5000, max_capacity)), 100)] +  # 1100-5000 GiB, step 100
            [float(value) for value in range(5000, int(max_capacity) + 1, 500)] +  # 5000-max GiB, step 500
            [max_capacity]
        ))
    capacities = capacities + [-1.0]  # -1.0 means infinite capacity
    query_path = phase / "capacity_query.jsonl"
    query = run_command(
        [str(binary_dir / "lite_hit_facts_query_main"), str(facts), str(query_path)] + [str(capacity) for capacity in capacities],
        phase,
        "query",
    )

    query_rows = [json.loads(line) for line in query_path.read_text().splitlines()]
    summary = next(row for row in query_rows if row.get("summary") and "instance_id" not in row)
    trace = list(rows(out / "trace.jsonl"))
    projected = [row for row in query_rows if not row.get("summary")]
    if len(projected) != len(trace) or summary["requests"] != len(trace):
        raise AssertionError("Facts/query request count differs from trace")

    with facts.open() as stream:
        facts_rows = list(csv.DictReader(stream))
    assert len(facts_rows) == len(trace)
    assert all(int(row["block_bytes"]) == layout["bundle_bytes"] for row in facts_rows)

    checked = 0
    for slot, capacity in enumerate(capacities):
        limit = None if capacity < 0 else math.floor(capacity * GIB) // layout["bundle_bytes"]
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
                raise AssertionError(f"Oracle mismatch: {capacity=}, {row['trace_id']=}, {hits=}, {actual=}")
            checked += 1
            for key in reversed(row["bundle_keys"]):
                cache.pop(key, None)
                cache[key] = None
                if limit is not None:
                    while len(cache) > limit:
                        cache.popitem(last=False)

    counts = summary["total_hit_tokens"]
    rates = [value / summary["total_input_tokens"] for value in counts]
    assert all(left <= right for left, right in zip(counts, counts[1:]))
    finite = list(zip(capacities[:-1], rates[:-1]))
    gain = finite[-1][1] - finite[0][1]
    elbow = max(finite, key=lambda point: ((point[1] - finite[0][1]) / gain if gain else 0) - point[0] / finite[-1][0])[0]
    cap95 = next((capacity for capacity, hit_rate in finite if hit_rate >= 0.95 * rates[-1]), None)

    with (phase / "capacity_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["capacity_gib", "hit_tokens", "input_tokens", "hit_rate", "marginal_pp_per_gib"])
        writer.writeheader()
        previous = None
        for capacity, hit_tokens, hit_rate in zip(capacities, counts, rates):
            slope = "" if previous is None or capacity < 0 else (hit_rate - previous[1]) * 100 / (capacity - previous[0])
            writer.writerow({
                "capacity_gib": "infinite" if capacity < 0 else capacity,
                "hit_tokens": hit_tokens,
                "input_tokens": summary["total_input_tokens"],
                "hit_rate": hit_rate,
                "marginal_pp_per_gib": slope,
            })
            previous = (capacity, hit_rate)

    save_json(phase / "summary.json", {
        **summary,
        "hit_rates_exact_from_integer_counts": rates,
        "trace_sha256": manifest["trace_sha256"],
        "layout_sha256": manifest["layout_sha256"],
        "replay_wall_s": replay["elapsed_wall_s"],
        "query_wall_s": query["elapsed_wall_s"],
        "queried_capacities": len(capacities),
        "facts_sha256": digest(facts),
        "elbow_capacity_gib": elbow,
        "capacity_at_95pct_infinite_hit_rate_gib": cap95,
        "elbow_rule": "max(normalized finite hit gain - normalized capacity), finite range 0..1024 GiB",
        "selected_capacity_gib": sorted(set([0.0, 50.0, elbow])) + [-1.0],
        "validation": {
            "independent_lru_request_capacity_checks": checked,
            "mismatches": 0,
            "facts_rows": len(facts_rows),
            "uniform_bundle_charge_checked": True,
        },
        "semantics": "request-start snapshot; immediate reverse write-back",
    })
    print(f"Facts replay {replay['elapsed_wall_s']:.3f}s; {len(capacities)}-capacity query {query['elapsed_wall_s']:.3f}s; elbow={elbow:g} GiB; oracle checks={checked}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="prepare,scan", help="comma-separated prepare,scan phases")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--keys-are-prefix-chained", action="store_true")
    parser.add_argument("--agentx-sessions", type=int, default=8)
    parser.add_argument("--requests-per-session", type=int, default=24)
    parser.add_argument("--agentx-window-seconds", type=float, default=3600.0)
    parser.add_argument("--max-input-tokens", type=int, default=262144)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--native-block-tokens", type=int, default=256)
    parser.add_argument("--checkpoint-tokens", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    if args.time_scale <= 0 or args.agentx_sessions <= 0 or args.requests_per_session <= 0:
        parser.error("time scale and AgentX selection limits must be positive")
    phases = {"prepare": prepare, "scan": scan}
    for step in args.steps.split(","):
        if step not in phases:
            parser.error(f"unknown phase {step}; curve-only workflow supports prepare,scan only")
        phases[step](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproducible Kimi-K3 cache-capacity workflow; all capacities are payload GiB."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from kv_cache_manager.optimizer.tools.trace_converter.utils.prefix_hash import apply_prefix_hash

GIB = 1 << 30


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def run_command(command: list[str], output: Path, name: str, env=None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {
        "command": command, "cwd": str(REPO), "returncode": result.returncode,
        "elapsed_wall_s": time.perf_counter() - start,
        "stdout_stderr": str(output / f"{name}.log"),
    }
    save_json(output / f"{name}.command.json", record)
    if result.returncode:
        raise RuntimeError(f"{name} failed: {(output / f'{name}.log').read_text()[-6000:]}")
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
        raise ValueError("KDA heads must be divisible by attention TP; no floor/replica approximation")
    if args.native_block_tokens <= 0 or args.checkpoint_tokens % args.native_block_tokens:
        raise ValueError("checkpoint interval must be a positive multiple of trace block size")
    if args.checkpoint_tokens <= 0:
        raise ValueError("checkpoint interval must be positive")
    heads = linear["num_heads"] // args.tp_size
    dim = linear["head_dim"]
    token_bytes = len(full) * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * 2
    ssm_bytes = len(kda) * heads * dim * dim * 4
    conv_bytes = len(kda) * 3 * (linear["short_conv_kernel_size"] - 1) * heads * dim * 2
    state_bytes = ssm_bytes + conv_bytes
    rank_bundle = args.checkpoint_tokens * token_bytes + state_bytes
    return {
        "schema_version": "hisim_hybrid_checkpoint_layout_v1",
        "model_name": "moonshotai/Kimi-K3", "model_config_path": str(model_path),
        "model_config_sha256": digest(model_path), "attention_tp_size": args.tp_size,
        "checkpoint_tokens": args.checkpoint_tokens, "native_block_tokens": args.native_block_tokens,
        "mla_layers": len(full), "kda_layers": len(kda),
        "mla_layer_ids_1based": full, "kda_layer_ids_1based": kda,
        "kv_dtype": "bfloat16", "conv_dtype": "bfloat16", "state_dtype": "float32",
        "mla_bytes_per_token_per_rank": token_bytes,
        "kda_ssm_bytes_per_checkpoint_per_rank": ssm_bytes,
        "kda_conv_bytes_per_checkpoint_per_rank": conv_bytes,
        "kda_state_bytes_per_checkpoint_per_rank": state_bytes,
        "bundle_bytes_per_rank": rank_bundle, "bundle_bytes": rank_bundle * args.tp_size,
        "storage_scope": "all_tp_rank_payloads", "capacity_unit": "GiB_payload",
        "policy": "atomic_checkpoint_bundle_lru", "commit_order": "tail_to_head",
        "g1_cross_request_cache": False, "speculative_decoding": False,
        "payload_accounting": "All TP rank tensor payloads retained, including replicated MLA; excludes storage index overhead",
        "restore_contract": "continuous full bundles and all KDA states at endpoint; no partial checkpoint hit",
        "backend_scope": "explicit KVCM bundle policy, not default SGLang Mamba dual-pool policy",
    }


def agentx_requests(args, instance_id: str) -> tuple[list[dict], dict]:
    """Flatten observed AgentX timestamps; preserve nested stream identity and lengths.

    This is a fixed-arrival counterfactual, not AgentX's closed-loop DAG benchmark.
    Hash ids already name prefix-chained 64-token blocks and are local per session.
    """
    rows, sessions, filtered = [], [], Counter()
    total_seen = 0
    parent_checks = 0
    subagent_rows = 0

    def leaves(items, stream="main", parent_start=0.0):
        for index, request in enumerate(items):
            path = f"{stream}/{index}"
            if request.get("type") == "subagent":
                yield from leaves(request.get("requests", []), path + "/subagent", float(request["t"]))
            elif all(k in request for k in ("in", "out", "hash_ids", "t")):
                if float(request["t"]) + 1e-6 < parent_start:
                    raise ValueError("Child time is not session-relative; explicit timestamp conversion is required")
                yield path, request
            else:
                filtered["not_a_complete_request"] += 1

    with args.trace.open() as f:
        for session_index, line in enumerate(f):
            if session_index >= args.agentx_sessions:
                break
            session = json.loads(line)
            if session.get("hash_id_scope") != "local" or session["block_size"] != args.native_block_tokens:
                raise ValueError("AgentX input must have one shared block size and session-local hash ids")
            flattened = sorted(leaves(session["requests"]), key=lambda item: (float(item[1]["t"]), item[0]))
            seen = len(flattened)
            total_seen += seen
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
            # Inductive check: each local prefix id has one depth and one parent.
            # This catches accidental use of non-prefix raw block hashes.
            parents = {}
            for path, req in kept:
                local_keys = req["hash_ids"]
                if len(local_keys) != req["in"] // args.native_block_tokens:
                    raise ValueError(f"AgentX length mismatch in {session['id']} {path}")
                previous = None
                for depth, key in enumerate(local_keys):
                    if not isinstance(key, int) or not 0 <= key < (1 << 32):
                        raise ValueError("AgentX local hash id outside supported uint32 range")
                    identity = (depth, previous)
                    if key in parents and parents[key] != identity:
                        raise ValueError("AgentX ids do not satisfy the prefix-chain contract")
                    parents[key] = identity
                    previous = key
                    parent_checks += 1
                keys = [((session_index + 1) << 32) | key for key in local_keys]
                is_subagent = "/subagent/" in path
                subagent_rows += is_subagent
                rows.append({
                    "type": "request", "trace_id": f"{session['id']}:{path}", "instance_id": instance_id,
                    "timestamp_ns": 1_000_000_000 + round(float(req["t"]) / args.time_scale * 1e9),
                    "keys": keys, "input_len": req["in"], "output_len": req["out"],
                    "query_type": "prefix_match", "block_mask": [],
                    "source_session_id": session["id"], "source_stream_path": path,
                    "source_model": req.get("model"), "source_type": req.get("type"),
                    "source_t_s": req["t"], "source_api_time_s": req.get("api_time"),
                    "source_ttft_s": req.get("ttft"), "source_think_time_s": req.get("think_time"),
                    "source_is_subagent": is_subagent,
                })
            sessions.append({"session_id": session["id"], "source_line_1based": session_index + 1,
                             "available_leaf_requests": seen, "selected_requests": len(kept)})
    rows.sort(key=lambda r: (r["timestamp_ns"], r["trace_id"]))
    return rows, {
        "kind": "agentx_observed_arrival_subset", "path": str(args.trace.resolve()),
        "sha256": digest(args.trace), "source_bytes": args.trace.stat().st_size,
        "methodology_url": "https://inferencex.semianalysis.com/agentx",
        "sessions": sessions, "leaf_requests_examined": total_seen, "filtered_requests": dict(filtered),
        "selected_subagent_requests": subagent_rows, "prefix_parent_checks": parent_checks,
        "hash_globalization": "(source_session_line_1based << 32) | local_prefix_hash_id; children share session namespace",
        "keys_are_prefix_chained": True, "model_mapping": "all selected source-model requests -> moonshotai/Kimi-K3",
        "arrival_policy": "all selected sessions start together; observed session-relative timestamps / time_scale; fixed across capacities",
        "time_scale": args.time_scale, "source_time_window_s": args.agentx_window_seconds,
        "requests_per_session_limit": args.requests_per_session,
        "max_input_tokens": args.max_input_tokens,
        "preserved_exactly": ["input_len", "output_len", "within_session_prefix_identity", "nested_stream_identity"],
        "not_preserved": ["closed_loop_completion_dependent_arrivals", "original_model_latency", "original_model_tokenizer"],
        "official_agentx_benchmark": False,
        "note": "No original payloads. No 25-75% random start, primer, one-hour profiling, forced speculative acceptance, or model quality claim",
    }


def prepare(args) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    is_agentx = False
    if args.trace is not None:
        with args.trace.open() as f:
            first = json.loads(next(f))
        is_agentx = "requests" in first and "hash_id_scope" in first
        if is_agentx:
            args.native_block_tokens = int(first["block_size"])
            args.keys_are_prefix_chained = True
    layout = make_layout(args)
    save_json(out / "layout.json", layout)
    interval, native = args.checkpoint_tokens, args.native_block_tokens
    stride = interval // native
    instance_id = f"kimi-k3.tp{args.tp_size}.checkpoint{interval}"
    rows = []
    if args.trace is None:
        if args.requests < 1 or args.request_rate <= 0:
            raise ValueError("requests and request rate must be positive")
        rng = random.Random(args.seed)
        # Shared system prefix; 16 hot domain prefixes; a long tail of 48 domains;
        # a unique request suffix. Ids represent synthetic token-block identity.
        domain_ids = list(range(64))
        weights = [12.0 / (1 + i * 0.08) if i < 16 else 0.35 for i in domain_ids]
        domains = rng.choices(domain_ids, weights=weights, k=args.requests)
        for i, domain in enumerate(domains):
            keys = list(range(1, 4 * stride + 1))
            keys += [1_000_000 + domain * 10000 + j for j in range(16 * stride)]
            keys += [100_000_000 + i * 10000 + j for j in range(4 * stride)]
            rows.append({
                "type": "request", "trace_id": f"request-{i:05d}", "instance_id": instance_id,
                "timestamp_ns": 1_000_000_000 + round(i / args.request_rate * 1e9),
                "keys": keys, "input_len": len(keys) * native + 32,
                "output_len": args.output_tokens, "query_type": "prefix_match", "block_mask": [],
                "synthetic_domain": domain,
            })
        source = {"kind": "synthetic_prefix_identity_trace", "seed": args.seed,
                  "hot_domains": 16, "total_domains": 64,
                  "prefix_bundles": 20, "unique_suffix_bundles": 4,
                  "request_rate_per_s": args.request_rate,
                  "note": "Constructed length and block-identity workload; no real user corpus or GPU measurement"}
    elif is_agentx:
        rows, source = agentx_requests(args, instance_id)
    else:
        previous = -1
        source_instances = set()
        for line in args.trace.read_text().splitlines():
            row = json.loads(line)
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
                raise ValueError("Generic trace has multiple source instances; split it before model remapping to preserve instance isolation")
            row["instance_id"] = instance_id
            row.setdefault("output_len", args.output_tokens)
            rows.append(row)
        source = {"kind": "reused_trace", "path": str(args.trace.resolve()),
                  "sha256": digest(args.trace), "keys_are_prefix_chained": args.keys_are_prefix_chained}
    if not rows:
        raise ValueError("Empty trace")
    seen = set()
    unique_bundles = set()
    for row in rows:
        rid = str(row["trace_id"])
        if rid in seen:
            raise ValueError(f"Duplicate trace_id {rid}")
        seen.add(rid)
        if args.trace is not None and args.keys_are_prefix_chained:
            chain = row["keys"]
        else:
            chain = apply_prefix_hash(row["keys"])
        row["bundle_keys"] = chain[stride - 1::stride]
        unique_bundles.update(row["bundle_keys"])
        # Keep every completed bundle for future longer requests; the current
        # query reserves a token to produce logits without changing source lengths.
        row["max_cache_hit_tokens"] = max(0, (row["input_len"] - 1) // interval * interval)
    trace_path = out / "trace.jsonl"
    with trace_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    spec_infos, kv_names, state_names = [], [], []
    for rank in range(args.tp_size):
        kv, state = f"mla.tp{rank}", f"kda.tp{rank}"
        kv_names.append(kv)
        state_names.append(state)
        spec_infos += [
            {"name": kv, "size": interval * layout["mla_bytes_per_token_per_rank"]},
            {"name": state, "size": layout["kda_state_bytes_per_checkpoint_per_rank"]},
        ]
    facts_dir = out / "stage1"
    facts_dir.mkdir(exist_ok=True)
    config = {
        "trace_file_path": str(trace_path), "output_result_path": str(facts_dir),
        "block_size": native, "pipeline_worker_count": 1, "reserve_last_token": True,
        "instance_groups": [{"name": "kimi-k3-bundles", "capacity_gb": [1],
                             "eviction_policy": "lru", "ttl_seconds": 0,
                             "enable_prefix_hash": not (args.trace is not None and args.keys_are_prefix_chained)}],
        "instances": [{"instance_group_name": "kimi-k3-bundles", "instance_id": instance_id,
                       "block_size": interval, "linear_step": 1,
                       "location_spec_infos": spec_infos,
                       "location_spec_groups": [{"name": "full", "spec_names": kv_names},
                                                {"name": "linear", "spec_names": state_names}],
                       "optimizer_state_info": {"full_location_spec_group_name": "full",
                                                "linear_location_spec_group_name": "linear"}}],
    }
    save_json(out / "litehit_config.json", config)
    save_json(out / "trace_manifest.json", {
        "source": source, "requests": len(rows), "input_tokens": sum(r["input_len"] for r in rows),
        "output_tokens": sum(r["output_len"] for r in rows), "trace_sha256": digest(trace_path),
        "layout_sha256": digest(out / "layout.json"), "unique_bundles": len(unique_bundles),
        "unbounded_resident_payload_gib": len(unique_bundles) * layout["bundle_bytes"] / GIB,
        "registration_capacity_note": "instance_groups.capacity_gb=[1] is an unused registration placeholder; facts replay has no capacity",
    })
    print(f"Prepared {len(rows)} requests; bundle={layout['bundle_bytes'] / (1 << 20):.3f} MiB over TP{args.tp_size}")


def scan(args) -> None:
    out, phase = args.output_dir, args.output_dir / "stage1"
    layout = load_json(out / "layout.json")
    manifest = load_json(out / "trace_manifest.json")
    if manifest["trace_sha256"] != digest(out / "trace.jsonl") or manifest["layout_sha256"] != digest(out / "layout.json"):
        raise ValueError("Trace/layout changed; prepare a consistent input set before scanning")
    binary_dir = REPO / "bazel-bin/kv_cache_manager/optimizer"
    facts = phase / "litehit_facts.csv"
    replay = run_command([str(binary_dir / "lite_hit_main"), str(out / "litehit_config.json")], phase, "replay")
    capacities = sorted(set([0.0, 50.0, 1024.0] + [float(x) for x in range(25, 1025, 25)])) + [-1.0]
    query_path = phase / "capacity_query.jsonl"
    query = run_command([str(binary_dir / "lite_hit_facts_query_main"), str(facts), str(query_path)]
                        + [str(c) for c in capacities], phase, "query")
    result_rows = [json.loads(line) for line in query_path.read_text().splitlines()]
    summary = next(r for r in result_rows if r.get("summary") and "instance_id" not in r)
    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    projected = [r for r in result_rows if not r.get("summary")]
    if len(projected) != len(trace) or summary["requests"] != len(trace):
        raise AssertionError("Facts/query request count differs from trace")
    with facts.open() as f:
        facts_rows = list(csv.DictReader(f))
    assert len(facts_rows) == len(trace)
    assert all(int(r["block_bytes"]) == layout["bundle_bytes"] for r in facts_rows)
    # Independent finite caches validate every request and every queried capacity;
    # they do not derive expected results from LiteHit's facts or RLE formula.
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
    counts = summary["total_hit_tokens"]
    rates = [v / summary["total_input_tokens"] for v in counts]
    assert all(a <= b for a, b in zip(counts, counts[1:]))
    finite = list(zip(capacities[:-1], rates[:-1]))
    gain = finite[-1][1] - finite[0][1]
    knee = max(finite, key=lambda p: ((p[1] - finite[0][1]) / gain if gain else 0) - p[0] / finite[-1][0])[0]
    cap95 = next((c for c, h in finite if h >= 0.95 * rates[-1]), None)
    selected = sorted(set([0.0, 50.0, knee])) + [-1.0]
    previous = None
    with (phase / "capacity_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["capacity_gib", "hit_tokens", "input_tokens", "hit_rate", "marginal_pp_per_gib"])
        writer.writeheader()
        for c, n, h in zip(capacities, counts, rates):
            slope = "" if previous is None or c < 0 else (h - previous[1]) * 100 / (c - previous[0])
            writer.writerow({"capacity_gib": "infinite" if c < 0 else c, "hit_tokens": n,
                             "input_tokens": summary["total_input_tokens"], "hit_rate": h,
                             "marginal_pp_per_gib": slope})
            previous = (c, h)
    save_json(phase / "summary.json", {
        **summary, "hit_rates_exact_from_integer_counts": rates,
        "trace_sha256": manifest["trace_sha256"], "layout_sha256": manifest["layout_sha256"],
        "replay_wall_s": replay["elapsed_wall_s"], "query_wall_s": query["elapsed_wall_s"],
        "queried_capacities": len(capacities), "facts_sha256": digest(facts),
        "elbow_capacity_gib": knee, "capacity_at_95pct_infinite_hit_rate_gib": cap95,
        "elbow_rule": "max(normalized finite hit gain - normalized capacity), finite range 0..1024 GiB",
        "selected_capacity_gib": selected,
        "validation": {"independent_lru_request_capacity_checks": checked, "mismatches": 0,
                       "facts_rows": len(facts_rows), "uniform_bundle_charge_checked": True},
        "semantics": "request-start snapshot; immediate reverse write-back. Stage2 models delayed write visibility separately",
    })
    print(f"Facts replay {replay['elapsed_wall_s']:.3f}s; {len(capacities)}-capacity query {query['elapsed_wall_s']:.3f}s; elbow={knee:g} GiB; oracle checks={checked}")



def bench(args) -> None:
    """Run the actual HiSim bench_serving entrypoint on representative capacities."""
    out = args.output_dir
    manifest = load_json(out / "trace_manifest.json")
    layout = load_json(out / "layout.json")
    if manifest["trace_sha256"] != digest(out / "trace.jsonl") or manifest["layout_sha256"] != digest(out / "layout.json"):
        raise ValueError("Trace/layout changed since prepare")
    stage1 = load_json(out / "stage1/summary.json")
    if stage1.get("trace_sha256") != manifest["trace_sha256"] or stage1.get("layout_sha256") != manifest["layout_sha256"]:
        raise ValueError("Stage1 facts do not carry the current trace/layout hashes; run scan before bench")
    capacities = stage1["selected_capacity_gib"]
    if args.capacities:
        capacities = [(-1.0 if c.strip().lower() in ("inf", "infinite") else float(c))
                      for c in args.capacities.split(",")]
    if any(c < 0 and c != -1 for c in capacities) or not capacities:
        raise ValueError("Use nonnegative GiB capacities or -1/infinite")
    if any(c not in stage1["capacity_gb"] for c in capacities):
        raise ValueError("--capacities must select points from stage1.capacity_gb; use the Facts query CLI for additional points")
    phase = out / "stage2"
    phase.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "hisim/src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    base = {
        "schema": "hisim.hybrid_checkpoint.v1",
        "layout_path": str(out / "layout.json"), "trace_path": str(out / "trace.jsonl"),
        "arrival_scale": args.bench_arrival_scale,
        "predictor": {"max_batch_size": args.max_concurrency, "max_num_tokens": args.max_prefill_tokens,
                      "memory_fraction": args.memory_fraction},
        "scheduler": {"max_concurrency": args.max_concurrency, "max_prefill_tokens": args.max_prefill_tokens,
                      "chunked_prefill_tokens": args.chunked_prefill_tokens or layout["checkpoint_tokens"],
                      "g1_page_tokens": 64, "active_kda_buffers": 2},
        "storage": {"capacity_bytes": None,
                    "read_bandwidth_bytes_per_s": args.read_gib_s * GIB,
                    "write_bandwidth_bytes_per_s": args.write_gib_s * GIB,
                    "h2d_bandwidth_bytes_per_s": args.h2d_gib_s * GIB,
                    "d2h_bandwidth_bytes_per_s": args.d2h_gib_s * GIB},
        "bandwidth_provenance": "Scenario assumptions, aggregate across all eight attention TP ranks; not measured hardware bandwidth",
        "workload_provenance": manifest["source"],
    }
    scenarios = [(f"capacity_{'infinite' if cap < 0 else format(cap, 'g') + 'gib'}", cap, args.max_concurrency)
                 for cap in capacities] if not args.concurrency_only else []
    for limit in args.concurrency_probes:
        if limit == args.max_concurrency:
            continue
        scenarios.append((f"elbow_concurrency_{limit}", stage1["elbow_capacity_gib"], limit))
    from analyze_results import validate_run, collect_summary
    if args.probe_requests:
        rows = (out / "trace.jsonl").read_text().splitlines()
        probe = phase / "probe"
        probe.mkdir(exist_ok=True)
        probe_trace = probe / "trace.jsonl"
        probe_trace.write_text("\n".join(rows[:args.probe_requests]) + "\n")
        config = json.loads(json.dumps(base))
        config.update(trace_path=str(probe_trace), output_dir=str(probe / "result"))
        config_path = probe / "config.json"
        save_json(config_path, config)
        print(f"HiSim preflight: {min(args.probe_requests, len(rows))} complete requests, infinite payload capacity", flush=True)
        command = [sys.executable, "-m", "hisim.simulation.bench_serving", "--hybrid-checkpoint-config", str(config_path)]
        run_command(command, probe, "bench_serving", env=env)
        validate_run(probe / "result", layout, probe_trace)
        print("HiSim preflight passed event, memory, and token accounting checks", flush=True)
    for name, cap, concurrency in scenarios:
        config = json.loads(json.dumps(base))
        destination = phase / name
        config["output_dir"] = str(destination)
        config["storage"]["capacity_bytes"] = None if cap < 0 else math.floor(cap * GIB)
        config["scheduler"]["max_concurrency"] = concurrency
        config["predictor"]["max_batch_size"] = max(args.max_concurrency, concurrency)
        config_path = phase / "configs" / f"{name}.json"
        save_json(config_path, config)
        print(f"HiSim {name}: admission limit={concurrency}, starting", flush=True)
        command = [sys.executable, "-m", "hisim.simulation.bench_serving", "--hybrid-checkpoint-config", str(config_path)]
        command_record = run_command(command, destination, "bench_serving", env=env)
        validate_run(destination, layout, out / "trace.jsonl")
        metrics = load_json(destination / "metrics.json")
        print(f"HiSim {name}: {command_record['elapsed_wall_s']:.2f}s CPU wall; "
              f"P50/P99 TTFT={metrics['p50_ttft_ms']/1000:.3f}/{metrics['p99_ttft_ms']/1000:.3f}s; "
              f"output={metrics['output_throughput']:.2f} tok/s; active peak={metrics['observed_peak_active']}", flush=True)
    collect_summary(out)


def plot(args) -> None:
    from analyze_results import plot_and_report
    plot_and_report(args.output_dir)
    print(f"Figures, offline HTML, and report: {args.output_dir / 'report'}", flush=True)


def provenance(args) -> None:
    paths = [Path(__file__), Path(__file__).with_name("analyze_results.py"), Path(__file__).with_name("SOP.md"),
             REPO / "hisim/src/hisim/simulation/hybrid_checkpoint.py",
             REPO / "hisim/src/hisim/simulation/bench_serving.py",
             REPO / "hisim/src/hisim/time_predictor/kimi_k3.py",
             REPO / "kv_cache_manager/optimizer/manager/lite_hit_offline_runner.cc",
             REPO / "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.h",
             REPO / "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.cc",
             REPO / "bazel-bin/kv_cache_manager/optimizer/lite_hit_main",
             REPO / "bazel-bin/kv_cache_manager/optimizer/lite_hit_facts_query_main"]
    save_json(args.output_dir / "source_manifest.json", {
        "python_executable": sys.executable, "python_version": sys.version,
        "repository": str(REPO), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "sources": [{"path": str(path), "sha256": digest(path)} for path in paths if path.exists()],
        "note": "Working-tree source hashes identify uncommitted changes; no commit or remote publish performed",
    })

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", default="prepare,scan,bench,plot", help="comma-separated prepare,scan,bench,plot phases, or all")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-config", type=Path,
                   default=Path("/workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync/aic-core/src/aiconfigurator_core/model_configs/moonshotai--Kimi-K3_config.json"))
    p.add_argument("--trace", type=Path)
    p.add_argument("--keys-are-prefix-chained", action="store_true")
    p.add_argument("--agentx-sessions", type=int, default=8)
    p.add_argument("--requests-per-session", type=int, default=24)
    p.add_argument("--agentx-window-seconds", type=float, default=3600.0)
    p.add_argument("--max-input-tokens", type=int, default=262144)
    p.add_argument("--time-scale", type=float, default=10.0,
                   help="divide AgentX observed relative timestamps by this factor; 1 preserves relative times")
    p.add_argument("--requests", type=int, default=512)
    p.add_argument("--request-rate", type=float, default=8.0)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--native-block-tokens", type=int, default=256)
    p.add_argument("--checkpoint-tokens", type=int, default=1024)
    p.add_argument("--capacities", help="Select any subset of Stage1 grid points; comma-separated, infinite or -1 means unbounded")
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--max-prefill-tokens", type=int, default=8192)
    p.add_argument("--chunked-prefill-tokens", type=int, default=0, help="0 uses checkpoint interval")
    p.add_argument("--memory-fraction", type=float, default=0.90)
    p.add_argument("--read-gib-s", type=float, default=64)
    p.add_argument("--write-gib-s", type=float, default=32)
    p.add_argument("--h2d-gib-s", type=float, default=128)
    p.add_argument("--d2h-gib-s", type=float, default=128)
    p.add_argument("--bench-arrival-scale", type=float, default=1.0, help="Additional factor on prepared arrival timestamps")
    p.add_argument("--probe-requests", type=int, default=16, help="Full-length preflight requests before the sweep; 0 skips")
    p.add_argument("--concurrency-only", action="store_true", help="Run only additional elbow admission probes")
    p.add_argument("--concurrency-probes", type=lambda v: [int(x) for x in v.split(",")], default=[],
                   help="Optional admission limits to test at the Stage1 elbow")
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.time_scale <= 0 or args.agentx_sessions <= 0 or args.requests_per_session <= 0:
        p.error("time scale and AgentX selection limits must be positive")
    if any(not math.isfinite(v) for v in (args.read_gib_s, args.write_gib_s, args.h2d_gib_s,
                                         args.d2h_gib_s, args.bench_arrival_scale, args.memory_fraction)):
        p.error("bandwidths, arrival scale and memory fraction must be finite")
    if min(args.max_concurrency, args.max_prefill_tokens, args.read_gib_s, args.write_gib_s,
           args.h2d_gib_s, args.d2h_gib_s, args.bench_arrival_scale) <= 0 or not 0 < args.memory_fraction <= 1:
        p.error("scheduler limits, bandwidths, arrival scale and memory fraction must be positive")
    if any(c > args.max_concurrency for c in args.concurrency_probes):
        p.error("concurrency probes must not exceed --max-concurrency; keep predictor/G1 overhead limits fixed")
    if args.concurrency_only and not args.concurrency_probes:
        p.error("--concurrency-only requires --concurrency-probes")
    if args.probe_requests < 0 or any(c <= 0 for c in args.concurrency_probes):
        p.error("probe request count must be nonnegative; concurrency limits must be positive")
    phases = {"prepare": prepare, "scan": scan, "bench": bench, "plot": plot}
    steps = list(phases) if args.steps == "all" else args.steps.split(",")
    for step in steps:
        if step not in phases:
            p.error(f"unknown phase {step}")
        phases[step](args)
    provenance(args)


if __name__ == "__main__":
    main()

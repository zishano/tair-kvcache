"""Independent ledger checks and report generation for Kimi-K3 capacity runs."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path

GIB = 1 << 30
EPS = 1e-8


def load(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7):
        raise AssertionError(f"{actual=} differs from {expected=}")


def validate_run(directory: Path, layout: dict, trace_path: Path) -> dict:
    """Check raw events, not merely simulator summary flags or hit ratios."""
    import numpy as np
    metrics = load(directory / "metrics.json")
    config = load(directory / "resolved_config.json")
    provenance = load(directory / "provenance.json")
    trace = list(rows(trace_path))
    requests = list(rows(directory / "request.jsonl"))
    by_id = {r["trace_id"]: r for r in requests}
    assert len(by_id) == len(trace) == metrics["completed"] == metrics["num_requests"]
    assert provenance["trace_sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert provenance["gpu_kernels_executed"] is False
    capacity = config["storage"]["capacity_bytes"]
    interval = layout["checkpoint_tokens"]
    bundle_bytes = layout["bundle_bytes"]
    origin_ns = min(r["timestamp_ns"] for r in trace)
    for source in trace:
        req = by_id[source["trace_id"]]
        assert (req["input_len"], req["output_len"], req["bundle_keys"]) == (
            source["input_len"], source["output_len"], list(map(str, source["bundle_keys"])))
        close(req["arrival_s"], (source["timestamp_ns"] - origin_ns) / 1e9 * config.get("arrival_scale", 1))
        times = req["token_times_s"]
        assert len(times) == req["output_len"] and all(a <= b for a, b in zip(times, times[1:]))
        assert req["arrival_s"] <= req["admission_s"] + EPS <= req["read_ready_s"] + 2 * EPS
        assert req["read_ready_s"] <= req["prefill_start_s"] + EPS <= times[0] + 2 * EPS
        close(req["finish_s"], times[-1])
        assert req["cached_prefix_tokens"] < req["input_len"]
        assert req["cached_prefix_tokens"] % interval == 0
        assert req["cached_prefix_tokens"] + req["computed_prompt_tokens"] == req["input_len"]
        close(req["ttft_s"], times[0] - req["arrival_s"])
        if req["output_len"] > 1:
            close(req["tpot_s"], (times[-1] - times[0]) / (req["output_len"] - 1))
        scheduler = config["scheduler"]
        token_slots = math.ceil((req["input_len"] + req["output_len"]) / scheduler.get("g1_page_tokens", 64)) * scheduler.get("g1_page_tokens", 64)
        mla = token_slots * layout["mla_bytes_per_token_per_rank"] * layout["attention_tp_size"]
        state = layout["kda_state_bytes_per_checkpoint_per_rank"] * layout["attention_tp_size"]
        snapshots = len(req["bundle_keys"]) if capacity is None or capacity >= bundle_bytes else 0
        expected_reserved = mla + (scheduler.get("active_kda_buffers", 2) + snapshots) * state
        assert req["g1_reserved_bytes"] == expected_reserved <= metrics["g1_capacity_bytes"]

    # Reconstruct per-request compute and output times from batches independently.
    computed, decoded = Counter(), Counter()
    end = 0.0
    batch_count = 0
    busy_s = 0.0
    query_ids = set()
    for batch in rows(directory / "iteration.jsonl"):
        assert batch["start_s"] + EPS >= end
        close(batch["end_s"] - batch["start_s"], batch["duration_s"])
        end = batch["end_s"]
        busy_s += batch["duration_s"]
        batch_count += 1
        query_ids.add(batch["aic_query_id"])
        assert batch["batch_size"] == len(batch["trace_ids"]) <= config["scheduler"]["max_concurrency"]
        assert batch["query_tokens"] == sum(batch["extend_lengths"])
        assert batch["g1_occupied_bytes"] <= metrics["g1_capacity_bytes"]
        if batch["phase"] == "prefill":
            assert batch["query_tokens"] <= config["scheduler"]["max_prefill_tokens"]
        for rid, query, prefix in zip(batch["trace_ids"], batch["extend_lengths"], batch["prefix_lengths"]):
            req = by_id[rid]
            assert batch["start_s"] + EPS >= req["read_ready_s"]
            if batch["phase"] == "prefill":
                assert prefix == req["cached_prefix_tokens"] + computed[rid]
                computed[rid] += query
                if req["cached_prefix_tokens"] + computed[rid] == req["input_len"]:
                    close(batch["end_s"], req["token_times_s"][0])
            else:
                assert query == 1 and prefix == req["input_len"] + decoded[rid]
                decoded[rid] += 1
                close(batch["end_s"], req["token_times_s"][decoded[rid]])
    for req in requests:
        assert computed[req["trace_id"]] == req["computed_prompt_tokens"]
        assert decoded[req["trace_id"]] == req["output_len"] - 1
    assert batch_count == metrics["batch_count"]
    assert query_ids == {r["query_id"] for r in rows(directory / "aic_queries.jsonl")}

    # Reconstruct storage membership and read leases from ordered operation events.
    resident = set()
    pins = Counter()
    snapshots = {}
    pending_writes = {}
    read_bytes = write_bytes = storage_checks = 0
    lanes = {lane: [] for lane in ("read", "write", "h2d", "d2h")}
    last_time = 0.0
    peak_occupancy = 0
    for event in rows(directory / "storage.jsonl"):
        assert event["time_s"] + EPS >= last_time
        last_time = event["time_s"]
        kind, rid = event["event"], event["trace_id"]
        req = by_id[rid]
        if kind == "checkpoint_materialized":
            key = tuple(event["key"])
            assert event["endpoint_tokens"] % interval == 0
            assert key == (req["instance_id"], req["bundle_keys"][event["endpoint_tokens"] // interval - 1])
            assert set(event["components"]) == {"mla", "kda_ssm", "kda_conv"}
            snapshots[(rid, key)] = event["time_s"]
        elif kind == "read_query":
            expected = []
            for key in req["bundle_keys"][:event["eligible_bundles"]]:
                key = (req["instance_id"], key)
                if key not in resident:
                    break
                expected.append(key)
            assert list(map(tuple, event["keys"])) == expected
            assert event["hit_bundles"] * interval == req["cached_prefix_tokens"]
            pins.update(expected)
            storage_checks += 1
        elif kind == "read_ready":
            for key in map(tuple, event["keys"]):
                assert pins[key] > 0
                pins[key] -= 1
        elif kind == "read_submit":
            size = req["cached_prefix_tokens"] // interval * bundle_bytes
            assert event["bytes"] == size
            read_bytes += size
            for lane, start_key, finish_key in [("read", "read_start_s", "read_end_s"), ("h2d", "h2d_start_s", "read_ready_s")]:
                close(event[finish_key] - event[start_key], size / config["storage"][f"{lane}_bandwidth_bytes_per_s"])
                lanes[lane].append((event[start_key], event[finish_key]))
            assert event["read_start_s"] + EPS >= event["time_s"]
            assert event["h2d_start_s"] + EPS >= event["read_end_s"]
        elif kind == "write_submit":
            keys = list(map(tuple, event["materialized_keys"]))
            assert all((rid, key) in snapshots and snapshots[(rid, key)] <= event["time_s"] + EPS for key in keys)
            assert event["bytes"] == len(keys) * bundle_bytes
            pending_writes[rid] = event
            write_bytes += event["bytes"]
            for lane, start_key, finish_key in [("d2h", "d2h_start_s", "d2h_end_s"), ("write", "write_start_s", "visible_at_s")]:
                close(event[finish_key] - event[start_key], event["bytes"] / config["storage"][f"{lane}_bandwidth_bytes_per_s"])
                lanes[lane].append((event[start_key], event[finish_key]))
            assert event["d2h_start_s"] + EPS >= event["time_s"]
            assert event["write_start_s"] + EPS >= event["d2h_end_s"]
        if kind in ("write_visible", "write_touch"):
            if kind == "write_visible":
                close(event["time_s"], pending_writes[rid]["visible_at_s"])
            for operation in event["operations"]:
                key = tuple(operation["key"])
                if operation["op"] == "insert":
                    assert key not in resident and (rid, key) in snapshots
                    assert kind == "write_visible"
                    resident.add(key)
                elif operation["op"] == "evict":
                    assert key in resident and pins[key] == 0
                    resident.remove(key)
                elif operation["op"] == "touch":
                    assert key in resident
                else:
                    raise AssertionError(f"Unknown storage operation {operation}")
        assert event["occupancy_bytes"] == len(resident) * bundle_bytes
        assert event["pinned_bytes"] == sum(v > 0 for v in pins.values()) * bundle_bytes
        assert event["read_pin_references"] == sum(pins.values())
        assert event["host_staging_bytes"] >= 0
        if capacity is not None:
            assert event["occupancy_bytes"] <= capacity
        peak_occupancy = max(peak_occupancy, event["occupancy_bytes"])
    assert sum(pins.values()) == 0
    assert read_bytes == metrics["storage_read_bytes"] and write_bytes == metrics["storage_write_bytes"]
    assert peak_occupancy == metrics["storage_peak_occupancy_bytes"]
    utilization = {}
    for lane, intervals in lanes.items():
        intervals.sort()
        assert all(a[1] <= b[0] + EPS for a, b in zip(intervals, intervals[1:]))
        utilization[lane] = sum(b - a for a, b in intervals) / metrics["storage_drain_complete_s"]
    queue_events = [e for e in rows(directory / "queue.jsonl") if "active" in e]
    assert max(e["active"] for e in queue_events) == metrics["observed_peak_active"]
    assert max(e["g1_occupied_bytes"] for e in queue_events) == metrics["g1_peak_occupied_bytes"]
    assert all(0 <= e["g1_occupied_bytes"] <= metrics["g1_capacity_bytes"] for e in queue_events)
    assert all(e["active"] + e["queued"] == e["outstanding"] for e in queue_events)
    assert queue_events[-1]["g1_occupied_bytes"] == 0
    duration = max(r["finish_s"] for r in requests)
    close(metrics["duration"], duration)
    close(metrics["output_throughput"], sum(r["output_len"] for r in requests) / duration)
    close(metrics["prefix_cache_reused_ratio"], sum(r["cached_prefix_tokens"] for r in requests) / sum(r["input_len"] for r in requests))
    for metric, source_key in (("ttft", "ttft_s"), ("tpot", "tpot_s")):
        values = [r[source_key] * 1000 for r in requests if r[source_key] is not None]
        for percentile in (50, 99):
            close(metrics[f"p{percentile}_{metric}_ms"], float(np.percentile(values, percentile)))
    report = {
        "passed": True, "request_count": len(requests), "batch_count": batch_count,
        "storage_query_checks": storage_checks, "storage_event_membership_and_pins_replayed": True,
        "token_and_shape_conservation": True, "metrics_recomputed_from_token_timestamps": True,
        "finite_capacity_and_g1_bounds_checked": True, "four_io_lanes_nonoverlap_checked": True,
        "compute_utilization": busy_s / duration, "io_utilization_over_storage_drain": utilization,
        "storage_read_gib": read_bytes / GIB, "storage_write_gib": write_bytes / GIB,
        "time_origin": "minimum selected trace arrival", "throughput_denominator_s": duration,
    }
    save(directory / "validation.json", report)
    return report


def collect_summary(out: Path) -> list[dict]:
    stage1 = load(out / "stage1/summary.json")
    manifest = load(out / "trace_manifest.json")
    if stage1.get("trace_sha256") != manifest["trace_sha256"] or stage1.get("layout_sha256") != manifest["layout_sha256"]:
        raise ValueError("Stage1 and prepared input hashes differ; scan the current prepared trace first")
    if hashlib.sha256((out / "trace.jsonl").read_bytes()).hexdigest() != manifest["trace_sha256"]:
        raise ValueError("Prepared trace changed after its manifest was written")
    if hashlib.sha256((out / "layout.json").read_bytes()).hexdigest() != manifest["layout_sha256"]:
        raise ValueError("Prepared layout changed after its manifest was written")
    lookup = dict(zip(stage1["capacity_gb"], stage1["hit_rates_exact_from_integer_counts"]))
    result = []
    controls = None
    for path in sorted((out / "stage2").glob("*/metrics.json")):
        directory = path.parent
        metrics = load(path)
        config = load(directory / "resolved_config.json")
        validation = load(directory / "validation.json")
        if not validation.get("passed"):
            raise ValueError(f"Unvalidated Stage2 run: {directory}")
        prov = load(directory / "provenance.json")
        if (prov["trace_sha256"], prov["layout_sha256"]) != (manifest["trace_sha256"], manifest["layout_sha256"]):
            raise ValueError(f"Stale Stage2 input provenance in {directory}; use a new output directory")
        current_controls = {
            "trace": prov["trace_sha256"], "layout": prov["layout_sha256"],
            "adapter": prov["adapter_sha256"], "predictor": prov["predictor"]["source_sha256"],
            "predictor_config": config.get("predictor", {}),
            "arrival_scale": config.get("arrival_scale", 1),
            "scheduler": {k: v for k, v in config["scheduler"].items() if k != "max_concurrency"},
            "storage": {k: v for k, v in config["storage"].items() if k != "capacity_bytes"},
        }
        if controls is None:
            controls = current_controls
        elif controls != current_controls:
            raise ValueError(f"Inconsistent workload/hardware/policy in {directory}; use a fresh output directory for a new experiment")
        capacity = config["storage"]["capacity_bytes"]
        capacity_gib = -1.0 if capacity is None else capacity / GIB
        row = {
            "scenario": directory.name, "capacity_gib": capacity_gib,
            "stage1_hit_rate": lookup.get(capacity_gib),
            "stage2_hit_rate": metrics["prefix_cache_reused_ratio"],
            "p50_ttft_ms": metrics["p50_ttft_ms"], "p99_ttft_ms": metrics["p99_ttft_ms"],
            "p50_tpot_ms": metrics["p50_tpot_ms"], "p99_tpot_ms": metrics["p99_tpot_ms"],
            "output_tokens_per_s": metrics["output_throughput"],
            "total_tokens_per_s": metrics["total_throughput"], "requests_per_s": metrics["request_throughput"],
            "duration_s": metrics["duration"], "mean_queue_ms": metrics["mean_queue_ms"],
            "configured_admission_limit": metrics["configured_max_concurrency"],
            "observed_peak_active": metrics["observed_peak_active"],
            "observed_peak_outstanding": metrics["observed_peak_outstanding"],
            "observed_peak_queue": metrics["observed_peak_queue"],
            "g1_peak_gib_all_ranks": metrics["g1_peak_occupied_bytes"] / GIB,
            "g1_budget_gib_all_ranks": metrics["g1_capacity_bytes"] / GIB,
            "storage_peak_gib": metrics["storage_peak_occupancy_bytes"] / GIB,
            "host_staging_peak_gib": metrics["host_staging_peak_bytes"] / GIB,
            "storage_evictions": metrics["storage_evictions"],
            "read_pin_rejected_bundles": metrics["storage_pin_rejected_bundles"],
            "compute_utilization": validation["compute_utilization"],
            "storage_read_utilization": validation["io_utilization_over_storage_drain"]["read"],
            "storage_write_utilization": validation["io_utilization_over_storage_drain"]["write"],
            "storage_read_gib": validation["storage_read_gib"], "storage_write_gib": validation["storage_write_gib"],
            "simulation_wall_s": metrics["time_cost"], "passed": validation["passed"],
        }
        result.append(row)
    result.sort(key=lambda r: (not r["scenario"].startswith("capacity_"), math.inf if r["capacity_gib"] < 0 else r["capacity_gib"], r["configured_admission_limit"]))
    save(out / "stage2/summary.json", result)
    if result:
        with (out / "stage2/summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    return result


def compare_hits(out: Path) -> dict:
    stage1 = load(out / "stage1/summary.json")
    slots = {float(c): i for i, c in enumerate(stage1["capacity_gb"])}
    projected = {r["trace_id"]: r for r in rows(out / "stage1/capacity_query.jsonl") if not r.get("summary")}
    interval = load(out / "layout.json")["checkpoint_tokens"]
    result = {}
    all_rows = []
    for path in sorted((out / "stage2").glob("capacity_*/request.jsonl")):
        config = load(path.parent / "resolved_config.json")
        cap = config["storage"]["capacity_bytes"]
        cap = -1.0 if cap is None else cap / GIB
        if cap not in slots:
            continue
        differences = []
        for req in rows(path):
            ideal = projected[req["trace_id"]]["hit_blocks"][slots[cap]] * interval
            actual = req["cached_prefix_tokens"]
            delta = actual - ideal
            differences.append(delta)
            all_rows.append({"scenario": path.parent.name, "trace_id": req["trace_id"],
                             "arrival_s": req["arrival_s"], "admission_s": req["admission_s"],
                             "stage1_hit_tokens": ideal, "stage2_hit_tokens": actual,
                             "stage2_minus_stage1_tokens": delta})
        result[path.parent.name] = {
            "more_hit_requests": sum(d > 0 for d in differences),
            "fewer_hit_requests": sum(d < 0 for d in differences),
            "equal_hit_requests": sum(d == 0 for d in differences),
            "positive_delta_tokens": sum(max(0, d) for d in differences),
            "negative_delta_tokens": sum(min(0, d) for d in differences),
            "net_delta_tokens": sum(differences),
        }
    save(out / "stage2/hit_comparison.json", result)
    with (out / "stage2/hit_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    return result


def plot_and_report(out: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    summary = collect_summary(out)
    primary = [r for r in summary if r["scenario"].startswith("capacity_")]
    if not primary:
        raise ValueError("Run bench before generating the report")
    differences = compare_hits(out)
    stage1 = load(out / "stage1/summary.json")
    manifest = load(out / "trace_manifest.json")
    layout = load(out / "layout.json")
    trace = list(rows(out / "trace.jsonl"))
    source = manifest["source"]
    agentx = source.get("kind") == "agentx_observed_arrival_subset"
    workload_label = "AgentX-derived" if agentx else "Synthetic" if source.get("kind") == "synthetic_prefix_identity_trace" else "Reused-trace"
    main_config = load(out / "stage2" / primary[0]["scenario"] / "resolved_config.json")
    admission_limit = main_config["scheduler"]["max_concurrency"]
    checkpoint = layout["checkpoint_tokens"]
    cap95_text = f"{cap95:g} GiB" if (cap95 := stage1["capacity_at_95pct_infinite_hit_rate_gib"]) is not None else "本次有限容量范围内未达到"
    report = out / "report"
    report.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2,
                         "figure.dpi": 130, "savefig.dpi": 180})
    colors = ["#5072A7", "#148A91", "#D37B12", "#7356A5", "#BD4A59", "#5A7A37"]
    caps = stage1["capacity_gb"][:-1]
    rates = np.array(stage1["hit_rates_exact_from_integer_counts"][:-1]) * 100
    infinite_rate = stage1["hit_rates_exact_from_integer_counts"][-1] * 100
    knee = stage1["elbow_capacity_gib"]
    cap95 = stage1["capacity_at_95pct_infinite_hit_rate_gib"]
    plateau = next((c for c, r in zip(caps, rates) if abs(r - infinite_rate) < 1e-9), None)
    labels = ["Infinite" if r["capacity_gib"] < 0 else f"{r['capacity_gib']:g}" for r in primary]
    x = np.arange(len(primary))

    def save_figure(fig, name):
        for extension in ("png", "svg", "pdf"):
            fig.savefig(report / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1]}, constrained_layout=True)
    axes[0].plot(caps, rates, color=colors[0], marker=".", linewidth=2, label="LiteHit: immediate request write-back")
    axes[0].axhline(infinite_rate, color=colors[3], linestyle="--", label=f"Infinite: {infinite_rate:.2f}%")
    for cap, color, name in [(knee, colors[2], "Geometric elbow"), (cap95, colors[1], "95% of infinite hit rate"),
                             (plateau, colors[5], "Sample plateau")]:
        if cap is None:
            continue
        rate = rates[caps.index(cap)]
        axes[0].scatter([cap], [rate], s=75, color=color, zorder=4)
        axes[0].axvline(cap, color=color, linestyle=":", alpha=.55)
        axes[0].annotate(f"{name}\n{cap:g} GiB / {rate:.2f}%", (cap, rate),
                         xytext=(cap - (85 if cap == knee else 15), rate - (19 if cap == knee else 31 if cap == cap95 else 46)),
                         arrowprops={"arrowstyle": "-", "color": color}, color=color, fontsize=10)
    axes[0].set(ylabel="Input-token hit rate (%)", ylim=(0, 100), title="Kimi-K3: capacity screening from one Facts replay")
    axes[0].legend(loc="lower right", fontsize=9)
    slopes = np.diff(rates) / np.diff(caps)
    axes[1].bar(caps[1:], slopes, width=np.diff(caps)*.82, color=colors[0])
    axes[1].set(xlabel="External cache payload capacity (GiB, all TP8 ranks)",
                ylabel="Marginal gain\n(pp / GiB)", xlim=(-10, 1040))
    fig.suptitle(f"{manifest['requests']} {workload_label} requests | {checkpoint:,}-token atomic MLA + KDA bundles | heuristic elbow on 0-1024 GiB", fontsize=10)
    save_figure(fig, "capacity_curve")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, metric, title, unit, scale in [
        (axes[0,0], "p50_ttft_ms", "P50 time to first token", "Seconds", .001),
        (axes[0,1], "p99_ttft_ms", "P99 time to first token", "Seconds", .001),
        (axes[1,0], "output_tokens_per_s", "Output throughput (whole TP8 replica)", "Output tokens / second", 1)]:
        values = [r[metric] * scale for r in primary]
        ax.bar(x, values, color=colors[:len(x)], width=.64)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values)*1.18))
    for percentile, style in [(50, "o-"), (99, "s--")]:
        axes[0,2].plot(x, [r[f"p{percentile}_tpot_ms"] for r in primary], style, label=f"P{percentile}")
    axes[0,2].set(title="Per-request mean TPOT percentiles", ylabel="Milliseconds / output token (log scale)", yscale="log")
    axes[0,2].legend()
    axes[1,1].bar(x-.17, [r["stage1_hit_rate"]*100 for r in primary], .32, label="LiteHit immediate write", color=colors[0])
    axes[1,1].bar(x+.17, [r["stage2_hit_rate"]*100 for r in primary], .32, label="HiSim delayed write", color=colors[1])
    axes[1,1].set(title="Cache hits under two timing policies", ylabel="Input-token hit rate (%)", ylim=(0, 100))
    axes[1,1].legend(fontsize=9)
    for offset, metric, color, name in [(-.2,"observed_peak_active",colors[1],"Active"),
                                      (0,"observed_peak_queue",colors[2],"Queued"),
                                      (.2,"observed_peak_outstanding",colors[0],"Outstanding")]:
        axes[1,2].bar(x+offset, [r[metric] for r in primary], .19, label=name, color=color)
    axes[1,2].set(title="Observed concurrency peaks", ylabel="Requests")
    axes[1,2].legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Cache payload capacity (GiB)")
    fig.suptitle(f"HiSim + AIC CPU simulation | fixed arrivals, admission limit {admission_limit} | long-context extrapolation; no SLA claim", fontsize=12)
    save_figure(fig, "serving_metrics")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, r in enumerate(primary):
        directory = out / "stage2" / r["scenario"]
        events = [e for e in rows(directory / "queue.jsonl") if "active" in e]
        times = [e["time_s"] for e in events]
        name = labels[i] + (" capacity" if r["capacity_gib"] < 0 else " GiB")
        axes[0,0].step(times, [e["queued"] for e in events], where="post", color=colors[i], label=name)
        axes[0,1].step(times, [e["g1_occupied_bytes"] / GIB for e in events], where="post", color=colors[i])
    axes[0,0].set(title="Pending admission queue", xlabel="Simulation time (s)", ylabel="Requests")
    axes[0,0].legend(fontsize=9)
    axes[0,1].axhline(primary[0]["g1_budget_gib_all_ranks"], color="#444444", linestyle="--", label="Physical KV/state budget")
    axes[0,1].set(title="G1 occupancy, including checkpoint snapshots", xlabel="Simulation time (s)", ylabel="GiB across all 8 ranks")
    axes[0,1].legend(fontsize=9)
    axes[1,0].bar(x-.17, [r["storage_peak_gib"] for r in primary], .32, label="Resident cache payload", color=colors[0])
    axes[1,0].bar(x+.17, [r["host_staging_peak_gib"] for r in primary], .32, label="Reserved host transfer buffers", color=colors[2])
    axes[1,0].set(title="Cache and host staging are separate budgets", ylabel="Peak GiB", xlabel="Cache payload capacity (GiB)")
    axes[1,0].set_xticks(x, labels)
    axes[1,0].legend(fontsize=9)
    for offset, field, color, name in [(-.2,"compute_utilization",colors[0],"Compute"),
                                     (0,"storage_read_utilization",colors[1],"Storage read"),
                                     (.2,"storage_write_utilization",colors[2],"Storage write")]:
        axes[1,1].bar(x+offset, [r[field]*100 for r in primary], .19, color=color, label=name)
    axes[1,1].set(title="Modeled resource busy time", ylabel="Utilization (%)", ylim=(0,115), xlabel="Cache payload capacity (GiB)")
    axes[1,1].set_xticks(x, labels)
    axes[1,1].legend(fontsize=9)
    fig.suptitle("Resource accounting | compute over serving duration; I/O over final storage-drain duration", fontsize=12)
    save_figure(fig, "resources")

    admission = [r for r in summary if r["scenario"].startswith("elbow_concurrency_") or
                 (r["scenario"].startswith("capacity_") and r["capacity_gib"] == knee)]
    admission.sort(key=lambda r:r["configured_admission_limit"])
    if len(admission) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, field, title, unit, factor in [
            (axes[0], "p99_ttft_ms", "P99 TTFT", "Seconds", .001),
            (axes[1], "p99_tpot_ms", "P99 per-request TPOT", "Milliseconds/token", 1),
            (axes[2], "output_tokens_per_s", "Output throughput", "Output tokens/second", 1)]:
            ax.plot([r["configured_admission_limit"] for r in admission], [r[field]*factor for r in admission], "o-", color=colors[2])
            ax.set(title=title, ylabel=unit, xlabel="Configured admission limit", xticks=[r["configured_admission_limit"] for r in admission])
        fig.suptitle(f"{knee:g} GiB payload | same arrivals and physical G1 budget | observed, not sustainable-concurrency proof", fontsize=11)
        save_figure(fig, "concurrency")

    # Self-contained interactive HTML: Plotly JavaScript is embedded, no CDN.
    interactive = make_subplots(rows=2, cols=2, subplot_titles=("Capacity / hit rate", "TTFT percentiles (s)", "TPOT percentiles (ms)", "Output throughput (tokens/s)"))
    interactive.add_trace(go.Scatter(x=caps,y=rates,mode="lines+markers",name="LiteHit grid",hovertemplate="%{x} GiB<br>%{y:.3f}%<extra></extra>"),row=1,col=1)
    for percentile in (50,99):
        interactive.add_trace(go.Scatter(x=labels,y=[r[f"p{percentile}_ttft_ms"]/1000 for r in primary],name=f"P{percentile} TTFT",mode="lines+markers"),row=1,col=2)
        interactive.add_trace(go.Scatter(x=labels,y=[r[f"p{percentile}_tpot_ms"] for r in primary],name=f"P{percentile} TPOT",mode="lines+markers"),row=2,col=1)
    interactive.add_trace(go.Bar(x=labels,y=[r["output_tokens_per_s"] for r in primary],name="Output tokens/s"),row=2,col=2)
    interactive.update_yaxes(type="log",row=2,col=1)
    interactive.update_layout(height=820,template="plotly_white",title="Kimi-K3 capacity study: modeled service metrics",legend={"orientation":"h","y":-0.1})
    interactive.update_xaxes(title_text="Cache payload capacity (GiB)")

    baseline = next((r for r in primary if r["capacity_gib"] == 0), None)
    elbow = next((r for r in primary if r["capacity_gib"] == knee), None)
    source = manifest["source"]
    origin = min(r["timestamp_ns"] for r in trace)
    arrival_span = (max(r["timestamp_ns"] for r in trace)-origin)/1e9
    markdown = [
        "# Kimi-K3 混合缓存容量评估", "",
        f"LiteHit 单遍生成 Facts、43 容量点投影与 HiSim `bench_serving` 仿真已完成。"
        f"当前样本的几何拐点是 **{knee:g} GiB**，达到无限容量命中率 95%：**{cap95_text}**。"
        + (f"容量曲线在 **{plateau:g} GiB** 达到本样本平台。" if plateau is not None else "本次有限容量网格尚未达到无限容量平台。")
        + "拐点依赖本次 0–1024 GiB 网格与归一化规则。", "",
        (f"在相同固定到达轨迹、准入上限 {admission_limit} 下，{knee:g} GiB 相比 0 GiB 的输出吞吐为 "
         f"**{elbow['output_tokens_per_s']/baseline['output_tokens_per_s']:.2f}×**，"
         f"P99 TTFT 下降 **{(1-elbow['p99_ttft_ms']/baseline['p99_ttft_ms'])*100:.1f}%**。" if baseline and elbow else
         "端到端结果覆盖下表中的指定容量，未运行的容量不推断服务指标。")
        + "这是含长上下文外推的端到端 CPU 仿真，使用显式缓存与调度策略；未执行 Kimi-K3 GPU kernels，不能作为已校准 SLA 结果。", "",
        "## 容量曲线", "", "![Capacity curve](capacity_curve.png)", "",
        f"Facts replay {stage1['replay_wall_s']:.4f} 秒；43 容量查询 {stage1['query_wall_s']:.4f} 秒。"
        f"{stage1['validation']['independent_lru_request_capacity_checks']:,} 个逐请求×容量独立 LRU 对照全部通过。"
        "计时是当前小样本及已启动环境上的 wall time，不是全量 1.85 GB trace 的性能成绩。", "",
        "## 同一负载下的服务指标", "", "![Serving metrics](serving_metrics.png)", "",
        "| 容量 GiB | LiteHit / HiSim 命中率 | TTFT P50 / P99 秒 | TPOT P50 / P99 毫秒 | 输出 token/s | 峰值 active / queued / outstanding |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    html_rows = []
    for r,label in zip(primary,labels):
        values=["∞" if label=="Infinite" else label,
                f"{r['stage1_hit_rate']*100:.2f}% / {r['stage2_hit_rate']*100:.2f}%",
                f"{r['p50_ttft_ms']/1000:.2f} / {r['p99_ttft_ms']/1000:.2f}",
                f"{r['p50_tpot_ms']:.2f} / {r['p99_tpot_ms']:.2f}", f"{r['output_tokens_per_s']:.2f}",
                f"{r['observed_peak_active']} / {r['observed_peak_queue']} / {r['observed_peak_outstanding']}"]
        markdown.append("| " + " | ".join(values) + " |")
        html_rows.append("<tr>"+"".join(f"<td>{html.escape(v)}</td>" for v in values)+"</tr>")
    markdown += ["", "TTFT 从原始到达时刻计到首 token，包含排队、缓存读/H2D 和 prompt 计算。"
                 "TPOT 对每请求先计算 `(最后 token − 首 token)/(output_len−1)`，再取请求间 P50/P99；它与逐 token ITL 分位数不同。"
                 "输出吞吐为全部 output tokens 除以最早到达至最后完成的时长，覆盖整个 TP8 副本，未按 GPU 再乘除。", "",
                 f"{manifest['requests']} 请求在 {arrival_span:.3f} 秒内到达，而各配置服务完成耗时为 "
                 f"{min(r['duration_s'] for r in primary):.1f}–{max(r['duration_s'] for r in primary):.1f} 秒。"
                 f"各容量平均准入等待为 {min(r['mean_queue_ms'] for r in primary)/1000:.3f}–{max(r['mean_queue_ms'] for r in primary)/1000:.3f} 秒。所有请求最终完成也不能证明最大可持续并发或 SLA。", "",
                 "## 两阶段命中差异", "",
                 "LiteHit 在请求序列上立即将全部完整 prompt 对象倒序提交；HiSim 在准入时查询，读取会 touch 并持有 pin，"
                 "新对象要等 prompt 完成、D2H 和存储写完成才可见。请求完成顺序、已读对象保护和有限容量淘汰顺序随调度变化。"
                 "因此有限容量下两阶段命中率不存在普遍的上界关系；无限容量下，延迟可见性使部分本可复用的前缀来不及命中。", "",
                 "| 容量场景 | 命中更多请求 | 命中更少请求 | 相同请求 | 净变化 input tokens |",
                 "|---|---:|---:|---:|---:|"]
    for name, diff in differences.items():
        markdown.append(f"| {name} | {diff['more_hit_requests']} | {diff['fewer_hit_requests']} | {diff['equal_hit_requests']} | {diff['net_delta_tokens']:+,} |")
    markdown += ["", "逐请求差异见 [hit_comparison.csv](../stage2/hit_comparison.csv)，可连接各容量的 `request.jsonl` 与 `storage.jsonl`。"
                 "独立案例审查见 [HiSim 事件审查](../../evidence/hisim_stage2_audit.md)。", "",
                 "## 并发与内存", "", "![Resources](resources.png)", "",
                 "配置准入上限、回放观察到的 active/outstanding 峰值和物理内存上限含义不同。"
                 f"本组主对照使用准入上限 {admission_limit}，各配置实际峰值见表；outstanding 包含队列，不能当成 GPU 同时执行的请求数。", "",
                 "G1 预算按 AIC 权重与 activation/runtime/communication 开销扣除后计算，"
                 f"为每 rank {primary[0]['g1_budget_gib_all_ranks']/8:.3f} GiB，合计 {primary[0]['g1_budget_gib_all_ranks']:.3f} GiB。"
                 "每个准入请求预留完整 input+max_output MLA、两个活跃 KDA 槽，以及每个完整输入 checkpoint 的 KDA snapshot；"
                 "snapshot 在 D2H 完成后释放。命中前缀仍要加载至 G1，跨请求 G1 prefix sharing 关闭。"
                 "这是保留 GPU checkpoint 到 prompt 结束再整组写出的保守策略，不能套用仅含两槽状态的理论并发上界。G1 使用共享字节预算；默认 SGLang 分离的 MLA/state 池还可能各自先耗尽，实际部署需另加对应池约束。", "",
                 "| 容量场景 | G1 峰值 GiB (TP8) | 常驻 cache 峰值 GiB | host staging 预留峰值 GiB | 驱逐对象数 | pin 导致放弃写入对象数 |",
                 "|---|---:|---:|---:|---:|---:|"]
    for r in primary:
        markdown.append(f"| {r['scenario']} | {r['g1_peak_gib_all_ranks']:.2f} | {r['storage_peak_gib']:.2f} | {r['host_staging_peak_gib']:.2f} | {r['storage_evictions']} | {r['read_pin_rejected_bundles']} |")
    markdown += ["", "外部 cache capacity 只计 tensor payload。Host 读写暂存空间单独记录且未设上限，"
                 "按传输入队时预留整个 buffer 计算，峰值是保守预留量。若 cache 和暂存共用主机 RAM，需要另外纳入暂存、索引与 allocator 开销；"
                 "不能将 250 GiB payload 解读成机器只需要 250 GiB RAM。", ""]
    if len(admission)>1:
        markdown += ["![Admission probes](concurrency.png)", "",
                     f"| {knee:g} GiB 准入上限 | 观察 active 峰值 | TTFT P99 秒 | TPOT P99 毫秒 | 输出 token/s |",
                     "|---:|---:|---:|---:|---:|"]
        for r in admission:
            markdown.append(f"| {r['configured_admission_limit']} | {r['observed_peak_active']} | {r['p99_ttft_ms']/1000:.2f} | {r['p99_tpot_ms']:.2f} | {r['output_tokens_per_s']:.2f} |")
        markdown += ["", f"这些点保持 AIC max_batch_size={main_config['predictor']['max_batch_size']}、max_num_tokens={main_config['predictor']['max_num_tokens']} 和物理 G1 预算一致，只改变准入上限。"
                     "没有给定 SLA 和持续稳态负载，因此此表用于并发参数比较，不给出最大可持续并发结论。", ""]
    markdown += ["## 缓存对象与源码实现", "",
                 "Kimi-K3 配置有 24 层 MLA 和 69 层 KDA。使用 attention TP8、BF16 MLA/conv 与 FP32 SSM：", "",
                 "```text", "MLA/rank/token = 24 × (512 + 64) × 2 = 27,648 bytes",
                 "SSM/rank/checkpoint = 69 × (96/8) × 128 × 128 × 4 = 54,263,808 bytes",
                 "conv/rank/checkpoint = 69 × 3 × (4−1) × (96/8) × 128 × 2 = 1,907,712 bytes",
                 f"bundle/TP8 = 8 × ({checkpoint} × 27,648 + 54,263,808 + 1,907,712)",
                 f"           = {layout['bundle_bytes']:,} bytes = {layout['bundle_bytes']/(1<<20):.6f} MiB", "```", "",
                 "每个对象是固定长度 MLA 段加对应终点的全部 KDA 状态，原子写入/淘汰。"
                 "LiteHit offline 仅开放等大小 `linear_step=1`，以 `size_full_linear` 收费；核心 Fenwick、RLE、Facts 格式、projector 均未重构。"
                 "`reserve_last_token=true` 仅裁剪 Facts 命中边界，仍提交全部完整输入对象。"
                 "此 KVCM bundle 策略和默认 SGLang Mamba 双池策略不同，也不假定每个 MLA 前缀都有 KDA 状态。源码依据见 [source_evidence.md](../../evidence/source_evidence.md)。", "",
                 "## 输入、硬件与有效范围", "",
                 (f"输入复用原 `traces.jsonl` 前 {len(source.get('sessions',[]))} 个 session，"
                  f"筛选源时间前 {source.get('source_time_window_s',3600):g} 秒及 input≤{source.get('max_input_tokens',262144)}，"
                  f"实际保留 {manifest['requests']} 请求（含 {source.get('selected_subagent_requests',0)} 个子请求）。"
                  f"局部 prefix hash 用 session 行号扩成全局身份，父子 stream 共用 session namespace；对原生 {layout['native_block_tokens']}-token key "
                  f"每 {checkpoint//layout['native_block_tokens']} 个取完整 checkpoint 终点。" if agentx else
                  f"输入类型：{source['kind']}，共 {manifest['requests']} 请求；生成或选择参数见 trace_manifest.json。"), "",
                 f"共 {manifest['input_tokens']:,} input tokens、{manifest['output_tokens']:,} output tokens。输入/输出长度保留原值。", "",
                 (f"源时间按 {source.get('time_scale',1):g} 倍压缩，并让所选 session 从同一原点开始。"
                  "这是 **AgentX-derived fixed-arrival counterfactual**，不包含官方闭环 DAG 调度、随机起点、primer、一小时 profiling 或模型质量评估，"
                  "也不能证明原 token/hash 在 Kimi tokenizer 下等价。" if agentx else "这是固定到达的长度与前缀身份回放，不是官方 AgentX 分数或 GPU 测量。")
                 + "所有容量复用同一固定到达轨迹；原 trace 的 TTFT/api_time 不参与 Kimi-K3 预测。", "",
                 "AIC 使用 8×B300-SXM，attention TP8、MoE TP1/EP8，backend 0.5.16，FP8 GEMM、MXFP4/MXFP8 MoE、无 speculative。"
                 "纯 ANALYTICAL 路径未实现本次所需 Rust KDA，因此使用 AIC silicon/empirical 数据库并记录实际来源，未伪装成纯解析覆盖。", "",
                 "长上下文边界：MLA prefill 表的 B1/B2 完整长度仅到 32768，B4/B8/B16/B32 上限分别为 16384/8192/4096/2048；"
                 "查询长度包含 past+new，因此减小 chunk 并不会消除长前缀外推。Decode B1/B8/B32 表到 131072。"
                 "TP8 的 12 local MLA heads 由 8/16-head 数据插值。异构 prefill 对 MLA 显式做平方工作量修正，仍是近似；"
                 "`silicon` 来源标签本身不意味着该长度被 GPU 实测覆盖。详见 [长上下文审查](../../evidence/aic_predictor/long_context_audit.md)。", "",
                 f"存储 read/write/H2D/D2H 带宽分别假设为 {'/'.join(format(main_config['storage'][f'{lane}_bandwidth_bytes_per_s']/GIB, 'g') for lane in ('read','write','h2d','d2h'))} GiB/s，均是全 TP8 聚合速率；四条 FIFO 通道独立，不模拟读写共享带宽争用。"
                 "当前是 FCFS 准入、prefill 优先、checkpoint 边界 chunk、独立 decode batch，无 preemption/mixed batch/CUDA graph/compute overlap。"
                 "GPU 本地 snapshot copy 没有单列耗时；实际 checkpoint 生成、异步复制和存储带宽需额外校准。", "",
                 "## 验证、交付与复跑", "",
                 "所有主容量与并发点都通过逐请求 token/shape 守恒、首/末 token 时间、独立 P50/P99 重算、"
                 "batch 和四条 I/O lane 不重叠、存储成员/pin/写可见事件重放，以及有限容量/G1 边界检查。"
                 "这验证实现与声明策略一致，不替代真实 Kimi-K3 serving 的时延校准。", "",
                 "- [可复跑 SOP](/workspace/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_capacity/SOP.md)",
                 "- [Stage1 CSV](../stage1/capacity_curve.csv) / [Stage1 summary](../stage1/summary.json)",
                 "- [Stage2 CSV](../stage2/summary.csv) / [Stage2 summary](../stage2/summary.json)",
                 "- [Trace manifest](../trace_manifest.json) / [Layout](../layout.json) / [Source manifest](../source_manifest.json)",
                 "- 每个 Stage2 子目录包含 `request.jsonl`、`iteration.jsonl`、`storage.jsonl`、`queue.jsonl`、`aic_queries.jsonl`、`validation.json` 与 `provenance.json`。",
                 "- 本目录提供 PNG/SVG/PDF 图表和可离线打开的 `index.html`。", "",
                 f"Trace SHA256: `{manifest['trace_sha256']}`", "",
                 f"Layout SHA256: `{manifest['layout_sha256']}`", "",
                 f"Facts SHA256: `{stage1['facts_sha256']}`", ""]
    audit_path = out.parent / "evidence/hisim_stage2_audit.md"
    if audit_path.exists() and manifest["trace_sha256"] in audit_path.read_text():
        markdown += ["## 独立事件审查补充", "",
                     "50 GiB 的一个请求在 582.605770 秒查询时仍命中 79 bundles；会将它们逐出的另一请求在 585.792351 秒才完成发布。"
                     "这给出有限容量即时写回曲线并非普遍上界的直接事件证据。250 GiB 的 33 条负差异中，29 条首个缺失对象尚未发布，4 条曾发布后逐出；"
                     "无限容量 38 条负差异全部来自发布晚于查询。", "",
                     "四个主容量点约 97% 的平均 TTFT 来自准入队列。尾部 TPOT 反映 prefill 优先造成的 token 间停顿："
                     "一个 50.243153 秒的 token 间隔由 50.207486 秒 prefill batches 与 0.035667 秒 decode 构成。"
                     "逐请求 ID、账本行号、时间与 G1 生命周期核对见 [独立审查](../../evidence/hisim_stage2_audit.md)。", ""]
    else:
        markdown = [line.replace("独立案例审查见 [HiSim 事件审查](../../evidence/hisim_stage2_audit.md)。", "") for line in markdown]
    optional_refs = {
        "[source_evidence.md](../../evidence/source_evidence.md)": out.parent / "evidence/source_evidence.md",
        "[长上下文审查](../../evidence/aic_predictor/long_context_audit.md)": out.parent / "evidence/aic_predictor/long_context_audit.md",
    }
    for label, path in optional_refs.items():
        if not path.exists():
            markdown = [line.replace(label, "随附 SOP 的源码与覆盖说明") for line in markdown]
    (report / "REPORT.md").write_text("\n".join(markdown))
    page = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Kimi-K3 容量与服务指标</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1280px;margin:auto;padding:24px;color:#172536;background:#f7f9fc}h1,h2{line-height:1.25}table{border-collapse:collapse;width:100%;background:white}th,td{border-bottom:1px solid #d7dfe8;text-align:right;padding:9px}img{max-width:100%;background:white}a{color:#176981}.note{padding:18px;background:#e8eef6;border-left:4px solid #5072a7}.fig{margin:30px 0}</style>",
            f"<h1>Kimi-K3 混合缓存容量评估</h1><p>几何拐点 <b>{knee:g} GiB</b>；达到无限命中率 95%：<b>{cap95_text}</b>。"
            f"{manifest['requests']} 条 {workload_label} 固定到达请求，全部容量与并发点完成并通过事件核对。</p>",
            "<p class='note'>HiSim + AIC CPU 仿真，包含长上下文外推。未执行 Kimi-K3 GPU kernels。时间指标包含准入排队，表中并发峰值不能代表最大可持续并发或 SLA 达标；外部容量单位为全部 TP8 rank 的 tensor payload GiB。</p>",
            "<p><a href='REPORT.md'>完整报告与边界说明</a> · <a href='../stage2/summary.csv'>指标 CSV</a> · <a href='../stage1/capacity_curve.csv'>容量曲线 CSV</a></p>",
            "<table><thead><tr><th>容量 GiB</th><th>LiteHit / HiSim 命中率</th><th>TTFT P50/P99 秒</th><th>TPOT P50/P99 毫秒</th><th>输出 token/s</th><th>峰值 active/queued/outstanding</th></tr></thead><tbody>",
            *html_rows, "</tbody></table>",
            "<h2>交互查看</h2><p>悬停查看数值，点击图例开关曲线；图内数据和 JavaScript 均已嵌入，可离线打开。</p>",
            interactive.to_html(full_html=False, include_plotlyjs=True),
            "<h2>容量、服务指标与资源</h2>"]
    for name in ["capacity_curve","serving_metrics","resources"] + (["concurrency"] if len(admission)>1 else []):
        page.append(f"<div class='fig'><img src='{name}.png' alt='{name}'><p><a href='{name}.svg'>SVG</a> · <a href='{name}.pdf'>PDF</a></p></div>")
    page.append(f"<p>实现：{checkpoint:,}-token MLA 段与终点 KDA SSM/conv 组成原子 bundle。有限容量 LRU、写完成可见、读 pin、完整 G1 预留。Host staging 单独计费且没有容量约束；带宽为场景假设。</p></html>")
    (report / "index.html").write_text("\n".join(page))
    save(report / "report_manifest.json", {"files": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(report.iterdir()) if p.is_file() and p.name != "report_manifest.json"],
        "stage2_all_validated": all(r["passed"] for r in summary), "official_agentx_benchmark": False, "gpu_measured": False})

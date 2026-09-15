"""Behavioral tests of timing/capacity semantics; no synthetic AIC accuracy claims."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hisim.simulation.hybrid_checkpoint import (
    AtomicCheckpointStorage,
    HybridCheckpointSimulation,
    HybridLayout,
    TraceRequest,
    read_trace,
)

LAYOUT = HybridLayout(
    checkpoint_tokens=4,
    bundle_bytes=12,
    mla_bytes_per_token=2,
    kda_checkpoint_bytes=4,
    attention_tp_size=1,
)


class ShapePredictor:
    def predict_batch(self, requests):
        if requests[0][2]:
            return 0.002 + 0.001 * sum(q for q, _, _ in requests)
        return 0.003 + 0.0001 * len(requests)


def config(capacity=None, concurrency=1, g1=10**6):
    return {
        "storage": {
            "capacity_bytes": capacity,
            **{
                f"{lane}_bandwidth_bytes_per_s": 10**6
                for lane in ("read", "write", "h2d", "d2h")
            },
        },
        "scheduler": {
            "max_concurrency": concurrency,
            "max_prefill_tokens": 8,
            "chunked_prefill_tokens": 4,
            "g1_capacity_bytes": g1,
            "g1_page_tokens": 1,
            "active_kda_buffers": 2,
        },
    }


def request(rid, group="a", arrival=0.0, output=3, instance="one", input_len=9):
    return TraceRequest(
        str(rid),
        instance,
        arrival,
        input_len,
        output,
        [f"{group}-{i}" for i in range(input_len // 4)],
    )


def simulate(cfg, requests):
    sim = HybridCheckpointSimulation(cfg, LAYOUT, ShapePredictor())
    return sim, sim.run(requests)


def test_finite_capacity_eviction_changes_actual_hits_and_ttft():
    rows = [request(1, "a", 0), request(2, "b", 0.2), request(3, "a", 0.4)]
    finite, finite_metrics = simulate(config(capacity=24), deepcopy(rows))
    infinite, infinite_metrics = simulate(config(), deepcopy(rows))
    assert finite_metrics["storage_evictions"] == 4
    assert finite.requests[2].cached_prefix_tokens == 0
    assert infinite.requests[2].cached_prefix_tokens == 8
    assert (
        infinite.request_records()[2]["ttft_s"] < finite.request_records()[2]["ttft_s"]
    )
    assert finite_metrics["storage_peak_occupancy_bytes"] <= 24
    assert infinite_metrics["storage_read_bytes"] == 24
    assert (
        infinite_metrics["prompt_compute_tokens"]
        < finite_metrics["prompt_compute_tokens"]
    )


def test_snapshot_is_contiguous_and_commit_order_keeps_chain_head_recent():
    storage = AtomicCheckpointStorage(12, 24)
    keys = [("one", str(i)) for i in range(3)]
    storage.commit(keys, set(keys))
    assert list(storage.entries) == [keys[1], keys[0]]
    lease = storage.query_and_pin(keys)
    assert lease == keys[:2]
    # Inspection did not opportunistically skip a missing key or revive it.
    assert storage.query_and_pin([keys[2], keys[0]]) == []
    storage.release(lease)


def test_read_pins_prevent_eviction_and_release_allows_progress():
    storage = AtomicCheckpointStorage(12, 12)
    a, b = ("one", "a"), ("one", "b")
    storage.commit([a], {a})
    lease = storage.query_and_pin([a])
    result = storage.commit([b], {b})
    assert result["dropped_pinned"] == [b]
    assert a in storage.entries and b not in storage.entries
    assert storage.peak_pinned_bytes == 12
    storage.release(lease)
    result = storage.commit([b], {b})
    assert result["evicted"] == [a]
    assert b in storage.entries


def test_mla_only_payload_cannot_be_published_or_hit():
    storage = AtomicCheckpointStorage(12, None)
    key = ("one", "a")
    with pytest.raises(ValueError, match="MLA, KDA SSM"):
        storage.commit([key], {key}, components=frozenset(("mla",)))
    assert storage.query_and_pin([key]) == []
    assert storage.occupancy_bytes == 0


def test_instance_ids_never_share_checkpoint_keys():
    sim, _ = simulate(
        config(), [request(1, arrival=0), request(2, arrival=0.2, instance="two")]
    )
    assert [r.cached_prefix_tokens for r in sim.requests] == [0, 0]


def test_unfinished_writes_are_invisible_after_prefill_but_before_commit():
    cfg = config(concurrency=2)
    cfg["storage"]["write_bandwidth_bytes_per_s"] = 12
    sim, _ = simulate(
        cfg, [request(1, arrival=0), request(2, arrival=0.1), request(3, arrival=2.2)]
    )
    assert sim.requests[0].prefill_done_s < 0.1
    assert [r.cached_prefix_tokens for r in sim.requests] == [0, 0, 8]
    first_write = next(e for e in sim.storage_events if e["event"] == "write_submit")
    assert first_write["visible_at_s"] > sim.requests[1].admission_s
    first_visible = next(e for e in sim.storage_events if e["event"] == "write_visible")
    assert first_visible["time_s"] == first_write["visible_at_s"]
    assert all(e["host_staging_bytes"] >= 0 for e in sim.storage_events)


def test_g1_full_prefix_and_checkpoint_buffers_force_admission_to_wait():
    # Every request reserves 12*2 MLA bytes plus (2 active+2 checkpoint)*4 state
    # bytes.  A fully cached prompt must still have the same admission budget.
    cfg = config(concurrency=4, g1=40)
    sim, metrics = simulate(cfg, [request(1), request(2), request(3)])
    assert metrics["observed_peak_active"] == 1
    assert metrics["g1_peak_occupied_bytes"] == 40
    assert metrics["g1_admission_wait_events"] > 0
    assert sim.requests[1].admission_s >= sim.requests[0].finish_s
    assert all(req.g1_reserved_bytes == 40 for req in sim.requests)
    assert sim.requests[1].cached_prefix_tokens == 8
    with pytest.raises(ValueError, match="exceeds G1 budget"):
        simulate(config(g1=39), [request(1)])


def test_gpu_source_buffers_live_until_d2h_even_after_inference_finishes():
    cfg = config(concurrency=4, g1=36)
    cfg["storage"]["d2h_bandwidth_bytes_per_s"] = 12
    sim, metrics = simulate(
        cfg, [request(1, output=1), request(2, group="b", output=1)]
    )
    assert sim.requests[0].finish_s < 1.0
    assert sim.requests[1].admission_s >= 2.0
    assert metrics["g1_admission_wait_events"] > 0
    assert metrics["g1_peak_occupied_bytes"] == 36
    assert sim.g1_occupied == 0


def test_queued_work_changes_p99_and_reports_observed_not_sustainable_concurrency():
    burst, burst_metrics = simulate(
        config(capacity=0), [request(i, group=str(i)) for i in range(5)]
    )
    spaced, spaced_metrics = simulate(
        config(capacity=0),
        [request(i, group=str(i), arrival=i * 0.2) for i in range(5)],
    )
    assert burst_metrics["p99_ttft_ms"] > spaced_metrics["p99_ttft_ms"]
    assert burst_metrics["observed_peak_outstanding"] == 5
    assert burst_metrics["observed_peak_active"] == 1
    assert burst_metrics["configured_max_concurrency"] == 1
    assert burst_metrics["maximum_sustainable_concurrency_measured"] is False
    assert burst.requests[-1].admission_s > burst.requests[0].finish_s


def test_prompt_output_and_batch_token_conservation_with_arrivals_and_hits():
    rows = [request(i, group=str(i % 2), arrival=i * 0.02, output=4) for i in range(8)]
    sim, metrics = simulate(config(concurrency=3), rows)
    prefill = sum(b["query_tokens"] for b in sim.batches if b["phase"] == "prefill")
    decode = sum(b["query_tokens"] for b in sim.batches if b["phase"] == "decode")
    assert prefill + metrics["cached_prompt_tokens"] == metrics["total_input"]
    assert decode + metrics["completed"] == metrics["total_output"]
    assert metrics["completed"] == len(rows)
    for req in sim.requests:
        assert len(req.token_times_s) == req.output_len
        assert all(a < b for a, b in zip(req.token_times_s, req.token_times_s[1:]))
        assert (
            req.arrival_s
            <= req.admission_s
            <= req.read_ready_s
            <= req.prefill_done_s
            <= req.finish_s
        )
    assert all(
        b["end_s"] <= sim.batches[i + 1]["start_s"]
        for i, b in enumerate(sim.batches[:-1])
    )
    assert all(b["query_tokens"] <= 8 for b in sim.batches if b["phase"] == "prefill")
    for event in sim.storage_events:
        if event["event"] == "checkpoint_materialized":
            assert event["endpoint_tokens"] % 4 == 0
            assert set(event["components"]) == {"mla", "kda_ssm", "kda_conv"}


def test_exact_prompt_checkpoint_requires_recomputing_last_checkpoint_for_logits():
    sim, _ = simulate(
        config(), [request(1, input_len=8), request(2, input_len=8, arrival=0.2)]
    )
    assert sim.requests[1].cached_prefix_tokens == 4
    assert sim.requests[1].computed_prompt_tokens == 4


def test_trace_accepts_existing_bundle_identity_and_validates_contract(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "trace_id": str(i),
            "instance_id": "one",
            "timestamp_ns": 10**9 + i * 10**8,
            "input_len": 9,
            "output_len": 3,
            "bundle_keys": ["opaque-prefix1", 12345],
            "keys": [100, 200, 300, 400],
        }
        for i in range(2)
    ]
    path.write_text("\n".join(json.dumps(row) for row in reversed(rows)))
    result = read_trace(path, LAYOUT)
    assert result[0].arrival_s == 0
    assert result[1].arrival_s == 0.1
    assert result[0].bundle_keys == ["opaque-prefix1", "12345"]
    rows[1]["bundle_keys"] = []
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="every complete input checkpoint"):
        read_trace(path, LAYOUT)


def test_commit_preserves_a_reused_head_when_new_suffix_exceeds_capacity():
    storage = AtomicCheckpointStorage(12, 12)
    a, b = ("one", "a"), ("one", "b")
    storage.commit([a], {a})
    result = storage.commit([a, b], {b})
    assert list(storage.entries) == [a]
    assert result["dropped_capacity"] == [b]
    assert result["evicted"] == []


def test_explicit_cache_limit_changes_query_without_dropping_committed_endpoints():
    a, b = request(1), request(2, arrival=0.2)
    b.max_cache_hit_tokens = 4
    sim, _ = simulate(config(), [a, b])
    assert sim.requests[1].cached_prefix_tokens == 4
    assert len(sim.storage.entries) == 2


def test_storage_event_log_replays_capacity_residency_and_read_pins():
    sim, _ = simulate(
        config(capacity=12, concurrency=2),
        [request(1), request(2, arrival=0.2), request(3, group="b", arrival=0.4)],
    )
    resident = {}
    for event in sim.storage_events:
        if event["event"] == "read_query":
            for encoded in event["keys"]:
                key = tuple(encoded)
                assert key in resident
                resident[key] += 1
        elif event["event"] == "read_ready":
            for encoded in event["keys"]:
                resident[tuple(encoded)] -= 1
        for operation in event.get("operations", []):
            key = tuple(operation["key"])
            if operation["op"] == "evict":
                assert resident.pop(key) == 0
            elif operation["op"] == "insert":
                assert key not in resident
                resident[key] = 0
            else:
                assert key in resident
        assert event["occupancy_bytes"] == len(resident) * 12
        assert event["pinned_bytes"] == sum(pins > 0 for pins in resident.values()) * 12
        assert event["read_pin_references"] == sum(resident.values())
        assert event["occupancy_bytes"] <= 12
        assert event["host_staging_bytes"] >= 0
    assert all(pins == 0 for pins in resident.values())

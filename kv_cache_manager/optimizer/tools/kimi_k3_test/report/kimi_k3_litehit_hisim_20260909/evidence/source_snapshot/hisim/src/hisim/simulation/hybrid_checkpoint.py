"""Controlled CPU HiSim adapter for atomic hybrid-attention checkpoints.

This is a queue/storage simulation with batch durations supplied by AIC.  It does
not execute SGLang, model kernels, a tokenizer, or KVCM RPCs.  The explicit policy
is a fixed-token bundle containing MLA KV plus KDA SSM and convolution state.
All byte counts and link bandwidths cover the whole attention TP replica.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol, Sequence

from hisim.simulation.types import RequestStats

BUNDLE_COMPONENTS = frozenset(("mla", "kda_ssm", "kda_conv"))
SCHEMA = "hisim.hybrid_checkpoint.v1"


@dataclass(frozen=True)
class HybridLayout:
    checkpoint_tokens: int
    bundle_bytes: int
    mla_bytes_per_token: int
    kda_checkpoint_bytes: int
    attention_tp_size: int

    @classmethod
    def from_dict(cls, raw: dict) -> "HybridLayout":
        tp = int(raw["attention_tp_size"])
        value = cls(
            checkpoint_tokens=int(raw["checkpoint_tokens"]),
            bundle_bytes=int(raw["bundle_bytes"]),
            mla_bytes_per_token=int(raw["mla_bytes_per_token_per_rank"]) * tp,
            kda_checkpoint_bytes=int(raw["kda_state_bytes_per_checkpoint_per_rank"])
            * tp,
            attention_tp_size=tp,
        )
        if min(asdict(value).values()) <= 0:
            raise ValueError("Every hybrid layout size must be positive")
        if value.bundle_bytes != (
            value.checkpoint_tokens * value.mla_bytes_per_token
            + value.kda_checkpoint_bytes
        ):
            raise ValueError(
                "bundle_bytes must equal all-rank MLA chunk plus KDA checkpoint bytes"
            )
        if raw.get("storage_scope", "all_tp_rank_payloads") != "all_tp_rank_payloads":
            raise ValueError(
                "Only all_tp_rank_payloads storage accounting is supported"
            )
        if raw.get("g1_cross_request_cache", False):
            raise ValueError("This adapter requires g1_cross_request_cache=false")
        return value


@dataclass
class Bundle:
    components: frozenset[str] = BUNDLE_COMPONENTS
    pins: int = 0


class AtomicCheckpointStorage:
    """Equal-charge LRU with immutable prefix queries and protected read leases."""

    def __init__(self, bundle_bytes: int, capacity_bytes: int | None):
        if bundle_bytes <= 0 or (capacity_bytes is not None and capacity_bytes < 0):
            raise ValueError("Invalid bundle size or storage capacity")
        self.bundle_bytes = bundle_bytes
        self.capacity_bytes = capacity_bytes
        self.max_bundles = (
            None if capacity_bytes is None else capacity_bytes // bundle_bytes
        )
        self.entries: OrderedDict[tuple[str, str], Bundle] = OrderedDict()
        self.evictions = 0
        self.pin_rejections = 0
        self.peak_occupancy_bytes = 0
        self.peak_pinned_bytes = 0
        self.peak_read_pin_references = 0

    @property
    def occupancy_bytes(self) -> int:
        return len(self.entries) * self.bundle_bytes

    @property
    def pinned_bytes(self) -> int:
        return (
            sum(entry.pins > 0 for entry in self.entries.values()) * self.bundle_bytes
        )

    @property
    def read_pin_references(self) -> int:
        return sum(entry.pins for entry in self.entries.values())

    def query_and_pin(self, keys: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        # Inspect the complete hit prefix before any LRU touch or pin mutation.
        hit = []
        for key in keys:
            entry = self.entries.get(key)
            if entry is None or entry.components != BUNDLE_COMPONENTS:
                break
            hit.append(key)
        for key in reversed(hit):
            self.entries[key].pins += 1
            self.entries.move_to_end(key)
        self.peak_pinned_bytes = max(self.peak_pinned_bytes, self.pinned_bytes)
        self.peak_read_pin_references = max(
            self.peak_read_pin_references, self.read_pin_references
        )
        return hit

    def release(self, keys: Sequence[tuple[str, str]]) -> None:
        for key in keys:
            entry = self.entries[key]
            if entry.pins <= 0:
                raise RuntimeError("Read lease released twice")
            entry.pins -= 1

    def commit(
        self,
        keys: Sequence[tuple[str, str]],
        materialized: set[tuple[str, str]],
        components: frozenset[str] = BUNDLE_COMPONENTS,
    ) -> dict:
        if components != BUNDLE_COMPONENTS:
            raise ValueError(
                "An atomic bundle must include MLA, KDA SSM, and KDA convolution state"
            )
        result: dict[str, list] = {
            "inserted": [],
            "touched": [],
            "evicted": [],
            "dropped_pinned": [],
            "dropped_capacity": [],
            "reused_prefix_evicted": [],
            "operations": [],
        }
        # Plan this completion against one residency snapshot.  Touching an
        # existing chain head must not lose its payload to an insertion from
        # the same tail-to-head commit before the head is visited.  All bundle
        # publications in this write group share one completion timestamp.
        target = OrderedDict(self.entries)
        for key in reversed(keys):
            if key in target:
                target.move_to_end(key)
            elif key in materialized:
                target[key] = Bundle()
            else:
                result["reused_prefix_evicted"].append(key)
        unpinned_survivors = (
            set(target)
            if self.max_bundles is None
            else set(list(target)[-self.max_bundles :]) if self.max_bundles else set()
        )
        if self.max_bundles is not None:
            while len(target) > self.max_bundles:
                victim = next(
                    (key for key, entry in target.items() if entry.pins == 0), None
                )
                if victim is None:
                    raise RuntimeError(
                        "Pinned storage payloads already exceed capacity"
                    )
                del target[victim]
                if victim not in self.entries:
                    if victim in unpinned_survivors:
                        self.pin_rejections += 1
                        result["dropped_pinned"].append(victim)
                    else:
                        result["dropped_capacity"].append(victim)
        # Evict the planned old victims first, then admit only surviving new
        # bundles.  Occupied payload bytes never transiently exceed capacity.
        for key in list(self.entries):
            if key not in target:
                del self.entries[key]
                self.evictions += 1
                result["evicted"].append(key)
                result["operations"].append({"op": "evict", "key": key})
        for key in reversed(keys):
            if key not in target:
                continue
            if key in self.entries:
                self.entries.move_to_end(key)
                result["touched"].append(key)
                result["operations"].append({"op": "touch", "key": key})
            else:
                self.entries[key] = target[key]
                result["inserted"].append(key)
                result["operations"].append({"op": "insert", "key": key})
        self.peak_occupancy_bytes = max(self.peak_occupancy_bytes, self.occupancy_bytes)
        return result


@dataclass
class TraceRequest:
    trace_id: str
    instance_id: str
    arrival_s: float
    input_len: int
    output_len: int
    bundle_keys: list[str]
    max_cache_hit_tokens: int | None = None
    admission_s: float | None = None
    read_ready_s: float | None = None
    prefill_start_s: float | None = None
    prefill_done_s: float | None = None
    finish_s: float | None = None
    cached_prefix_tokens: int = 0
    computed_prompt_tokens: int = 0
    token_times_s: list[float] = field(default_factory=list)
    stage: str = "not_arrived"
    g1_mla_bytes: int = 0
    g1_active_kda_bytes: int = 0
    g1_checkpoint_staging_bytes: int = 0
    g1_reserved_bytes: int = 0
    g1_base_released: bool = False
    g1_staging_released: bool = False
    read_lease: list[tuple[str, str]] = field(default_factory=list)

    @property
    def keys(self) -> list[tuple[str, str]]:
        return [(self.instance_id, key) for key in self.bundle_keys]

    @property
    def processed_prompt_tokens(self) -> int:
        return self.cached_prefix_tokens + self.computed_prompt_tokens


class BatchPredictor(Protocol):
    """Durations are seconds; query shapes are per request, never hit ratios."""

    def predict_batch(self, requests: list[tuple[int, int, bool]]) -> float: ...


def read_trace(
    path: Path, layout: HybridLayout, arrival_scale: float = 1.0
) -> list[TraceRequest]:
    if not math.isfinite(arrival_scale) or arrival_scale <= 0:
        raise ValueError("arrival_scale must be finite and positive")
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Trace must contain at least one request")
    rows.sort(key=lambda row: int(row["timestamp_ns"]))
    origin_ns = int(rows[0]["timestamp_ns"])
    requests = []
    seen = set()
    for row in rows:
        rid = str(row["trace_id"])
        if rid in seen:
            raise ValueError(f"Duplicate trace_id: {rid}")
        seen.add(rid)
        input_len, output_len = int(row["input_len"]), int(row["output_len"])
        if input_len < 1 or output_len < 1:
            raise ValueError(f"{rid}: positive input_len and output_len are required")
        keys = [str(key) for key in row["bundle_keys"]]
        if len(keys) != input_len // layout.checkpoint_tokens:
            raise ValueError(
                f"{rid}: bundle_keys must contain every complete input checkpoint"
            )
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"{rid}: duplicate bundle keys violate the chained-prefix contract"
            )
        max_cache_hit_tokens = int(
            row.get(
                "max_cache_hit_tokens",
                (input_len - 1) // layout.checkpoint_tokens * layout.checkpoint_tokens,
            )
        )
        if (
            max_cache_hit_tokens < 0
            or max_cache_hit_tokens >= input_len
            or max_cache_hit_tokens % layout.checkpoint_tokens
        ):
            raise ValueError(
                f"{rid}: max_cache_hit_tokens must be an aligned prefix shorter than input_len"
            )
        requests.append(
            TraceRequest(
                trace_id=rid,
                instance_id=str(row["instance_id"]),
                arrival_s=(int(row["timestamp_ns"]) - origin_ns) / 1e9 * arrival_scale,
                input_len=input_len,
                output_len=output_len,
                bundle_keys=keys,
                max_cache_hit_tokens=max_cache_hit_tokens,
            )
        )
    return requests


class HybridCheckpointSimulation:
    """Nonpreemptive FCFS, prefill-first batching and serialized duplex I/O lanes.

    G1 reserves the complete prompt plus maximum output MLA tokens, two (by
    default) active KDA buffers, and every possible input endpoint checkpoint
    until its D2H copy completes.  The last reservation is conservative but
    prevents an implicit unbounded GPU checkpoint buffer.  We do not share G1
    prefixes across requests.  Host staging has no admission limit; its peak is
    reported so this assumption can be assessed.
    """

    def __init__(self, config: dict, layout: HybridLayout, predictor: BatchPredictor):
        self.config, self.layout, self.predictor = config, layout, predictor
        storage = config["storage"]
        capacity = storage.get("capacity_bytes")
        self.storage = AtomicCheckpointStorage(
            layout.bundle_bytes, None if capacity is None else int(capacity)
        )
        self.bandwidths = {}
        for key in ("read", "write", "h2d", "d2h"):
            rate = float(storage[f"{key}_bandwidth_bytes_per_s"])
            if math.isnan(rate) or rate <= 0:
                raise ValueError(f"{key} bandwidth must be positive bytes/s")
            self.bandwidths[key] = rate
        scheduler = config["scheduler"]
        self.max_concurrency = int(scheduler["max_concurrency"])
        self.max_prefill_tokens = int(scheduler.get("max_prefill_tokens", 8192))
        self.chunked_prefill_tokens = int(scheduler.get("chunked_prefill_tokens", 1024))
        self.g1_capacity_bytes = int(scheduler["g1_capacity_bytes"])
        self.g1_page_tokens = int(scheduler.get("g1_page_tokens", 64))
        self.active_kda_buffers = int(scheduler.get("active_kda_buffers", 2))
        if (
            min(
                self.max_concurrency,
                self.max_prefill_tokens,
                self.chunked_prefill_tokens,
                self.g1_capacity_bytes,
                self.g1_page_tokens,
                self.active_kda_buffers,
            )
            <= 0
        ):
            raise ValueError("Scheduler limits must be positive")
        self.events: list[tuple] = []
        self.event_sequence = 0
        self.now = 0.0
        self.compute_busy = False
        self.requests: list[TraceRequest] = []
        self.waiting: deque[TraceRequest] = deque()
        self.active: OrderedDict[str, TraceRequest] = OrderedDict()
        self.arrived = 0
        self.completed = 0
        self.g1_occupied = 0
        self.g1_peak = 0
        self.host_staging = 0
        self.host_staging_peak = 0
        self.host_read_staging = 0
        self.host_write_staging = 0
        self.lane_free = dict.fromkeys(self.bandwidths, 0.0)
        self.batches: list[dict] = []
        self.predictor_queries: list[dict] = []
        self._predictor_query_cache: dict[tuple, tuple[float, int]] = {}
        self.storage_events: list[dict] = []
        self.queue_events: list[dict] = []
        self.read_bytes = 0
        self.write_bytes = 0
        self.peak_active = 0
        self.peak_outstanding = 0
        self.peak_queue = 0
        self.g1_blocked_admissions = 0
        self._last_queue_snapshot: tuple | None = None
        self._last_g1_blocked: tuple | None = None

    def _push(self, at: float, kind: str, payload: Any) -> None:
        if at < self.now - 1e-12 or not math.isfinite(at):
            raise RuntimeError(f"Invalid event time {at} for {kind} at {self.now}")
        priority = {
            "d2h_complete": 0,
            "read_ready": 0,
            "write_complete": 1,
            "compute_done": 2,
            "arrival": 3,
        }[kind]
        self.event_sequence += 1
        heapq.heappush(self.events, (at, priority, self.event_sequence, kind, payload))

    def _storage_event(self, kind: str, req: TraceRequest, **values: Any) -> None:
        self.storage_events.append(
            {
                "event": kind,
                "time_s": self.now,
                "trace_id": req.trace_id,
                "instance_id": req.instance_id,
                "occupancy_bytes": self.storage.occupancy_bytes,
                "pinned_bytes": self.storage.pinned_bytes,
                "read_pin_references": self.storage.read_pin_references,
                "host_staging_bytes": self.host_staging,
                "host_read_staging_bytes": self.host_read_staging,
                "host_write_staging_bytes": self.host_write_staging,
                **values,
            }
        )

    def _observe_queue(self) -> None:
        self.peak_active = max(self.peak_active, len(self.active))
        self.peak_outstanding = max(
            self.peak_outstanding, self.arrived - self.completed
        )
        self.peak_queue = max(self.peak_queue, len(self.waiting))
        self.g1_peak = max(self.g1_peak, self.g1_occupied)
        snapshot = (
            len(self.active),
            len(self.waiting),
            self.arrived - self.completed,
            self.g1_occupied,
        )
        if snapshot != self._last_queue_snapshot:
            self.queue_events.append(
                {
                    "time_s": self.now,
                    "active": snapshot[0],
                    "queued": snapshot[1],
                    "outstanding": snapshot[2],
                    "g1_occupied_bytes": snapshot[3],
                }
            )
            self._last_queue_snapshot = snapshot
        if self.g1_occupied < 0 or self.g1_occupied > self.g1_capacity_bytes:
            raise RuntimeError("G1 accounting exceeded its configured capacity")

    def _prepare_reservation(self, req: TraceRequest) -> None:
        tokens = (
            math.ceil((req.input_len + req.output_len) / self.g1_page_tokens)
            * self.g1_page_tokens
        )
        req.g1_mla_bytes = tokens * self.layout.mla_bytes_per_token
        req.g1_active_kda_bytes = (
            self.active_kda_buffers * self.layout.kda_checkpoint_bytes
        )
        req.g1_checkpoint_staging_bytes = (
            len(req.bundle_keys) * self.layout.kda_checkpoint_bytes
            if self.storage.max_bundles != 0
            else 0
        )
        req.g1_reserved_bytes = (
            req.g1_mla_bytes + req.g1_active_kda_bytes + req.g1_checkpoint_staging_bytes
        )
        if req.g1_reserved_bytes > self.g1_capacity_bytes:
            raise ValueError(
                f"{req.trace_id}: full MLA plus active/staging KDA reservation "
                f"({req.g1_reserved_bytes} bytes) exceeds G1 budget ({self.g1_capacity_bytes})"
            )

    def _transfer(self, lane: str, ready: float, size: int) -> tuple[float, float]:
        start = max(ready, self.lane_free[lane])
        finish = start + size / self.bandwidths[lane]
        self.lane_free[lane] = finish
        return start, finish

    def _admit(self) -> None:
        while self.waiting and len(self.active) < self.max_concurrency:
            req = self.waiting[0]
            if self.g1_occupied + req.g1_reserved_bytes > self.g1_capacity_bytes:
                state = (req.trace_id, self.g1_occupied)
                if state != self._last_g1_blocked:
                    self.g1_blocked_admissions += 1
                    self.queue_events.append(
                        {
                            "event": "g1_admission_wait",
                            "time_s": self.now,
                            "trace_id": req.trace_id,
                            "required_bytes": req.g1_reserved_bytes,
                            "available_bytes": self.g1_capacity_bytes
                            - self.g1_occupied,
                        }
                    )
                    self._last_g1_blocked = state
                break
            self.waiting.popleft()
            req.admission_s = self.now
            self.active[req.trace_id] = req
            self.g1_occupied += req.g1_reserved_bytes
            # A checkpoint is KV/state, not final logits.  Always compute at
            # least one prompt token; exact-boundary prompts use the prior one.
            eligible = (req.input_len - 1) // self.layout.checkpoint_tokens
            if req.max_cache_hit_tokens is not None:
                eligible = min(
                    eligible, req.max_cache_hit_tokens // self.layout.checkpoint_tokens
                )
            req.read_lease = self.storage.query_and_pin(req.keys[:eligible])
            req.cached_prefix_tokens = (
                len(req.read_lease) * self.layout.checkpoint_tokens
            )
            self._storage_event(
                "read_query",
                req,
                eligible_bundles=eligible,
                hit_bundles=len(req.read_lease),
                keys=req.read_lease,
                cached_prefix_tokens=req.cached_prefix_tokens,
            )
            if req.read_lease:
                size = len(req.read_lease) * self.layout.bundle_bytes
                read_start, read_end = self._transfer("read", self.now, size)
                h2d_start, ready = self._transfer("h2d", read_end, size)
                self.read_bytes += size
                # Reserve complete transfer buffers when queued, including
                # their wait time.  The reported host peak is conservative.
                self.host_read_staging += size
                self.host_staging += size
                self.host_staging_peak = max(self.host_staging_peak, self.host_staging)
                req.stage = "reading"
                self._storage_event(
                    "read_submit",
                    req,
                    bytes=size,
                    query_time_s=self.now,
                    read_start_s=read_start,
                    read_end_s=read_end,
                    h2d_start_s=h2d_start,
                    read_ready_s=ready,
                    storage_queue_s=read_start - self.now,
                    h2d_queue_s=h2d_start - read_end,
                )
                self._push(ready, "read_ready", req)
            else:
                req.stage = "prefill"
                req.read_ready_s = self.now
            self._observe_queue()

    def _release_staging(self, req: TraceRequest) -> None:
        if not req.g1_staging_released:
            self.g1_occupied -= req.g1_checkpoint_staging_bytes
            req.g1_staging_released = True

    def _release_finished_base(self, req: TraceRequest) -> None:
        # MLA source views also participate in the D2H bundle.  Keep their
        # reservation until both inference and the copy have completed.
        if (
            req.finish_s is not None
            and req.g1_staging_released
            and not req.g1_base_released
        ):
            self.g1_occupied -= req.g1_mla_bytes + req.g1_active_kda_bytes
            req.g1_base_released = True

    def _submit_write(self, req: TraceRequest) -> None:
        materialized = set(
            req.keys[req.cached_prefix_tokens // self.layout.checkpoint_tokens :]
        )
        if self.storage.max_bundles == 0:
            self._storage_event("write_disabled", req, bytes=0)
            self._release_staging(req)
            return
        if not materialized:
            details = self.storage.commit(req.keys, materialized)
            self._storage_event("write_touch", req, bytes=0, **details)
            self._release_staging(req)
            return
        # All computed endpoint snapshots are retained until this D2H.  A
        # concurrently submitted duplicate write still pays for its transfer.
        size = len(materialized) * self.layout.bundle_bytes
        d2h_start, d2h_end = self._transfer("d2h", self.now, size)
        write_start, visible = self._transfer("write", d2h_end, size)
        self.write_bytes += size
        self.host_write_staging += size
        self.host_staging += size
        self.host_staging_peak = max(self.host_staging_peak, self.host_staging)
        self._storage_event(
            "write_submit",
            req,
            bytes=size,
            materialized_keys=sorted(materialized),
            d2h_start_s=d2h_start,
            d2h_end_s=d2h_end,
            write_start_s=write_start,
            visible_at_s=visible,
            d2h_queue_s=d2h_start - self.now,
            storage_queue_s=write_start - d2h_end,
            commit_order="tail_to_head",
        )
        self._push(d2h_end, "d2h_complete", (req, size))
        self._push(visible, "write_complete", (req, materialized, size))

    def _finish_request(self, req: TraceRequest) -> None:
        req.finish_s = self.now
        req.stage = "complete"
        del self.active[req.trace_id]
        self.completed += 1
        self._release_finished_base(req)
        self._observe_queue()

    def _start_batch(self) -> None:
        if self.compute_busy:
            return
        ready = [req for req in self.active.values() if req.stage == "prefill"]
        phase = "prefill" if ready else "decode"
        if phase == "prefill":
            chosen, lengths, prefixes = [], [], []
            budget = self.max_prefill_tokens
            for req in ready:
                # End batches at checkpoint boundaries so the declared KDA
                # endpoint snapshot is produced by an actual forward step.
                to_checkpoint = self.layout.checkpoint_tokens - (
                    req.processed_prompt_tokens % self.layout.checkpoint_tokens
                )
                extend = min(
                    req.input_len - req.processed_prompt_tokens,
                    self.chunked_prefill_tokens,
                    budget,
                    to_checkpoint,
                )
                if extend <= 0:
                    break
                chosen.append(req)
                lengths.append(extend)
                prefixes.append(req.processed_prompt_tokens)
                budget -= extend
        else:
            chosen = [req for req in self.active.values() if req.stage == "decode"]
            if not chosen:
                return
            lengths = [1] * len(chosen)
            # Before consuming the last emitted token, existing KV contains
            # prompt + all earlier emitted tokens.  seq_len includes this Q.
            prefixes = [req.input_len + len(req.token_times_s) - 1 for req in chosen]
        query = tuple(
            (extend, prefix, phase == "prefill")
            for extend, prefix in zip(lengths, prefixes)
        )
        if query not in self._predictor_query_cache:
            if hasattr(self.predictor, "query_batch"):
                details = self.predictor.query_batch(query)
                duration = float(details["latency_s"])
            else:
                duration = float(self.predictor.predict_batch(list(query)))
                details = {"latency_s": duration, "requests": query}
            query_id = len(self.predictor_queries)
            self.predictor_queries.append({"query_id": query_id, **details})
            self._predictor_query_cache[query] = (duration, query_id)
        duration, query_id = self._predictor_query_cache[query]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"AIC returned invalid {phase} batch duration: {duration}")
        for req in chosen:
            if phase == "prefill" and req.prefill_start_s is None:
                req.prefill_start_s = self.now
        batch = {
            "batch_id": len(self.batches),
            "phase": phase,
            "aic_query_id": query_id,
            "start_s": self.now,
            "end_s": self.now + duration,
            "duration_s": duration,
            "trace_ids": [req.trace_id for req in chosen],
            "batch_size": len(chosen),
            "extend_lengths": lengths,
            "prefix_lengths": prefixes,
            "sequence_lengths": [
                prefix + extend for prefix, extend in zip(prefixes, lengths)
            ],
            "query_tokens": sum(lengths),
            "g1_occupied_bytes": self.g1_occupied,
        }
        self.batches.append(batch)
        self.compute_busy = True
        self._push(batch["end_s"], "compute_done", (batch, chosen))

    def _event(self, kind: str, payload: Any) -> None:
        if kind == "arrival":
            req = payload
            req.stage = "queued"
            self.waiting.append(req)
            self.arrived += 1
        elif kind == "read_ready":
            req = payload
            self.storage.release(req.read_lease)
            size = len(req.read_lease) * self.layout.bundle_bytes
            self.host_read_staging -= size
            self.host_staging -= size
            req.read_ready_s = self.now
            req.stage = "prefill"
            self._storage_event(
                "read_ready",
                req,
                released_pins=len(req.read_lease),
                keys=req.read_lease,
            )
            req.read_lease = []
        elif kind == "d2h_complete":
            req, size = payload
            self._release_staging(req)
            self._release_finished_base(req)
            self._storage_event("d2h_complete", req, bytes=size)
        elif kind == "write_complete":
            req, materialized, size = payload
            # D2H completion has higher priority if both lanes finish together.
            self.host_staging -= size
            self.host_write_staging -= size
            if self.host_staging < 0 or self.host_write_staging < 0:
                raise RuntimeError("Write completed before its host payload was ready")
            details = self.storage.commit(req.keys, materialized)
            self._storage_event("write_visible", req, bytes=size, **details)
        elif kind == "compute_done":
            batch, chosen = payload
            self.compute_busy = False
            for req, extend in zip(chosen, batch["extend_lengths"]):
                if batch["phase"] == "prefill":
                    req.computed_prompt_tokens += extend
                    if (
                        self.storage.max_bundles != 0
                        and req.processed_prompt_tokens % self.layout.checkpoint_tokens
                        == 0
                    ):
                        endpoint = (
                            req.processed_prompt_tokens // self.layout.checkpoint_tokens
                        )
                        self._storage_event(
                            "checkpoint_materialized",
                            req,
                            endpoint_tokens=req.processed_prompt_tokens,
                            key=req.keys[endpoint - 1],
                            components=sorted(BUNDLE_COMPONENTS),
                            kda_snapshot_bytes=self.layout.kda_checkpoint_bytes,
                            bundle_bytes=self.layout.bundle_bytes,
                        )
                    if req.processed_prompt_tokens != req.input_len:
                        continue
                    req.prefill_done_s = self.now
                    req.token_times_s.append(self.now)
                    self._submit_write(req)
                    req.stage = "decode"
                else:
                    req.token_times_s.append(self.now)
                if len(req.token_times_s) == req.output_len:
                    self._finish_request(req)
        else:
            raise RuntimeError(f"Unknown event {kind}")
        self._observe_queue()

    def run(self, requests: list[TraceRequest]) -> dict:
        if self.requests:
            raise RuntimeError(
                "Create a new simulation for every capacity/configuration"
            )
        if not requests:
            raise ValueError("Simulation requires at least one request")
        self.requests = requests
        for req in requests:
            self._prepare_reservation(req)
            self._push(req.arrival_s, "arrival", req)
        start_wall = time.perf_counter()
        while self.events:
            self.now = self.events[0][0]
            # Resolve simultaneous completion/arrival events before selecting a
            # new batch; newly scheduled zero-duration I/O is resolved too.
            while True:
                while self.events and self.events[0][0] <= self.now:
                    _, _, _, kind, payload = heapq.heappop(self.events)
                    self._event(kind, payload)
                self._admit()
                if not self.events or self.events[0][0] > self.now:
                    break
            self._start_batch()
        if self.completed != len(requests) or self.g1_occupied or self.host_staging:
            raise RuntimeError(
                "Simulation ended with incomplete requests or live buffers"
            )
        if self.storage.read_pin_references:
            raise RuntimeError("Simulation ended with active read leases")
        self.wall_seconds = time.perf_counter() - start_wall
        return self.metrics()

    def metrics(self) -> dict:
        # Import lazily so the entrypoint can select the supported AIC source
        # before the legacy HiSim utility imports its AIC module.
        from hisim.simulation.utils import calc_metrics

        stats = []
        for req in self.requests:
            if len(req.token_times_s) != req.output_len:
                raise RuntimeError("Output token conservation failed")
            if req.cached_prefix_tokens + req.computed_prompt_tokens != req.input_len:
                raise RuntimeError("Prompt token conservation failed")
            token_gaps = [req.token_times_s[0] - req.arrival_s]
            token_gaps.extend(
                b - a for a, b in zip(req.token_times_s, req.token_times_s[1:])
            )
            stats.append(
                RequestStats(
                    rid=req.trace_id,
                    last_event_time=req.finish_s,
                    input_length=req.input_len,
                    output_length=req.output_len,
                    final_reused_tokens=req.cached_prefix_tokens,
                    prefetch_complete_tokens=req.cached_prefix_tokens,
                    queue_start=req.arrival_s,
                    queue_end=req.admission_s,
                    created_time=req.arrival_s,
                    gen_token_latencies=token_gaps,
                )
            )
        result = calc_metrics(stats)
        result.update(
            {
                "schema": SCHEMA,
                "measurement_kind": "controlled_hisim_aic_cpu_simulation",
                "p50_ttft_ms": result["median_ttft_ms"],
                "p50_tpot_ms": result["median_tpot_ms"],
                "configured_max_concurrency": self.max_concurrency,
                "observed_peak_active": self.peak_active,
                "observed_peak_outstanding": self.peak_outstanding,
                "observed_peak_queue": self.peak_queue,
                "maximum_sustainable_concurrency_measured": False,
                "g1_capacity_bytes": self.g1_capacity_bytes,
                "g1_peak_occupied_bytes": self.g1_peak,
                "g1_admission_wait_events": self.g1_blocked_admissions,
                "g1_cross_request_cache": False,
                "storage_capacity_bytes": self.storage.capacity_bytes,
                "storage_effective_capacity_bytes": (
                    None
                    if self.storage.max_bundles is None
                    else self.storage.max_bundles * self.layout.bundle_bytes
                ),
                "storage_peak_occupancy_bytes": self.storage.peak_occupancy_bytes,
                "storage_final_occupancy_bytes": self.storage.occupancy_bytes,
                "storage_peak_pinned_bytes": self.storage.peak_pinned_bytes,
                "storage_peak_read_pin_references": self.storage.peak_read_pin_references,
                "storage_evictions": self.storage.evictions,
                "storage_pin_rejected_bundles": self.storage.pin_rejections,
                "storage_read_bytes": self.read_bytes,
                "storage_write_bytes": self.write_bytes,
                "host_staging_peak_bytes": self.host_staging_peak,
                "storage_drain_complete_s": self.now,
                "batch_count": len(self.batches),
                "unique_aic_query_shapes": len(self.predictor_queries),
                "prompt_compute_tokens": sum(
                    r.computed_prompt_tokens for r in self.requests
                ),
                "cached_prompt_tokens": sum(
                    r.cached_prefix_tokens for r in self.requests
                ),
                "decode_compute_tokens": sum(r.output_len - 1 for r in self.requests),
                "time_cost": self.wall_seconds,
            }
        )
        return result

    def request_records(self) -> list[dict]:
        records = []
        for req in self.requests:
            row = asdict(req)
            row.pop("read_lease")
            row["ttft_s"] = req.token_times_s[0] - req.arrival_s
            row["tpot_s"] = (
                (req.token_times_s[-1] - req.token_times_s[0]) / (req.output_len - 1)
                if req.output_len > 1
                else None
            )
            row["queue_s"] = req.admission_s - req.arrival_s
            records.append(row)
        return records

    def write_outputs(self, directory: Path, metrics: dict, provenance: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in (
            ("request.jsonl", self.request_records()),
            ("iteration.jsonl", self.batches),
            ("storage.jsonl", self.storage_events),
            ("queue.jsonl", self.queue_events),
            ("aic_queries.jsonl", self.predictor_queries),
        ):
            with (directory / name).open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        for name, value in (
            ("metrics.json", metrics),
            ("provenance.json", provenance),
            ("resolved_config.json", self.config),
        ):
            (directory / name).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_config(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())

    def resolve(value: str) -> Path:
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (config_path.parent / path).resolve()
        )

    layout_path = resolve(config["layout_path"])
    trace_path = resolve(config["trace_path"])
    output_path = resolve(config["output_dir"])
    layout_raw = json.loads(layout_path.read_text())
    layout = HybridLayout.from_dict(layout_raw)
    # The predictor module is implemented independently so that generic HiSim
    # model parsing and the original HTTP/SGLang serving path are unaffected.
    from hisim.time_predictor.kimi_k3 import KimiK3TimePredictor

    predictor = KimiK3TimePredictor(**config.get("predictor", {}))
    memory = predictor.memory_profile
    if memory["rank_count"] != layout.attention_tp_size:
        raise ValueError("Predictor and layout attention TP sizes differ")
    if (
        memory["mla_kv_bytes_per_token_per_rank"] * layout.attention_tp_size
        != layout.mla_bytes_per_token
    ):
        raise ValueError("Predictor and layout MLA token bytes differ")
    if (
        memory["kda_state_bytes_per_slot_per_rank"] * layout.attention_tp_size
        != layout.kda_checkpoint_bytes
    ):
        raise ValueError("Predictor and layout KDA checkpoint bytes differ")
    scheduler = config["scheduler"]
    if scheduler["max_concurrency"] > predictor.max_batch_size:
        raise ValueError("Scheduler concurrency exceeds predictor max_batch_size")
    if scheduler.get("max_prefill_tokens", 8192) > predictor.max_num_tokens:
        raise ValueError("Scheduler prefill budget exceeds predictor max_num_tokens")
    physical_g1_budget = memory["kv_budget_bytes"] * layout.attention_tp_size
    scheduler.setdefault("g1_capacity_bytes", physical_g1_budget)
    if scheduler["g1_capacity_bytes"] > physical_g1_budget:
        raise ValueError(
            "G1 capacity exceeds AIC's available physical KV/state memory budget"
        )
    simulation = HybridCheckpointSimulation(config, layout, predictor)
    requests = read_trace(trace_path, layout, float(config.get("arrival_scale", 1.0)))
    metrics = simulation.run(requests)
    provenance = {
        "schema": SCHEMA,
        "measurement_kind": "controlled_hisim_aic_cpu_simulation",
        "native_sglang_scheduler": False,
        "gpu_kernels_executed": False,
        "physical_g1_budget_bytes_all_ranks": physical_g1_budget,
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "layout_path": str(layout_path),
        "layout_sha256": _sha256(layout_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256(Path(__file__)),
        "scheduler": "FCFS admission; prefill-first; chunk/token/checkpoint-boundary limits; decode-only otherwise; no preemption",
        "storage": "atomic MLA+KDA-SSM+KDA-convolution bundles; snapshot query and read pins; write-completion visibility; tail-to-head commit",
        "storage_accounting_scope": "all_tp_rank_payloads",
        "query_time": "admission, after FCFS/max_concurrency/G1 gates",
        "predictor_query_reuse": "exact per-request shape memoization; each batch references aic_query_id; predictor provenance query/source counters cover unique queried shapes",
        "simulation_clock_origin": "minimum trace timestamp; arrivals in seconds scaled by arrival_scale",
        "write_source": "all newly computed complete input checkpoints; reused-prefix touches only if still resident",
        "g1_policy": "full input plus maximum output MLA; active KDA buffers and conservative per-input-checkpoint KDA staging reservation; no cross-request sharing",
        "io_policy": "independent FIFO storage read/write and H2D/D2H lanes; aggregate bytes/s; read pins retained through H2D",
        "host_staging_policy": "unbounded host staging; complete read/write transfer buffers reserved at submission until completion; conservative reserved peak recorded; no cross-request host cache",
        "limitations": [
            "Controlled queue adapter, not native SGLang Kimi-K3 or GPU measurements.",
            "AIC predictions require calibration before production SLA or sustainable-capacity claims.",
            "No CUDA graphs, scheduler overlap, mixed prefill/decode, speculative decoding, or request preemption.",
            "Checkpoint materialization is a declared simulation policy, not a claim about this SGLang version's default Mamba cache.",
            "GPU-local checkpoint snapshot copies are not timed separately; endpoint state exists at forward completion and D2H/storage transfers are timed.",
            "Trace-driven active/outstanding peaks are observations, not maximum sustainable concurrency.",
        ],
        "layout": layout_raw,
    }
    if hasattr(predictor, "provenance"):
        value = predictor.provenance
        provenance["predictor"] = value() if callable(value) else value
    simulation.write_outputs(output_path, metrics, provenance)
    return metrics


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-checkpoint-config",
        type=Path,
        required=True,
        help="JSON config for the controlled HiSim+AIC hybrid checkpoint simulation",
    )
    args = parser.parse_args(argv)
    result = run_config(args.hybrid_checkpoint_config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    cli()

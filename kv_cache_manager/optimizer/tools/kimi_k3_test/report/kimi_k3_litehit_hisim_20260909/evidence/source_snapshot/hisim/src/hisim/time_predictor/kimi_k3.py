"""Kimi-K3 CPU latency queries from an explicitly selected AIC checkout.

The separate ``aiconfigurator_core`` namespace allows HiSim's legacy
``aiconfigurator`` installation to remain loaded and unchanged. No package is
installed and neither sys.path nor the environment is rewritten.

AIC's static batch API has homogeneous sequence lengths. This adapter uses
ceil(mean(new_tokens)) and ceil(mean(past_kv_length)), retains the real request
shapes in its audit result, and explicitly scales only MLA context attention
by its shape-derived work ratio. AIC's MLA operator ignores the generic dense
attention imbalance knob. This correction is recorded beside the unmodified
native latency; it is shape aggregation, not a fit to throughput. Mixed prefill/decode batches are deliberately rejected; the
CPU serving adapter schedules the two phases separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import json
import math
from numbers import Integral
from pathlib import Path
import subprocess
import sys
import threading
from typing import Iterable, Mapping


DEFAULT_AIC_ROOT = Path("/workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync")
MODEL_PATH = "moonshotai/Kimi-K3"
_IMPORT_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_core(source_root: Path):
    """Load the unique core package without replacing legacy aiconfigurator."""
    package = source_root / "aic-core/src/aiconfigurator_core"
    init = package / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"AIC core package missing: {init}")
    with _IMPORT_LOCK:
        loaded = sys.modules.get("aiconfigurator_core")
        if loaded is not None:
            if Path(loaded.__file__).resolve() != init.resolve():
                raise RuntimeError("aiconfigurator_core is already loaded from another checkout")
            return loaded
        spec = importlib.util.spec_from_file_location(
            "aiconfigurator_core", init, submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        return module


@dataclass(frozen=True)
class KimiK3Request:
    new_tokens: int
    past_kv_length: int
    is_prefill: bool

    def __post_init__(self):
        for name in ("new_tokens", "past_kv_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.new_tokens < 1 or self.past_kv_length < 0:
            raise ValueError("new_tokens must be positive and past_kv_length nonnegative")
        if not isinstance(self.is_prefill, bool):
            raise ValueError("is_prefill must be an explicit boolean")
        if not self.is_prefill and self.new_tokens != 1:
            raise ValueError("no-speculative decode requires exactly one new token")


def _requests(values: Iterable) -> tuple[KimiK3Request, ...]:
    result = []
    for value in values:
        if isinstance(value, KimiK3Request):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(KimiK3Request(**value))
        else:
            result.append(KimiK3Request(*value))
    return tuple(result)


def _runtime(requests: tuple[KimiK3Request, ...]) -> tuple[dict, dict]:
    phases = {request.is_prefill for request in requests}
    if len(phases) != 1:
        raise ValueError("issue separate prefill and decode batches; mixed phases are unsupported")
    batch_size = len(requests)
    new_tokens = math.ceil(sum(r.new_tokens for r in requests) / batch_size)
    past = math.ceil(sum(r.past_kv_length for r in requests) / batch_size)
    prefill = requests[0].is_prefill
    exact_work = sum(r.new_tokens * (r.new_tokens + 1 + 2 * r.past_kv_length) for r in requests)
    aggregate_work = batch_size * new_tokens * (new_tokens + 1 + 2 * past)
    # AIC MLA's prefix model uses (q + p)^2 - p^2, omitting the discrete
    # causal diagonal term q. Match that model exactly for the applied scale;
    # retain exact discrete pair counts separately so the approximation is
    # auditable even for one-token prefills.
    exact_aic_work = sum(r.new_tokens * (r.new_tokens + 2 * r.past_kv_length) for r in requests)
    aggregate_aic_work = batch_size * new_tokens * (new_tokens + 2 * past)
    attention_scale = exact_aic_work / aggregate_aic_work if prefill else 1.0
    runtime = dict(
        batch_size=batch_size,
        isl=past + new_tokens if prefill else past,
        prefix=past if prefill else 0,
        osl=1 if prefill else 2,
        mode="static_ctx" if prefill else "static_gen",
        stride=1,
        seq_imbalance_correction_scale=1.0,
        gen_seq_imbalance_correction_scale=1.0,
    )
    aggregation = dict(
        method="aic_homogeneous_batch_ceil_mean",
        homogeneous=len({(r.new_tokens, r.past_kv_length) for r in requests}) == 1,
        actual_new_tokens=sum(r.new_tokens for r in requests),
        modeled_new_tokens=batch_size * new_tokens,
        actual_past_tokens=sum(r.past_kv_length for r in requests),
        modeled_past_tokens=batch_size * past,
        causal_attention_work_ratio=exact_work / aggregate_work if prefill else 1.0,
        exact_causal_attention_pairs=exact_work // 2 if prefill else None,
        aggregate_causal_attention_pairs=aggregate_work // 2 if prefill else None,
        exact_aic_attention_work=exact_aic_work if prefill else None,
        aggregate_aic_attention_work=aggregate_aic_work if prefill else None,
        mla_attention_work_scale=attention_scale,
        mla_correction_application="explicit adapter scaling; upstream MLA ignores the generic imbalance knob",
        decode_attention_sequence_length=None if prefill else past + 1,
    )
    return runtime, aggregation


class KimiK3TimePredictor:
    """Native Kimi-K3 operation model, returning simulated forward seconds.

    ``memory_profile`` reports one rank. MLA KV is TP replicated; KDA state is
    TP sharded. It exposes ONE state slot and leaves the live/checkpoint slot
    policy to the scheduler, instead of charging AIC's flat five-slot proxy.
    """

    def __init__(
        self,
        aic_source_root: str | Path = DEFAULT_AIC_ROOT,
        *,
        system: str = "b300_sxm",
        tp_size: int = 8,
        moe_ep_size: int = 8,
        backend_version: str = "0.5.16",
        max_batch_size: int = 32,
        max_num_tokens: int = 8192,
        memory_fraction: float = 0.90,
        audit_jsonl_path: str | Path | None = None,
    ):
        if max_batch_size < 1 or max_num_tokens < 1:
            raise ValueError("batch/token limits must be positive")
        self.aic_source_root = Path(aic_source_root).resolve()
        _load_core(self.aic_source_root)
        sdk = "aiconfigurator_core.sdk."
        engine_module = importlib.import_module(sdk + "engine")
        builders = importlib.import_module(sdk + "config_builders")
        models = importlib.import_module(sdk + "models")
        perf_database = importlib.import_module(sdk + "perf_database")
        memory = importlib.import_module(sdk + "memory")
        self.max_batch_size = int(max_batch_size)
        self.max_num_tokens = int(max_num_tokens)
        self.system = system
        self.backend_version = backend_version
        self.engine_config = dict(
            tp_size=tp_size,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode="fp8",
            moe_quant_mode="w4a8_mxfp4_mxfp8",
            kvcache_quant_mode="bfloat16",
            fmha_quant_mode="bfloat16",
            nextn=0,
        )
        self.systems_path = self.aic_source_root / "aic-core/src/aiconfigurator_core/systems"
        self._engine = engine_module.EngineHandle.compile(
            MODEL_PATH, system, "sglang", backend_version=backend_version,
            systems_path=str(self.systems_path), database_mode="SILICON", **self.engine_config,
        )
        model_config = builders.build_model_config(**{k: v for k, v in self.engine_config.items() if k != "nextn"})
        builders.apply_nextn(model_config, 0)
        self._model = models.get_model(MODEL_PATH, model_config, "sglang")
        cfg = self._model.extra_params
        self.max_context_length = int(self._model._context_length)
        if (self._model._num_layers, cfg.layer_types.count("linear_attention"), cfg.layer_types.count("full_attention")) != (93, 69, 24):
            raise RuntimeError("selected AIC checkout does not contain the expected 93-layer Kimi-K3 model")
        if cfg.kda_num_heads % tp_size:
            raise ValueError("KDA head count must be divisible by tensor parallel size")
        self._database = perf_database.get_database_view(
            system, "sglang", backend_version, systems_paths=str(self.systems_path), database_mode="SILICON",
        )
        native_memory = memory.estimate_kv_cache(
            MODEL_PATH, system, "sglang", backend_version,
            systems_path=str(self.systems_path), max_num_tokens=self.max_num_tokens,
            max_batch_size=self.max_batch_size, memory_fraction_kind="of_total",
            memory_fraction_value=memory_fraction, allow_naive_fallback=False,
            **self.engine_config,
        )
        heads = cfg.kda_num_heads // tp_size
        layers = cfg.layer_types.count("linear_attention")
        ssm_bytes = layers * heads * cfg.kda_head_dim**2 * 4
        conv_bytes = layers * (cfg.kda_conv_kernel - 1) * 3 * heads * cfg.kda_head_dim * 2
        self.memory_profile = dict(
            source="AIC native weights and reservations; explicit physical hybrid state geometry",
            gpu_capacity_bytes=native_memory["total_gpu_capacity_bytes"],
            memory_fraction_of_total=memory_fraction,
            **native_memory["memory_breakdown"],
            kv_budget_bytes=native_memory["total_kv_size_bytes"],
            mla_kv_bytes_per_token_per_rank=int(self._model.get_kvcache_elements_per_token() * 2),
            kda_ssm_bytes_per_slot_per_rank=ssm_bytes,
            kda_conv_bytes_per_slot_per_rank=conv_bytes,
            kda_state_bytes_per_slot_per_rank=ssm_bytes + conv_bytes,
            aic_native_state_slots_per_request=self._model.KDA_STATE_SLOTS_PER_REQUEST,
            scheduler_state_slots_per_request="explicit caller input; not inferred from AIC five-slot proxy",
            rank_count=tp_size,
            max_num_tokens=self.max_num_tokens,
            max_batch_size=self.max_batch_size,
            max_context_length=self.max_context_length,
        )
        self.audit_jsonl_path = Path(audit_jsonl_path) if audit_jsonl_path else None
        if self.audit_jsonl_path:
            self.audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._query_cache = {}
        self._query_count = 0
        self._source_latency_ms = Counter()
        self._provenance = None

    def query_batch(self, requests: Iterable) -> dict:
        """Return latency and the exact standard-AIC inputs/per-op outputs."""
        requests = _requests(requests)
        if not requests:
            return dict(latency_s=0.0, latency_ms=0.0, requests=[], per_op=[], source_latency_ms={})
        if len(requests) > self.max_batch_size:
            raise ValueError("batch exceeds the configured admission limit")
        if sum(r.new_tokens for r in requests) > self.max_num_tokens:
            raise ValueError("batch exceeds the activation-memory token budget")
        if any(r.new_tokens + r.past_kv_length > self.max_context_length for r in requests):
            raise ValueError("request exceeds Kimi-K3 model context length")
        runtime, aggregation = _runtime(requests)
        key = (tuple(runtime.items()), aggregation["mla_attention_work_scale"])
        if key not in self._query_cache:
            phases = self._engine._run_static_per_op_with_metadata(**runtime)
            entries = [entry for phase in phases for entry in phase]
            per_op = []
            for name, latency, energy, source, fallback in entries:
                if not math.isfinite(latency) or latency < 0:
                    raise RuntimeError(f"AIC returned invalid latency for {name}: {latency}")
                if any(part in name for part in ("kda_conv1d", "kda_scan", "kda_recurrent")) and source != "silicon":
                    raise RuntimeError(f"KDA measured-table coverage unavailable: {name} returned {source}")
                # Native ContextMlaOp ignores seq_imbalance_correction_scale.
                # Do not average away q/p covariance: apply its own squared-
                # length work model to MLA only, retaining native values.
                scale = aggregation["mla_attention_work_scale"] if name == "context_attention" else 1.0
                per_op.append(dict(name=name, latency_ms=latency * scale, energy_wms=energy * scale,
                                   native_latency_ms=latency, shape_work_scale=scale,
                                   source=source, moe_comm_fallback=fallback))
            self._query_cache[key] = per_op
        per_op = self._query_cache[key]
        source_ms = Counter()
        for op in per_op:
            source_ms[op["source"]] += op["latency_ms"]
        total_ms = sum(op["latency_ms"] for op in per_op)
        self._query_count += 1
        self._source_latency_ms.update(source_ms)
        result = dict(
            latency_s=total_ms / 1000, latency_ms=total_ms,
            requests=[asdict(r) for r in requests], aic_runtime=runtime,
            aggregation=aggregation, per_op=per_op, source_latency_ms=dict(source_ms),
            silicon_latency_fraction=source_ms["silicon"] / total_ms if total_ms else 0,
        )
        if self.audit_jsonl_path:
            with self.audit_jsonl_path.open("a") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
        return result

    def predict_batch(self, requests: Iterable) -> float:
        return self.query_batch(requests)["latency_s"]

    def predict_infer_time(self, batch) -> float:
        """Compatibility with a HiSim ScheduleBatch; explicit phase wins."""
        phase = batch.is_prefill()
        return self.predict_batch([
            (r.input_length, r.past_kv_length, getattr(r, "is_prefill", phase)) for r in batch.reqs
        ])

    def memory_fit(self, *, g1_tokens: int, active_requests: int,
                   active_state_slots_per_request: int = 2, additional_reserved_bytes: int = 0) -> dict:
        """Check physical resident MLA tokens plus actual KDA slots per rank."""
        values = (g1_tokens, active_requests, active_state_slots_per_request, additional_reserved_bytes)
        if any(not isinstance(v, Integral) or v < 0 for v in values):
            raise ValueError("resident tokens, request/slot counts and reservation must be nonnegative integers")
        p = self.memory_profile
        required = (g1_tokens * p["mla_kv_bytes_per_token_per_rank"]
                    + active_requests * active_state_slots_per_request * p["kda_state_bytes_per_slot_per_rank"]
                    + additional_reserved_bytes)
        return dict(fits=required <= p["kv_budget_bytes"], required_kv_bytes_per_rank=required,
                    available_kv_bytes_per_rank=p["kv_budget_bytes"],
                    remaining_kv_bytes_per_rank=p["kv_budget_bytes"] - required)

    def provenance(self) -> dict:
        if self._provenance is None:
            self._database._materialize_source_reports()
            core = self.aic_source_root / "aic-core/src/aiconfigurator_core"
            files = [self.aic_source_root / "pyproject.toml", core / "_aiconfigurator_core.abi3.so",
                     core / "model_configs/moonshotai--Kimi-K3_config.json",
                     core / "sdk/models/kimi_k3.py", core / "sdk/engine.py", core / "sdk/memory.py",
                     core / "sdk/backends/base_backend.py", self.systems_path / f"{self.system}.yaml",
                     self.systems_path / "query_versions.yaml", Path(__file__)]
            for records in self._database.data_provenance.values():
                for record in records:
                    path = Path(record["path"])
                    if path.is_file():
                        files.append(path)
                        files.extend(p for p in (path.parent / "collection_meta.yaml", path.parent / "reuse.yaml") if p.is_file())
            version = subprocess.run(["git", "-C", str(self.aic_source_root), "rev-parse", "HEAD"],
                                     text=True, capture_output=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(self.aic_source_root), "status", "--short"],
                                   text=True, capture_output=True, check=True).stdout.strip()
            self._provenance = dict(
                aic_source_root=str(self.aic_source_root), aic_git_commit=version,
                aic_git_status=dirty, model=MODEL_PATH, system=self.system, backend="sglang",
                backend_version=self._database.version, database_mode="SILICON",
                engine_config=self.engine_config, engine_spec_sha256=hashlib.sha256(self._engine.spec_bytes).hexdigest(),
                memory_profile=self.memory_profile,
                source_sha256={str(path): _sha256(path) for path in sorted(set(files))},
                admitted_table_sources=self._database.data_provenance,
                limitations=[
                    "CPU operation-level simulation; no GPU serving measurements were collected by this workflow.",
                    "AIC SILICON queries include built-in empirical elementwise/normalization/AttnRes operations; silicon tags may include interpolation or extrapolation.",
                    "SGLang 0.5.16 KDA data are required; Rust ANALYTICAL mode does not implement KDA in this checkout.",
                    "Heterogeneous lengths use standard homogeneous AIC queries with ceil means. Only MLA context latency/energy are explicitly corrected by sum(q*(q+2*p))/(batch*q_mean*(q_mean+2*p_mean)); AIC itself ignores the generic MLA imbalance knob. Discrete causal pair counts and native unscaled times are retained. Decode uses ceil mean past plus one current token.",
                    "For the packaged B300/SGLang BF16 MLA tables, prefill full sequence lengths above 32768 at batch 1/2 and decode lengths above 131072 are extrapolated; the native source tag remains silicon. Smaller measured prefill limits apply at larger batches.",
                    "Table-source records describe admitted priority-ordered files, not the selected row for every interpolated query.",
                    "AIC uses its configured power_law expert-routing distribution; synthetic traces do not supply measured expert routing.",
                    "No speculative model, no latency correction fit, no communication/compute overlap added by this adapter.",
                    "Memory is per rank. G1 tokens and KDA active/checkpoint/staging state slots must be budgeted explicitly by the scheduler.",
                ],
            )
        return {**self._provenance, "queries": self._query_count,
                "unique_query_shapes": len(self._query_cache),
                "modeled_source_latency_ms_across_queries": dict(self._source_latency_ms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aic-source-root", type=Path, default=DEFAULT_AIC_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = KimiK3TimePredictor(args.aic_source_root)
    cases = {"prefill_1k": [(1024, 0, True)], "prefill_8k": [(8192, 0, True)],
             "prefill_cached_7k": [(1024, 7168, True)],
             **{f"decode_b{b}": [(1, 8192, False)] * b for b in (1, 8, 32)}}
    results = {name: predictor.query_batch(requests) for name, requests in cases.items()}
    payload = dict(provenance=predictor.provenance(), probes=results,
                   memory_fit_example=predictor.memory_fit(g1_tokens=32 * (16384 + 128), active_requests=32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "latency_ms": {k: v["latency_ms"] for k, v in results.items()},
                      "memory_profile": predictor.memory_profile}, indent=2))


if __name__ == "__main__":
    main()

"""Integration checks for the explicitly bound Kimi-K3 AIC predictor.

These use the repository's CPU native extension and real operator tables;
no GPU, fitted model, HF model download, or AIC package install is needed.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from hisim.time_predictor.kimi_k3 import (
    DEFAULT_AIC_ROOT,
    KimiK3Request,
    KimiK3TimePredictor,
)


@pytest.fixture(scope="module")
def predictor():
    if not (DEFAULT_AIC_ROOT / "aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so").exists():
        pytest.skip("the requested AIC checkout and compiled core are unavailable")
    return KimiK3TimePredictor()


def test_cached_prefill_preserves_prefix_attention_cost(predictor):
    fresh = predictor.query_batch([(8192, 0, True)])
    cached = predictor.query_batch([(1024, 7168, True)])
    short = predictor.query_batch([(1024, 0, True)])
    assert 0 < short["latency_s"] < cached["latency_s"] < fresh["latency_s"]
    assert cached["aic_runtime"]["isl"] == 8192
    assert cached["aic_runtime"]["prefix"] == 7168
    assert cached["latency_ms"] == pytest.approx(sum(o["latency_ms"] for o in cached["per_op"]))
    kda_ops = [o for o in cached["per_op"] if any(k in o["name"] for k in ("conv1d", "kda_scan"))]
    assert kda_ops and all(o["source"] == "silicon" for o in kda_ops)


def test_decode_batch_and_current_token_match_native_aic(predictor):
    queries = [predictor.query_batch([(1, 8192, False)] * batch) for batch in (1, 8, 32)]
    times = [query["latency_s"] for query in queries]
    assert 0 < times[0] < times[1] < times[2] < 32 * times[0]
    assert queries[1]["aggregation"]["decode_attention_sequence_length"] == 8193
    native_ms = predictor._engine.predict_decode_latency(8, 8192, 2)
    assert times[1] * 1000 == pytest.approx(native_ms, rel=1e-12)


def test_heterogeneous_batch_records_exact_inputs_and_causal_work(predictor):
    result = predictor.query_batch([(1024, 3072, True), (2048, 8192, True)])
    runtime = result["aic_runtime"]
    assert runtime["batch_size"] == 2
    assert runtime["isl"] - runtime["prefix"] == 1536
    assert runtime["prefix"] == 5632
    exact = 1024 * (1025 + 2 * 3072) + 2048 * (2049 + 2 * 8192)
    aggregate = 2 * 1536 * (1537 + 2 * 5632)
    assert runtime["seq_imbalance_correction_scale"] == 1.0
    assert result["aggregation"]["causal_attention_work_ratio"] == pytest.approx(exact / aggregate)
    exact_aic = 1024 * (1024 + 2 * 3072) + 2048 * (2048 + 2 * 8192)
    aggregate_aic = 2 * 1536 * (1536 + 2 * 5632)
    mla = next(o for o in result["per_op"] if o["name"] == "context_attention")
    assert mla["latency_ms"] == pytest.approx(mla["native_latency_ms"] * exact_aic / aggregate_aic)
    assert not result["aggregation"]["homogeneous"]
    assert result["aggregation"]["actual_past_tokens"] == 11264
    assert result["latency_s"] > 0


def test_physical_capacity_separates_mla_and_kda_slots(predictor):
    memory = predictor.memory_profile
    assert memory["mla_kv_bytes_per_token_per_rank"] == 24 * (512 + 64) * 2
    one_state = 69 * (12 * 128 * 128 * 4 + 3 * 3 * 12 * 128 * 2)
    assert memory["kda_state_bytes_per_slot_per_rank"] == one_state == 56171520
    assert memory["weights_bytes"] == 189461135360
    active = predictor.memory_fit(g1_tokens=32 * 16512, active_requests=32, active_state_slots_per_request=2)
    assert active["fits"]
    assert active["required_kv_bytes_per_rank"] == 32 * 16512 * 27648 + 32 * 2 * one_state
    assert not predictor.memory_fit(g1_tokens=10**8, active_requests=32)["fits"]
    assert predictor.memory_fit(g1_tokens=0, active_requests=1, active_state_slots_per_request=1)["required_kv_bytes_per_rank"] == one_state


def test_invalid_or_mixed_queries_are_not_silently_reinterpreted(predictor):
    assert predictor.predict_batch([]) == 0
    with pytest.raises(ValueError, match="separate prefill and decode"):
        predictor.predict_batch([(10, 100, True), (1, 100, False)])
    with pytest.raises(ValueError, match="exactly one"):
        KimiK3Request(2, 100, False)
    with pytest.raises(ValueError, match="explicit boolean"):
        KimiK3Request(1, 100, 0)
    with pytest.raises(ValueError, match="activation-memory"):
        predictor.predict_batch([(8193, 0, True)])
    with pytest.raises(ValueError, match="admission"):
        predictor.predict_batch([(1, 10, False)] * 33)


def test_provenance_contains_frozen_model_and_actual_source_hashes(predictor):
    provenance = predictor.provenance()
    assert provenance["model"] == "moonshotai/Kimi-K3"
    assert provenance["backend_version"] == "0.5.16"
    assert provenance["engine_config"]["nextn"] == 0
    assert len(provenance["engine_spec_sha256"]) == 64
    data_files = provenance["source_sha256"]
    assert any("kda/sglang/0.5.16/kda_perf.parquet" in name for name in data_files)
    assert all(len(value) == 64 and Path(name).is_file() for name, value in data_files.items())
    assert json.loads(json.dumps(provenance))["queries"] > 0


def test_loading_core_preserves_existing_aiconfigurator_and_sys_path():
    if not DEFAULT_AIC_ROOT.exists():
        pytest.skip("the requested AIC checkout is unavailable")
    source = """
import sys
import aiconfigurator
from hisim.time_predictor.kimi_k3 import _load_core, DEFAULT_AIC_ROOT
legacy = aiconfigurator
old_path = list(sys.path)
core = _load_core(DEFAULT_AIC_ROOT)
assert sys.modules['aiconfigurator'] is legacy
assert sys.path == old_path
assert str(DEFAULT_AIC_ROOT) in core.__file__
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_extreme_anticorrelated_prefill_does_not_lose_q_prefix_covariance(predictor):
    # One token against an almost 1M prefix and 8191 cold tokens. Averaging
    # first without the MLA correction overstates the squared-length work
    # by more than 100x, although the total GEMM token count is exact.
    result = predictor.query_batch([(1, 1048575, True), (8191, 0, True)])
    exact_pairs = 1048576 + 8191 * 8192 // 2
    exact_aic_work = 2097151 + 8191**2
    aggregate_aic_work = 2 * 4096 * (4096 + 2 * 524288)
    agg = result["aggregation"]
    assert agg["exact_causal_attention_pairs"] == exact_pairs
    assert agg["actual_new_tokens"] == agg["modeled_new_tokens"] == 8192
    assert agg["mla_attention_work_scale"] == pytest.approx(exact_aic_work / aggregate_aic_work)
    mla = next(o for o in result["per_op"] if o["name"] == "context_attention")
    assert mla["latency_ms"] < mla["native_latency_ms"] / 100
    assert all(o["shape_work_scale"] == 1 for o in result["per_op"] if o["name"] != "context_attention")
    with pytest.raises(ValueError, match="model context length"):
        predictor.predict_batch([(1, 1048576, False)])

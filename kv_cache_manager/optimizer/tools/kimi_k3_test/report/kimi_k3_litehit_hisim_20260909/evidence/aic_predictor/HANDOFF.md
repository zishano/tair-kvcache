# AIC Kimi-K3 predictor handoff (implemented and tested)

Owner: kimi_aic_predictor. Implemented file: /workspace/tair-kvcache/hisim/src/hisim/time_predictor/kimi_k3.py.

Public API now available (standalone constructor, no legacy ModelInfo required):
- `KimiK3TimePredictor(aic_source_root="/workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync", system="b300_sxm", tp_size=8, moe_ep_size=8, backend_version="0.5.16", max_batch_size=32, max_num_tokens=8192)`; standalone CPU predictor, no legacy ModelInfo needed.
- `predict_batch(requests)` -> float seconds, requests list of `(new_tokens, past_kv_length, is_prefill)` triples; prefill-only or decode-only batches. Mixed-phase batches are rejected, so the scheduler must issue separate phases. Parent scheduler should issue separate prefill and continuous-decode batches.
- `query_batch(requests)` -> dict with `latency_s`, per-op ms/source and all request/aggregate inputs.
- `memory_profile` -> per-rank weights, reservation, MLA bytes/token and single active KDA state bytes; no hidden five-slot reservation.
- `provenance()` -> source/version/hashes and complete numeric assumptions.

Actual CPU probes at AIC ec215200d05cf6899d0d472b75ceafa6b53c432f:
B300-SXM 8 ranks, TP8/MoE TP1 EP8, gemm fp8, MoE w4a8_mxfp4_mxfp8, BF16 MLA KV, BF16 FMHA, nextn=0, SGLang 0.5.16, AIC SILICON+normal built-in op fallbacks.
8192-new prefill batch1: 363.7890892582ms. 1024-new + 7168-past prefill batch1: 88.5547324402ms.
8192-past decode batch1/8/32: 14.6258809863 / 19.5918088595 / 30.0645890185ms (current static_gen API uses isl8192, osl2; verified exact current-token convention: AIC static_gen with isl=past and osl=2 queries attention length past+1).
Per-op source values are silicon + empirical (elementwise/norm/AttnRes), no SOL in these six 0.5.16 probes.
0.5.14 has KDA SOL fallback; do NOT choose current alias. ANALYTICAL is NOT runnable: `unsupported ANALYTICAL Rust path: operator context_kda_conv1d (Kda) is not yet ported to the Rust KernelSim path`.

Native memory query at max_num_tokens8192/max_batch32/memfraction0.9: physical HBM=288400343040 bytes/rank; weights=189461135360; activations=1740059443; runtime overhead=4509715660; comm overhead=411041792. Native kv budget=63438356480. Its kv_size_per_token_bytes=280885248 incorrectly serves as all-in sequence length-one footprint for hybrid purposes; MUST NOT use as linear per-token MLA bytes.
Model source says 24 MLA*576*2=27648 MLA bytes/token/rank, TP replicated. Single KDA state=69*((96/8)*128*128*4+(4-1)*3*(96/8)*128*2)=56171520 bytes/rank. AIC native memory charges 5 such slots per request; parent scheduler should instead charge its actual active/pinned slots.

Source isolation: only explicit loading of unique `aiconfigurator_core` package from new checkout; never replace venv old `aiconfigurator` facade or pip install.
Raw probe files in this directory.


Delivered smoke artifact: `/tmp/litehit_kimi_k3_aic_probe/predictor_probe.json` (100 KiB) includes full source/data/extension hashes, priority-ordered admitted data source reports, six complete per-op queries and physical memory-fit example.
Command: `/workspace/tair-kvcache/.venv/bin/python -m hisim.time_predictor.kimi_k3 --output /tmp/litehit_kimi_k3_aic_probe/predictor_probe.json`
Tests: `/workspace/tair-kvcache/.venv/bin/python -m pytest -q test/test_kimi_k3_predictor.py` in hisim; 7 passed in 6.68s.

Final API details:
`predictor = KimiK3TimePredictor(max_batch_size=32, max_num_tokens=8192, memory_fraction=0.90)`
`predictor.predict_batch([(1024, 7168, True)]) -> 0.08855473244023185` seconds.
`predictor.query_batch([(1,8192,False)]*8)` returns full JSON-friendly query result.
`predictor.memory_fit(g1_tokens=32*16512, active_requests=32, active_state_slots_per_request=2, additional_reserved_bytes=0)` requires 18,203,738,112 bytes/rank vs 63,438,356,480 available: fits with 45,234,618,368 bytes spare.
`memory_profile` is a dict property; all sizes rank-local. `provenance()` is a method returning a JSON-friendly dict with lifetime query/source counters.

Heterogeneous prefill uses AIC's homogeneous batch interface with ceil means and the built-in sequence-imbalance scalar equal to exact sum of causal attention work divided by aggregated work. For homogeneous batches it is exactly 1. Decode uses ceil mean past length +1 current token. All actual request lengths and aggregates are retained. This is an explicit model approximation, not a fitted throughput multiplier. Mixed phases are not modeled together.

Limitations surfaced in artifacts: per-op silicon tags include interpolation/extrapolation; primitive elementwise/AttnRes costs are empirical. AIC source resolution admits lower-fidelity cross-backend MLA BMM fallback data from TRT-LLM (warning emitted when materializing provenance); source records show admitted candidates, not exact per-query interpolant row. KDA itself must return silicon provenance or the adapter raises. No GPU run has been performed. The new adapter does not mutate any AIC source, change the venv installation, replace legacy aiconfigurator, or change sys.path.


## Long-context audit update

The prior statement about using AIC's generic imbalance knob is superseded: upstream MLA ignores it. The owned adapter now explicitly applies the AIC squared-length workload ratio to MLA only, preserving native latency and the correction in per-op outputs. B1 and homogeneous results are unchanged. 8 tests pass. See `long_context_audit.md` and `long_context_audit.json` for 64K through1M queries, significant MLA extrapolation boundaries, and actual G1/admission limits. Restart a worker if it had already imported the earlier module.

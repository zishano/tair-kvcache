HiSim controlled hybrid-checkpoint adapter is implemented and tested.

Files changed by this agent:
- `/workspace/tair-kvcache/hisim/src/hisim/simulation/hybrid_checkpoint.py` (new)
- `/workspace/tair-kvcache/hisim/src/hisim/simulation/bench_serving.py` (explicit 10-line dispatch branch; original HTTP mode unchanged)
- `/workspace/tair-kvcache/hisim/test/test_hybrid_checkpoint.py` (new, 15 behavioral tests)

Copy/edit `/tmp/kimi_hisim_adapter_smoke/agentx_template.json` for the main AgentX workload. The complete schema is shown there. Execute:

```bash
/workspace/tair-kvcache/.venv/bin/python -m hisim.simulation.bench_serving --hybrid-checkpoint-config /path/to/config.json
```

`storage.capacity_bytes` is whole-TP8 payload capacity, integer bytes, or JSON null for unlimited; nonmultiples of bundle_bytes are floored to whole bundles. All four link bandwidths are whole-TP8 aggregate bytes/s. The example values are declared simulation assumptions, not measured storage hardware rates. The predictor is KimiK3TimePredictor, native AIC core isolated from the legacy AIC facade. Predictor limits should stay max_batch_size32 / max_num_tokens8192 throughout a c1/2/4/8/16/32 sweep so the native memory overhead denominator stays fixed.

Omit scheduler.g1_capacity_bytes to use AIC's physical remaining KV/state budget (63,438,356,480 bytes/rank * 8 for default B300-SXM TP8, memory_fraction0.9). An explicit smaller aggregate budget is allowed; a larger one is rejected. Layout TP and per-rank MLA/KDA sizes must match the predictor. All timestamps are simulation seconds, normalized to the trace's minimum timestamp. A trace can supply max_cache_hit_tokens; it must be aligned and shorter than input_len. Query eligibility also obeys floor((input_len-1)/checkpoint_tokens). Commit includes all complete prompt bundles.

Scheduling is FCFS, nonpreemptive admission; prefill-first (no mixed phase), max_prefill_tokens batch budget, per-request chunk and checkpoint-boundary stops; decode when no prefill request is ready. Real per-request (new_tokens,past_kv_length,is_prefill) vectors are passed to the AIC predictor. Same-shaped queries are memoized; each iteration.aic_query_id joins to aic_queries.jsonl. The corrected heterogeneous MLA predictor was loaded in fresh processes for the final smoke.

Critical memory policy: reserve the WHOLE input + maximum output MLA footprint per admitted request, plus active_kda_buffers (default2) KDA states AND floor(input/checkpoint) KDA checkpoint snapshot buffers. The latter is deliberately conservative, because this adapter retains GPU endpoint snapshots through prompt prefill and sends them as one write group afterward. This is stronger than a two-state-only memory_fit bound; do not report the two-state-only bound as this adapter's actual admission capacity. G1 source bytes remain reserved until D2H completes even if generation finishes earlier. All AgentX sample requests individually fit this policy; the largest (input170368/output6903) reserves14,338,252,800 bytes/rank. Host transfer buffers are unbounded, reserved at enqueue, and have a conservative peak recorded. No cross-request G1 or host prefix reuse.

Storage snapshots/pins protect read objects through H2D. New bundles become query-visible only after prefill completion, D2H, and FIFO write completion. All objects contain MLA, KDA SSM, KDA conv; partial bundles cannot be published. Tail-to-head write-group completion is planned atomically against a residency snapshot so insertion of a new suffix cannot evict the same commit's reused head before touching it. Only newly computed suffix bundles pay write traffic; surviving reused-prefix objects are touched, and a prefix evicted by OTHER requests during inference is not recreated. Ordered operations in storage.jsonl make residency and pin counts replayable.

Outputs per run: metrics.json, request.jsonl (arrival/admission/read_ready/prefill_done/token_times/finish/reuse/G1), iteration.jsonl (times/phase/exact query shapes/query id), aic_queries.jsonl (full per-op answers for unique shapes), storage.jsonl (query/materialization/transfers/completion/ordered evictions/occupancy/pins), queue.jsonl (active/outstanding/queued/G1 waits), resolved_config.json, provenance.json (source and input hashes, model/policy, limitations). Metrics reuse HiSim calc_metrics; p50/p99 TPOT percentiles are over per-request mean intertoken time. Max concurrency is configured_max_concurrency; observed peak active/outstanding/queue are observations, not maximum sustainable capability.

Validation:
- `/workspace/tair-kvcache/.venv/bin/python -m pytest -q test/test_hybrid_checkpoint.py`: 15 passed (1.00s).
- Original `bench_serving --help` path succeeds.
- Final real-AIC smoke is first16 rows of the parent's synthetic trace, same16 requests/393728 input/512 output at all three capacities. Results and conservation/storage-replay audit are in verification.json (all PASS). All request/compute/cache token totals, per-batch AIC joins, exact source/input hashes, resident capacity, and read pin replay were checked.
- Three fresh bench_serving commands completed for disabled.json, finite_8bundles.json and unlimited.json. These are functional CPU simulation smoke results, not the main AgentX report.

Observed smoke results (c4, 8rps offered trace):
| Capacity | Cached input tokens | P50 TTFT ms | P99 TTFT ms | P50 TPOT ms | P99 TPOT ms | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 12598.945 | 20354.343 | 16.872 | 28.919 | 22.861 |
| 8 bundles=5,406,916,608 bytes | 53248 | 11314.627 | 18055.768 | 17.993 | 29.544 | 25.294 |
| Unlimited | 98304 | 10135.813 | 16169.244 | 19.466 | 81.939 | 27.751 |

TTFT and throughput improve, while P99 TPOT gets worse under the declared prefill-first policy because different readiness/admission timing interrupts decode; do not assume every service metric is monotone in cache capacity. All cases have peak active4, outstanding16, queue12. This small trace is overloaded and is not an SLA test.

Remaining main-agent work: run the actual AgentX subset and concurrency/capacity sweep, assemble plots/report/SOP. Include long-prefix MLA table extrapolation limits from `/tmp/litehit_kimi_k3_aic_probe/long_context_audit.md` in the report provenance; the per-op silicon tag does not prove long context was measured. The adapter is CPU HiSim+AIC with declared storage policy, not native SGLang Kimi-K3 and not GPU measured. Local GPU checkpoint snapshot copies are not separately timed; forward time and D2H/storage transfer time are modeled. No other agents were spawned, no AIC or LiteHit source was modified by this agent, and no full AgentX sweep was launched by this agent.

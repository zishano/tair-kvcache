# Long-context AIC Kimi-K3 audit

This CPU probe is pinned to AIC ec215200d05cf6899d0d472b75ceafa6b53c432f, B300-SXM TP8 / MoE EP8 / no speculative decoding. No GPU E2E measurement was collected.

| Past tokens | Prefill new8192 B1 ms | Decode B1 ms | Decode B8 ms | Decode B32 ms |
|---:|---:|---:|---:|---:|
| 65536 | 427.784268 | 15.048376 | 22.158333 | 38.902174 |
| 131072 | 493.225664 | 15.498161 | 24.579178 | 48.661914 |
| 196608 | 558.661089 | 16.093257 | 27.297355 | 58.784654 |
| 262144 | 624.093956 | 16.683490 | 30.018818 | 68.910611 |
| 1040384 | 1401.099912 | 23.741114 | 62.406316 | 189.233316 |

Every prefill probe has 27 silicon-labelled and 11 empirical operators; decode has 28 silicon-labelled and 11 empirical operators. No SOL or missing KDA rows appeared. The silicon label includes interpolation and extrapolation and does not certify the long context length was measured.

## Coverage

B300/SGLang BF16 context MLA tables for the surrounding 8/16-head slices stop at full sequence length 32768 for batch 1/2 (batch4:16384, batch8:8192, batch16:4096, batch32:2048). Kimi-K3 TP8 has 12 local MLA heads, interpolated between 8 and 16. AIC queries full_s = past + new before applying the prefix correction. Thus 8K chunks do NOT make a 64K-1M prefix an in-range prefill query. The requested 64K/128K/256K past probes extrapolate the context table by 2.25x/4.25x/8.25x in full sequence length; the final 1M chunk is 32x.

Generation tables span sequence length isl+step 2..131072 at batches 1/8/32. Decode past=65536 is in range. Past=131072 queries 131073, just one token outside the endpoint. Past=196608/262144/1040384 are approximately 1.5x/2x/7.94x outside it.

KDA depends on the new chunk/state, not total past length. Prefill8192 and decode batches1/8/32 use available KDA silicon tables; the long-context extrapolation is MLA.

## Memory bounds

Memory is rank-local; one complete MLA token costs 27648 bytes and one KDA state slot costs 56171520 bytes. The existing 90%-HBM native weight/activation/runtime reservation leaves 63438356480 bytes/rank for G1+states. With TWO actual active state slots per admitted request:

| Resident context/request | Physical request limit |
|---:|---:|
| 65536 | 32 |
| 66048 | 32 |
| 73728 | 29 |
| 131072 | 16 |
| 139264 | 16 |
| 196608 | 11 |
| 204800 | 10 |
| 262144 | 8 |
| 270336 | 8 |
| 1048576 | 2 |

These limits use the whole remaining KV budget. A smaller configured G1, retained checkpoints, transfer staging, or extra reservations reduces the limit. The predictor can return CPU timing for B32 at 128K/256K even though memory_fit rejects those batches; the scheduler must enforce admission. B8 fits at 256K but not near 1M. Model context is capped at 1048576, even though one-request physical memory permits more.

## Adapter correction found and fixed

The upstream ContextMla dispatch ignores seq_imbalance_correction_scale: native MLA latency with scale1 and scale3 is identical. The adapter now keeps that generic scale at1 and explicitly multiplies only context_attention latency/energy by sum(q*(q+2*p))/(B*q_mean*(q_mean+2*p_mean)); native_latency_ms and shape_work_scale are retained. This follows AIC's own squared-length prefix model, with exact discrete causal pair counts recorded separately. Homogeneous and B1 results are unchanged. This is still a workload-based approximation to a true variable-length attention kernel, not an E2E calibration.

Extreme anti-correlated case: requests (q1,p1048575) and (q8191,p0) attend 1048576 + 8191*8192/2 = 34598912 exact causal pairs. AIC squared work is 69189632 vs aggregate work 8623489024; explicit scale = 0.008023391901750972. Uncorrected homogeneous averaging overstates that work by 124.635567x. Other token-major operators remain unscaled and see total8192 tokens. Eight deterministic random pairs and two additional hand-checkable pairs were independently enumerated; all exact causal-work checks passed. Ceil means can add fewer than B modeled query tokens; both actual and modeled counts are logged.

The adapter also rejects past+new >1048576. Tests: 8 passed in6.56s. If a HiSim worker imported the previous module already, restart it before recording provenance for the new run.

Artifacts: long_context_audit.json (full per-op results, source counts, source hashes, bounds, and all arithmetic checks); long_context_audit.py (reproducible command script).

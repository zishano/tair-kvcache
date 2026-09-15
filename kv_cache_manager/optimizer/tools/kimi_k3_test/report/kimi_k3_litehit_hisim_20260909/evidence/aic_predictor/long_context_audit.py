import json
import math
from collections import Counter
from pathlib import Path
import random
import time

import pandas as pd
from hisim.time_predictor.kimi_k3 import KimiK3TimePredictor, KimiK3Request, _runtime

output = Path('/tmp/litehit_kimi_k3_aic_probe/long_context_audit.json')
predictor = KimiK3TimePredictor()
records = []
for past in (65536, 131072, 196608, 262144, 1040384):
    for batch, prefill in [(1, True), (1, False), (8, False), (32, False)]:
        new = 8192 if prefill else 1
        t0 = time.perf_counter()
        result = predictor.query_batch([(new, past, prefill)] * batch)
        result.update(past=past, batch=batch, phase='prefill' if prefill else 'decode',
                      source_counts=dict(Counter(op['source'] for op in result['per_op'])),
                      wall_seconds=time.perf_counter() - t0,
                      memory_fit=predictor.memory_fit(g1_tokens=batch * (past + new), active_requests=batch),
                      mla_query_length=past + new)
        records.append(result)
        print(result['phase'], 'past', past, 'batch', batch, 'latency_ms', round(result['latency_ms'], 6),
              'source_counts', result['source_counts'], 'fits', result['memory_fit']['fits'], flush=True)

profile = predictor.memory_profile
memory_limits = []
for b in (1, 8, 32):
    free_mla = profile['kv_budget_bytes'] - b * 2 * profile['kda_state_bytes_per_slot_per_rank']
    token_limit = free_mla // profile['mla_kv_bytes_per_token_per_rank']
    memory_limits.append(dict(active_requests=b, state_slots_per_request=2, mla_token_capacity=int(token_limit),
                              per_request_context_limit=int(token_limit // b),
                              model_context_limit=predictor.max_context_length))
concurrency_limits = []
for context in (65536, 65536 + 512, 73728, 131072, 139264, 196608, 204800, 262144, 270336, 1048576):
    request_bytes = context * profile['mla_kv_bytes_per_token_per_rank'] + 2 * profile['kda_state_bytes_per_slot_per_rank']
    concurrency_limits.append(dict(context_tokens=context, bytes_per_request_per_rank=request_bytes,
                                   physical_limit=int(profile['kv_budget_bytes'] // request_bytes),
                                   configured_admission_limit=predictor.max_batch_size))

base = predictor.systems_path / 'data/b300_sxm/mla/sglang/0.5.14'
coverage = {}
for name in ('context_mla_perf.parquet', 'generation_mla_perf.parquet'):
    df = pd.read_parquet(base / name)
    df = df[(df.kv_cache_dtype == 'bfloat16') & df.num_heads.isin([8, 16])]
    df['sequence_length'] = df.isl + df.step
    coverage[name] = dict(path=str(base/name), matching_heads=sorted(df.num_heads.unique().tolist()),
                         query_heads=12,
                         rows=df.groupby(['num_heads','batch_size']).sequence_length.agg(['min','max','count']).reset_index().to_dict('records'))

manual_shapes = [(KimiK3Request(1, 1048575, True), KimiK3Request(8191, 0, True)),
                 (KimiK3Request(4096, 0, True), KimiK3Request(4096, 262144, True)),
                 (KimiK3Request(1024, 3072, True), KimiK3Request(2048, 8192, True))]
rng = random.Random(20260909)
for _ in range(8):
    q0 = rng.randint(1, 8191)
    q1 = rng.randint(1, 8192-q0)
    manual_shapes.append((KimiK3Request(q0, rng.randint(0, 262144), True),
                          KimiK3Request(q1, rng.randint(0, 262144), True)))
manual = []
for shapes in manual_shapes:
    rt, agg = _runtime(shapes)
    # Independent sum of the number of keys attended by each query token.
    counted_pairs = sum(sum(r.past_kv_length + token for token in range(1, r.new_tokens+1)) for r in shapes)
    assert agg['exact_causal_attention_pairs'] == counted_pairs
    assert math.isclose(agg['aggregate_aic_attention_work'] * agg['mla_attention_work_scale'], agg['exact_aic_attention_work'], rel_tol=1e-14)
    manual.append(dict(requests=[r.__dict__ for r in shapes], aggregation=agg, enumerated_causal_pairs=counted_pairs))

# Runtime's generic scale is demonstrably ignored by upstream MLA: retained
# here as the regression evidence motivating the explicit adapter correction.
rt, agg = _runtime(manual_shapes[2])
rows1 = predictor._engine.run_static_per_op(**rt)[0]
rt['seq_imbalance_correction_scale'] = 3.0
rows3 = predictor._engine.run_static_per_op(**rt)[0]
mla1 = next(o[1] for o in rows1 if o[0] == 'context_attention')
mla3 = next(o[1] for o in rows3 if o[0] == 'context_attention')
assert mla1 == mla3

payload = dict(probes=records, memory_limits=memory_limits, concurrency_limits=concurrency_limits,
               mla_table_coverage=coverage, heterogeneous_work_checks=manual,
               upstream_generic_scale_probe=dict(scale_1_mla_ms=mla1, scale_3_mla_ms=mla3),
               provenance=predictor.provenance())
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print('memory_limits', json.dumps(memory_limits))
print('concurrency_limits', json.dumps(concurrency_limits))
print('output', output)

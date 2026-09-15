# Hybrid checkpoint source evidence

Kimi-K3 requires both the MLA prefix and KDA endpoint state. The fixed-interval atomic bundle policy makes every LiteHit object equal in bytes. It is an explicit modeled policy; the SGLang cache below does not create a state checkpoint when splitting arbitrary prefix nodes.

AIC also explicitly describes a single elastic KV/state budget, while default SGLang may reserve separate pools. The controlled HiSim adapter uses one aggregate G1 byte budget with explicit active-state and checkpoint-snapshot reservations. Actual split-pool deployments require separate pool admission gates.

## /sgl-workspace/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py:917

```text
917:         value: List[torch.Tensor] = []
918:         best_value_len = 0
919:         best_last_node = node
920:         while len(key) > 0 and child_key in node.children.keys():
921:             child = node.children[child_key]
922:             # update best_value_len and best_last_node if needed
923:             if node.mamba_value is not None:
924:                 best_value_len = len(value)
925:                 best_last_node = node
926: 
927:             prefix_len = self.key_match_fn(child.key, key)
928:             if prefix_len < len(child.key):
929:                 new_node = self._split_node(child.key, child, prefix_len)
930:                 value.append(new_node.value)
931:                 node = new_node
932:                 break
933:             else:
934:                 value.append(child.value)
935:                 node = child
936:                 key = key[prefix_len:]
937: 
938:                 if len(key):
939:                     child_key = self.get_child_key_fn(key)
940:         # handle best_value_len and best_last_node, for the case that last node is fully matched
941:         if node.mamba_value is not None:
942:             best_value_len = len(value)
943:             best_last_node = node
944: 
945:         return value, best_last_node, best_value_len
946: 
```

## /sgl-workspace/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py:1030

```text
1030:     def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
1031:         # new_node -> child
1032:         new_node = TreeNode()
1033:         new_node.children = {self.get_child_key_fn(key[split_len:]): child}
1034:         new_node.parent = child.parent
1035:         new_node.mamba_value = None  # mamba cache can not be split
1036:         new_node.full_lock_ref = child.full_lock_ref
1037:         new_node.mamba_lock_ref = 0
1038:         new_node.key = child.key[:split_len]
1039:         new_node.value = child.value[:split_len].clone()
1040: 
```

## /workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync/aic-core/src/aiconfigurator_core/sdk/models/kimi_k3.py:768

```text
768:         """Per-token paged KV: only MLA layers hold token-linear latent KV
769:         (kv_lora_rank + qk_rope_head_dim, TP-replicated). With DSPARK the
770:         draft model's GQA KV (5 layers, TP-sharded) adds per-token elements."""
771:         cfg: common.KimiK3Config = self.extra_params
772:         n_mla = cfg.layer_types.count("full_attention")
773:         elements = n_mla * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
774:         if self._nextn > 0:
775:             if self._backend_name == "vllm":
776:                 # Inferact draft is MLA-style: latent KV, TP-replicated.
777:                 elements += self.DRAFT_NUM_LAYERS * (self.VLLM_DRAFT_KV_LORA_RANK + self.VLLM_DRAFT_QK_ROPE_DIM)
778:             else:
779:                 tp = self.config.tp_size
780:                 draft_kv_heads = max(1, self.DRAFT_NUM_KV_HEADS // tp)
781:                 elements += self.DRAFT_NUM_LAYERS * 2 * draft_kv_heads * self.DRAFT_HEAD_DIM
782:         return elements
783: 
784:     def _kda_state_bytes_per_request(self) -> float:
785:         """Constant KDA state-pool bytes per admitted request on one GPU:
786:         (SSM fp32 state + conv bf16 window) per KDA layer, TP-sharded, times
787:         the radix-cache slot multiplier."""
788:         cfg: common.KimiK3Config = self.extra_params
789:         tp = self.config.tp_size
790:         n_kda = cfg.layer_types.count("linear_attention")
791:         heads_local = max(1, cfg.kda_num_heads // tp)
792:         ssm_bytes = heads_local * cfg.kda_head_dim * cfg.kda_head_dim * 4
793:         conv_bytes = (cfg.kda_conv_kernel - 1) * 3 * heads_local * cfg.kda_head_dim * 2
794:         return n_kda * (ssm_bytes + conv_bytes) * self.KDA_STATE_SLOTS_PER_REQUEST
```

## /workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync/aic-core/src/aiconfigurator_core/sdk/models/kimi_k3.py:796

```text
796:     def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
797:         """KDA state is charged INSIDE the kvcache budget: MLA KV and KDA
798:         state draw from one elastic byte pool here. That matches sglang's
799:         opt-in --enable-unified-memory mode; the DEFAULT is two separately
800:         sized pools (MLA KV vs KDA state) with a boot-time ratio, where a
801:         mismatched workload saturates one pool while the other idles — so
802:         this estimate is optimistic for default-mode deployments (owner
803:         decision 2026-07-31: annotate, don't model the split)."""
804:         seq_len = max(0, seq_len)
805:         token_bytes = seq_len * self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()
806:         return token_bytes + self._kda_state_bytes_per_request()
```

## /workspace/tair-kvcache/kv_cache_manager/optimizer/manager/lite_hit_offline_runner.cc:62

```text
62:     }
63:     const std::size_t threads = std::min<std::size_t>(static_cast<std::size_t>(worker_count), count);
64:     std::vector<std::thread> pool;
65:     pool.reserve(threads);
66:     for (std::size_t t = 0; t < threads; ++t) {
67:         pool.emplace_back([&, t] {
68:             for (std::size_t i = t; i < count; i += threads) {
69:                 fn(i);
70:             }
71:         });
72:     }
73:     for (auto &worker : pool) {
74:         worker.join();
75:     }
76: }
77: 
78: } // namespace
79: 
80: bool LiteHitOfflineRunner::Run() {
81:     // Empty registry_uri keeps everything in-memory: registration is validated
82:     // exactly like online but nothing is persisted.
83:     auto registry = std::make_shared<OptimizerRegistryManager>("");
84:     OnlineOptimizerManager manager(registry);
85: 
86:     std::unordered_map<std::string, const OptimizerInstanceGroup *> groups_by_name;
87:     for (const auto &group : config_.instance_groups()) {
88:         ErrorCode ec = manager.CreateInstanceGroup(group);
89:         if (ec != EC_OK) {
90:             KVCM_LOG_ERROR("LiteHitOfflineRunner: CreateInstanceGroup[%s] failed, ec=%d",
91:                            group.name().c_str(),
92:                            static_cast<int>(ec));
93:             return false;
94:         }
95:         groups_by_name.emplace(group.name(), &group);
96:     }
97: 
98:     // Per-instance lanes. Registration through the manager validates the
99:     // config and yields the full and full+linear byte charges. Every lane
100:     // has one uniform per-block charge recorded into every fact row.
```

## /workspace/tair-kvcache/kv_cache_manager/optimizer/manager/lite_hit_offline_runner.cc:220

```text
220:         }
221: 
222:         // Lane commits stay in input order; only same-lane order is
223:         // semantically required, and input order trivially satisfies it.
224:         for (BatchItem &item : batch) {
225:             item.record.fact = item.lane->core.ProcessRequest(item.normalized.block_keys, item.record.timestamp_ns);
226:             if (config_.reserve_last_token()) {
227:                 const uint64_t tokens = item.normalized.input_token_len;
228:                 uint64_t remaining = tokens == 0 ? 0 : (tokens - 1) / item.lane->block_size_tokens;
229:                 auto &curve = item.record.fact.hit_curve;
230:                 std::size_t keep = 0;
231:                 for (auto &segment : curve) {
232:                     if (remaining == 0) {
233:                         break;
234:                     }
235:                     segment.run_length = std::min(segment.run_length, remaining);
236:                     remaining -= segment.run_length;
237:                     ++keep;
238:                 }
239:                 curve.resize(keep);
240:             }
241:         }
242: 
243:         ParallelForIndex(batch.size(), worker_count, [&](std::size_t i) {
244:             BatchItem &item = batch[i];
245:             item.record.input_token_len = item.normalized.input_token_len;
246:             item.record.block_size_tokens = item.lane->block_size_tokens;
```

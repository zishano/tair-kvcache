**HiSim Stage 2 独立结果审查：容量、命中时序与指标解释**

审查日期：2026-09-09 UTC。范围为 C32 下 0、50、250 GiB、无限容量四个现有运行；不包含新增 C8/C16 探针。本次只读取既有 trace、Facts、request/storage/iteration/queue ledger 与 provenance，未启动 LiteHit/HiSim，也未修改代码。

结论：四点结果在声明的 CPU HiSim+AIC 模型下自洽。50 GiB 的 Stage 2 命中略高有具体时序证据，不能将“立即写回”的有限容量曲线称为 Stage 2 上界。较大容量的负差异包含尚未发布的 checkpoint，以及有限容量逐出后重写尚未完成两类情况。延迟改善主要通过缩短准入队列体现；数秒级尾部 TPOT 来源于 prefill 优先调度造成的 token 间停顿。

**控制组与计数核对**

四点都保留 163 请求、10,473,344 input tokens、184,322 output tokens。最后一个请求在 190.9056 秒到达。去掉 output_dir 和 storage.capacity_bytes 后，四份 resolved_config 完全一致；trace、layout、adapter 以及 predictor 的全部 source_sha256 字典分别一致。共同 trace SHA256 为 `9602b6877341fec39f1e407d7a79960d3e5e87f1eeeaf5a42bb204ac7fa9644e`。50/250 GiB 实际能放入 79/397 个 675,864,576-byte bundle；容量指 TP8 全 payload，不包含存储索引开销。

从 [Stage 1 逐请求投影](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage1/capacity_query.jsonl) 和每点 request ledger 按 trace_id 对齐，使用整数 token 数重算：

| 容量 | Stage 1 hit tokens / rate | Stage 2 hit tokens / rate | Stage 2 - Stage 1 | 正 / 负 / 相同请求数 |
|---|---:|---:|---:|---:|
| 0 | 0 / 0% | 0 / 0% | 0 | 0 / 0 / 163 |
| 50 GiB | 2,908,160 / 27.767254% | 2,959,360 / 28.256114% | +51,200；+0.488860 pp | 20 / 24 / 119 |
| 250 GiB | 7,898,112 / 75.411559% | 6,934,528 / 66.211212% | -963,584；-9.200347 pp | 0 / 33 / 130 |
| 无限 | 8,860,672 / 84.602129% | 8,132,608 / 77.650538% | -728,064；-6.951591 pp | 0 / 38 / 125 |

50 GiB 的正差异合计 +833,536 tokens，负差异合计 -782,336 tokens，净值仅 +51,200。因此“略高”并不意味着两阶段几乎逐请求相同。

**50 GiB 正差异：延迟发布也会延迟逐出**

为定位有限容量的逐出者，本审查仅从既有 trace 重建 79-bundle 的立即写回 LRU 元数据；163 条重建命中与已有 Stage 1 查询逐条相同。20 条正差异按首个分歧 bundle 的事件解释：18 条对应 Stage 1 中的逐出请求在 Stage 2 查询之后才发布；1 条对应之后重新写入；1 条对应复用前缀失效后只写新后缀的策略差异。

- **+80,896 tokens 的直接反例。** 请求 `03e110ac6921c2fe9ac7772a9654314a2beb:main/9/subagent/6`，[Facts 第129行](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage1/litehit_facts.csv:129) 为 `[[166,85]]`，50 GiB 只有79个位置，故 Stage 1 命中0。立即 LRU 中，前驱 `0196085d85d2075a50b74cd8795ffbdcea9a:main/7`（[trace 第127行](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/trace.jsonl:127)）已逐出该前缀。在 Stage 2 中，它在581.809125秒提交写入，计划585.792351秒才可见（[write_submit](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:5983)）；目标查询却在582.605770秒执行，仍命中79 bundles（[read_query](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:5985)）。该前缀头 key `25769804657` 到585.792351秒才被此长请求真正逐出（[write_visible/evict](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:5996)）。读取在583.771233秒已完成，命中不依赖读已删除对象。
- **延迟重新写入。** `002001296e8a8c38ad9d7cc436d691afc602:main/7` 增加47,104 tokens。前缀头 `4294967321` 在55.210344秒被逐出，56.590450秒又由较早请求 `...:main/4` 写回，随后被 `main/6`、`main/5` 刷新，63.680452秒的目标查询得以命中。可查 [逐出](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:1272)、[重新写入](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:1300)、[目标请求](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/request.jsonl:33)。这反映提交次序改变了 LRU，不能只比较到达顺序。
- **只重写新后缀。** `00ca01c4aaeccee751ae6285beb54dea262f:main/17` 增加67,584 tokens，即66 bundles。此前 `0196085d85d2075a50b74cd8795ffbdcea9a:main/11/subagent/4` 在756.509516秒读到67 bundles；它们在另一次提交中失效后，该请求在759.322361秒只发布新算出的13 bundles，记录 `reused_prefix_evicted=67`，没有凭空恢复已逐出的复用前缀（[读取](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:7432)、[只写13个新bundle](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:7451)）。Stage 1 的立即提交会刷新该请求全部80个完整bundle，足以逐出另一条链。Stage 2 保留了另一条链的66个bundle，使761.360380秒的后续查询命中（[目标请求](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/request.jsonl:153)）。这是明确的生命周期差异，不能简化成 IO 时间一项。

**负差异：先前可复用不等于本次查询已可读取**

按首个缺失 bundle 的历史状态归类，50 GiB 的24条负差异中，7条在查询前从未发布（缺117,760 tokens），17条曾发布后被逐出（缺664,576）；250 GiB 的33条分别为29条（643,072）和4条（320,512）。这首先是可核查的驻留状态分类；“从未发布”本身不排除有限容量写入被拒绝，因而不全部归因于带宽。

- **50 GiB 的 -80,896 tokens。** `0470d446a4514dfe0c6ad0be92853bd13287:main/23` 的 [Facts](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage1/litehit_facts.csv:117) 是 `[[1,113]]`，故粗筛命中79 bundles。HiSim 中前缀头 `30064771094` 在464.811949秒被其他请求逐出；目标在477.923933秒查询时未命中，前驱 `...:main/22` 的重写到479.168839秒才发布。参见 [逐出](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:4878)、[查询](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:4995)、[较晚重写完成](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_50gib/storage.jsonl:5010)。
- **250 GiB 的 -160,768 tokens。** `0196085d85d2075a50b74cd8795ffbdcea9a:main/4` 的 [Facts](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage1/litehit_facts.csv:119) 是 `[[1,157]]`，粗筛全命中157 bundles。HiSim 中其前缀头 `17179869199` 在211.490827秒被逐出；目标在305.010453秒查询时未命中。前驱 `main/3` 已在301.973062秒完成prefill并提交写入，但要到305.833348秒才可见，晚了0.822895秒。对应 [逐出](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_250gib/storage.jsonl:2361)、[写提交](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_250gib/storage.jsonl:2928)、[查询](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_250gib/storage.jsonl:2930)、[重新发布](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_250gib/storage.jsonl:2940)。无限容量下同一请求仍命中160,768 tokens；有限容量历史与查询时序共同导致这里的缺口。
- **无限容量的38条负差异全部有发布时序证据。** 对每一条取 `bundle_keys[Stage2_hit_tokens/1024]`，其第一次 insert 时间均严格晚于本次 admission/query，合计解释728,064个缺失tokens，且该点没有逐出或pin导致的写拒绝。例如 `002001296e8a8c38ad9d7cc436d691afc602:main/4` 粗筛命中66,560 tokens，却在24.270594秒查询到0；它的前缀首次到26.095010秒才由 `main/1` 发布。后者prefill已在23.351010秒完成，故这个例子确实需要区分“算完”和“写完”。可查 [producer write_submit](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/storage.jsonl:537)、[目标查询](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/storage.jsonl:566)、[首次发布](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/storage.jsonl:610)。

**延迟、吞吐与并发的解释**

本审查从 token 时间戳重算四点 P99 TTFT/TPOT，与 metrics 一致。吞吐分母是从首个到达到最后生成完成的整个仿真区间，包含队列与尾部排空；输出吞吐为整个8-GPU replica的tokens/s。

| 容量 | 完成区间 s | P50 / P99 TTFT s | P50 / P99 TPOT ms | 输出 tokens/s | peak active / outstanding / queued |
|---|---:|---:|---:|---:|---:|
| 0 | 1214.823 | 351.019 / 884.923 | 173.064 / 3779.530 | 151.728 | 32 / 153 / 121 |
| 50 GiB | 959.707 | 285.686 / 698.849 | 121.193 / 3328.934 | 192.061 | 32 / 151 / 119 |
| 250 GiB | 607.318 | 169.530 / 364.220 | 63.796 / 2454.638 | 303.502 | 32 / 141 / 109 |
| 无限 | 505.271 | 135.953 / 273.838 | 51.438 / 2454.638 | 364.798 | 32 / 139 / 107 |

按逐请求可加分解，平均 TTFT 的“准入排队 / 存储读取 / 参与的prefill batch计算 / ready后其余调度等待”分别为：0容量 `374.740 / 0 / 8.310 / 3.383`秒；50 GiB `303.540 / 0.271 / 6.452 / 0.607`秒；250 GiB `169.919 / 0.639 / 3.831 / 0.543`秒；无限 `129.385 / 0.769 / 3.092 / 0.544`秒。准入排队约占平均TTFT的97%，这些量不能解释成单请求模型forward耗时。

TPOT是每请求 `(末token时间-首token时间)/(output_len-1)` 的平均，再对163个请求取分位数。250 GiB与无限容量的P99完全相同，因为决定P99插值的请求 `[03e110...main/0](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/request.jsonl:6)` 和 `[002001...main/0](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/request.jsonl:1)` 的TPOT分别为2317.685503和2678.085873ms，两个容量下完全一致；163点P99取两者的0.38插值，得到2454.637644ms。

尾部停顿有逐批证据：无限容量中 [006c98...main/0](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/request.jsonl:2) 只输出14 tokens。第一个到第二个token的间隔为50.243153秒，其中164个prefill batches合计50.207486秒，最后一次decode为0.035667秒；从 [iteration第2行](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/iteration.jsonl:2) 到 [第166行](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/iteration.jsonl:166) 可逐项相加。该结果符合声明的prefill优先策略，不能把3.9秒/request的平均TPOT说成一次decode kernel需3.9秒。

所有点的32只是设定上限和观测到的active峰值；outstanding还包含队列。队列最多107–121条，且到达区间仅190.9056秒、排空需505–1215秒；本组数据没有测得最大可持续并发或满足某个SLA的容量。样本P99主要由最高两三条请求决定。

**G1与外部存储的计费边界**

本审查独立重建四点共652条请求的G1几何与生命周期：按64token页对齐的完整input+max_output MLA、两个活跃KDA状态、每个完整输入checkpoint的KDA snapshot缓冲；在admission计入，在D2H结束释放snapshot，基础KV/状态到 `max(finish,D2H结束)` 才释放。无新payload或禁用写出时，staging在prefill结束释放，并用该时刻代替D2H结束。重算峰值与记录逐字节相同，且所有时刻不超过507,506,851,840-byte聚合预算（每rank59.081573GiB）。

| 容量 | 每rank G1观测峰值 GiB | G1 wait events | Host staging保守峰值 GiB（全replica） |
|---|---:|---:|---:|
| 0 | 59.057041 | 58 | 0 |
| 50 GiB | 58.967717 | 163 | 154.214745 |
| 250 GiB | 59.003551 | 168 | 197.646652 |
| 无限 | 59.031086 | 173 | 152.326401 |

完整prefix即使命中也占G1；没有跨请求G1共享。0容量禁用checkpoint写出，因此没有checkpoint snapshot缓冲；其余容量按声明策略保留GPU snapshots直到D2H。这是容量0与启用外存之间的派生生命周期变化。不能套用“只含两个KDA槽”的静态并发界来解释该adapter：最大请求input170368/output6903的完整预留达到14,338,252,800 bytes/rank。

Host staging目前无准入容量限制，上表是额外的保守预留需求，未计入扫描的外部cache容量。50 GiB有125次bundle因read pin保护而未能写入，250/无限没有；pin计数是写入准入结果，不是读取已淘汰数据。阶段指标只能在这些明确的G1、staging、四条独立FIFO链路假设下使用。

**数值可信范围**

这里是声明的FCFS/prefill优先CPU HiSim adapter加AIC批次预测，provenance明确记录 `native_sglang_scheduler=false`、`gpu_kernels_executed=false`。AIC的SILICON标签可包含插值/外推，且部分elementwise/norm/AttnRes为empirical；GPU本地checkpoint snapshot复制未单独计时，D2H和存储读写已经计时。输入是8个AgentX session的163条已观测请求，包含75条源trace子请求，时间统一除10；保留原输入/输出长度并映射到Kimi-K3，未重建闭环完成依赖或官方AgentX采样/primer流程。

长上下文外推实际发生，不能因chunk只有1024而忽略。无限容量1262个prefill batches中，980个AIC full-sequence查询超过32768——已超出审计所列B1/B2 context MLA最高长度，其他较大batch的覆盖还更窄。最大query的`isl=170368, prefix=169984, batch=1`，见 [aic_queries第4825行](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/capacity_infinite/aic_queries.jsonl:4825)。12788个decode batches中3431个`past+1>131072`，最大聚合长度144316。具体表覆盖见 [long_context_audit.md](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/evidence/aic_predictor/long_context_audit.md)。这些结果支持容量敏感性与事件因果分析；部署SLA或绝对GPU延迟需要后续校准，当前数据不提供该校准。

# Kimi-K3：LiteHit 容量粗筛与 HiSim 性能精算 SOP

本流程先用 `lite_hit_main` 单遍读取 trace，生成与容量无关的 Facts，再用 `lite_hit_facts_query_main` 投影容量曲线；随后将代表容量写入 HiSim 的实际存储配置，由 `hisim.simulation.bench_serving` 的 `--hybrid-checkpoint-config` 分支运行有调度、读写完成事件和显存约束的 CPU 仿真。

结果用于分析指定 workload、缓存策略和带宽假设下的容量收益。HiSim 在这里使用受控调度适配器和 Kimi-K3 AIC 算子模型，没有运行 GPU kernel，也没有运行原生 SGLang Kimi-K3 scheduler。TTFT、TPOT 和吞吐量均为模拟值；AIC 中的 silicon 算子数据也不等于本轮端到端 GPU 实测。

基线目录：[/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample)。以下重跑命令使用新目录，保留这份基线证据。

## 1. 环境、构建与检查

在 Bash 中设置本次任务变量；选择尚未使用的 `K3_RUN`。Python 固定使用 `/workspace/tair-kvcache/.venv`。

```bash
K3_REPO=/workspace/tair-kvcache
K3_PY="$K3_REPO/.venv/bin/python"
K3_WORKFLOW="$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_capacity/run_workflow.py"
K3_AIC_ROOT=/workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync
K3_MODEL="$K3_AIC_ROOT/aic-core/src/aiconfigurator_core/model_configs/moonshotai--Kimi-K3_config.json"
K3_TRACE=/workspace/dynosim_aic_sweep/traces.jsonl
K3_BASELINE=/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample
K3_RUN=/workspace/dynosim_aic_sweep/reports/kimi_k3_iterations/agentx_rerun_01
cd "$K3_REPO"
```

需要已有的 AIC checkout、其中编译好的 `aiconfigurator_core/_aiconfigurator_core.abi3.so` 与随仓库提供的算子数据库。venv 默认的旧 `aiconfigurator` 安装保留；Kimi-K3 predictor 显式加载新 checkout 的独立 `aiconfigurator_core` 命名空间。HiSim 子进程的 `PYTHONPATH` 由 workflow 设置为本仓库 `hisim/src`。

LiteHit 构建目标与对应测试目标如下。首次准备环境或修改 C++ 后执行；单纯增加 Facts 查询容量无需重新构建。

```bash
bazel build --jobs=4 \
  //kv_cache_manager/optimizer:lite_hit_main \
  //kv_cache_manager/optimizer:lite_hit_facts_query_main

bazel test --jobs=4 --test_output=errors \
  //kv_cache_manager/optimizer/test:LiteHitTest \
  //kv_cache_manager/optimizer/test:LiteHitTtlTest \
  //kv_cache_manager/optimizer/test:LiteHitOfflineRunnerTest \
  //kv_cache_manager/optimizer/test:OptimizerConfigTest
```

修改 HiSim 或 predictor 后可运行这两个有针对性的测试文件：

```bash
PYTHONPATH="$K3_REPO/hisim/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$K3_PY" -m pytest -q \
  "$K3_REPO/hisim/test/test_hybrid_checkpoint.py" \
  "$K3_REPO/hisim/test/test_kimi_k3_predictor.py"
```

查看 CLI 参数：

```bash
"$K3_PY" "$K3_WORKFLOW" --help
```

## 2. 输入选择与缓存对象

基线使用真实 AgentX session JSONL 的一个有界样本：前 8 个 session，每个 session 最多选择 24 个叶请求，保留 `t <= 3600 s`、`0 < in <= 262144`、`out > 0` 的请求。过滤、时间排序和采样后得到 163 个请求，其中 75 个来自嵌套 subagent；总输入 10,473,344 tokens，总输出 184,322 tokens。具体过滤计数、源文件 hash 和每 session 选择数见 [trace_manifest.json](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/trace_manifest.json)。

这是 **AgentX-derived fixed-arrival counterfactual**。各 session 同时起步，使用源 session-relative 时间戳；基线 `--time-scale 10` 将这些时间除以 10。输入长度、输出长度、session 内的 prefix identity 和嵌套 stream path 保留，源模型统一映射为 Kimi-K3；没有原始 token payload，也没有重新执行 Kimi tokenizer。源模型的质量、时延、完成事件驱动的后续到达和工具/DAG 因果关系不保留。这不是官方 AgentX 的闭环 DAG 分数，也未实现官方随机起点、primer、一小时 profiling 或 speculative acceptance 协议。

`hash_id_scope=local` 表示本源格式的 session-local 前缀编号。父请求和嵌套子请求共享该 session 命名空间，转换使用：

```text
全局 native key = (源文件 session 行号，从 1 开始 << 32) | local_prefix_hash_id
```

转换会校验每个 local id 的 depth 与 parent 一致；子请求时间不能早于其 subagent 开始时间。源块为 64 tokens，编号已经是 prefix-chained key，避免二次 prefix hashing。checkpoint 为 1024 tokens 时，每 16 个 native blocks 取末端 key。一个 cache bundle 同时包含这一段 MLA KV 和该 endpoint 的全部 KDA 状态，按完整 bundle 淘汰和恢复。

Kimi-K3 config 为 93 层：69 KDA + 24 MLA；TP8、BF16 MLA/conv、FP32 SSM 下，每 rank 的 payload 为：

```text
MLA bytes/token = 24 × (512 + 64) × 2 = 27,648
KDA SSM bytes/checkpoint = 69 × (96 / 8) × 128 × 128 × 4 = 54,263,808
KDA conv bytes/checkpoint = 69 × 3 × (4 - 1) × (96 / 8) × 128 × 2 = 1,907,712
KDA state bytes/checkpoint = 56,171,520
bundle bytes/rank = checkpoint_tokens × 27,648 + 56,171,520
```

当 checkpoint interval 为 1024：`bundle_bytes_per_rank=84,483,072`；8 ranks 合计 `bundle_bytes=675,864,576`。容量包括所有 TP rank payload，连同 TP-replicated MLA 副本一起计费；不包含存储索引元数据。所有容量均为 **GiB payload**，`1 GiB = 2^30 bytes`。底层 query JSON 的历史字段名 `capacity_gb` 在本流程中仍使用此 GiB 口径。

为生成 logits，当前请求至少重算一个 prompt token，最大可恢复前缀为：

```text
max_cache_hit_tokens = floor((input_len - 1) / checkpoint_tokens) × checkpoint_tokens
```

只裁剪本次可命中前缀；所有完整输入 bundles 仍提交，供后续更长请求使用。若输入刚好对齐 checkpoint，必须从前一个 checkpoint 重算末段，不能将状态倒退一个 token。当前流程只为完整输入 checkpoints 提供可复用 keys，没有为 decode 输出凭空生成新 key。

## 3. 分阶段执行

### prepare：生成一致的 trace、layout 和 LiteHit 配置

```bash
"$K3_PY" "$K3_WORKFLOW" \
  --steps prepare --output-dir "$K3_RUN" \
  --trace "$K3_TRACE" --model-config "$K3_MODEL" \
  --agentx-sessions 8 --requests-per-session 24 \
  --agentx-window-seconds 3600 --max-input-tokens 262144 \
  --time-scale 10 --tp-size 8 --checkpoint-tokens 1024
```

产生 `trace.jsonl`、`trace_manifest.json`、`layout.json`、`litehit_config.json`。对 AgentX 输入会自动读取 64-token native block 大小；上述 `--max-input-tokens` 是选择过滤上限，不会截断已选择请求的长度。

不传 `--trace` 可构造可复跑的 prefix-identity trace，例如：

```bash
"$K3_PY" "$K3_WORKFLOW" \
  --steps prepare --output-dir "$K3_REPO/output/kimi_k3_synthetic_01" \
  --model-config "$K3_MODEL" --requests 512 --request-rate 8 \
  --output-tokens 32 --seed 20260909 --checkpoint-tokens 1024
```

复用普通 `get/request` JSONL 时，`keys` 必须恰好覆盖 `floor(input_len/native_block_tokens)` 个完整块；用 `--native-block-tokens` 指定粒度。仅当这些 keys 确为 prefix-chained 且命名空间明确时传 `--keys-are-prefix-chained`。`prepare` 会拒绝重复 trace id、错误块数或不合法的 checkpoint 粒度。普通 trace 必须只含一个源 instance，避免模型映射时合并原本隔离的实例。

### scan：单遍 Facts 与全部容量投影

```bash
"$K3_PY" "$K3_WORKFLOW" --steps scan --output-dir "$K3_RUN"
```

默认查询 0、25、50、…、1000、1024 GiB，以及无限容量，共 43 点。每条请求、每个容量均与独立有限 LRU oracle 对照。基线首次记录的 Facts replay 约 0.018 s、43 容量查询约 0.007 s（初次日志保存在 evidence/agentx_sample_initial_stage1；最新复跑计时见 summary.json）；这些是本机 CPU wall time，未计入环境准备，也不是服务时延。

基线验证为 163 条 Facts、7,009 次 request×capacity 检查、零差异。`scan` 还检查 trace/layout hash、uniform bundle charge、请求数量及曲线单调性。关键曲线点如下：

| 容量 | 本样本 token 命中率 | 含义 |
|---:|---:|---|
| 50 GiB | 27.7673% | 较小容量代表点 |
| 250 GiB | 75.4116% | 0..1024 GiB 范围、25 GiB 网格上的归一化几何拐点 |
| 525 GiB | 81.1410% | 首个达到无限容量命中率 95% 的网格点 |
| 675 GiB | 84.6021% | 本样本开始达到无限容量平台 |
| 无限 | 84.6021% | 冷请求及 checkpoint/末 token 边界仍会 miss |

几何拐点规则为 `argmax(normalized finite hit gain - normalized capacity)`。250 GiB 后仍有约 9.19 个百分点提升；它不是“后续收益消失”的精确阈值。默认 Stage 2 选择 `0,50,250,infinite`。

单独运行 C++ Facts 工具也可复用 `prepare` 结果：

```bash
"$K3_REPO/bazel-bin/kv_cache_manager/optimizer/lite_hit_main" \
  "$K3_RUN/litehit_config.json"

"$K3_REPO/bazel-bin/kv_cache_manager/optimizer/lite_hit_facts_query_main" \
  "$K3_RUN/stage1/litehit_facts.csv" \
  "$K3_RUN/stage1/manual_query.jsonl" \
  0 50 250 525 675 1024 -1
```

改变查询容量只需第二条命令。C++ query 使用 `-1` 表示无限；容量以 `floor(capacity_bytes/bundle_bytes)` 转为完整 bundle 数。`lite_hit_main` 从配置里的绝对路径读取 trace 并写 Facts，因此改用新 output 目录时应重新 `prepare`。

### bench：注入容量并运行实际 HiSim 入口

```bash
"$K3_PY" "$K3_WORKFLOW" \
  --steps bench --output-dir "$K3_RUN" \
  --capacities 0,50,250,infinite \
  --max-concurrency 32 --max-prefill-tokens 8192 \
  --chunked-prefill-tokens 1024 --memory-fraction 0.90 \
  --read-gib-s 64 --write-gib-s 32 --h2d-gib-s 128 --d2h-gib-s 128 \
  --bench-arrival-scale 1 --probe-requests 16
```

`--capacities` 选择 Stage 1 已查询网格的任意子集；网格外容量可直接使用 Facts query CLI 投影。默认先用前 16 个**完整长度请求**做预探针并验证，再依次执行四个容量点。已有适用的预探针时可显式传 `--probe-requests 0`。`--max-prefill-tokens 8192` 是每次 forward 的总 token 预算；默认 chunk 大小为 checkpoint interval，所以基线每个请求的单步 prefill 最多 1024 tokens，并在 checkpoint 边界停下产生状态。

四个实际配置是：

| 场景 | JSON 的 `storage.capacity_bytes` | 基线配置链接 |
|---|---:|---|
| 0 GiB | `0` | [capacity_0gib.json](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/configs/capacity_0gib.json) |
| 50 GiB | `53687091200` | [capacity_50gib.json](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/configs/capacity_50gib.json) |
| 250 GiB | `268435456000` | [capacity_250gib.json](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/configs/capacity_250gib.json) |
| 无限 | `null` | [capacity_infinite.json](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/configs/capacity_infinite.json) |

配置注入的是有限外部存储，GPU G1 预算仍受物理显存限制。无限外部容量也保留相同有限带宽、写入延迟和 G1 约束。四条链路是相互独立的 FIFO：storage read/write 和 H2D/D2H；**64/32/128/128 GiB/s 均为场景假设**，是全部 8 ranks 的聚合 bytes/s，没有乘第二次 TP，也不是本次硬件测量。

workflow 实际调用的模块入口可单独复跑如下。先将选定配置复制为新文件并修改其中 `output_dir`，避免覆盖基线事件日志；本例用于重跑当前新目录里的配置：

```bash
PYTHONPATH="$K3_REPO/hisim/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$K3_PY" -m hisim.simulation.bench_serving \
  --hybrid-checkpoint-config "$K3_RUN/stage2/configs/capacity_250gib.json"
```

本 venv 没有要求安装 `hisim` shell 脚本；以上 module 命令就是这里的 HiSim `bench_serving` 入口。正式比较优先使用 workflow，它还执行独立 validation 并收集 summary。

### plot：从已完成产物生成图表与报告

```bash
"$K3_PY" "$K3_WORKFLOW" --steps plot --output-dir "$K3_RUN"
```

`plot` 消费已有 Stage 1/2 结果，输出到 `report/`，不重新仿真。报告包含容量曲线、端到端指标及并发诊断，并提供可离线查看的 HTML 和可导出的图表；具体文件以生成目录为准。只有在相应 bench 成功且 `validation.json` 的 `passed` 为 true 后，才将场景纳入完成结果。

### all：一次执行完整流程

下面与分步流程二选一。`--steps all` 和默认阶段序列均包含 `prepare,scan,bench,plot`。

```bash
"$K3_PY" "$K3_WORKFLOW" \
  --steps all --output-dir "$K3_RUN" \
  --trace "$K3_TRACE" --model-config "$K3_MODEL" \
  --agentx-sessions 8 --requests-per-session 24 \
  --agentx-window-seconds 3600 --max-input-tokens 262144 --time-scale 10 \
  --tp-size 8 --checkpoint-tokens 1024 \
  --max-concurrency 32 --max-prefill-tokens 8192 --chunked-prefill-tokens 1024 \
  --memory-fraction 0.90 \
  --read-gib-s 64 --write-gib-s 32 --h2d-gib-s 128 --d2h-gib-s 128 \
  --bench-arrival-scale 1 --probe-requests 16
```

四个主场景完成后，可只追加几何拐点容量处的 admission-limit 探针：

```bash
"$K3_PY" "$K3_WORKFLOW" \
  --steps bench,plot --output-dir "$K3_RUN" \
  --concurrency-only --concurrency-probes 8,16 --probe-requests 0
```

这会增加 `elbow_concurrency_8`、`elbow_concurrency_16`；默认配置 32 已在 `capacity_250gib` 中覆盖。这些是有限 trace 上的 admission-limit 对比，不是稳定性或 SLA 极限测试。Probe 上限不能超过 `--max-concurrency`，以保持 predictor 的显存开销参数一致。

## 4. HiSim 的时序与显存口径

固定计算配置为 **8×B300-SXM、attention TP8、MoE TP1/EP8、FP8 GEMM、MXFP4/MXFP8 MoE、BF16 MLA KV/conv、FP32 KDA state、无 draft/speculative**。AIC checkout 基线 commit 为 `ec215200d05cf6899d0d472b75ceafa6b53c432f`；数据库显式固定 SGLang **0.5.16**，不能用当时指向 0.5.14 的 `current` 代替。相关源码、模型配置、native extension 与数据文件 hash 写入各场景 provenance。

调度采用 FCFS admission、prefill-first、独立 prefill/decode batches、无抢占。真正的 cache 查询发生在 admission 通过之后；read pins 保留到 H2D 完成；新 checkpoint 经过 forward 完成、D2H、storage write 完成后才可被其他请求看见。容量或 pins 不允许的写入会被拒绝/淘汰，并写入事件账本。

Stage 1 用请求开始快照命中、随后立即按 tail-to-head 写回全部完整 bundles；Stage 2 有排队、物理资源约束和延迟写可见性，因此两阶段命中率无需相同。Stage 2 不能仅把 Stage 1 的 hit-rate 数字乘到 AIC prefill 时延上。

G1 按完整驻留上下文预留，不能只算本次未命中部分。基线每 rank 的物理 HBM 为 288,400,343,040 bytes，AIC 权重为 189,461,135,360 bytes；扣除 activation、runtime、communication 预留后，90%-HBM 预算留下 **63,438,356,480 bytes/rank**，8 ranks 合计 **507,506,851,840 bytes**。

对输入 `L`、输出上限 `O`、checkpoint interval `Δ`，基线每请求的保守 G1 预留为：

```text
T = ceil((L + O) / 64) × 64
C = floor(L / Δ)                         # 外部存储至少容得下一个 bundle 时
C = 0                                   # 0 容量或小于一个 bundle 时
G1/request/rank = T × 27,648 + (2 + C) × 56,171,520
G1/request/all-ranks = 8 × G1/request/rank
```

其中 `2` 是两个实际 active KDA buffers，`C` 为全部可能输入 checkpoint 状态的保守 staging 预留，保留到相应 D2H 完成。G1 不跨请求共享 prefix；使用共享字节预算。AIC 源码也注明这是单一 elastic pool 假设，默认 SGLang 分离的 MLA/state 池还可能各自先耗尽，部署时需补上各自的准入边界。源码摘录见 [source_evidence.md](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/evidence/source_evidence.md)。这里没有使用 AIC 的“每请求固定 5 槽”代理；只含两个 active slots 的简单长上下文并发估计，也不能代替上述含 snapshots 的实际 admission 预算。

Host staging 采用**无容量上限的 host buffer 假设**：完整读/写传输 buffer 从提交时保留到完成，记录保守峰值，不构成跨请求 host cache。这没有引入无限 GPU staging；GPU 上全部 input snapshots 已预留。应检查 `host_staging_peak_bytes` 是否符合计划部署，不能忽略它而仅看外部 cache GiB。

模型未单独计时 GPU-local checkpoint snapshot copy；将 endpoint 状态视为 forward 完成时可用，D2H 和 storage I/O 则显式计时。因此结果包含这项实现成本近似。

## 5. 输出、指标与结果读取

| 产物 | 用途 |
|---|---|
| `trace_manifest.json`、`layout.json` | 数据来源、筛选、prefix 身份与逐项字节口径 |
| `source_manifest.json` | workflow、HiSim/C++ 相关源码及二进制 hash、仓库 HEAD、Python 版本 |
| `stage1/litehit_facts.csv` | 与容量无关、可以反复投影的 Facts |
| `stage1/capacity_query.jsonl`、`capacity_curve.csv`、`summary.json` | 逐请求/逐容量结果、边际收益、拐点与 oracle 验证 |
| `stage2/configs/*.json`、`*/resolved_config.json` | 实际注入的容量、带宽、调度与物理 G1 预算 |
| `stage2/*/request.jsonl` | 每请求到达、admission、token 时间、命中/重算及 G1 预留 |
| `stage2/*/iteration.jsonl`、`aic_queries.jsonl` | 真正 scheduled batch、每请求 query/past 长度、逐算子 AIC 来源 |
| `stage2/*/storage.jsonl`、`queue.jsonl` | checkpoint materialization、读写事件、pins、驻留、队列及 host staging |
| `stage2/*/metrics.json`、`validation.json`、`provenance.json` | 指标、独立事件/资源/守恒验证、模型与仿真边界 |
| `stage2/summary.csv`、`summary.json`、`report/` | 场景比较表、可视化和报告 |

TTFT 为 `first_output_token_time - arrival_time`，包括 admission 排队、读等待与 prefill。TPOT 先对每个输出长度大于 1 的请求计算 `(last_token_time-first_token_time)/(output_len-1)`，再对**请求级 TPOT** 取 P50/P99。ITL 是所有相邻输出 token 间隔的分布，与请求级 TPOT 分位数不同。`metrics.json` 的时延字段为 ms，图表可能显示 s。

`output_throughput` 为总输出 tokens 除以从首个 trace 到达到最后请求完成的模拟时长；`total_throughput` 包含输入与输出，输入计数也包含命中的 prompt tokens，不代表实际计算了这些 tokens。总吞吐是整组 8 GPU 服务吞吐，需要每 GPU 口径时显式除以 8。`time_cost`/command `elapsed_wall_s` 是 CPU 执行时间；`storage_drain_complete_s` 包含最后请求完成后的存储 drain，不能与服务吞吐分母混用。

并发字段必须分开解释：

| 口径 | 解释 |
|---|---|
| Configured | `configured_max_concurrency`，admission 配置上限 |
| Observed active | `observed_peak_active`，本有限 trace 中实际已接纳且未完成的请求峰值 |
| Observed outstanding/queue | 已到达未完成总数、尚未接纳数；不是实际 GPU batch 大小 |
| Physical | 用实际 `L/O/Δ`、完整 G1 与 snapshots 预算计算的可容纳量；随工作集变化 |
| Sustainable | 在明确 SLA、到达模型、时长、排队稳定性和误差校准下测出的持续可服务并发 |

本轮 `maximum_sustainable_concurrency_measured=false`。没有指定 SLA、没有稳定负载的饱和边界探测，不能将配置 32 或观测峰值 32 声称为“最大可持续并发”。

基线四点的已落盘模拟结果可作为重跑对照，详细信息以 [stage2/summary.csv](/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/stage2/summary.csv) 和场景 provenance 为准：

| 外部容量 | P50 TTFT (s) | P99 TTFT (s) | P50 TPOT (ms) | P99 TPOT (ms) | 输出 tok/s（8 GPUs 合计） |
|---:|---:|---:|---:|---:|---:|
| 0 GiB | 351.019 | 884.923 | 173.064 | 3779.530 | 151.728 |
| 50 GiB | 285.686 | 698.849 | 121.193 | 3328.934 | 192.061 |
| 250 GiB | 169.530 | 364.220 | 63.796 | 2454.638 | 303.502 |
| 无限 | 135.953 | 273.838 | 51.438 | 2454.638 | 364.798 |

这组高 TTFT 包含压缩到达时间下的明显排队；增加容量的收益不能直接外推到不同负载压力的生产服务。Validation 通过说明本模拟的 token、事件、容量与指标账本一致，不证明 AIC 已在该 workload 上达到真实硬件误差目标。

## 6. 长上下文与迭代边界

AIC 的 KDA 表可查询当前 1024/8192-token chunk 和 B1/B8/B32 decode，但 MLA 需要完整 past 长度。B300/SGLang BF16 MLA 的 12 个本地 heads 从 8/16-head 表切片插值，关键长度边界为：

- Prefill 表在 B1/B2 下最多完整序列 32,768；B4/B8/B16/B32 上限分别为 16,384/8,192/4,096/2,048。AIC 先查询 `past + new`，再做 prefix correction。长 prompt 切成小 chunk 仍可能超出测量长度范围。
- Decode 表在 B1/B8/B32 下最多序列长度 131,072；一次 decode 的查询长度是 `past + 1`。64K 通常位于范围内，256K 与接近 1M 需要明显外推。
- Native `silicon` 标签也包含插值/外推，不能据该标签认定长上下文测过。纯 `ANALYTICAL` Rust 路径在此 AIC checkout 尚未实现 KDA，不能作为完整 Kimi-K3 的替代入口。
- Predictor 会保留真实请求长度和聚合输入。异构 prefill 的 MLA 显式使用工作量修正，并保留 native 时延和修正因子；这是 batch 近似，不是拟合吞吐系数。总上下文超过模型上限 1,048,576 会拒绝。

因此真实 64K–200K AgentX 样本可用于**明确标注长上下文外推的容量敏感性仿真**，不能由本结果宣称该长度上的 GPU 实测或生产 SLA 已达标。接近 1M 的请求应独立报告外推和显存约束；选择过滤上限与模型可执行上限也应分开。

迭代时以新 `K3_RUN` 保存每组证据，并将其参数写入命令记录：

- 改 checkpoint interval：新目录重新 `prepare,scan,bench,plot`，例如 `--checkpoint-tokens 2048`；chunk 大小用默认 0 跟随 interval，或显式设置相容值。布局字节、前缀粒度、G1 snapshots 与拐点都会改变。
- 改带宽：同样在新目录生成一致输入后运行，显式改变 `--read-gib-s` 等参数。它们始终是全 TP replica 聚合带宽假设；对照 source/layout/trace hash 一致后再比较。
- 改到达压力：`prepare --time-scale 20` 比基线 10 更压缩；`bench --bench-arrival-scale 2` 将 prepared 到达间隔拉长两倍。二者含义相反，应显式记录，不能混作同一个参数。
- 改输入样本：改变 session 数、每 session 请求数、时间窗口或长度过滤后，重新扫描；250/525/675 GiB 是本基线结果，不能固定套用新 trace。
- 查更多容量：保留现有 Facts，直接运行 query 工具；需要新增 E2E 点时在新实验目录使用 `--capacities 0,50,250,525,675,infinite`。
- 研究可持续并发：先定义 TTFT/TPOT SLA、稳定到达过程、观测时长与统计方法，再扩大 admission/arrival 探针；有队列峰值或少量并发对比不足以回答稳定极限。

同一 output 目录中的同名场景会被重写。基线目录只读复查；参数或代码改变后使用新目录，并对照 `source_manifest.json`、trace/layout hash、resolved config、AIC provenance、validation 与最终指标一起解释差异。

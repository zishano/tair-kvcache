# Kimi-K3 完整工作流脚本使用说明

本文档说明如何使用 `run_full_workflow.sh` 一次性执行完整的 Kimi-K3 容量粗筛与性能精算流程。

## 概述

`run_full_workflow.sh` 脚本基于 [SOP.md](SOP.md) 的 **"all: 一次执行完整流程"** 部分，自动执行以下四个阶段：

1. **prepare** - 生成 trace、layout 和 LiteHit 配置
2. **scan** - 单遍 Facts 与全部容量投影
3. **bench** - 注入容量并运行 HiSim 仿真
4. **plot** - 生成图表与报告

## 快速开始

### 1. 使用默认配置（最简单）

```bash
cd /mnt/nvme1n1/data/lmk/Github/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_test

# 直接运行（使用所有默认值）
./run_full_workflow.sh
```

默认输出目录：`$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test/output/full_run`

### 2. 自定义输出目录

```bash
# 方式 1: 通过环境变量
K3_RUN=/workspace/my_experiments/run_01 ./run_full_workflow.sh

# 方式 2: 先导出再运行
export K3_RUN=/workspace/my_experiments/run_01
./run_full_workflow.sh
```

### 3. 自定义多个参数

```bash
# 修改时间压缩比例和最大并发数
K3_RUN=/workspace/run_02 \
TIME_SCALE=5 \
MAX_CONCURRENCY=64 \
./run_full_workflow.sh
```

### 4. 跳过确认提示（自动化场景）

```bash
# 当输出目录已存在时自动覆盖，不提示确认
K3_ASSUME_YES=true ./run_full_workflow.sh
```

## 环境变量配置

### 必需配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `K3_RUN` | `$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test/output/full_run` | 输出目录（推荐自定义） |

### 基础路径（通常使用默认值）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `K3_REPO` | `/workspace/tair-kvcache` | 代码仓库根目录 |
| `K3_PY` | `$K3_REPO/.venv/bin/python` | Python 解释器路径 |
| `K3_WORKFLOW` | `$K3_REPO/.../run_workflow.py` | Workflow 脚本路径 |
| `K3_AIC_ROOT` | `/workspace/dynosim_aic_sweep/aiconfigurator_analytical_sync` | AIC 根目录 |
| `K3_MODEL` | `$K3_AIC_ROOT/.../Kimi-K3_config.json` | 模型配置文件 |
| `K3_TRACE` | `/workspace/dynosim_aic_sweep/traces.jsonl` | Trace 输入文件 |
| `K3_BASELINE` | `/workspace/.../agentx_sample` | 基线对比目录 |

### Trace 采样参数

| 变量 | 默认值 | 说明 | SOP 对应 |
|------|--------|------|----------|
| `AGENTX_SESSIONS` | `8` | AgentX session 数量 | `--agentx-sessions 8` |
| `REQUESTS_PER_SESSION` | `24` | 每个 session 最多请求数 | `--requests-per-session 24` |
| `AGENTX_WINDOW_SECONDS` | `3600` | 时间窗口（秒） | `--agentx-window-seconds 3600` |
| `MAX_INPUT_TOKENS` | `262144` | 输入 token 上限 | `--max-input-tokens 262144` |
| `TIME_SCALE` | `10` | 时间压缩比例 | `--time-scale 10` |

### 模型和缓存配置

| 变量 | 默认值 | 说明 | SOP 对应 |
|------|--------|------|----------|
| `TP_SIZE` | `8` | Tensor Parallel 大小 | `--tp-size 8` |
| `CHECKPOINT_TOKENS` | `1024` | Checkpoint 间隔 tokens | `--checkpoint-tokens 1024` |

### HiSim 仿真参数

| 变量 | 默认值 | 说明 | SOP 对应 |
|------|--------|------|----------|
| `MAX_CONCURRENCY` | `32` | 最大并发数 | `--max-concurrency 32` |
| `MAX_PREFILL_TOKENS` | `8192` | 每次 forward 的总 token 预算 | `--max-prefill-tokens 8192` |
| `CHUNKED_PREFILL_TOKENS` | `1024` | Chunk 大小 | `--chunked-prefill-tokens 1024` |
| `MEMORY_FRACTION` | `0.90` | 显存使用比例 | `--memory-fraction 0.90` |

### 存储带宽配置（GiB/s）

| 变量 | 默认值 | 说明 | SOP 对应 |
|------|--------|------|----------|
| `READ_GIB_S` | `64` | Storage read 带宽 | `--read-gib-s 64` |
| `WRITE_GIB_S` | `32` | Storage write 带宽 | `--write-gib-s 32` |
| `H2D_GIB_S` | `128` | Host to Device 带宽 | `--h2d-gib-s 128` |
| `D2H_GIB_S` | `128` | Device to Host 带宽 | `--d2h-gib-s 128` |

### Bench 参数

| 变量 | 默认值 | 说明 | SOP 对应 |
|------|--------|------|----------|
| `BENCH_ARRIVAL_SCALE` | `1` | 到达时间缩放 | `--bench-arrival-scale 1` |
| `PROBE_REQUESTS` | `16` | 预探针请求数 | `--probe-requests 16` |
| `CAPACITIES` | `0,50,250,infinite` | 容量点选择 | 默认 Stage 2 选择 |

### 可选功能：并发探针

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RUN_CONCURRENCY_PROBES` | `false` | 是否运行并发探针 |
| `CONCURRENCY_PROBES` | `8,16` | 并发探针点（仅当上面为 true 时生效） |

### 其他控制变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `K3_ASSUME_YES` | `false` | 跳过确认提示，自动继续 |

## 使用示例

### 示例 1: 复现基线结果

```bash
# 使用与基线相同的参数
K3_RUN=/workspace/my_experiments/baseline_rerun \
./run_full_workflow.sh
```

### 示例 2: 修改 checkpoint interval

```bash
# 测试 2048 tokens checkpoint
K3_RUN=/workspace/experiments/checkpoint_2048 \
CHECKPOINT_TOKENS=2048 \
CHUNKED_PREFILL_TOKENS=2048 \
./run_full_workflow.sh
```

### 示例 3: 修改存储带宽

```bash
# 测试更高的读写带宽
K3_RUN=/workspace/experiments/high_bandwidth \
READ_GIB_S=128 \
WRITE_GIB_S=64 \
./run_full_workflow.sh
```

### 示例 4: 修改到达压力

```bash
# 更压缩的到达时间（5 倍压缩）
K3_RUN=/workspace/experiments/high_pressure \
TIME_SCALE=5 \
./run_full_workflow.sh
```

### 示例 5: 测试不同容量点

```bash
# 增加更多容量点
K3_RUN=/workspace/experiments/more_capacities \
CAPACITIES="0,50,100,150,200,250,300,infinite" \
./run_full_workflow.sh
```

### 示例 6: 完整自定义 + 并发探针

```bash
K3_RUN=/workspace/experiments/custom_full \
TIME_SCALE=10 \
MAX_CONCURRENCY=64 \
CHECKPOINT_TOKENS=1024 \
READ_GIB_S=100 \
WRITE_GIB_S=50 \
CAPACITIES="0,50,250,500,infinite" \
RUN_CONCURRENCY_PROBES=true \
CONCURRENCY_PROBES="16,32,48" \
K3_ASSUME_YES=true \
./run_full_workflow.sh
```

### 示例 7: 批量运行多个实验

```bash
#!/bin/bash
# batch_experiments.sh

# 实验 1: 基线
K3_RUN=/workspace/batch/exp_baseline \
K3_ASSUME_YES=true \
./run_full_workflow.sh

# 实验 2: 高带宽
K3_RUN=/workspace/batch/exp_high_bw \
READ_GIB_S=128 \
WRITE_GIB_S=64 \
K3_ASSUME_YES=true \
./run_full_workflow.sh

# 实验 3: 大 checkpoint
K3_RUN=/workspace/batch/exp_ckpt_2048 \
CHECKPOINT_TOKENS=2048 \
CHUNKED_PREFILL_TOKENS=2048 \
K3_ASSUME_YES=true \
./run_full_workflow.sh
```

## 输出结构

成功执行后，输出目录结构如下：

```
$K3_RUN/
├── trace_manifest.json          # Trace 来源和筛选信息
├── layout.json                  # 缓存布局配置
├── litehit_config.json         # LiteHit 配置
├── trace.jsonl                 # 处理后的 trace
├── source_manifest.json        # 源码和二进制 hash
├── stage1/                     # Stage 1: 容量粗筛
│   ├── litehit_facts.csv      # Facts（容量无关）
│   ├── capacity_query.jsonl   # 容量查询结果
│   ├── capacity_curve.csv     # 容量曲线数据
│   └── summary.json           # 关键点（elbow、95%）
├── stage2/                     # Stage 2: HiSim 精算
│   ├── configs/               # 各容量点的配置
│   │   ├── capacity_0gib.json
│   │   ├── capacity_50gib.json
│   │   ├── capacity_250gib.json
│   │   └── capacity_infinite.json
│   ├── capacity_0gib/         # 0 GiB 容量仿真结果
│   │   ├── request.jsonl
│   │   ├── iteration.jsonl
│   │   ├── storage.jsonl
│   │   ├── metrics.json
│   │   └── validation.json
│   ├── capacity_50gib/        # 50 GiB 容量仿真结果
│   ├── capacity_250gib/       # 250 GiB 容量仿真结果
│   ├── capacity_infinite/     # 无限容量仿真结果
│   ├── summary.csv            # 场景比较表
│   └── summary.json           # 汇总指标
└── report/                     # 图表和报告
    ├── capacity_curve.png     # 容量曲线图
    ├── capacity_curve.svg
    ├── capacity_curve.pdf
    ├── *.html                 # HTML 报告
    └── ...                    # 其他图表
```

## 关键输出文件说明

### Stage 1 关键文件

| 文件 | 说明 |
|------|------|
| `stage1/summary.json` | 包含 elbow、95% 点、无限容量命中率等关键指标 |
| `stage1/capacity_curve.csv` | 容量曲线完整数据（CSV 格式） |
| `stage1/litehit_facts.csv` | Facts 文件，可用于后续快速投影 |

### Stage 2 关键文件

| 文件 | 说明 |
|------|------|
| `stage2/summary.csv` | 各容量点的性能对比表（TTFT、TPOT、吞吐等） |
| `stage2/summary.json` | 汇总指标（JSON 格式） |
| `stage2/*/metrics.json` | 每个容量点的详细指标 |
| `stage2/*/validation.json` | 验证结果（必须 passed=true） |

### 报告文件

| 文件 | 说明 |
|------|------|
| `report/*.png` | 高分辨率位图（DPI 180） |
| `report/*.svg` | 矢量图（可无损缩放） |
| `report/*.pdf` | PDF 格式（适合论文） |
| `report/*.html` | 交互式 HTML 报告 |

## 运行时间估计

根据基线数据（AgentX 8 sessions × 24 requests，163 条请求）：

| 阶段 | 时间 | 说明 |
|------|------|------|
| prepare | < 1 秒 | 生成配置文件 |
| scan | < 1 秒 | Facts 生成 + 43 点查询 |
| bench | 10-60 分钟 | HiSim 仿真（取决于容量点数量） |
| plot | < 10 秒 | 生成图表 |
| **总计** | **10-60 分钟** | 主要时间在 bench 阶段 |

**注意**: 
- 实际运行时间取决于 trace 大小、请求数量、容量点数量
- bench 阶段是最耗时的部分
- 使用 `K3_ASSUME_YES=true` 可以避免交互等待

## 常见问题

### Q1: 如何只运行部分阶段？

使用原始的 `run_workflow.py` 脚本，指定 `--steps`：

```bash
# 只运行 prepare + scan
"$K3_PY" "$K3_WORKFLOW" \
  --steps prepare,scan \
  --output-dir "$K3_RUN" \
  ...
```

### Q2: 如何重新绘图而不重跑仿真？

```bash
# 只运行 plot
"$K3_PY" "$K3_WORKFLOW" \
  --steps plot \
  --output-dir "$K3_RUN"
```

### Q3: 如何验证输出结果？

检查关键文件：

```bash
# 检查 Stage 1 是否成功
cat "$K3_RUN/stage1/summary.json"

# 检查 Stage 2 是否成功
cat "$K3_RUN/stage2/summary.json"

# 检查验证是否通过
jq '.passed' "$K3_RUN/stage2/*/validation.json"

# 查看图表
ls -lh "$K3_RUN/report/"
```

### Q4: 如何对比两次运行的结果？

```bash
# 对比 Stage 1 关键点
diff <(jq -S . run1/stage1/summary.json) \
     <(jq -S . run2/stage1/summary.json)

# 对比 Stage 2 指标
diff run1/stage2/summary.csv run2/stage2/summary.csv
```

### Q5: 脚本执行失败怎么办？

1. 检查环境变量是否正确
2. 检查输入文件是否存在
3. 查看脚本输出的错误信息
4. 检查 Python 环境和依赖
5. 查看详细日志：`$K3_RUN/stage*/*/logs/`

### Q6: 如何修改更多高级参数？

对于脚本中未暴露的参数，直接编辑脚本或使用原始的 `run_workflow.py`：

```bash
"$K3_PY" "$K3_WORKFLOW" --help  # 查看所有可用参数
```

## 与 SOP.md 的对应关系

本脚本严格遵循 [SOP.md](SOP.md) 第 3.4 节 "all: 一次执行完整流程" 的命令：

```bash
# SOP.md 原始命令（第 197-208 行）
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

本脚本将所有参数都提取为环境变量，方便自定义和批量实验。

## 参考资料

- [SOP.md](SOP.md) - 完整的标准操作流程
- [README.md](README.md) - 测试工具的技术文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [INDEX.md](INDEX.md) - 文件索引

## 更新日志

- 2024-09-14: 初始版本，基于 SOP.md 创建

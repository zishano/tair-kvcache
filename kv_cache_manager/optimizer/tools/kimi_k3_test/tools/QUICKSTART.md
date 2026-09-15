# 快速开始指南

## 一、使用现有基线数据（最简单）

如果你已经有 `run_workflow.py` 生成的输出（包含 `stage1/summary.json`），可以直接绘图：

```bash
cd /mnt/nvme1n1/data/lmk/Github/tair-kvcache

# 使用基线数据
./kv_cache_manager/optimizer/tools/kimi_k3_test/run_example.sh \
    /workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample \
    ./test_output
```

输出：`./test_output/capacity_curve_axes0.{png,svg,pdf}`

## 二、从头计算（需要 LiteHit 工具）

### 步骤 1: 准备输入数据

首先运行原始 `run_workflow.py` 的 `prepare` 阶段：

```bash
K3_REPO=/workspace/tair-kvcache
K3_PY="$K3_REPO/.venv/bin/python"
K3_WORKFLOW="$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_capacity/run_workflow.py"

"$K3_PY" "$K3_WORKFLOW" \
  --steps prepare \
  --output-dir /path/to/prepared \
  --trace /path/to/traces.jsonl \
  --model-config /path/to/Kimi-K3_config.json \
  --agentx-sessions 8 \
  --requests-per-session 24 \
  --time-scale 10 \
  --tp-size 8 \
  --checkpoint-tokens 1024
```

这会生成：
- `trace.jsonl`
- `layout.json`
- `trace_manifest.json`
- `litehit_config.json`

### 步骤 2: 使用测试工具

```bash
./kv_cache_manager/optimizer/tools/kimi_k3_test/run_example.sh \
    /path/to/prepared \
    ./test_output
```

## 三、手动运行（分步）

### 方法 A：使用现有 stage1 数据

```bash
K3_PY=/workspace/tair-kvcache/.venv/bin/python
K3_TEST=/mnt/nvme1n1/data/lmk/Github/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_test

PREPARED=/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample
OUTPUT=./my_output

"$K3_PY" "$K3_TEST/plot_capacity_curve_axes0.py" \
  --capacity-data "$PREPARED/stage1/summary.json" \
  --trace-manifest "$PREPARED/trace_manifest.json" \
  --layout "$PREPARED/layout.json" \
  --output-dir "$OUTPUT"
```

### 方法 B：完整计算

```bash
K3_PY=/workspace/tair-kvcache/.venv/bin/python
K3_TEST=/mnt/nvme1n1/data/lmk/Github/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_test

PREPARED=/path/to/prepared
OUTPUT=./my_output

# 步骤 1: 计算容量数据
"$K3_PY" "$K3_TEST/compute_capacity_data.py" \
  --trace "$PREPARED/trace.jsonl" \
  --layout "$PREPARED/layout.json" \
  --litehit-config "$PREPARED/litehit_config.json" \
  --output-dir "$OUTPUT"

# 步骤 2: 绘制图表
"$K3_PY" "$K3_TEST/plot_capacity_curve_axes0.py" \
  --capacity-data "$OUTPUT/capacity_data.json" \
  --trace-manifest "$PREPARED/trace_manifest.json" \
  --layout "$PREPARED/layout.json" \
  --output-dir "$OUTPUT"
```

## 四、查看输出

### 生成的文件

**使用现有数据时**：
```
<output_dir>/
└── capacity_curve_axes0.{png,svg,pdf}
```

**完整计算时**：
```
<output_dir>/
├── capacity_data.json           # 容量曲线数据
├── capacity_curve.csv           # CSV 表格
├── capacity_query.jsonl         # 查询结果
├── litehit_facts.csv           # Facts 文件
└── capacity_curve_axes0.{png,svg,pdf}  # 图表
```

### 查看图表

```bash
# Linux
xdg-open ./test_output/capacity_curve_axes0.png

# macOS
open ./test_output/capacity_curve_axes0.png
```

## 五、常见问题

### Q1: 找不到 LiteHit 工具

**错误**：`lite_hit_main` 或 `lite_hit_facts_query_main` 找不到

**解决**：编译 LiteHit 工具
```bash
cd /workspace/tair-kvcache
bazel build --jobs=4 \
  //kv_cache_manager/optimizer:lite_hit_main \
  //kv_cache_manager/optimizer:lite_hit_facts_query_main
```

### Q2: 没有 prepare 阶段的输出

**解决**：使用现有基线数据
```bash
./kv_cache_manager/optimizer/tools/kimi_k3_test/run_example.sh \
    /workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample
```

### Q3: 想要修改绘图样式

编辑 `plot_capacity_curve_axes0.py` 中的样式参数：
- 第 40-47 行：字体、网格等
- 第 50 行：颜色调色板
- 第 53 行：图表尺寸

## 六、与原始工具的对比

| 功能 | 原始工具 | 测试工具 |
|------|---------|---------|
| 计算容量曲线 | ✅ `run_workflow.py --steps scan` | ✅ `compute_capacity_data.py` |
| 绘制 axes[0] | ✅ `analyze_results.py` | ✅ `plot_capacity_curve_axes0.py` |
| 绘制 axes[1] | ✅ | ❌ |
| HiSim 仿真 | ✅ `run_workflow.py --steps bench` | ❌ |
| 完整报告 | ✅ `analyze_results.py` | ❌ |
| 代码行数 | ~600 行 | ~250 行 |

**使用场景**：
- **原始工具**：完整的容量分析和服务指标仿真
- **测试工具**：快速验证容量曲线，只需要 axes[0] 图表

## 七、示例输出

运行成功后会显示：

```
==========================================
Kimi-K3 Capacity Curve Test (axes[0] only)
==========================================

Prepared input: /workspace/.../agentx_sample
Output directory: ./test_output

Found existing stage1 data, using it directly for plotting...

Step: Plotting capacity curve (axes[0])...
Loading data...
Generating capacity curve plot (axes[0] only)...
Saved: ./test_output/capacity_curve_axes0.png
Saved: ./test_output/capacity_curve_axes0.svg
Saved: ./test_output/capacity_curve_axes0.pdf

✓ Plot saved to: ./test_output

==========================================
SUCCESS! Using existing stage1 data
==========================================
Output: ./test_output/capacity_curve_axes0.{png,svg,pdf}
```

## 八、下一步

- 查看完整文档：[README.md](README.md)
- 了解原始工具：[../kimi_k3_capacity/SOP.md](../kimi_k3_capacity/SOP.md)
- 修改绘图样式：编辑 `plot_capacity_curve_axes0.py`
- 分析数据：查看 `capacity_data.json` 和 `capacity_curve.csv`

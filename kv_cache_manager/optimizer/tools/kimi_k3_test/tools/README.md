# Kimi-K3 容量曲线测试工具

这是一个精简版的容量曲线计算和绘图工具，只包含**绘制 axes[0] 图表**所需的核心功能。

## 文件说明

- **`compute_capacity_data.py`** - 计算容量曲线数据（从 `run_workflow.py` 的 `scan()` 函数提取）
- **`plot_capacity_curve_axes0.py`** - 绘制 axes[0] 容量曲线图（从 `analyze_results.py` 的 `plot_and_report()` 函数提取）
- **`README.md`** - 本文件

## 功能说明

### 1. compute_capacity_data.py

从 `run_workflow.py` 的第 330-407 行 `scan()` 函数提取，包含：

- 运行 `lite_hit_main` 生成 Facts
- 运行 `lite_hit_facts_query_main` 查询 43 个容量点
- 独立 LRU 缓存验证每个容量点
- 计算 elbow（几何拐点）和 95% 点
- 保存容量曲线数据到 JSON

**输入**：
- `--trace`: trace.jsonl 文件
- `--layout`: layout.json 文件
- `--litehit-config`: litehit_config.json 文件
- `--output-dir`: 输出目录

**输出**：
- `capacity_data.json` - 容量曲线数据
- `capacity_curve.csv` - 容量曲线 CSV 表格
- `litehit_facts.csv` - LiteHit Facts 文件
- `capacity_query.jsonl` - 容量查询结果

### 2. plot_capacity_curve_axes0.py

从 `analyze_results.py` 的第 380-395 行提取，只绘制 axes[0] 图表：

```python
axes[0].set(ylabel="Input-token hit rate (%)", 
            ylim=(0, 100), 
            title="Kimi-K3: capacity screening from one Facts replay")
```

**输入**：
- `--capacity-data`: capacity_data.json（由 compute_capacity_data.py 生成）
- `--trace-manifest`: trace_manifest.json
- `--layout`: layout.json
- `--output-dir`: 输出目录

**输出**：
- `capacity_curve_axes0.png` - PNG 格式
- `capacity_curve_axes0.svg` - SVG 格式
- `capacity_curve_axes0.pdf` - PDF 格式

## 使用方法

### 前置条件

1. 已经构建了 LiteHit 工具：
```bash
cd /workspace/tair-kvcache
bazel build --jobs=4 \
  //kv_cache_manager/optimizer:lite_hit_main \
  //kv_cache_manager/optimizer:lite_hit_facts_query_main
```

2. 已经运行了 `run_workflow.py` 的 `prepare` 阶段，生成了：
   - `trace.jsonl`
   - `layout.json`
   - `trace_manifest.json`
   - `litehit_config.json`

### 完整流程

```bash
# 设置环境变量
K3_REPO=/workspace/tair-kvcache
K3_PY="$K3_REPO/.venv/bin/python"
K3_TEST="$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test"

# 假设你已经有 prepare 阶段的输出
PREPARED_DIR=/path/to/prepared/output
OUTPUT_DIR=/path/to/test/output

# 步骤 1: 计算容量曲线数据
"$K3_PY" "$K3_TEST/compute_capacity_data.py" \
  --trace "$PREPARED_DIR/trace.jsonl" \
  --layout "$PREPARED_DIR/layout.json" \
  --litehit-config "$PREPARED_DIR/litehit_config.json" \
  --output-dir "$OUTPUT_DIR"

# 步骤 2: 绘制 axes[0] 容量曲线图
"$K3_PY" "$K3_TEST/plot_capacity_curve_axes0.py" \
  --capacity-data "$OUTPUT_DIR/capacity_data.json" \
  --trace-manifest "$PREPARED_DIR/trace_manifest.json" \
  --layout "$PREPARED_DIR/layout.json" \
  --output-dir "$OUTPUT_DIR"
```

### 使用现有的基线数据

如果你想使用已经存在的基线数据：

```bash
K3_BASELINE=/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample

# 使用基线的 stage1 数据重新绘图
"$K3_PY" "$K3_TEST/plot_capacity_curve_axes0.py" \
  --capacity-data "$K3_BASELINE/stage1/summary.json" \
  --trace-manifest "$K3_BASELINE/trace_manifest.json" \
  --layout "$K3_BASELINE/layout.json" \
  --output-dir ./output_test
```

注意：基线的 `stage1/summary.json` 包含与 `capacity_data.json` 相同的字段。

## 数据格式

### capacity_data.json

```json
{
  "capacity_gb": [0, 25, 50, ..., 1000, 1024, -1],
  "hit_rates": [0.0, 0.452, 0.678, ..., 0.985],
  "hit_rates_percent": [0.0, 45.2, 67.8, ..., 98.5],
  "elbow_capacity_gib": 250.0,
  "capacity_at_95pct_infinite_gib": 500.0,
  "total_input_tokens": 10473344,
  "total_hit_tokens": [0, 4734567, ...],
  "replay_wall_s": 2.456,
  "query_wall_s": 1.234,
  "facts_sha256": "abc123...",
  "validation": {
    "independent_lru_checks": 7052,
    "mismatches": 0
  }
}
```

### 绘图数据来源

绘制 axes[0] 图表所需的数据：

1. **容量列表** (`capacity_gb[:-1]`): 0, 25, 50, ..., 1000, 1024 GiB
2. **命中率** (`hit_rates_percent[:-1]`): 各容量点的命中率百分比
3. **无限容量命中率** (`hit_rates_percent[-1]`): 无限容量时的命中率
4. **Elbow 点** (`elbow_capacity_gib`): 几何拐点容量
5. **95% 点** (`capacity_at_95pct_infinite_gib`): 达到无限容量命中率 95% 的容量
6. **Plateau 点**: 曲线达到平台的容量（如果存在）

## 与原始代码的对应关系

### compute_capacity_data.py ← run_workflow.py

| 功能 | 原始位置 | 提取位置 |
|------|---------|---------|
| 运行 LiteHit replay | `run_workflow.py:338` | `compute_capacity_data.py:78-83` |
| 查询容量点 | `run_workflow.py:339-342` | `compute_capacity_data.py:86-93` |
| 独立 LRU 验证 | `run_workflow.py:356-375` | `compute_capacity_data.py:106-126` |
| 计算 elbow | `run_workflow.py:380-382` | `compute_capacity_data.py:131-133` |
| 保存 CSV | `run_workflow.py:385-393` | `compute_capacity_data.py:136-149` |

### plot_capacity_curve_axes0.py ← analyze_results.py

| 功能 | 原始位置 | 提取位置 |
|------|---------|---------|
| 绘制容量曲线 | `analyze_results.py:382` | `plot_capacity_curve_axes0.py:58-60` |
| 绘制无限容量线 | `analyze_results.py:383` | `plot_capacity_curve_axes0.py:63-65` |
| 标注关键点 | `analyze_results.py:384-393` | `plot_capacity_curve_axes0.py:68-86` |
| 设置标签和标题 | `analyze_results.py:394` | `plot_capacity_curve_axes0.py:89-93` |
| 添加图例 | `analyze_results.py:395` | `plot_capacity_curve_axes0.py:96` |

## 关键算法

### Elbow 计算（几何拐点）

```python
# 归一化收益曲线的几何方法
finite = list(zip(capacities[:-1], rates[:-1]))
gain = finite[-1][1] - finite[0][1]
knee = max(finite, key=lambda p: 
    ((p[1] - finite[0][1]) / gain if gain else 0) - p[0] / finite[-1][0]
)[0]
```

含义：在 0-1024 GiB 范围内，找到 `(归一化命中率增益 - 归一化容量)` 最大的点。

### 95% 点计算

```python
cap95 = next((c for c, h in finite if h >= 0.95 * rates[-1]), None)
```

含义：第一个达到无限容量命中率 95% 的容量点。

## 输出示例

运行后会看到：

```
Running LiteHit Facts replay...
Querying 43 capacity points...
Validating with independent LRU cache...
✓ Facts replay: 2.456s
✓ Query 43 capacities: 1.234s
✓ Elbow capacity: 250 GiB
✓ 95% infinite capacity: 500.0 GiB
✓ Independent LRU checks: 7052

Capacity curve data saved to: /path/to/output/capacity_data.json

Loading data...
Generating capacity curve plot (axes[0] only)...
Saved: /path/to/output/capacity_curve_axes0.png
Saved: /path/to/output/capacity_curve_axes0.svg
Saved: /path/to/output/capacity_curve_axes0.pdf

✓ Plot saved to: /path/to/output
```

## 依赖

- Python 3.8+
- numpy
- matplotlib
- LiteHit 工具（需要编译）

## 限制

- **只计算和绘制 axes[0] 图表**，不包括：
  - axes[1]（边际收益柱状图）
  - HiSim 仿真（bench 阶段）
  - 服务指标绘图（serving_metrics、resources 等）
  - 完整报告生成

- **需要独立运行 prepare 阶段**，本工具只替代 scan 和 plot 的一部分。

## 参考

- 完整工作流：`/workspace/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_capacity/run_workflow.py`
- 完整分析：`/workspace/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_capacity/analyze_results.py`
- 详细文档：`/workspace/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_capacity/SOP.md`

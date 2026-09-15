# 容量曲线绘图工具

本目录包含用于提取和绘制 Kimi-K3 容量曲线的独立工具。

## 文件说明

### 核心工作流
- **`run_workflow.py`** - 主工作流程脚本，执行完整的容量分析流程（prepare → scan → bench → plot）
- **`analyze_results.py`** - 结果验证和完整报告生成

### 数据提取和绘图工具
- **`extract_plot_data.py`** - 从 `run_workflow.py` 输出中提取绘图所需数据
- **`plot_from_extracted_data.py`** - 使用提取的数据生成容量曲线图
- **`plot_capacity_curve.py`** - 直接从原始输出目录绘制容量曲线（旧版本）

### 文档
- **`SOP.md`** - 详细的标准操作流程文档

## 使用方法

### 方法一：完整工作流（推荐）

按照 `SOP.md` 中的说明运行完整流程：

```bash
# 1. 设置环境变量
K3_REPO=/workspace/tair-kvcache
K3_PY="$K3_REPO/.venv/bin/python"
K3_WORKFLOW="$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_capacity/run_workflow.py"
K3_RUN=/path/to/output/directory

# 2. 运行完整流程（prepare → scan → bench → plot）
"$K3_PY" "$K3_WORKFLOW" \
  --steps all \
  --output-dir "$K3_RUN" \
  --trace /path/to/trace.jsonl \
  --model-config /path/to/model/config.json \
  ...其他参数...
```

### 方法二：仅提取和绘制容量曲线

如果您已经运行了 `prepare` 和 `scan` 阶段，可以使用新工具单独提取数据并绘图：

```bash
# 步骤 1: 提取绘图数据
python extract_plot_data.py /path/to/output/directory

# 这将生成: /path/to/output/directory/plot_data/capacity_curve_data.json

# 步骤 2: 使用提取的数据绘图
python plot_from_extracted_data.py /path/to/output/directory/plot_data/capacity_curve_data.json

# 这将生成图表:
#   - capacity_curve.png
#   - capacity_curve.svg
#   - capacity_curve.pdf
```

### 方法三：直接从输出目录绘图（旧版本）

```bash
python plot_capacity_curve.py /path/to/output/directory
```

## 输入数据结构

### 对于 `extract_plot_data.py`

需要以下文件（由 `run_workflow.py` 的 prepare 和 scan 阶段生成）：

```
<output_directory>/
├── stage1/
│   └── summary.json          # Stage1 容量分析结果
├── trace_manifest.json        # 追踪清单
└── layout.json                # 布局配置
```

### 对于 `plot_from_extracted_data.py`

需要提取的数据文件：

```
<output_directory>/plot_data/
└── capacity_curve_data.json   # 由 extract_plot_data.py 生成
```

## 输出说明

### `extract_plot_data.py` 输出

生成的 `capacity_curve_data.json` 包含：

```json
{
  "capacities_gib": [0, 25, 50, ...],
  "hit_rates_percent": [0.0, 45.2, 67.8, ...],
  "infinite_hit_rate_percent": 98.5,
  "elbow_capacity_gib": 250.0,
  "capacity_at_95pct_infinite_gib": 500.0,
  "plateau_capacity_gib": 750.0,
  "checkpoint_tokens": 1024,
  "num_requests": 163,
  "workload_type": "AgentX-derived",
  "input_tokens": 10473344,
  "output_tokens": 184322,
  "bundle_bytes": 675864576,
  "attention_tp_size": 8,
  "metadata": { ... }
}
```

### `plot_from_extracted_data.py` 输出

生成三种格式的容量曲线图：
- `capacity_curve.png` - 高分辨率 PNG（DPI 180）
- `capacity_curve.svg` - 矢量图（可缩放）
- `capacity_curve.pdf` - PDF 格式（适合论文）

图表包含：
1. **上图**：容量 vs 命中率曲线，标注关键点（elbow、95% 点、plateau）
2. **下图**：边际收益柱状图（每 GiB 增加的命中率百分点）

## 数据来源

绘图数据来自 `run_workflow.py` 中 `scan()` 函数的处理：

1. **prepare 阶段**：生成 trace、layout 和配置文件
2. **scan 阶段**：
   - 使用 `lite_hit_main` 生成 Facts
   - 使用 `lite_hit_facts_query_main` 查询 43 个容量点
   - 计算 elbow（几何拐点）和 95% 点
   - 独立 LRU 验证每个请求和容量

关键数据字段：
- `capacity_gb`: 容量列表（0, 25, 50, ..., 1000, 1024, infinite）
- `hit_rates_exact_from_integer_counts`: 精确命中率（基于整数 token 计数）
- `elbow_capacity_gib`: 几何拐点容量（归一化收益最大点）
- `capacity_at_95pct_infinite_hit_rate_gib`: 达到无限容量命中率 95% 的容量

## 注意事项

1. **容量单位**：所有容量均为 GiB payload（2^30 bytes），包含所有 TP rank 的 tensor payload
2. **命中率**：基于 input tokens 的前缀命中率，不包括 output tokens
3. **Elbow 计算**：在 0-1024 GiB 范围内，基于归一化收益曲线的几何方法
4. **独立验证**：每个容量点都通过独立的 LRU 缓存模拟验证，不依赖 LiteHit 的内部公式

## 示例：基线数据位置

```bash
# 基线报告目录
K3_BASELINE=/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample

# 提取基线数据
python extract_plot_data.py "$K3_BASELINE"

# 重新绘制基线图表
python plot_from_extracted_data.py "$K3_BASELINE/plot_data/capacity_curve_data.json"
```

## 依赖

- Python 3.8+
- numpy
- matplotlib

安装依赖：
```bash
pip install numpy matplotlib
```

## 参考

- 完整文档：[SOP.md](SOP.md)
- 主工作流：[run_workflow.py](run_workflow.py)
- 结果分析：[analyze_results.py](analyze_results.py)

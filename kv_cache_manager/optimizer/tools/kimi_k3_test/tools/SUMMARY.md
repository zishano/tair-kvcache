# 项目总结

## 已完成的工作

成功从 `run_workflow.py` 和 `analyze_results.py` 中提取了**只绘制 axes[0] 容量曲线图**的精简版代码，保存到新目录：

```
/mnt/nvme1n1/data/lmk/Github/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_test/
```

## 创建的核心文件

### 1. **compute_capacity_data.py** (主要工作)
- **来源**: `run_workflow.py` 的 `scan()` 函数（第 330-407 行）
- **功能**: 计算容量曲线数据
- **包含**:
  - 运行 `lite_hit_main` 生成 Facts
  - 运行 `lite_hit_facts_query_main` 查询 43 个容量点
  - 独立 LRU 缓存验证
  - 计算 elbow（几何拐点）和 95% 点
- **输出**: `capacity_data.json`, `capacity_curve.csv`

### 2. **plot_capacity_curve_axes0.py** (主要工作)
- **来源**: `analyze_results.py` 的 `plot_and_report()` 函数（第 380-395 行）
- **功能**: 只绘制 axes[0] 图表
- **图表内容**:
  ```python
  axes[0].set(ylabel="Input-token hit rate (%)", 
              ylim=(0, 100), 
              title="Kimi-K3: capacity screening from one Facts replay")
  ```
- **输出**: `capacity_curve_axes0.{png,svg,pdf}`

### 3. **run_example.sh** (便捷工具)
- 一键运行脚本
- 自动检测输入类型（stage1 数据或 litehit_config）
- 智能选择执行路径

### 4. 文档文件
- **README.md** - 详细技术文档
- **QUICKSTART.md** - 快速开始指南
- **PLOTTING_README.md** - 早期创建的绘图说明（可选）

## 使用方法

### 最简单方式：使用现有数据

```bash
cd /mnt/nvme1n1/data/lmk/Github/tair-kvcache

# 使用基线数据直接绘图
./kv_cache_manager/optimizer/tools/kimi_k3_test/run_example.sh \
    /workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample \
    ./test_output

# 查看结果
ls ./test_output/capacity_curve_axes0.*
```

### 完整流程：从头计算

```bash
K3_PY=/workspace/tair-kvcache/.venv/bin/python
K3_TEST=/mnt/nvme1n1/data/lmk/Github/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_test

# 假设已经运行了 prepare 阶段
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

## 代码对应关系

### compute_capacity_data.py ← run_workflow.py

| 原始代码行 | 功能 | 提取后位置 |
|-----------|------|-----------|
| 336-337 | 获取 binary 路径 | 78 |
| 338 | 运行 lite_hit_main | 80-83 |
| 339-342 | 运行 facts_query | 86-93 |
| 343-348 | 加载结果 | 96-100 |
| 356-375 | 独立 LRU 验证 | 106-126 |
| 377-382 | 计算 elbow/cap95 | 129-133 |
| 385-393 | 保存 CSV | 136-149 |
| 394-405 | 保存 summary | 151-168 |

### plot_capacity_curve_axes0.py ← analyze_results.py

| 原始代码行 | 功能 | 提取后位置 |
|-----------|------|-----------|
| 362-364 | 设置绘图样式 | 40-47 |
| 365 | 颜色调色板 | 50 |
| 366-367 | 数据提取 | 24-30 |
| 380-381 | 创建子图 | 53 |
| 382 | 绘制容量曲线 | 58-60 |
| 383 | 绘制无限容量线 | 63-65 |
| 384-393 | 标注关键点 | 68-86 |
| 394 | 设置轴标签和标题 | 89-93 |
| 395 | 添加图例 | 96 |
| 400 | 总标题 | 102-107 |

## 与原始工具的区别

| 特性 | 原始工具 | 新工具 (kimi_k3_test) |
|------|---------|---------------------|
| **包含功能** | prepare + scan + bench + plot | 只有 scan + 部分 plot |
| **绘制图表** | axes[0] + axes[1] + 多个服务指标图 | **只有 axes[0]** |
| **HiSim 仿真** | ✅ 完整的 bench_serving | ❌ 不包含 |
| **代码行数** | ~600 行 | ~250 行 |
| **依赖** | HiSim + AIC predictor | 只需 LiteHit |
| **运行时间** | 数分钟到数小时 | 数秒到数分钟 |
| **输出** | 完整报告 + HTML + 多张图 | 容量数据 + 1 张图 |

## 优势

1. **精简**: 只包含 axes[0] 相关代码，易于理解和维护
2. **独立**: 不依赖 HiSim 仿真，只需要 LiteHit
3. **快速**: 可以使用现有 stage1 数据，秒级出图
4. **灵活**: 数据和绘图分离，便于自定义

## 输出示例

### capacity_data.json
```json
{
  "capacity_gb": [0, 25, 50, ..., 1024, -1],
  "hit_rates_percent": [0.0, 45.2, 67.8, ..., 98.5],
  "elbow_capacity_gib": 250.0,
  "capacity_at_95pct_infinite_gib": 500.0,
  ...
}
```

### 生成的图表
- `capacity_curve_axes0.png` - 高分辨率 PNG
- `capacity_curve_axes0.svg` - 矢量图
- `capacity_curve_axes0.pdf` - PDF 格式

图表包含：
- 容量 vs 命中率曲线
- 无限容量基准线
- 三个关键点标注：elbow、95% 点、plateau 点

## 文件结构

```
kimi_k3_test/
├── compute_capacity_data.py      # 计算容量数据（从 run_workflow.py 提取）
├── plot_capacity_curve_axes0.py  # 绘制 axes[0]（从 analyze_results.py 提取）
├── run_example.sh                # 一键运行脚本
├── README.md                     # 详细技术文档
├── QUICKSTART.md                 # 快速开始指南
└── SUMMARY.md                    # 本文件
```

## 下一步建议

1. **测试工具**:
   ```bash
   ./run_example.sh /path/to/baseline ./test_output
   ```

2. **自定义样式**: 编辑 `plot_capacity_curve_axes0.py` 修改颜色、字体等

3. **分析数据**: 查看 `capacity_data.json` 和 `capacity_curve.csv`

4. **集成到工作流**: 将这些工具集成到你的自动化流程中

## 相关文档

- 原始工具文档: `../kimi_k3_capacity/SOP.md`
- 完整工作流: `../kimi_k3_capacity/run_workflow.py`
- 完整分析: `../kimi_k3_capacity/analyze_results.py`

## 技术细节

### Elbow 计算公式
```python
knee = max(finite, key=lambda p: 
    ((p[1] - finite[0][1]) / gain if gain else 0) - p[0] / finite[-1][0]
)[0]
```
含义：归一化命中率增益 - 归一化容量，最大值点

### 95% 点计算
```python
cap95 = next((c for c, h in finite if h >= 0.95 * rates[-1]), None)
```
含义：第一个达到无限容量命中率 95% 的点

### 独立验证
每个容量点都通过独立的 OrderedDict LRU 缓存模拟验证，确保结果准确性。

## 联系与支持

如有问题，请查看：
1. [QUICKSTART.md](QUICKSTART.md) - 快速开始
2. [README.md](README.md) - 详细文档
3. 原始工具的 SOP.md - 完整流程说明

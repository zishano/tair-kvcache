# Kimi-K3 容量曲线测试工具 - 文件索引

## 📁 核心文件（主要成果）

### 1️⃣ compute_capacity_data.py
**用途**: 计算容量曲线数据  
**来源**: `run_workflow.py` 的 `scan()` 函数（第 330-407 行）  
**输入**: trace.jsonl, layout.json, litehit_config.json  
**输出**: capacity_data.json, capacity_curve.csv  
**代码行数**: ~170 行

### 2️⃣ plot_capacity_curve_axes0.py
**用途**: 绘制 axes[0] 容量曲线图  
**来源**: `analyze_results.py` 的 `plot_and_report()` 函数（第 380-395 行）  
**输入**: capacity_data.json, trace_manifest.json, layout.json  
**输出**: capacity_curve_axes0.{png,svg,pdf}  
**代码行数**: ~140 行

### 3️⃣ run_example.sh
**用途**: 一键运行示例脚本  
**功能**: 自动检测输入类型并执行相应流程  
**使用**: `./run_example.sh <input_dir> <output_dir>`  
**代码行数**: ~80 行

## 📚 文档文件

### QUICKSTART.md ⭐ 推荐首先阅读
快速开始指南，包含：
- 3种使用方法
- 常见问题解答
- 示例命令

### README.md
完整技术文档，包含：
- 详细功能说明
- 数据格式说明
- 代码对应关系表
- 算法细节

### SUMMARY.md
项目总结，包含：
- 完成的工作概览
- 代码提取对照表
- 与原始工具的对比
- 输出示例

### INDEX.md
本文件 - 文件索引和导航

## 🔧 辅助文件（早期版本，可选）

### extract_plot_data.py
数据提取工具（早期创建的独立版本）

### plot_from_extracted_data.py
使用提取数据绘图（早期创建的独立版本）

### plot_capacity_curve.py
直接绘图工具（早期创建的版本）

### PLOTTING_README.md
早期创建的绘图说明文档

## 📊 快速使用指南

### 场景 1: 我有现成的 stage1 数据
```bash
./run_example.sh /path/to/baseline_output ./test_output
```

### 场景 2: 我要从头计算
```bash
# 步骤 1: 计算数据
python compute_capacity_data.py \
  --trace trace.jsonl \
  --layout layout.json \
  --litehit-config litehit_config.json \
  --output-dir output

# 步骤 2: 绘图
python plot_capacity_curve_axes0.py \
  --capacity-data output/capacity_data.json \
  --trace-manifest trace_manifest.json \
  --layout layout.json \
  --output-dir output
```

### 场景 3: 使用基线数据快速测试
```bash
./run_example.sh \
  /workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample \
  ./quick_test
```

## 🎯 文件阅读顺序

### 新手入门
1. **QUICKSTART.md** - 快速开始
2. **run_example.sh** - 看示例脚本
3. **plot_capacity_curve_axes0.py** - 理解绘图逻辑

### 深入理解
1. **SUMMARY.md** - 项目总览
2. **README.md** - 完整文档
3. **compute_capacity_data.py** - 数据计算逻辑

### 原始代码对照
1. **SUMMARY.md** 的代码对应表
2. **README.md** 的算法说明
3. 原始 `run_workflow.py` 和 `analyze_results.py`

## 🔍 核心功能说明

### compute_capacity_data.py 做什么？
1. 运行 `lite_hit_main` 生成 Facts（容量无关的访问记录）
2. 运行 `lite_hit_facts_query_main` 查询 43 个容量点（0, 25, 50, ..., 1024, ∞）
3. 用独立的 LRU 缓存验证每个容量点（7000+ 次检查）
4. 计算关键指标：elbow（几何拐点）、95% 点
5. 保存结果到 JSON 和 CSV

### plot_capacity_curve_axes0.py 做什么？
只绘制**一张图**（axes[0]），包含：
- 容量 vs 命中率曲线（蓝色实线）
- 无限容量基准线（紫色虚线）
- 三个关键点标注：
  - Elbow（橙色） - 几何拐点
  - 95% 点（青色） - 达到无限容量 95% 的点
  - Plateau（绿色） - 曲线平台点（如果有）

## 📈 输出说明

### capacity_data.json 包含
```json
{
  "capacity_gb": [0, 25, 50, ..., 1024, -1],
  "hit_rates_percent": [0.0, 45.2, 67.8, ..., 98.5],
  "elbow_capacity_gib": 250.0,
  "capacity_at_95pct_infinite_gib": 500.0,
  "total_input_tokens": 10473344,
  "validation": {"independent_lru_checks": 7052}
}
```

### 图表文件
- `capacity_curve_axes0.png` - 高分辨率 PNG（DPI 180）
- `capacity_curve_axes0.svg` - 矢量图（无损缩放）
- `capacity_curve_axes0.pdf` - PDF 格式（论文用）

## ⚙️ 技术细节

### 代码提取统计
- **原始代码总行数**: ~600 行（run_workflow.py + analyze_results.py 相关部分）
- **提取后代码行数**: ~250 行（compute + plot）
- **精简比例**: 58% 减少
- **保留功能**: 100%（axes[0] 相关）

### 关键算法

**Elbow 计算**（几何拐点）:
```python
knee = max(finite, key=lambda p: 
    ((p[1] - finite[0][1]) / gain) - p[0] / finite[-1][0]
)[0]
```

**95% 点计算**:
```python
cap95 = next((c for c, h in finite if h >= 0.95 * rates[-1]), None)
```

## 🔗 相关资源

### 原始工具
- `../kimi_k3_capacity/run_workflow.py` - 完整工作流
- `../kimi_k3_capacity/analyze_results.py` - 完整分析和报告
- `../kimi_k3_capacity/SOP.md` - 标准操作流程

### 基线数据位置
```
/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample/
├── stage1/summary.json
├── trace_manifest.json
└── layout.json
```

## ❓ 常见问题

**Q: 这些工具和原始工具有什么区别？**  
A: 这些工具只包含 axes[0] 图表，不包含 HiSim 仿真、axes[1] 和其他服务指标图。

**Q: 我需要什么数据才能运行？**  
A: 最简单：现有的 stage1/summary.json。完整：trace.jsonl + layout.json + litehit_config.json。

**Q: 图表样式可以修改吗？**  
A: 可以，编辑 `plot_capacity_curve_axes0.py` 的第 40-50 行。

**Q: 可以只计算数据不绘图吗？**  
A: 可以，只运行 `compute_capacity_data.py`。

**Q: 可以用其他数据源绘图吗？**  
A: 可以，只要数据格式符合 capacity_data.json 的结构。

## 📝 版本历史

- **2024-09-14**: 初始版本
  - 从 run_workflow.py 提取 scan() 函数
  - 从 analyze_results.py 提取 axes[0] 绘图代码
  - 创建完整文档和示例脚本

## 🎉 总结

✅ **已完成**：精简版工具，只绘制 axes[0] 容量曲线  
✅ **代码行数**：~250 行（相比原始 ~600 行减少 58%）  
✅ **功能完整**：包含数据计算、验证、绘图的完整流程  
✅ **文档齐全**：3 个文档 + 1 个示例脚本  
✅ **即时可用**：可直接使用基线数据测试

## 🚀 立即开始

```bash
cd /mnt/nvme1n1/data/lmk/Github/tair-kvcache/kv_cache_manager/optimizer/tools/kimi_k3_test

# 阅读快速指南
cat QUICKSTART.md

# 运行示例
./run_example.sh \
  /workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample \
  ./my_test_output

# 查看结果
ls -lh ./my_test_output/capacity_curve_axes0.*
```

---

**需要帮助？** 查看 [QUICKSTART.md](QUICKSTART.md) 或 [README.md](README.md)

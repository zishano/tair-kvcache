#!/bin/bash
# Kimi-K3 完整工作流脚本 - 一次执行 prepare + scan + bench + plot
# 基于 SOP.md 的 "all: 一次执行完整流程" 部分

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错

# ============================================================================
# 颜色输出
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# ============================================================================
# 1. 环境变量设置
# ============================================================================
echo "============================================================================"
echo "  Kimi-K3 完整工作流：LiteHit 容量粗筛 + HiSim 性能精算"
echo "============================================================================"
echo ""

# 基础路径
K3_REPO="${K3_REPO:-/mnt/nvme1n1/data/lmk/Github/tair-kvcache}"
K3_PY="${K3_PY:-$K3_REPO/.venv/bin/python}"
K3_WORKFLOW="${K3_WORKFLOW:-$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test/run_workflow.py}"

# AIC 和模型配置
K3_MODEL="${K3_MODEL:-$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test/config/moonshotai--Kimi-K3_config.json}"

# Trace 输入
K3_TRACE="${K3_TRACE:-$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test/trace/traces.jsonl}"

# 基线对比目录（只读）
K3_BASELINE="${K3_BASELINE:-$K3_REPO/reports/kimi_k3_litehit_hisim_20260909/agentx_sample}"

# 输出目录（默认使用当前项目下的独立目录，也可通过 K3_RUN 覆盖）
K3_RUN="${K3_RUN:-$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test/output/full_run}"

# ============================================================================
# 2. 工作流参数配置
# ============================================================================

# --- Trace 采样参数 ---
AGENTX_SESSIONS="${AGENTX_SESSIONS:-8}"                # AgentX session 数量
REQUESTS_PER_SESSION="${REQUESTS_PER_SESSION:-24}"     # 每个 session 最多请求数
AGENTX_WINDOW_SECONDS="${AGENTX_WINDOW_SECONDS:-3600}" # 时间窗口（秒）
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-262144}"         # 输入 token 上限
TIME_SCALE="${TIME_SCALE:-10}"                         # 时间压缩比例

# --- 模型和缓存配置 ---
TP_SIZE="${TP_SIZE:-8}"                                # Tensor Parallel 大小
CHECKPOINT_TOKENS="${CHECKPOINT_TOKENS:-1024}"         # Checkpoint 间隔 tokens

# --- HiSim 仿真参数 ---
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"               # 最大并发数
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"       # 每次 forward 的总 token 预算
CHUNKED_PREFILL_TOKENS="${CHUNKED_PREFILL_TOKENS:-1024}" # Chunk 大小
MEMORY_FRACTION="${MEMORY_FRACTION:-0.90}"             # 显存使用比例

# --- 存储带宽配置（GiB/s）---
READ_GIB_S="${READ_GIB_S:-64}"                         # Storage read 带宽
WRITE_GIB_S="${WRITE_GIB_S:-32}"                       # Storage write 带宽
H2D_GIB_S="${H2D_GIB_S:-128}"                          # Host to Device 带宽
D2H_GIB_S="${D2H_GIB_S:-128}"                          # Device to Host 带宽

# --- Bench 参数 ---
BENCH_ARRIVAL_SCALE="${BENCH_ARRIVAL_SCALE:-1}"       # 到达时间缩放
PROBE_REQUESTS="${PROBE_REQUESTS:-16}"                # 预探针请求数

# --- 容量点选择 ---
# 默认：0, 50, 250, infinite（根据 SOP 的默认 Stage 2 选择）
CAPACITIES="${CAPACITIES:-0,50,250,infinite}"

# --- 可选：并发探针 ---
RUN_CONCURRENCY_PROBES="${RUN_CONCURRENCY_PROBES:-false}" # 是否运行并发探针
CONCURRENCY_PROBES="${CONCURRENCY_PROBES:-8,16}"      # 并发探针点

# ============================================================================
# 3. 参数验证和环境检查
# ============================================================================

info "验证环境和参数..."

if [ -z "$K3_RUN" ]; then
    error "K3_RUN 不能为空"
    exit 1
fi

# 检查 Python 可执行文件
if [ ! -f "$K3_PY" ]; then
    error "Python 可执行文件不存在: $K3_PY"
    exit 1
fi

# 检查 workflow 脚本
if [ ! -f "$K3_WORKFLOW" ]; then
    error "Workflow 脚本不存在: $K3_WORKFLOW"
    exit 1
fi

# 检查模型配置文件
if [ ! -f "$K3_MODEL" ]; then
    error "模型配置文件不存在: $K3_MODEL"
    exit 1
fi

# 检查 trace 文件
if [ ! -f "$K3_TRACE" ]; then
    error "Trace 文件不存在: $K3_TRACE"
    exit 1
fi

# 检查输出目录
if [ -d "$K3_RUN" ]; then
    warning "输出目录已存在: $K3_RUN"
    warning "现有内容将被覆盖"
    if [ "${K3_ASSUME_YES:-false}" = "true" ]; then
        info "K3_ASSUME_YES=true，自动继续"
    else
        read -p "是否继续? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "已取消"
            exit 0
        fi
    fi
else
    info "将创建输出目录: $K3_RUN"
    mkdir -p "$K3_RUN"
fi

success "环境检查通过"

# ============================================================================
# 4. 显示配置摘要
# ============================================================================

echo ""
echo "============================================================================"
echo "  配置摘要"
echo "============================================================================"
echo ""
echo "【路径配置】"
echo "  K3_REPO:       $K3_REPO"
echo "  K3_PY:         $K3_PY"
echo "  K3_WORKFLOW:   $K3_WORKFLOW"
echo "  K3_MODEL:      $K3_MODEL"
echo "  K3_TRACE:      $K3_TRACE"
echo "  K3_BASELINE:   $K3_BASELINE"
echo "  K3_RUN:        $K3_RUN"
echo ""
echo "【Trace 采样】"
echo "  AgentX Sessions:        $AGENTX_SESSIONS"
echo "  Requests per Session:   $REQUESTS_PER_SESSION"
echo "  Window (seconds):       $AGENTX_WINDOW_SECONDS"
echo "  Max Input Tokens:       $MAX_INPUT_TOKENS"
echo "  Time Scale:             $TIME_SCALE"
echo ""
echo "【模型和缓存】"
echo "  TP Size:                $TP_SIZE"
echo "  Checkpoint Tokens:      $CHECKPOINT_TOKENS"
echo ""
echo "【HiSim 仿真】"
echo "  Max Concurrency:        $MAX_CONCURRENCY"
echo "  Max Prefill Tokens:     $MAX_PREFILL_TOKENS"
echo "  Chunked Prefill Tokens: $CHUNKED_PREFILL_TOKENS"
echo "  Memory Fraction:        $MEMORY_FRACTION"
echo ""
echo "【存储带宽 (GiB/s)】"
echo "  Read:                   $READ_GIB_S"
echo "  Write:                  $WRITE_GIB_S"
echo "  H2D:                    $H2D_GIB_S"
echo "  D2H:                    $D2H_GIB_S"
echo ""
echo "【Bench 参数】"
echo "  Arrival Scale:          $BENCH_ARRIVAL_SCALE"
echo "  Probe Requests:         $PROBE_REQUESTS"
echo "  Capacities:             $CAPACITIES"
echo ""
echo "【可选功能】"
echo "  Run Concurrency Probes: $RUN_CONCURRENCY_PROBES"
if [ "$RUN_CONCURRENCY_PROBES" = "true" ]; then
    echo "  Concurrency Probes:     $CONCURRENCY_PROBES"
fi
echo ""
echo "============================================================================"
echo ""

# ============================================================================
# 5. 切换到工作目录
# ============================================================================

info "切换到工作目录: $K3_REPO"
cd "$K3_REPO"

# ============================================================================
# 6. 执行完整工作流 (--steps all)
# ============================================================================

echo ""
echo "============================================================================"
echo "  执行完整工作流: prepare + scan + bench + plot"
echo "============================================================================"
echo ""

START_TIME=$(date +%s)

info "开始执行..."
info "命令: $K3_PY $K3_WORKFLOW --steps all ..."
echo ""

"$K3_PY" "$K3_WORKFLOW" \
  --steps all \
  --output-dir "$K3_RUN" \
  --trace "$K3_TRACE" \
  --model-config "$K3_MODEL" \
  --agentx-sessions "$AGENTX_SESSIONS" \
  --requests-per-session "$REQUESTS_PER_SESSION" \
  --agentx-window-seconds "$AGENTX_WINDOW_SECONDS" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --time-scale "$TIME_SCALE" \
  --tp-size "$TP_SIZE" \
  --checkpoint-tokens "$CHECKPOINT_TOKENS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --chunked-prefill-tokens "$CHUNKED_PREFILL_TOKENS" \
  --memory-fraction "$MEMORY_FRACTION" \
  --read-gib-s "$READ_GIB_S" \
  --write-gib-s "$WRITE_GIB_S" \
  --h2d-gib-s "$H2D_GIB_S" \
  --d2h-gib-s "$D2H_GIB_S" \
  --bench-arrival-scale "$BENCH_ARRIVAL_SCALE" \
  --probe-requests "$PROBE_REQUESTS"

WORKFLOW_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
    success "完整工作流执行成功！"
    success "耗时: ${ELAPSED} 秒"
else
    error "完整工作流执行失败，退出码: $WORKFLOW_EXIT_CODE"
    exit $WORKFLOW_EXIT_CODE
fi

# ============================================================================
# 7. 可选：运行并发探针
# ============================================================================

if [ "$RUN_CONCURRENCY_PROBES" = "true" ]; then
    echo ""
    echo "============================================================================"
    echo "  执行并发探针: bench + plot (concurrency only)"
    echo "============================================================================"
    echo ""

    info "开始执行并发探针..."
    info "命令: $K3_PY $K3_WORKFLOW --steps bench,plot --concurrency-only ..."
    echo ""

    PROBE_START_TIME=$(date +%s)

    "$K3_PY" "$K3_WORKFLOW" \
      --steps bench,plot \
      --output-dir "$K3_RUN" \
      --concurrency-only \
      --concurrency-probes "$CONCURRENCY_PROBES" \
      --probe-requests 0

    PROBE_EXIT_CODE=$?

    PROBE_END_TIME=$(date +%s)
    PROBE_ELAPSED=$((PROBE_END_TIME - PROBE_START_TIME))

    echo ""
    if [ $PROBE_EXIT_CODE -eq 0 ]; then
        success "并发探针执行成功！"
        success "耗时: ${PROBE_ELAPSED} 秒"
    else
        error "并发探针执行失败，退出码: $PROBE_EXIT_CODE"
        exit $PROBE_EXIT_CODE
    fi
fi

# ============================================================================
# 8. 输出结果摘要
# ============================================================================

echo ""
echo "============================================================================"
echo "  执行完成 - 结果摘要"
echo "============================================================================"
echo ""

info "输出目录: $K3_RUN"
echo ""

# 检查关键输出文件
echo "【关键输出文件】"
if [ -f "$K3_RUN/trace_manifest.json" ]; then
    success "✓ trace_manifest.json"
else
    warning "✗ trace_manifest.json (未找到)"
fi

if [ -f "$K3_RUN/layout.json" ]; then
    success "✓ layout.json"
else
    warning "✗ layout.json (未找到)"
fi

if [ -f "$K3_RUN/stage1/summary.json" ]; then
    success "✓ stage1/summary.json"
else
    warning "✗ stage1/summary.json (未找到)"
fi

if [ -f "$K3_RUN/stage2/summary.json" ]; then
    success "✓ stage2/summary.json"
else
    warning "✗ stage2/summary.json (未找到)"
fi

if [ -d "$K3_RUN/report" ]; then
    success "✓ report/ 目录"
    REPORT_FILES=$(find "$K3_RUN/report" -type f | wc -l)
    info "  包含 $REPORT_FILES 个文件"
else
    warning "✗ report/ 目录 (未找到)"
fi

echo ""

# 显示 Stage 1 关键点（如果可用）
if [ -f "$K3_RUN/stage1/summary.json" ]; then
    echo "【Stage 1 容量曲线关键点】"
    info "从 stage1/summary.json 提取..."

    # 使用 Python 简单解析 JSON（如果有 jq 更好）
    if command -v jq &> /dev/null; then
        ELBOW=$(jq -r '.elbow_capacity_gib // "N/A"' "$K3_RUN/stage1/summary.json")
        CAP95=$(jq -r '.capacity_at_95pct_infinite_gib // "N/A"' "$K3_RUN/stage1/summary.json")
        INFINITE=$(jq -r '.hit_rate_at_infinite_capacity_percent // "N/A"' "$K3_RUN/stage1/summary.json")

        echo "  Elbow (几何拐点):     ${ELBOW} GiB"
        echo "  95% 点:              ${CAP95} GiB"
        echo "  无限容量命中率:      ${INFINITE}%"
    else
        warning "  (需要 jq 工具来解析 JSON)"
        info "  请查看: $K3_RUN/stage1/summary.json"
    fi
    echo ""
fi

# 显示 Stage 2 指标（如果可用）
if [ -f "$K3_RUN/stage2/summary.csv" ]; then
    echo "【Stage 2 性能指标】"
    info "查看详细指标: $K3_RUN/stage2/summary.csv"

    if command -v column &> /dev/null; then
        echo ""
        column -t -s',' "$K3_RUN/stage2/summary.csv" | head -10
    else
        head -5 "$K3_RUN/stage2/summary.csv"
    fi
    echo ""
fi

# ============================================================================
# 9. 后续步骤建议
# ============================================================================

echo "============================================================================"
echo "  后续步骤"
echo "============================================================================"
echo ""
echo "1. 查看容量曲线图:"
echo "   ls -lh $K3_RUN/report/*.{png,svg,pdf}"
echo ""
echo "2. 查看 Stage 1 容量数据:"
echo "   cat $K3_RUN/stage1/summary.json"
echo "   cat $K3_RUN/stage1/capacity_curve.csv"
echo ""
echo "3. 查看 Stage 2 性能指标:"
echo "   cat $K3_RUN/stage2/summary.json"
echo "   cat $K3_RUN/stage2/summary.csv"
echo ""
echo "4. 查看完整报告:"
echo "   find $K3_RUN/report -type f"
echo ""
echo "5. 对比基线结果:"
echo "   diff $K3_BASELINE/stage1/summary.json $K3_RUN/stage1/summary.json"
echo ""
echo "============================================================================"
echo ""

TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END_TIME - START_TIME))

success "全部完成！总耗时: ${TOTAL_ELAPSED} 秒 ($((TOTAL_ELAPSED / 60)) 分钟)"
echo ""

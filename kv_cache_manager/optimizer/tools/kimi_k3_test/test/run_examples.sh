#!/bin/bash
# Kimi-K3 完整工作流使用示例
# 展示 run_full_workflow.sh 的典型使用场景

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_SCRIPT="$SCRIPT_DIR/run_full_workflow.sh"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

# 检查脚本是否存在
if [ ! -f "$WORKFLOW_SCRIPT" ]; then
    echo "错误: 找不到 $WORKFLOW_SCRIPT"
    exit 1
fi

echo "============================================================================"
echo "  Kimi-K3 完整工作流使用示例"
echo "============================================================================"
echo ""
echo "本脚本展示几个典型使用场景，请根据需要选择："
echo ""
echo "  1) 基线复现（使用默认参数）"
echo "  2) 修改 checkpoint interval（测试 2048 tokens）"
echo "  3) 修改存储带宽（测试高带宽场景）"
echo "  4) 修改到达压力（测试高压力场景）"
echo "  5) 增加容量点（测试更多容量）"
echo "  6) 修改最大并发（测试 64 并发）"
echo "  7) 完整自定义（所有参数）"
echo "  8) 批量运行（顺序执行多个实验）"
echo "  0) 退出"
echo ""
read -p "请选择 [0-8]: " choice

case $choice in
    1)
        info "场景 1: 基线复现（使用默认参数）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_baseline \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_baseline" \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_baseline/report/"
        fi
        ;;

    2)
        info "场景 2: 修改 checkpoint interval（测试 2048 tokens）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_checkpoint_2048 \\"
        echo "  CHECKPOINT_TOKENS=2048 \\"
        echo "  CHUNKED_PREFILL_TOKENS=2048 \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_checkpoint_2048" \
            CHECKPOINT_TOKENS=2048 \
            CHUNKED_PREFILL_TOKENS=2048 \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_checkpoint_2048/report/"
        fi
        ;;

    3)
        info "场景 3: 修改存储带宽（测试高带宽场景）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_high_bandwidth \\"
        echo "  READ_GIB_S=128 \\"
        echo "  WRITE_GIB_S=64 \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_high_bandwidth" \
            READ_GIB_S=128 \
            WRITE_GIB_S=64 \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_high_bandwidth/report/"
        fi
        ;;

    4)
        info "场景 4: 修改到达压力（测试高压力场景）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_high_pressure \\"
        echo "  TIME_SCALE=5 \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        echo "说明: TIME_SCALE=5 表示 5 倍时间压缩，比默认的 10 更高压力"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_high_pressure" \
            TIME_SCALE=5 \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_high_pressure/report/"
        fi
        ;;

    5)
        info "场景 5: 增加容量点（测试更多容量）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_more_capacities \\"
        echo "  CAPACITIES='0,50,100,150,200,250,300,infinite' \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        echo "说明: 默认只测试 0,50,250,infinite 四个点"
        echo "      这里增加到 8 个点，运行时间会更长"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_more_capacities" \
            CAPACITIES="0,50,100,150,200,250,300,infinite" \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_more_capacities/report/"
        fi
        ;;

    6)
        info "场景 6: 修改最大并发（测试 64 并发）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_high_concurrency \\"
        echo "  MAX_CONCURRENCY=64 \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_high_concurrency" \
            MAX_CONCURRENCY=64 \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_high_concurrency/report/"
        fi
        ;;

    7)
        info "场景 7: 完整自定义（所有参数）"
        echo ""
        echo "执行命令:"
        echo "  K3_RUN=./output/example_custom \\"
        echo "  TIME_SCALE=10 \\"
        echo "  CHECKPOINT_TOKENS=1024 \\"
        echo "  MAX_CONCURRENCY=64 \\"
        echo "  READ_GIB_S=100 \\"
        echo "  WRITE_GIB_S=50 \\"
        echo "  CAPACITIES='0,50,250,500,infinite' \\"
        echo "  RUN_CONCURRENCY_PROBES=true \\"
        echo "  CONCURRENCY_PROBES='16,32,48' \\"
        echo "  K3_ASSUME_YES=true \\"
        echo "  $WORKFLOW_SCRIPT"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            K3_RUN="./output/example_custom" \
            TIME_SCALE=10 \
            CHECKPOINT_TOKENS=1024 \
            MAX_CONCURRENCY=64 \
            READ_GIB_S=100 \
            WRITE_GIB_S=50 \
            CAPACITIES="0,50,250,500,infinite" \
            RUN_CONCURRENCY_PROBES=true \
            CONCURRENCY_PROBES="16,32,48" \
            K3_ASSUME_YES=true \
            "$WORKFLOW_SCRIPT"
            success "完成！查看结果: ./output/example_custom/report/"
        fi
        ;;

    8)
        info "场景 8: 批量运行（顺序执行多个实验）"
        echo ""
        echo "将顺序执行以下实验："
        echo "  1. 基线（默认参数）"
        echo "  2. 高带宽"
        echo "  3. 大 checkpoint"
        echo ""
        echo "说明: 使用 K3_ASSUME_YES=true 跳过确认提示"
        echo ""
        read -p "是否执行? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            info "实验 1/3: 基线"
            K3_RUN="./output/batch_baseline" \
            K3_ASSUME_YES=true \
            "$WORKFLOW_SCRIPT"
            success "实验 1 完成"

            echo ""
            info "实验 2/3: 高带宽"
            K3_RUN="./output/batch_high_bw" \
            READ_GIB_S=128 \
            WRITE_GIB_S=64 \
            K3_ASSUME_YES=true \
            "$WORKFLOW_SCRIPT"
            success "实验 2 完成"

            echo ""
            info "实验 3/3: 大 checkpoint"
            K3_RUN="./output/batch_ckpt_2048" \
            CHECKPOINT_TOKENS=2048 \
            CHUNKED_PREFILL_TOKENS=2048 \
            K3_ASSUME_YES=true \
            "$WORKFLOW_SCRIPT"
            success "实验 3 完成"

            echo ""
            success "批量运行完成！"
            echo ""
            echo "查看结果:"
            echo "  ls -lh ./output/batch_*/report/"
            echo ""
            echo "对比指标:"
            echo "  diff ./output/batch_baseline/stage2/summary.csv \\"
            echo "       ./output/batch_high_bw/stage2/summary.csv"
        fi
        ;;

    0)
        info "退出"
        exit 0
        ;;

    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "============================================================================"
echo "  提示"
echo "============================================================================"
echo ""
echo "查看完整文档:"
echo "  cat FULL_WORKFLOW_README.md"
echo ""
echo "查看所有可配置参数:"
echo "  grep -E '^[A-Z_]+=' run_full_workflow.sh | head -20"
echo ""
echo "============================================================================"

#!/bin/bash
# Example script to run the simplified capacity curve workflow
# This demonstrates how to use the kimi_k3_test tools

set -e  # Exit on error

# Configuration
K3_REPO="/mnt/nvme1n1/data/lmk/Github/tair-kvcache"
K3_PY="$K3_REPO/.venv/bin/python"
K3_TEST="$K3_REPO/kv_cache_manager/optimizer/tools/kimi_k3_test"

# Input from prepare stage (you need to run prepare first)
PREPARED_DIR="${1:-/workspace/dynosim_aic_sweep/reports/kimi_k3_litehit_hisim_20260909/agentx_sample}"

# Output directory for test
OUTPUT_DIR="${2:-./kimi_k3_test_output}"

echo "=========================================="
echo "Kimi-K3 Capacity Curve Test (axes[0] only)"
echo "=========================================="
echo ""
echo "Prepared input: $PREPARED_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if prepared files exist
if [ ! -f "$PREPARED_DIR/trace.jsonl" ]; then
    echo "Error: trace.jsonl not found in $PREPARED_DIR"
    echo "Please run the prepare stage first or specify a valid prepared directory."
    exit 1
fi

if [ ! -f "$PREPARED_DIR/layout.json" ]; then
    echo "Error: layout.json not found in $PREPARED_DIR"
    exit 1
fi

if [ ! -f "$PREPARED_DIR/trace_manifest.json" ]; then
    echo "Error: trace_manifest.json not found in $PREPARED_DIR"
    exit 1
fi

# For baseline data, use stage1/summary.json instead of litehit_config.json
if [ -f "$PREPARED_DIR/stage1/summary.json" ]; then
    echo "Found existing stage1 data, using it directly for plotting..."
    echo ""

    echo "Step: Plotting capacity curve (axes[0])..."
    "$K3_PY" "$K3_TEST/plot_capacity_curve_axes0.py" \
        --capacity-data "$PREPARED_DIR/stage1/summary.json" \
        --trace-manifest "$PREPARED_DIR/trace_manifest.json" \
        --layout "$PREPARED_DIR/layout.json" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "SUCCESS! Using existing stage1 data"
    echo "=========================================="
    echo "Output: $OUTPUT_DIR/capacity_curve_axes0.{png,svg,pdf}"

elif [ -f "$PREPARED_DIR/litehit_config.json" ]; then
    echo "Found litehit_config.json, running full computation..."
    echo ""

    # Step 1: Compute capacity curve data
    echo "Step 1: Computing capacity curve data..."
    "$K3_PY" "$K3_TEST/compute_capacity_data.py" \
        --trace "$PREPARED_DIR/trace.jsonl" \
        --layout "$PREPARED_DIR/layout.json" \
        --litehit-config "$PREPARED_DIR/litehit_config.json" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "Step 2: Plotting capacity curve (axes[0])..."
    "$K3_PY" "$K3_TEST/plot_capacity_curve_axes0.py" \
        --capacity-data "$OUTPUT_DIR/capacity_data.json" \
        --trace-manifest "$PREPARED_DIR/trace_manifest.json" \
        --layout "$PREPARED_DIR/layout.json" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "SUCCESS! Full computation completed"
    echo "=========================================="
    echo "Data: $OUTPUT_DIR/capacity_data.json"
    echo "Plot: $OUTPUT_DIR/capacity_curve_axes0.{png,svg,pdf}"
    echo "CSV:  $OUTPUT_DIR/capacity_curve.csv"
else
    echo "Error: Neither stage1/summary.json nor litehit_config.json found"
    echo "Please ensure you have run the prepare stage or have existing stage1 data."
    exit 1
fi

echo ""
echo "View the plot:"
echo "  xdg-open $OUTPUT_DIR/capacity_curve_axes0.png"

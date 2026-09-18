#!/usr/bin/env bash
# Run only the stages required for the capacity_curve figure:
# prepare -> scan -> plot_capacity_curve_only.py.
# It intentionally skips bench/HiSim and full report generation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${K3_REPO:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
PYTHON="${K3_PY:-$REPO/.venv/bin/python}"
WORKFLOW="${K3_WORKFLOW:-$SCRIPT_DIR/run_workflow_curve_only.py}"
MODEL="${K3_MODEL:-$SCRIPT_DIR/config/moonshotai--Kimi-K3_config.json}"
# TRACE="${K3_TRACE:-$SCRIPT_DIR/trace/merged.jsonl}"
# TRACE="${K3_TRACE:-$SCRIPT_DIR/trace/traces.jsonl}"
# OUTPUT="${K3_RUN:-$SCRIPT_DIR/output/capacity_curve_only}"
# TRACE="${K3_TRACE:-$SCRIPT_DIR/trace_local/merged.jsonl}"
# TRACE="${K3_TRACE:-$SCRIPT_DIR/test/merged.jsonl}"
# TRACE="${K3_TRACE:-$SCRIPT_DIR/trace_60agent/traces.jsonl}"
# OUTPUT="${K3_RUN:-$SCRIPT_DIR/output/capacity_curve_only_60agent}"
TRACE="${K3_TRACE:-$SCRIPT_DIR/trace_main1/9f148b4f-e71.json}"
OUTPUT="${K3_RUN:-$SCRIPT_DIR/output/capacity_curve_only_main1}"

# === Trace 采样参数（已设置为不限制，处理整条trace的所有请求）===
SESSIONS="${AGENTX_SESSIONS:-999999}" # 从 AgentX trace 中选择多少个 session（对话）进行分析
REQUESTS_PER_SESSION="${REQUESTS_PER_SESSION:-999999999}" # 每个 session 最多保留多少个请求
WINDOW_SECONDS="${AGENTX_WINDOW_SECONDS:-999999999}" # 时间窗口（秒）：只选择 session 开始后此时间内的请求
MAX_INPUT="${MAX_INPUT_TOKENS:-999999999}" # 输入 token 上限：超过此值的请求会被过滤）
TIME_SCALE="${TIME_SCALE:-10}" # 时间压缩比例：将原始到达时间缩放，用于加速仿真
TP_SIZE="${TP_SIZE:-32}" # Tensor Parallel 大小：模型分片数量
CHECKPOINT="${CHECKPOINT_TOKENS:-1024}" # Checkpoint 间隔（tokens）：每隔多少 tokens 生成一个 KV cache checkpoint

# SESSIONS="${AGENTX_SESSIONS:-8}" # 从 AgentX trace 中选择多少个 session（对话）进行分析
# REQUESTS_PER_SESSION="${REQUESTS_PER_SESSION:-24}" # 每个 session 最多保留多少个请求
# WINDOW_SECONDS="${AGENTX_WINDOW_SECONDS:-3600}" # 时间窗口（秒）：只选择 session 开始后此时间内的请求
# MAX_INPUT="${MAX_INPUT_TOKENS:-262144}" # 输入 token 上限：超过此值的请求会被过滤）
# TIME_SCALE="${TIME_SCALE:-10}" # 时间压缩比例：将原始到达时间缩放，用于加速仿真
# TP_SIZE="${TP_SIZE:-8}" # Tensor Parallel 大小：模型分片数量
# CHECKPOINT="${CHECKPOINT_TOKENS:-1024}" # Checkpoint 间隔（tokens）：每隔多少 tokens 生成一个 KV cache checkpoint

if [[ ! -x "$PYTHON" ]]; then
  printf '[ERROR] Python executable not found: %s\n' "$PYTHON" >&2
  exit 1
fi
for path in "$WORKFLOW" "$MODEL" "$TRACE"; do
  if [[ ! -f "$path" ]]; then
    printf '[ERROR] Required file not found: %s\n' "$path" >&2
    exit 1
  fi
done
for binary in lite_hit_main lite_hit_facts_query_main; do
  if [[ ! -x "$REPO/bazel-bin/kv_cache_manager/optimizer/$binary" ]]; then
    printf '[ERROR] Missing executable: %s\n' "$REPO/bazel-bin/kv_cache_manager/optimizer/$binary" >&2
    printf '        Build it with .venv/bin/bazel build //kv_cache_manager/optimizer:%s\n' "$binary" >&2
    exit 1
  fi
done

if [[ -e "$OUTPUT" && "${K3_ASSUME_YES:-false}" != true ]]; then
  read -r -p "Output exists and will be regenerated: $OUTPUT (y/N) " answer
  [[ "$answer" =~ ^[Yy]$ ]] || exit 0
fi

mkdir -p "$OUTPUT"
printf '[INFO] Running prepare + scan only (HiSim bench is skipped)\n'
"$PYTHON" "$WORKFLOW" \
  --steps prepare,scan \
  --output-dir "$OUTPUT" \
  --trace "$TRACE" \
  --model-config "$MODEL" \
  --agentx-sessions "$SESSIONS" \
  --requests-per-session "$REQUESTS_PER_SESSION" \
  --agentx-window-seconds "$WINDOW_SECONDS" \
  --max-input-tokens "$MAX_INPUT" \
  --time-scale "$TIME_SCALE" \
  --tp-size "$TP_SIZE" \
  --checkpoint-tokens "$CHECKPOINT"

printf '[INFO] Plotting the capacity curve only\n'
"$PYTHON" "$SCRIPT_DIR/plot_capacity_curve_only.py" "$OUTPUT"
printf '[SUCCESS] Results: %s\n' "$OUTPUT/capacity_curve_only/capacity_curve.{png,svg,pdf}"

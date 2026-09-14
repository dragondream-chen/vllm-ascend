#!/usr/bin/env bash
set -euo pipefail

# Run from an empty working directory to avoid Python module shadowing.

: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${NODE0_IP:?Set NODE0_IP}"
: "${LOCAL_IP:?Set LOCAL_IP}"
: "${NIC_NAME:?Set NIC_NAME}"
: "${MODEL_PATH:?Set MODEL_PATH}"

# Weight loading and graph capture can exceed the default 600-second frontend timeout.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

DP_START_RANK=$((NODE_RANK * 2))
HEADLESS_ARGS=()
if [[ "$NODE_RANK" != "0" ]]; then
  HEADLESS_ARGS+=(--headless --data-parallel-start-rank "$DP_START_RANK")
fi

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  "${HEADLESS_ARGS[@]}" \
  --data-parallel-address "$NODE0_IP" \
  --data-parallel-rpc-port "${DP_RPC_PORT:-13399}" \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name deepseek-v41 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --block-size 128 \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --additional-config '{"enable_engram":true,"engram_storage":"int8","enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

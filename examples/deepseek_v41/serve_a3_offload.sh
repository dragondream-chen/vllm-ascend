#!/usr/bin/env bash
set -euo pipefail

# Single A3 (16 logical devices), with compressed Engram tables on CPU.
# Build the native extension containing engram_int8_lookup_cpu before launch.
# Run from an empty working directory to avoid Python module shadowing.

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

export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --data-parallel-address "$LOCAL_IP" \
  --data-parallel-rpc-port "${DP_RPC_PORT:-13399}" \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name deepseek-v41 \
  --max-model-len 131072 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-memory-bytes 1073741824 \
  --no-enable-prefix-caching \
  --block-size 128 \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --additional-config '{"enable_engram":true,"enable_engram_ple_offload":true,"engram_storage":"int8","enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

#!/bin/bash

# Script to start vLLM server for GRPO training
# This should be run in a separate terminal before starting training

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_GPU_MEMORY="${VLLM_GPU_MEMORY:-0.95}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"

echo "Starting vLLM server..."
echo "Model: ${MODEL_NAME}"
echo "Host: ${VLLM_HOST}:${VLLM_PORT}"
echo "GPU Memory Utilization: ${VLLM_GPU_MEMORY}"
echo "Tensor Parallel Size: ${VLLM_TENSOR_PARALLEL}"

uv run trl vllm-serve \
    --model "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY}" \
    --tensor_parallel_size "${VLLM_TENSOR_PARALLEL}" \
    --trust_remote_code

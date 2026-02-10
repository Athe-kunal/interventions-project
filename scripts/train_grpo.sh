#!/bin/bash

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/grpo_experiment_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/training.log}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.2}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-localhost}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"

mkdir -p "${OUTPUT_DIR}"

echo "Starting GRPO training..."
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "vLLM Mode: ${VLLM_MODE}"
if [ "${VLLM_MODE}" != "disabled" ]; then
    echo "vLLM GPU Memory Utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
    echo "vLLM Tensor Parallel Size: ${VLLM_TENSOR_PARALLEL_SIZE}"
    if [ "${VLLM_MODE}" = "server" ]; then
        echo "vLLM Server: ${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}"
    fi
fi

# Run training with accelerate
CUDA_VISIBLE_DEVICES=2,3 ACCELERATE_LOG_LEVEL=info \
    uv run accelerate launch \
    --main_process_port 29503 \
    --config_file scripts/accelerate/ds_zero2_4gpu.yaml \
    run.py \
    --config configs/config.yaml \
    --model.model_name_or_path "${MODEL_NAME}" \
    --training.output_dir "${OUTPUT_DIR}" \
    --training.run_name $(basename ${OUTPUT_DIR}) \
    --training.gradient_accumulation_steps 2 \
    --training.max_completion_length 16384 \
    --training.max_prompt_length 512 \
    --training.per_device_train_batch_size 4 \
    --training.save_steps 64 \
    --training.max_steps 1024 \
    --training.use_vllm $([ "${VLLM_MODE}" != "disabled" ] && echo "true" || echo "false") \
    --training.vllm_mode "${VLLM_MODE}" \
    --training.vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --training.vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --training.vllm_server_host "${VLLM_SERVER_HOST}" \
    --training.vllm_server_port "${VLLM_SERVER_PORT}" \
    --training.epsilon_high 0.28 \
    --training.lr_scheduler_type cosine \
    --training.use_liger_kernel true \
    --training.loss_type dr_grpo \
    --logging.wandb_project grpo-full-interventions \
    --dataset.example_numbers 1000000000 \
    &> "${LOG_FILE}"

echo "Training complete! Check: ${LOG_FILE}"

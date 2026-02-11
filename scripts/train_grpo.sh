#!/bin/bash

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/grpo_experiment_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/training.log}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-open-r1/DAPO-Math-17k-Processed}"
# vLLM mode: colocate, server, or disabled
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
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

export VLLM_SKIP_WARMUP=1

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    uv run accelerate launch \
    --main_process_port 29503 \
    --config_file scripts/accelerate/ds_zero2_4gpu.yaml \
    run.py \
    --config.model.model_name_or_path "${MODEL_NAME}" \
    --config.dataset.dataset_name_or_path "${DATASET_NAME}" \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "$(basename "${OUTPUT_DIR}")" \
    --config.training.gradient_accumulation_steps 8 \
    --config.training.max_completion_length 16384 \
    --config.training.max_prompt_length 512 \
     --config.training.num_train_epochs 1 \
    --config.training.per_device_train_batch_size 4 \
    --config.training.num_generations 8 \
    --config.training.save_steps 64 \
    --config.training.max_steps 1024 \
    --config.training.use_vllm $([ "${VLLM_MODE}" != "disabled" ] && echo "true" || echo "false") \
    --config.training.vllm_mode "${VLLM_MODE}" \
    --config.training.vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --config.training.vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --config.interventions.intervention_type LoreftIntervention \
    --config.interventions.intervention_layers all \
    --config.interventions.low_rank_dimension 128 \
    --config.interventions.act_fn gelu \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type cosine \
    --config.training.use_liger_kernel false \
    --config.training.loss_type dr_grpo \
    --config.training.report_to "" \
    --config.dataset.example_numbers 1000000000 \
    2>&1 | tee "${LOG_FILE}"

echo "Training complete! Check: ${LOG_FILE}"

#!/bin/bash

# Training script for GRPO with interventions
# Based on DeepSeek-R1 training configuration
#
# Customize interventions by editing the parameters below:
#   - intervention_type: "LoreftIntervention" or "DireftIntervention"
#   - intervention_layers: "all", "odd_only", "even_only", or "last_only"
#   - low_rank_dimension: rank for low-rank interventions (e.g., 128, 256)
#   - dropout: dropout rate for interventions
#   - act_fn: activation function ("gelu", "relu", or null for linear)

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/grpo_experiment_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/training.log}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"

# vLLM Configuration
# Set VLLM_MODE=server to use vLLM server (must start server first with scripts/start_vllm_server.sh)
# Set VLLM_MODE=colocate to run vLLM in the same process (simpler, may have memory issues)
# Set VLLM_MODE=disabled to disable vLLM (slower generation but no setup needed)
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-localhost}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Starting GRPO training..."
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Model: ${MODEL_NAME}"
echo "vLLM Mode: ${VLLM_MODE}"

if [ "${VLLM_MODE}" = "server" ]; then
    echo "vLLM Server: ${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}"
    echo "Make sure vLLM server is running! (./scripts/start_vllm_server.sh)"
fi

echo ""

# Run training with accelerate
CUDA_VISIBLE_DEVICES=2,3 ACCELERATE_LOG_LEVEL=info \
    uv run accelerate launch \
    --main_process_port 29503 \
    --config_file scripts/accelerate/ds_zero2_4gpu.yaml \
    run.py train \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "${MODEL_NAME}" \
    --config.model.dtype "bfloat16" \
    --config.model.intervention_type "LoreftIntervention" \
    --config.model.intervention_layers "all" \
    --config.model.low_rank_dimension 128 \
    --config.model.dropout 0.0 \
    --config.model.act_fn "gelu" \
    --config.model.init_orth true \
    --config.training.learning_rate 1e-5 \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "$(basename ${OUTPUT_DIR})" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 8 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 16384 \
    --config.training.num_generations 8 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 4 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 64 \
    --config.training.max_steps 1024 \
    --config.training.use_vllm $([ "${VLLM_MODE}" != "disabled" ] && echo "true" || echo "false") \
    --config.training.vllm_mode "${VLLM_MODE}" \
    --config.training.vllm_gpu_memory_utilization 0.3 \
    --config.training.vllm_tensor_parallel_size 2 \
    --config.training.vllm_server_host "${VLLM_SERVER_HOST}" \
    --config.training.vllm_server_port "${VLLM_SERVER_PORT}" \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "cosine" \
    --config.training.use_liger_kernel true \
    --config.training.loss_type "dr_grpo" \
    --config.training.report_to '["tensorboard"]' \
    --config.logging.wandb_project "grpo-full-interventions" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    --config.dataset.example_numbers 1000000000 \
    &> "${LOG_FILE}"

echo "Training complete!"
echo "Check logs at: ${LOG_FILE}"
echo "Model saved to: ${OUTPUT_DIR}/final_model"

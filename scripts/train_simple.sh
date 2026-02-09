#!/bin/bash

# Simple training script for quick testing (single GPU)
# This script uses minimal resources for testing purposes
#
# Customize interventions by editing the parameters below:
#   - intervention_type: "LoreftIntervention" or "DireftIntervention"
#   - intervention_layers: "all", "odd_only", "even_only", or "last_only"
#   - low_rank_dimension: rank for low-rank interventions (e.g., 128, 256)
#   - dropout: dropout rate for interventions
#   - act_fn: activation function ("gelu", "relu", or null for linear)

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/test_run}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting simple GRPO training (single GPU)..."

python run.py train \
    --config.common.seed 42 \
    --config.model.model_name_or_path "Qwen/Qwen3-1.7B" \
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
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 2 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 512 \
    --config.training.num_generations 4 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 1 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 10 \
    --config.training.max_steps 20 \
    --config.training.use_vllm false \
    --config.training.loss_type "dr_grpo" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    --config.dataset.example_numbers 50

echo "Training complete! Model saved to: ${OUTPUT_DIR}/final_model"

#!/bin/bash

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/test_run}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting simple GRPO training (single GPU)..."

uv run python run.py \
    --config configs/config.yaml \
    --training.output_dir "${OUTPUT_DIR}" \
    --model.low_rank_dimension 32 \
    --training.gradient_accumulation_steps 2 \
    --training.max_completion_length 512 \
    --training.num_generations 2 \
    --training.save_steps 10 \
    --training.max_steps 20 \
    --training.use_vllm false \
    --training.loss_type dr_grpo \
    --dataset.example_numbers 50

echo "Training complete! Model saved to: ${OUTPUT_DIR}/final_model"

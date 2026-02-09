# Interventions RL - GRPO Training with LoReFT/DiReFT

Train language models using GRPO (Group Relative Policy Optimization) with interventions (LoReFT/DiReFT) for improved reasoning capabilities.

## Overview

This project combines:
- **GRPO**: Group Relative Policy Optimization for reinforcement learning
- **Interventions**: LoReFT (Low-Rank Representation Fine-Tuning) or DiReFT (Directional Representation Fine-Tuning)
- **Math Reasoning**: Training on mathematical problem-solving datasets

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Test Your Setup

```bash
./scripts/train_simple.sh
```

This runs a quick 2-minute test to verify everything works.

### 3. Run Full Training

```bash
./scripts/train_grpo.sh
```

This runs full-scale training with 4 GPUs and vLLM acceleration.

## Usage Examples

### Basic Training

```bash
python run.py train \
    --config.model.model_name_or_path "Qwen/Qwen3-1.7B" \
    --config.training.output_dir "./outputs/basic"
```

### Custom Interventions

```bash
python run.py train \
    --config.model.intervention_type "DireftIntervention" \
    --config.model.low_rank_dimension 256 \
    --config.training.output_dir "./outputs/direft"
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file scripts/accelerate/ds_zero2_4gpu.yaml \
    run.py train \
    --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --config.training.use_vllm true \
    --config.training.output_dir "./outputs/multi_gpu"
```

### Production Training

```bash
# Set environment variables
export OUTPUT_DIR="./outputs/production_run"
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Run with full configuration
./scripts/train_grpo.sh
```

## Configuration

### Interventions Config (YAML)

Edit `interventions_rl/model/interventions_config.yaml`:

```yaml
InterventionsConfig:
  intervention_type: "LoreftIntervention"  # or "DireftIntervention"
  intervention_layers: "all"                # all, odd_only, even_only, last_only
  low_rank_dimension: 128
  dropout: 0.0
  act_fn: "gelu"                           # gelu, relu, or null
  init_orth: true
```

**Example:**
```bash
python run.py train \
    --config.model.model_name_or_path "Qwen/Qwen3-1.7B" \
    --config.training.learning_rate 1e-5 \
    --config.training.max_steps 100 \
    --config.training.report_to '["wandb"]' \
    --config.logging.wandb_project "my-project"
```

## Intervention Features Explained

### Interventions

**Types:**
- **LoReFT**: Low-rank transformations to hidden states (memory-efficient)
- **DiReFT**: Directional vectors for targeted steering

**Layer Selection** (`intervention_layers`):
- `all` - All transformer layers
- `odd_only` - Layers 1, 3, 5, ...
- `even_only` - Layers 0, 2, 4, ...
- `last_only` - Only the last layer


To load:

```python
import torch
from interventions_rl.model import qwen3, interventions_utils

# Load config
config = interventions_utils.InterventionsConfig.parse_file(
    "outputs/experiment/final_model/interventions_config.json"
)

# Load model
model = qwen3.Qwen3ForCausalLM(
    interventions_config=config,
    config=hf_config,
)
model.load_state_dict(torch.load("outputs/experiment/final_model/model.pt"))
```


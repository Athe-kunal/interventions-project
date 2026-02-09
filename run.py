#!/usr/bin/env python3
"""
GRPO Training Script with Interventions Support

Usage:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file accelerate_config.yaml \
        run.py train \
        --config.model.model_name_or_path "Qwen/Qwen3-1.7B" \
        --config.training.output_dir "./outputs" \
        ...
"""

import argparse
import json
import os
import sys
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal

import torch
from loguru import logger
from trl import GRPOTrainer, GRPOConfig
from transformers import set_seed

from interventions_rl.model import qwen3, llama, interventions_utils
from interventions_rl.model.load_model import load_interventions_model
from interventions_rl.data import open_r1


@dataclass
class CommonConfig:
    """Common configuration settings"""
    seed: int = 42
    debug: bool = False


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    model_type: Literal["qwen3", "llama"] = "qwen3"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    interventions_config_path: Optional[str] = None
    
    # Manual interventions config (alternative to yaml file)
    intervention_type: Optional[str] = None
    intervention_layers: Optional[str] = None
    low_rank_dimension: Optional[int] = None
    dropout: Optional[float] = None
    act_fn: Optional[str] = None
    init_orth: Optional[bool] = None


@dataclass
class PeftConfig:
    """PEFT/LoRA configuration"""
    use_peft: bool = False
    type: str = "lora"
    task_type: str = "CAUSAL_LM"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    total_step: Optional[int] = None


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    dataset_name_or_path: str = "open-r1/DAPO-Math-17k-Processed"
    example_numbers: int = 1000
    test_split_ratio: float = 0.1


@dataclass
class LoggingConfig:
    """Logging configuration"""
    trackio_space_id: Optional[str] = None
    trackio_project: Optional[str] = None
    wandb_project: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration extending GRPOConfig parameters"""
    # Basic training params
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: Optional[dict] = None
    
    # GRPO specific
    beta: float = 0.0
    num_generations: int = 8
    num_generations_eval: Optional[int] = None
    max_completion_length: int = 256
    max_prompt_length: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    top_entropy_quantile: float = 1.0
    loss_type: str = "dapo"
    
    # Generation
    use_vllm: bool = False
    vllm_mode: str = "server"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    
    # Logging and saving
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 500
    remove_unused_columns: bool = False
    report_to: Optional[list[str]] = None
    
    # Other
    use_liger_kernel: bool = False
    mask_truncated_completions: bool = False


@dataclass
class Config:
    """Main configuration object"""
    common: CommonConfig = field(default_factory=CommonConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def parse_nested_config(args: list[str]) -> Config:
    """Parse command-line arguments with nested config structure"""
    config = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train'], help='Command to run')
    
    # Parse all --config.* arguments
    i = 0
    parsed_args = {'command': None}
    while i < len(args):
        if args[i] == 'train':
            parsed_args['command'] = 'train'
            i += 1
        elif args[i].startswith('--config.'):
            key = args[i][9:]  # Remove '--config.' prefix
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                value = args[i + 1]
                parsed_args[key] = value
                i += 2
            else:
                parsed_args[key] = True
                i += 1
        else:
            i += 1
    
    # Apply parsed arguments to config
    for key, value in parsed_args.items():
        if key == 'command':
            continue
        
        parts = key.split('.')
        if len(parts) != 2:
            continue
        
        section, param = parts
        
        # Get the appropriate config section
        if section == 'common':
            section_obj = config.common
        elif section == 'model':
            section_obj = config.model
        elif section == 'peft':
            section_obj = config.peft
        elif section == 'dataset':
            section_obj = config.dataset
        elif section == 'training':
            section_obj = config.training
        elif section == 'logging':
            section_obj = config.logging
        else:
            logger.warning(f"Unknown config section: {section}")
            continue
        
        # Convert value to appropriate type
        if hasattr(section_obj, param):
            field_type = type(getattr(section_obj, param))
            
            # Handle special cases
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value.replace("'", '"'))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON value for {key}: {value}")
            elif field_type == bool:
                value = value.lower() in ('true', '1', 'yes')
            elif field_type == int:
                value = int(value)
            elif field_type == float:
                value = float(value)
            
            setattr(section_obj, param, value)
        else:
            logger.warning(f"Unknown parameter: {section}.{param}")
    
    return config


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "half": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def get_model_class(model_type: str):
    """Get model class based on model type"""
    if model_type.lower() == "qwen3":
        return qwen3.Qwen3ForCausalLM
    elif model_type.lower() == "llama":
        return llama.LlamaForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_interventions_config(config: ModelConfig) -> interventions_utils.InterventionsConfig:
    """Load interventions configuration from yaml or manual config"""
    if config.interventions_config_path:
        logger.info(f"Loading interventions config from: {config.interventions_config_path}")
        return interventions_utils.read_config_from_yaml(config.interventions_config_path)
    elif config.intervention_type:
        logger.info("Using manual interventions config from command line")
        return interventions_utils.InterventionsConfig(
            intervention_type=config.intervention_type,
            intervention_layers=config.intervention_layers or "all",
            low_rank_dimension=config.low_rank_dimension or 128,
            dropout=config.dropout or 0.0,
            act_fn=config.act_fn,
            init_orth=config.init_orth if config.init_orth is not None else True,
        )
    else:
        # Load default from yaml file
        default_yaml = "interventions_rl/model/interventions_config.yaml"
        if os.path.exists(default_yaml):
            logger.info(f"Loading default interventions config from: {default_yaml}")
            return interventions_utils.read_config_from_yaml(default_yaml)
        else:
            raise ValueError(
                "No interventions config provided. Please specify:\n"
                "  - Recommended: Manual config via command line:\n"
                "      --config.model.intervention_type 'LoreftIntervention' \\\n"
                "      --config.model.intervention_layers 'all' \\\n"
                "      --config.model.low_rank_dimension 128\n"
                "  - Or: --config.model.interventions_config_path <path_to_yaml>\n"
                f"  - Or: Ensure default config exists at: {default_yaml}"
            )


def save_model_with_interventions(
    model,
    output_dir: str,
    interventions_config: interventions_utils.InterventionsConfig,
    tokenizer=None,
):
    """Save model and interventions configuration"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to: {model_path}")
    
    # Save interventions config
    config_path = output_path / "interventions_config.json"
    with open(config_path, 'w') as f:
        f.write(interventions_config.to_json())
    logger.info(f"Saved interventions config to: {config_path}")
    
    # Save tokenizer if provided
    if tokenizer:
        tokenizer.save_pretrained(output_path / "tokenizer")
        logger.info(f"Saved tokenizer to: {output_path / 'tokenizer'}")


def train(config: Config):
    """Main training function"""
    # Set seed
    set_seed(config.common.seed)
    logger.info(f"Set random seed to {config.common.seed}")
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save full config
    config_save_path = Path(config.training.output_dir) / "training_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    logger.info(f"Saved training config to: {config_save_path}")
    
    # Load interventions config
    interventions_config = load_interventions_config(config.model)
    logger.info(f"Interventions config: {interventions_config}")
    
    # Determine model type from model name if not specified
    if "qwen" in config.model.model_name_or_path.lower():
        model_type = "qwen3"
    elif "llama" in config.model.model_name_or_path.lower():
        model_type = "llama"
    else:
        model_type = config.model.model_type
    
    model_class = get_model_class(model_type)
    logger.info(f"Using model class: {model_class.__name__}")
    
    # Load model with interventions
    dtype = get_dtype(config.model.dtype)
    logger.info(f"Loading model: {config.model.model_name_or_path} with dtype {dtype}")
    
    report, model = load_interventions_model(
        hf_model_name_or_path=config.model.model_name_or_path,
        model_class=model_class,
        ic_config=interventions_config,
        map_dtype=dtype,
        map_device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        trust_remote_code=config.model.trust_remote_code,
    )
    
    logger.info(f"Model loading report: {report.summary()}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.dataset_name_or_path}")
    dataset = open_r1.load_openr1_dataset(
        config.dataset.dataset_name_or_path,
        example_numbers=config.dataset.example_numbers,
        test_split_ratio=config.dataset.test_split_ratio,
    )
    
    logger.info(f"Train dataset size: {len(dataset.train_dataset)}")
    if dataset.test_dataset:
        logger.info(f"Test dataset size: {len(dataset.test_dataset)}")
    
    # Prepare training arguments
    training_args_dict = asdict(config.training)
    
    # Handle special fields
    if config.training.lr_scheduler_kwargs:
        training_args_dict['lr_scheduler_kwargs'] = config.training.lr_scheduler_kwargs
    
    # Create GRPOConfig
    grpo_config = GRPOConfig(**training_args_dict)
    
    logger.info(f"GRPO Config created with output_dir: {grpo_config.output_dir}")
    
    # Setup reporting
    if config.training.report_to:
        grpo_config.report_to = config.training.report_to
        
        # Setup wandb if specified
        if "wandb" in config.training.report_to and config.logging.wandb_project:
            os.environ["WANDB_PROJECT"] = config.logging.wandb_project
            logger.info(f"W&B project: {config.logging.wandb_project}")
    
    # Create trainer
    logger.info("Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=dataset.reward_functions,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.test_dataset,
        args=grpo_config,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model with interventions
    logger.info("Training complete. Saving model...")
    final_output_dir = Path(config.training.output_dir) / "final_model"
    save_model_with_interventions(
        model=model,
        output_dir=str(final_output_dir),
        interventions_config=interventions_config,
        tokenizer=trainer.tokenizer if hasattr(trainer, 'tokenizer') else None,
    )
    
    logger.info(f"Training complete! Model saved to: {final_output_dir}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2 or sys.argv[1] not in ['train']:
        print("Usage: python run.py train [options]")
        print("\nExample:")
        print("  python run.py train \\")
        print("    --config.model.model_name_or_path 'Qwen/Qwen3-1.7B' \\")
        print("    --config.training.output_dir './outputs' \\")
        print("    --config.training.num_train_epochs 1")
        sys.exit(1)
    
    # Parse configuration
    config = parse_nested_config(sys.argv[1:])
    
    if config.common.debug:
        logger.info("Debug mode enabled")
        logger.info(f"Configuration: {config}")
    
    # Execute command
    command = sys.argv[1]
    if command == 'train':
        train(config)
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

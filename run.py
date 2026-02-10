#!/usr/bin/env python3
"""
GRPO Training Script with Interventions Support

Usage:
    python run.py --config configs/config.yaml
    python run.py --config configs/config.yaml --output_dir ./custom_output
"""

import argparse
import os
import yaml
from pathlib import Path

import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import set_seed

from interventions_rl.model import qwen3, llama, interventions_utils
from interventions_rl.model.load_model import load_interventions_model
from interventions_rl.data import open_r1


def _get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "half": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def _get_model_class(model_name: str):
    if "qwen" in model_name.lower():
        return qwen3.Qwen3ForCausalLM
    elif "llama" in model_name.lower():
        return llama.LlamaForCausalLM
    else:
        raise ValueError(f"Cannot infer model type from: {model_name}")


def _load_interventions_config(
    model_cfg: dict,
) -> interventions_utils.InterventionsConfig:
    if model_cfg.get("interventions_config_path"):
        return interventions_utils.read_config_from_yaml(
            model_cfg["interventions_config_path"]
        )
    elif model_cfg.get("intervention_type"):
        return interventions_utils.InterventionsConfig(
            intervention_type=model_cfg["intervention_type"],
            intervention_layers=model_cfg.get("intervention_layers", "all"),
            low_rank_dimension=model_cfg.get("low_rank_dimension", 128),
            dropout=model_cfg.get("dropout", 0.0),
            act_fn=model_cfg.get("act_fn"),
            init_orth=model_cfg.get("init_orth", True),
        )
    else:
        default_yaml = "interventions_rl/model/interventions_config.yaml"
        if os.path.exists(default_yaml):
            return interventions_utils.read_config_from_yaml(default_yaml)
        raise ValueError("No interventions config provided")


def _save_trainable_weights(
    model, output_dir: str, interventions_config, tokenizer=None
):
    """Save only trainable model weights and config"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save only trainable parameters
    trainable_state_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(trainable_state_dict, output_path / "trainable_weights.pt")

    # Save config
    with open(output_path / "interventions_config.json", "w") as f:
        f.write(interventions_config.to_json())

    if tokenizer:
        tokenizer.save_pretrained(output_path / "tokenizer")


def train(config_path: str, **overrides):
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            d = cfg
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        else:
            cfg[key] = value

    set_seed(cfg.get("seed", 42))

    # Setup output directory
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Load interventions config
    interventions_config = _load_interventions_config(cfg["model"])

    # Load model
    model_class = _get_model_class(cfg["model"]["model_name_or_path"])
    dtype = _get_dtype(cfg["model"].get("dtype", "bfloat16"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, model = load_interventions_model(
        hf_model_name_or_path=cfg["model"]["model_name_or_path"],
        model_class=model_class,
        ic_config=interventions_config,
        map_dtype=dtype,
        map_device=device,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )

    # Load dataset
    dataset = open_r1.load_openr1_dataset(
        cfg["dataset"]["dataset_name_or_path"],
        example_numbers=cfg["dataset"].get("example_numbers", 1000),
        test_split_ratio=cfg["dataset"].get("test_split_ratio", 0.1),
    )

    # Setup GRPO config - pass training args directly
    training_cfg = cfg["training"].copy()
    if cfg.get("logging", {}).get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg["logging"]["wandb_project"]
        training_cfg.setdefault("report_to", ["wandb"])

    grpo_config = GRPOConfig(**training_cfg)

    # Train
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=dataset.reward_functions,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.test_dataset,
        args=grpo_config,
    )
    trainer.train()

    # Save trainable weights only
    _save_trainable_weights(
        model=model,
        output_dir=str(output_dir / "final_model"),
        interventions_config=interventions_config,
        tokenizer=getattr(trainer, "tokenizer", None),
    )


def _parse_value(value_str: str):
    """Parse string value to appropriate type"""
    # Try bool
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    # Try list/dict
    if value_str.startswith("[") or value_str.startswith("{"):
        try:
            import json

            return json.loads(value_str)
        except:
            pass
    # Keep as string
    return value_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config file"
    )
    args, unknown = parser.parse_known_args()

    # Parse additional overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = _parse_value(unknown[i + 1])
                overrides[key] = value
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1

    train(args.config, **overrides)

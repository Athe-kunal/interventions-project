import argparse
import json
import os
import re
import shutil

from loguru import logger
import torch
import transformers
from safetensors.torch import save_file

from interventions_rl.model import qwen, llama, interventions_utils
from interventions_rl.model.load_model import load_interventions_model

# Model source files needed for trust_remote_code
_MODEL_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "interventions_rl", "model")
_MODEL_SOURCE_FILES = [
    "interventions.py",
    "interventions_utils.py",
    "llama.py",
    "qwen.py",
]

# Map model class to the module path for auto_map in config.json
_AUTO_MAP = {
    "qwen3": "qwen.Qwen3ForCausalLM",
    "qwen2": "qwen.Qwen2ForCausalLM",
    "llama": "llama.LlamaForCausalLM",
}


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _get_model_class(model_name: str):
    name = model_name.lower()
    if "qwen3" in name:
        return qwen.Qwen3ForCausalLM, "qwen3"
    elif "qwen" in name:
        return qwen.Qwen2ForCausalLM, "qwen2"
    elif "llama" in name:
        return llama.LlamaForCausalLM, "llama"
    else:
        raise ValueError(f"Cannot infer model type from: {model_name}")


def _copy_model_sources(vllm_output_dir: str) -> None:
    """Copy model source files and rewrite imports to be local (no interventions_rl.model prefix)."""
    for filename in _MODEL_SOURCE_FILES:
        src = os.path.join(_MODEL_SOURCE_DIR, filename)
        dst = os.path.join(vllm_output_dir, filename)
        with open(src, "r") as f:
            content = f.read()
        # Rewrite package imports to local imports for trust_remote_code
        content = content.replace(
            "from interventions_rl.model import ", "from . import "
        )
        content = content.replace("from interventions_rl.model.", "from .")
        with open(dst, "w") as f:
            f.write(content)
    # Write __init__.py so the directory is importable
    init_src = os.path.join(_MODEL_SOURCE_DIR, "__init__.py")
    init_dst = os.path.join(vllm_output_dir, "__init__.py")
    with open(init_src, "r") as f:
        content = f.read()
    content = content.replace("from .", "from .")  # already relative
    with open(init_dst, "w") as f:
        f.write(content)
    logger.info(
        f"Copied {len(_MODEL_SOURCE_FILES) + 1} model source files to {vllm_output_dir}"
    )


def _patch_config_auto_map(vllm_output_dir: str, model_key: str) -> None:
    """Patch config.json with auto_map so transformers loads our custom model class."""
    config_path = os.path.join(vllm_output_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoModelForCausalLM": _AUTO_MAP[model_key],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Patched config.json with auto_map -> {_AUTO_MAP[model_key]}")


def convert_to_vllm_compatible_checkpoint(
    model_name: str,
    output_dir: str,
    intervention_type: str = "LoreftIntervention",
    intervention_layers: str = "all",
    low_rank_dimension: int = 128,
    dropout: float = 0.0,
    act_fn: str | None = "gelu",
    init_orth: bool = True,
    dtype: str = "bfloat16",
) -> str:
    """Build interventions model and save as a vLLM-compatible checkpoint.

    Saves the full state dict (base + intervention weights), model source files,
    config.json with auto_map, and tokenizer so vLLM can load it with:
        vllm serve <path> --model-impl transformers --trust-remote-code
    """
    safe_name = sanitize_filename(model_name.replace("/", "_"))
    vllm_output_dir = os.path.join(output_dir, safe_name)
    os.makedirs(vllm_output_dir, exist_ok=True)

    # Build interventions config
    iv_config = interventions_utils.InterventionsConfig(
        intervention_type=intervention_type,
        intervention_layers=intervention_layers,
        low_rank_dimension=low_rank_dimension,
        dropout=dropout,
        act_fn=act_fn,
        init_orth=init_orth,
    )

    # Load model with interventions
    map_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model_class, model_key = _get_model_class(model_name)

    logger.info(f"Loading {model_name} with {intervention_type} interventions...")
    report, model = load_interventions_model(
        hf_model_name_or_path=model_name,
        model_class=model_class,
        ic_config=iv_config,
        map_dtype=map_dtype,
        map_device=torch.device("cpu"),
    )
    logger.info(f"Model loaded: {report.summary()}")

    # Save full state dict (base + intervention weights, clone shared tensors)
    state_dict = {}
    seen_data_ptrs: dict[int, str] = {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr in seen_data_ptrs:
            state_dict[k] = v.clone().contiguous()
        else:
            seen_data_ptrs[ptr] = k
            state_dict[k] = v.contiguous()
    save_file(state_dict, os.path.join(vllm_output_dir, "model.safetensors"))
    logger.info(f"Saved {len(state_dict)} params to model.safetensors")

    # Save interventions config
    with open(os.path.join(vllm_output_dir, "interventions_config.json"), "w") as f:
        f.write(iv_config.to_json())

    # Download HF config and tokenizer
    _download_config_from_hf(model_name=model_name, vllm_output_dir=vllm_output_dir)

    # Copy model source files for trust_remote_code
    _copy_model_sources(vllm_output_dir)

    # Patch config.json with auto_map
    _patch_config_auto_map(vllm_output_dir, model_key)

    logger.info(f"vLLM-compatible checkpoint saved to: {vllm_output_dir}")
    logger.info(
        f"Serve with: vllm serve {vllm_output_dir} --model-impl transformers --trust-remote-code"
    )
    return vllm_output_dir


def _download_config_from_hf(model_name: str, vllm_output_dir: str) -> None:
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.save_pretrained(vllm_output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(vllm_output_dir)

    logger.info("Config and tokenizer saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build interventions model and save as vLLM checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF model name (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models", help="Output directory"
    )
    parser.add_argument("--intervention_type", type=str, default="LoreftIntervention")
    parser.add_argument("--intervention_layers", type=str, default="all")
    parser.add_argument("--low_rank_dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--act_fn", type=str, default="gelu")
    parser.add_argument("--init_orth", action="store_true", default=True)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"]
    )
    args = parser.parse_args()

    path = convert_to_vllm_compatible_checkpoint(
        model_name=args.model_name,
        output_dir=args.output_dir,
        intervention_type=args.intervention_type,
        intervention_layers=args.intervention_layers,
        low_rank_dimension=args.low_rank_dimension,
        dropout=args.dropout,
        act_fn=args.act_fn,
        init_orth=args.init_orth,
        dtype=args.dtype,
    )
    print(f"\nCheckpoint ready at: {path}")
    print(
        f"Serve with: uv run vllm serve {path} --model-impl transformers --trust-remote-code"
    )

import pydantic
import torch
import torch.nn as nn
import configmate
from typing import Annotated, Literal, cast, Self
from interventions_rl.model import interventions

InterventionLayers = Literal["all", "odd_only", "even_only", "last_only"]
InterventionType = Literal["LoreftIntervention", "DireftIntervention"]

InterventionTypeRegistry = {
    "LoreftIntervention": interventions.LoreftIntervention,
    "DireftIntervention": interventions.DireftIntervention,
}


class InterventionsConfig(pydantic.BaseModel):
    """Configuration settings controlling REFT / DIREFT interventions applied to MoE layers."""

    intervention_type: Annotated[
        Literal["LoreftIntervention", "DireftIntervention"],
        pydantic.Field(
            description=(
                "Specifies which intervention module to inject. "
                "'LoreftIntervention' applies low-rank feature transformations "
                "(LoReFT-style REFT), while 'DireftIntervention' applies "
                "directional interventions (DiReFT) using directional vectors."
            ),
        ),
    ]

    intervention_layers: Annotated[
        Literal["all", "odd_only", "even_only", "last_only"],
        pydantic.Field(
            description=(
                "Controls which transformer layers receive interventions:\n"
                "- 'all': apply to every layer\n"
                "- 'odd_only': apply only to odd-numbered layers\n"
                "- 'even_only': apply only to even-numbered layers\n"
                "- 'last_only': apply only to the last layer before the language model head\n"
                "Useful for ablations and reducing compute overhead."
            ),
        ),
    ]

    low_rank_dimension: Annotated[
        int,
        pydantic.Field(
            description="Rank of the low-rank projection used for intervention layers."
        ),
    ]

    dropout: Annotated[
        float,
        pydantic.Field(
            description="Dropout probability applied inside the intervention layer. Useful for regularization"
        ),
    ]

    act_fn: Annotated[
        str | None,
        pydantic.Field(
            description=(
                "Optional activation function used inside the intervention module. "
                "Examples: 'gelu', 'relu', 'silu'. If None, the intervention is linear."
            ),
        ),
    ]

    init_orth: Annotated[
        bool,
        pydantic.Field(
            default=True,
            description="Whether to orthogonally initialize the low-rank projection matrices.",
        ),
    ]

    @pydantic.model_validator(mode="after")
    def validate_interventions_config(self) -> Self:
        assert (
            self.low_rank_dimension >= 1
        ), f"{self.low_rank_dimension=} cannot be less than 1"
        return self

    def to_json(self, **kwargs) -> str:
        """Convenient shortcut for JSON serialization."""
        return self.model_dump_json(indent=2, **kwargs)


def interventions_based_layer_idx(
    interventions_config: InterventionsConfig, layer_idx: int, num_layers: int
) -> bool:
    if (
        interventions_config.intervention_layers == "last_only"
        and layer_idx == num_layers - 1
    ):
        return True
    if interventions_config.intervention_layers == "all":
        return True
    elif interventions_config.intervention_layers == "even_only":
        if layer_idx % 2 == 0:
            return True
    elif interventions_config.intervention_layers == "odd_only":
        if layer_idx % 2 != 0:
            return True
    return False


def build_intervention(
    *,
    icfg: InterventionsConfig | None,
    layer_idx: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Create the requested intervention or Identity if not applicable."""
    if icfg is None:
        return nn.Identity()
    if not interventions_based_layer_idx(icfg, layer_idx, num_layers):
        return nn.Identity()

    # Enforce allowed types
    if icfg is not None and icfg.intervention_type not in InterventionTypeRegistry:
        raise ValueError(
            f"Unsupported intervention_type={icfg.intervention_type!r}. "
            f"Allowed: Loreft and Direft"
        )

    cls = InterventionTypeRegistry[icfg.intervention_type]
    mod = cls(
        embed_dim=hidden_size,
        low_rank_dimension=icfg.low_rank_dimension,
        dropout=icfg.dropout,
        act_fn=icfg.act_fn,
        init_orth=icfg.init_orth,
    )
    result = cast(torch.nn.Module, mod)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def read_config_from_yaml(path: str) -> InterventionsConfig:
    assert path.endswith(".yaml"), f"Config path must end with .yaml, got: {path}"
    config = configmate.get_config(
        path, section="InterventionsConfig", validation=InterventionsConfig
    )
    return config


VLLM_ARCHITECTURE_MAPPING: dict[str, str] = {
    "Qwen2ForCausalLM": "Qwen2InterventionsForCausalLM",
    "Qwen3ForCausalLM": "Qwen3InterventionsForCausalLM",
    "LlamaForCausalLM": "LlamaInterventionsForCausalLM",
}


def read_interventions_config_from_hf(hf_config) -> InterventionsConfig:
    """Extract InterventionsConfig from a HuggingFace PretrainedConfig.

    Expects config.json to contain an ``interventions_config`` dict
    (written by ``prepare_model_for_vllm``).
    """
    ic_dict = getattr(hf_config, "interventions_config", None)
    if ic_dict is None:
        raise ValueError(
            "interventions_config not found in model config.json. "
            "Use interventions_utils.prepare_model_for_vllm() to create "
            "a model directory with the required config."
        )
    return InterventionsConfig(**ic_dict)


def prepare_model_for_vllm(
    model_name_or_path: str,
    interventions_config: InterventionsConfig,
    output_dir: str,
) -> str:
    """Create a local model directory whose config.json contains the
    interventions architecture name and config, with symlinks to the
    original model files (weights, tokenizer, etc.).

    Returns output_dir so it can be used as the vLLM model path.
    """
    import os
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    src = Path(model_name_or_path)
    if not src.is_dir():
        src = Path(snapshot_download(model_name_or_path))

    config = AutoConfig.from_pretrained(str(src), trust_remote_code=True)

    original_arch = (config.architectures or [None])[0]
    new_arch = VLLM_ARCHITECTURE_MAPPING.get(original_arch)
    if new_arch is None:
        raise ValueError(
            f"No interventions architecture registered for: {original_arch}. "
            f"Supported: {list(VLLM_ARCHITECTURE_MAPPING.keys())}"
        )

    config.architectures = [new_arch]
    config.interventions_config = interventions_config.model_dump()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(out))

    for src_file in src.iterdir():
        if src_file.name == "config.json":
            continue
        dst_file = out / src_file.name
        if not dst_file.exists():
            os.symlink(src_file.resolve(), dst_file)

    return str(out)

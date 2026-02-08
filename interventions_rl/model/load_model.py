from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence
import tqdm
import fnmatch
import torch
from loguru import logger
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from interventions_rl.model import interventions_utils

from interventions_rl.model import llama, qwen3


@dataclass
class TransferReport:
    copied: list[str]
    skipped_shape: list[tuple[str, torch.Size, torch.Size]]
    skipped_missing: list[str]
    skipped_intervention: list[str]

    def summary(self) -> str:
        return (
            f"Copied: {len(self.copied)} | "
            f"Skipped (shape): {len(self.skipped_shape)} | "
            f"Skipped (missing): {len(self.skipped_missing)} | "
            f"Skipped (intervention): {len(self.skipped_intervention)}"
        )

    @classmethod
    def from_empty(cls) -> TransferReport:
        return TransferReport(
            copied=[], skipped_shape=[], skipped_missing=[], skipped_intervention=[]
        )


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def build_partial_state_dict(
    src_sd: Mapping[str, torch.Tensor],
    dst_module: nn.Module,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    dst_sd: MutableMapping[str, torch.Tensor] = dst_module.state_dict()
    out: dict[str, torch.Tensor] = {}

    report = TransferReport.from_empty()

    # tqdm over keys so we can get len(src_sd)
    for src_name in tqdm.tqdm(
        src_sd.keys(),
        total=len(src_sd),
        desc="Building partial state dict",
    ):
        src_tensor: torch.Tensor = src_sd[src_name]
        matched_dst: str | None = src_name if src_name in dst_sd else None

        if matched_dst is None:
            report.skipped_missing.append(src_name)
            logger.info(f"Missing for {src_name=}")
            continue

        dst_t: torch.Tensor = dst_sd[matched_dst]

        if src_tensor.shape != dst_t.shape:
            report.skipped_shape.append((matched_dst, src_tensor.shape, dst_t.shape))
            logger.error(
                f"The shape for {src_name=} with {src_tensor.shape=} "
                f"didn't match {matched_dst=} with {dst_t.shape=}"
            )
            continue

        if dtype is not None or device is not None:
            src_tensor = src_tensor.to(
                device=device if device is not None else dst_t.device,
                dtype=dtype if dtype is not None else dst_t.dtype,
            )

        out[matched_dst] = src_tensor
        report.copied.append(matched_dst)

    return out, report


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params: int = 0
    trainable_params: int = 0
    for p in model.parameters():
        numel: int = p.numel()
        total_params += numel
        if p.requires_grad:
            trainable_params += numel
    return total_params, trainable_params


def load_hf_into_custom_model(
    *,
    hf_model_name_or_path: str | None = None,
    pt_file: str | None = None,
    custom_model: nn.Module,
    hf_config: AutoConfig,
    full_parameter_finetuning: bool = False,
    map_dtype: torch.dtype | None = None,
    map_device: torch.device | None = None,
    trust_remote_code: bool = False,
) -> tuple[TransferReport, nn.Module]:

    if (hf_model_name_or_path is None) == (pt_file is None):
        raise ValueError(
            "You must specify exactly one of hf_model_name_or_path or pt_file."
        )

    # 1) Load state dict from HF or torch file
    if hf_model_name_or_path is not None:
        hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            hf_model_name_or_path,
            dtype=map_dtype if map_dtype is not None else None,
            trust_remote_code=trust_remote_code,
        )
        src_sd = hf_model.state_dict()
    else:
        assert pt_file
        src_sd = torch.load(pt_file, map_location="cpu")
        if isinstance(src_sd, dict) and "model_state_dict" in src_sd:
            src_sd = src_sd["model_state_dict"]

    # 2) Build filtered state dict compatible with your custom model
    filtered_sd, report = build_partial_state_dict(
        src_sd=src_sd,
        dst_module=custom_model,
        device=map_device,
        dtype=map_dtype,
    )
    missing, unexpected = custom_model.load_state_dict(filtered_sd, strict=False)

    # Merge loader feedback into the report
    report.skipped_missing.extend(missing)
    if unexpected:
        report.skipped_missing.extend(unexpected)

    # 4) Freeze everything, then unfreeze only intervention layers unless doing full-parameter finetuning
    if full_parameter_finetuning:
        for param in custom_model.parameters():
            param.requires_grad = True
    else:
        for param in custom_model.parameters():
            param.requires_grad = False

        for name, param in custom_model.named_parameters():
            if not "intervention" in name:
                continue
            if interventions_utils.interventions_based_layer_idx(
                custom_model.interventions_config, name, hf_config.num_hidden_layers
            ):
                param.requires_grad = True

    # 5) Print parameter stats
    total_params, trainable_params = count_parameters(custom_model)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    logger.info(
        f"Parameter stats — total: {total_params}, trainable: {trainable_params}"
    )
    logger.info(f"Trainable parameters: {trainable_params / total_params * 100:.2f}%")
    return report, custom_model


def load_interventions_model(
    hf_model_name_or_path: str,
    model_class: type[qwen3.Qwen3ForCausalLM] | type[llama.LlamaForCausalLM],
    ic_config: interventions_utils.InterventionsConfig,
    map_dtype: torch.dtype = torch.float32,
    map_device: torch.device = torch.device("cuda"),
    trust_remote_code: bool = True,
) -> tuple[TransferReport, nn.Module]:
    hf_config = AutoConfig.from_pretrained(
        hf_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    custom_model = model_class(
        interventions_config=ic_config,
        config=hf_config,
    )
    report, module = load_hf_into_custom_model(
        hf_model_name_or_path=hf_model_name_or_path,
        custom_model=custom_model,
        hf_config=hf_config,
        map_dtype=map_dtype,
        map_device=map_device,
        trust_remote_code=trust_remote_code,
    )
    return report, module


if __name__ == "__main__":
    from interventions_rl.model import qwen3

    hf_model_name_or_path = "Qwen/Qwen3-1.7B"
    hf_config = AutoConfig.from_pretrained(
        hf_model_name_or_path,
        trust_remote_code=True,
    )
    custom_model = qwen3.Qwen3ForCausalLM(
        interventions_config=interventions_utils.InterventionsConfig(
            intervention_type="LoreftIntervention",
            intervention_layers="all",
            low_rank_dimension=128,
            dropout=0.0,
            act_fn="gelu",
        ),
        config=hf_config,
    )

    # 2) Load HF weights into the overlapping parts, skipping interventions
    report, module = load_hf_into_custom_model(
        hf_model_name_or_path=hf_model_name_or_path,
        custom_model=custom_model,
        hf_config=hf_config,
        map_dtype=torch.bfloat16,  # optional casting
        map_device=torch.device("cuda"),  # optional device move
        trust_remote_code=True,
    )

    print(report.summary())

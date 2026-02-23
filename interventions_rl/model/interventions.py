import torch
from collections import OrderedDict

from pyvene import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN


# -----------------------------
# Helpers
# -----------------------------
def _get_device(kwargs: dict) -> torch.device | None:
    dev = kwargs.get("device", None)
    if dev is None:
        return None
    return dev if isinstance(dev, torch.device) else torch.device(dev)


def _get_compute_dtype(kwargs: dict) -> torch.dtype:
    # This is your "training" dtype (bf16/fp16/fp32). rotate weights will stay fp32.
    return kwargs.get("dtype", torch.float32)


@torch.no_grad()
def force_rotate_layers_fp32(module: torch.nn.Module) -> None:
    """
    Call this ONCE after you do model.to(bf16) and/or after FSDP wrapping.
    It forces all modules with .rotate_layer to stay in fp32, preventing orgqr_cuda bf16 crashes.
    """
    for m in module.modules():
        if hasattr(m, "rotate_layer"):
            getattr(m, "rotate_layer").to(dtype=torch.float32)


# -----------------------------
# Core building block
# -----------------------------
class LowRankRotateLayer(torch.nn.Module):
    """
    A linear transformation with orthogonal initialization.
    IMPORTANT: this module MUST remain fp32 when using orthogonal parametrization on CUDA,
    because torch.linalg.householder_product / orgqr_cuda doesn't support bf16.
    """

    def __init__(self, n: int, m: int, init_orth: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(n, m, dtype=torch.float32), requires_grad=True
        )
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly disable autocast so no bf16 sneaks into the orthogonal parametrization path.
        x = x.to(dtype=torch.float32)
        weight_fp32 = self.weight.to(torch.float32)
        return x.matmul(weight_fp32)


class _BaseReftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    Common utilities:
    - learned_source / proj_layer can be compute_dtype (bf16/fp16/fp32)
    - rotate_layer MUST be fp32
    - forward() blocks that touch rotate_layer run with autocast disabled
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        self._device = _get_device(kwargs)
        self._compute_dtype = _get_compute_dtype(kwargs)

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        act_key = kwargs.get("act_fn", None)
        self.act_fn = ACT2FN["linear"] if act_key is None else ACT2FN[act_key]

    def _make_rotate_layer(
        self, low_rank_dimension: int, init_orth: bool = True
    ) -> torch.nn.Module:
        rotate = LowRankRotateLayer(
            self.embed_dim, low_rank_dimension, init_orth=init_orth
        )
        # Keep a plain fp32 parameter here. Runtime orthogonal parametrization uses
        # householder_product/orgqr kernels that are not implemented for bf16 on CUDA,
        # and AMP/FSDP/DeepSpeed can still cast parametrization internals to bf16.
        rotate = rotate.to(device=self._device, dtype=torch.float32)  # MUST stay fp32
        return rotate

    def _no_autocast(self):
        return torch.autocast(device_type="cuda", enabled=False)


# -----------------------------
# Interventions
# -----------------------------
class LoreftIntervention(_BaseReftIntervention):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        r = int(kwargs["low_rank_dimension"])

        self.learned_source = torch.nn.Linear(self.embed_dim, r).to(
            device=self._device, dtype=self._compute_dtype
        )
        self.rotate_layer = self._make_rotate_layer(r, init_orth=True)

    def forward(
        self, base: torch.Tensor, source: torch.Tensor | None = None, subspaces=None
    ) -> torch.Tensor:
        # Everything touching rotate_layer runs in fp32 with autocast disabled.
        with self._no_autocast():
            base_f = base.to(torch.float32)
            rotated_base = self.rotate_layer(base_f)  # fp32
            learned_source_fp32 = self.learned_source.to(torch.float32)
            delta = self.act_fn(learned_source_fp32(base_f)) - rotated_base  # fp32
            rotate_layer = self.rotate_layer.weight.to(torch.float32)
            out_f = base_f + delta.matmul(rotate_layer.T)  # fp32
            out_f = self.dropout(out_f)

        return out_f.to(dtype=torch.bfloat16)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        if destination is None:
            destination = OrderedDict()

        for k, v in self.learned_source.state_dict(keep_vars=keep_vars).items():
            destination[prefix + "learned_source." + k] = v

        destination[prefix + "rotate_layer"] = (
            self.rotate_layer.weight if keep_vars else self.rotate_layer.weight.data
        )
        return destination

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        learned_source_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("learned_source."):
                learned_source_sd[k[len("learned_source.") :]] = v
            elif k in ("weight", "bias"):
                learned_source_sd[k] = v

        if learned_source_sd:
            self.learned_source.load_state_dict(learned_source_sd, strict=False)

        rotate_layer_key = "rotate_layer"
        if rotate_layer_key not in state_dict:
            if strict:
                raise KeyError(f"Missing key: {rotate_layer_key}")
            return

        overload_w = state_dict[rotate_layer_key].to(
            device=self.learned_source.weight.device, dtype=torch.float32
        )
        overload_w_width = int(overload_w.shape[-1])

        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(device=self.learned_source.weight.device, dtype=torch.float32)
        with torch.no_grad():
            rotate_layer.weight.copy_(overload_w)
        self.rotate_layer = rotate_layer

        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) is True


class NoreftIntervention(_BaseReftIntervention):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        r = int(kwargs["low_rank_dimension"])
        add_bias = bool(kwargs.get("add_bias", True))

        self.proj_layer = torch.nn.Linear(self.embed_dim, r, bias=add_bias).to(
            device=self._device, dtype=self._compute_dtype
        )
        self.learned_source = torch.nn.Linear(self.embed_dim, r).to(
            device=self._device, dtype=self._compute_dtype
        )

    def forward(
        self, base: torch.Tensor, source: torch.Tensor | None = None, subspaces=None
    ) -> torch.Tensor:
        # No orthogonal parametrization here, but keeping fp32 block is fine for numerical stability.
        with self._no_autocast():
            base_f = base.to(torch.float32)
            proj_base = self.proj_layer(base_f)
            delta = self.act_fn(self.learned_source(base_f)) - proj_base
            proj_layer_fp32 = self.proj_layer.to(torch.float32)
            out_f = base_f + delta.matmul(proj_layer_fp32)
            out_f = self.dropout(out_f)

        return out_f.to(dtype=base.dtype)


class ConsreftIntervention(_BaseReftIntervention):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        r = int(kwargs["low_rank_dimension"])

        self.rotate_layer = self._make_rotate_layer(r, init_orth=True)
        self.learned_source = torch.nn.Parameter(
            torch.rand(r, dtype=torch.float32), requires_grad=True
        )

    def forward(
        self, base: torch.Tensor, source: torch.Tensor | None = None, subspaces=None
    ) -> torch.Tensor:
        with self._no_autocast():
            base_f = base.to(torch.float32)
            rotated_base = self.rotate_layer(base_f)
            out_f = base_f + (self.learned_source - rotated_base).matmul(
                self.rotate_layer.weight.T
            )
        return out_f.to(dtype=base.dtype)


class LobireftIntervention(_BaseReftIntervention):
    """
    LobiReFT(h) = h + R^T(b)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        r = int(kwargs["low_rank_dimension"])

        self.rotate_layer = self._make_rotate_layer(r, init_orth=True)
        self.learned_source = torch.nn.Parameter(
            torch.rand(r, dtype=torch.float32), requires_grad=True
        )

    def forward(
        self, base: torch.Tensor, source: torch.Tensor | None = None, subspaces=None
    ) -> torch.Tensor:
        with self._no_autocast():
            base_f = base.to(torch.float32)
            out_f = base_f + self.learned_source.matmul(self.rotate_layer.weight.T)
            out_f = self.dropout(out_f)
        return out_f.to(dtype=base.dtype)


class DireftIntervention(_BaseReftIntervention):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        r = int(kwargs["low_rank_dimension"])

        self.rotate_layer = self._make_rotate_layer(r, init_orth=True)
        self.learned_source = torch.nn.Linear(self.embed_dim, r).to(
            device=self._device, dtype=self._compute_dtype
        )

    def forward(
        self, base: torch.Tensor, source: torch.Tensor | None = None, subspaces=None
    ) -> torch.Tensor:
        with self._no_autocast():
            base_f = base.to(torch.float32)
            out_f = base_f + self.act_fn(self.learned_source(base_f)).matmul(
                self.rotate_layer.weight.T
            )
            out_f = self.dropout(out_f)
        return out_f.to(dtype=base.dtype)


class NodireftIntervention(_BaseReftIntervention):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        r = int(kwargs["low_rank_dimension"])
        add_bias = bool(kwargs.get("add_bias", True))

        self.proj_layer = torch.nn.Linear(self.embed_dim, r, bias=add_bias).to(
            device=self._device, dtype=self._compute_dtype
        )
        self.learned_source = torch.nn.Linear(self.embed_dim, r).to(
            device=self._device, dtype=self._compute_dtype
        )

    def forward(
        self, base: torch.Tensor, source: torch.Tensor | None = None, subspaces=None
    ) -> torch.Tensor:
        with self._no_autocast():
            base_f = base.to(torch.float32)
            proj_layer_fp32 = self.proj_layer.to(torch.float32)
            out_f = base_f + self.act_fn(self.learned_source(base_f)).matmul(
                proj_layer_fp32
            )
            out_f = self.dropout(out_f)
        return out_f.to(dtype=base.dtype)

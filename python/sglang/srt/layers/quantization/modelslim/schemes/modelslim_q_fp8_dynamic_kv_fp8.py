"""ModelSlim Q-FP8 dynamic / KV-FP8 attention scale loading."""

from __future__ import annotations

import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_scheme import (
    ModelSlimKVSchemeBase,
)


def _modelslim_kv_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """Load a scalar/local scale or shard a full-head tensor on dimension zero."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
        return

    loaded_weight = loaded_weight.to(param.dtype)
    if loaded_weight.shape != param.shape:
        if loaded_weight.ndim == 0:
            raise ValueError(
                "Cannot shard a scalar ModelSlim KV weight into parameter shape "
                f"{tuple(param.shape)}."
            )
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard ModelSlim KV weight {tuple(loaded_weight.shape)} "
                f"across tensor parallel size {tp_size}."
            )
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)

    if loaded_weight.shape != param.shape:
        raise ValueError(
            f"Attempted to load ModelSlim KV weight {tuple(loaded_weight.shape)} "
            f"into parameter {tuple(param.shape)}."
        )
    param.data.copy_(loaded_weight)


class ModelSlimQFP8DynamicKVFP8Scheme(ModelSlimKVSchemeBase):
    """Register checkpoint Q/K/V scales and derived K descales for NPU MLA."""

    def __init__(self, quant_config, prefix: str) -> None:
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: nn.Module,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:
        if num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError(
                "ModelSlim KV head counts must be positive, got "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}."
            )

        head_counts = {
            "fa_q": num_heads,
            "fa_k": num_kv_heads,
            "fa_v": num_kv_heads,
        }
        for module_name, head_count in head_counts.items():
            module = layer._modules.get(module_name)
            if module is None:
                module = nn.Module()
                layer.add_module(module_name, module)

            # NaN is an explicit not-loaded sentinel. A missing checkpoint scale
            # must never quietly become a unit scale.
            if "scale" not in module._parameters:
                scale = nn.Parameter(
                    torch.full((head_count, 1), torch.nan, dtype=torch.float32),
                    requires_grad=False,
                )
                scale.weight_loader = _modelslim_kv_weight_loader
                module.register_parameter("scale", scale)
            elif module.scale.shape != (head_count, 1):
                raise ValueError(
                    f"Existing {module_name}.scale has shape "
                    f"{tuple(module.scale.shape)}, expected {(head_count, 1)}."
                )
            if "offset" not in module._parameters:
                offset = nn.Parameter(
                    torch.zeros((head_count, 1), dtype=torch.float32),
                    requires_grad=False,
                )
                offset.weight_loader = _modelslim_kv_weight_loader
                module.register_parameter("offset", offset)
            elif module.offset.shape != (head_count, 1):
                raise ValueError(
                    f"Existing {module_name}.offset has shape "
                    f"{tuple(module.offset.shape)}, expected {(head_count, 1)}."
                )

        runtime_shapes = {
            "fak_descale_float": (1, num_kv_heads),
            "fak_descale_reciprocal": (1, num_kv_heads),
        }
        for name, shape in runtime_shapes.items():
            if name not in layer._parameters:
                layer.register_parameter(
                    name,
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )
            elif layer._parameters[name].shape != shape:
                raise ValueError(
                    f"Existing {name} has shape "
                    f"{tuple(layer._parameters[name].shape)}, expected {shape}."
                )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for name in ("fa_q", "fa_k", "fa_v"):
            scale = layer._modules[name]._parameters["scale"]
            if not torch.isfinite(scale).all():
                raise RuntimeError(
                    f"Missing ModelSlim {name}.scale for {self.prefix}; "
                    "Q_FP8_DYNAMIC_KV_FP8 does not permit a unit-scale fallback."
                )
            if (scale <= 0).any():
                raise ValueError(
                    f"ModelSlim {name}.scale for {self.prefix} must be positive."
                )

        fa_k_scale = layer.fa_k.scale.reshape(1, -1).to(torch.float32)
        layer.fak_descale_float.data.copy_(fa_k_scale)
        layer.fak_descale_reciprocal.data.copy_(torch.reciprocal(fa_k_scale))

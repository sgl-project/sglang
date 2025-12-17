from typing import Union

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear


def fuse_linear_projections(
    q_proj: Union[nn.Linear, ReplicatedLinear],
    k_proj: Union[nn.Linear, ReplicatedLinear],
    v_proj: Union[nn.Linear, ReplicatedLinear],
    use_bias: bool,
    linear_cls: type = None,
) -> Union[nn.Linear, ReplicatedLinear]:
    device = q_proj.weight.data.device
    dtype = q_proj.weight.data.dtype
    bias_dtype = q_proj.bias.data.dtype

    concatenated_weights = torch.cat(
        [q_proj.weight.data, k_proj.weight.data, v_proj.weight.data]
    )
    in_features = concatenated_weights.shape[1]
    out_features = concatenated_weights.shape[0]

    if linear_cls is None:
        linear_cls = type(q_proj)

    fused_layer = linear_cls(
        in_features, out_features, bias=use_bias, quant_config=q_proj.quant_config
    )
    fused_layer.weight.data = concatenated_weights.to(device=device, dtype=dtype)

    if hasattr(q_proj, "weight_scale_inv"):
        concatenated_weights_scale = torch.cat(
            [
                q_proj.weight_scale_inv.data,
                k_proj.weight_scale_inv.data,
                v_proj.weight_scale_inv.data,
            ]
        )
        fused_layer.weight_scale_inv.data = concatenated_weights_scale

    if use_bias:
        concatenated_bias = torch.cat(
            [q_proj.bias.data, k_proj.bias.data, v_proj.bias.data]
        )
        fused_layer.bias.data = concatenated_bias.to(device=device, dtype=bias_dtype)

    return fused_layer


def delete_projection_layers(module: nn.Module, layer_names: list[str]) -> None:
    for name in layer_names:
        if hasattr(module, name):
            delattr(module, name)

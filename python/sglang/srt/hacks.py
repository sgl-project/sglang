from typing import Tuple

import torch
from tqdm import trange

from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

# def hack_requant_moe_weight(that, weights):
#     print("hi hack_requant_moe_weight")
#
#     weights_dict = dict(list(weights))
#     del weights
#
#     moe_layers = range(
#         that.config.first_k_dense_replace,
#         that.config.num_hidden_layers,
#         that.config.moe_layer_freq,
#     )
#
#     module_names = [
#         "down_proj",
#         "gate_proj",
#         "up_proj",
#     ]
#
#     for moe_layer in moe_layers:
#         for expert_index in trange(
#             that.config.n_routed_experts, desc=f"layer={moe_layer}"
#         ):
#             for module_name in module_names:
#                 partial_name = (
#                     f"model.layers.{moe_layer}.mlp.experts.{expert_index}.{module_name}"
#                 )
#                 name_weight = partial_name + ".weight"
#                 name_weight_scale_inv = partial_name + ".weight_scale_inv"
#
#                 weight_new, weight_scale_inv_new = _requant_moe_weight(
#                     that, weights_dict[name_weight], weights_dict[name_weight_scale_inv]
#                 )
#
#                 weights_dict[name_weight] = weight_new
#                 weights_dict[name_weight_scale_inv] = weight_scale_inv_new
#
#     return list(weights_dict.items())
#
#
# def _requant_moe_weight(that, weight: torch.Tensor, weight_scale_inv: torch.Tensor):
#     weight_block_size = that.quant_config.weight_block_size
#
#     assert weight_block_size == [128, 128]
#
#     weight_dequant = block_quant_dequant(
#         weight,
#         # TODO does "inv" have trouble?
#         weight_scale_inv,
#         weight_block_size,
#         # TODO correct?
#         torch.float32,
#     )
#
#     return per_block_cast_to_fp8(weight_dequant)
#


def hack_requant_moe_weight_at_post_load_weights(that):
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    moe_layers = list(
        range(
            that.config.first_k_dense_replace,
            that.config.num_hidden_layers,
            that.config.moe_layer_freq,
        )
    )
    for layer_id in trange(that.config.num_hidden_layers):
        layer = that.model.layers[layer_id]
        # print([(name, param.shape, param.dtype) for name, param in self_attn.named_parameters()])

        for module in [
            layer.self_attn.fused_qkv_a_proj_with_mqa,
            layer.self_attn.q_b_proj,
            layer.self_attn.kv_b_proj,
            layer.self_attn.o_proj,
        ]:
            _requant_grouped_moe_weight_inplace(
                that, module.weight, module.weight_scale_inv
            )

        if layer_id in moe_layers:
            shared_experts = layer.mlp.shared_experts
            for module in [
                shared_experts.gate_up_proj,
                shared_experts.down_proj,
            ]:
                _requant_grouped_moe_weight_inplace(
                    that, module.weight, module.weight_scale_inv
                )

            experts = layer.mlp.experts
            assert isinstance(experts, DeepEPMoE)
            for w in [
                experts.w13_weight_fp8,
                experts.w2_weight_fp8,
            ]:
                _requant_grouped_moe_weight_inplace(that, w[0], w[1])
        else:
            mlp = layer.mlp
            assert isinstance(mlp, DeepseekV2MLP)
            for module in [
                mlp.gate_up_proj,
                mlp.down_proj,
            ]:
                _requant_grouped_moe_weight_inplace(
                    that, module.weight, module.weight_scale_inv
                )


def _requant_grouped_moe_weight_inplace(that, weight, weight_scale_inv):
    weight[...], weight_scale_inv[...] = _requant_grouped_moe_weight(
        that, weight, weight_scale_inv
    )


def _requant_grouped_moe_weight(
    that, weight: torch.Tensor, weight_scale_inv: torch.Tensor
):
    weight_block_size = that.quant_config.weight_block_size
    assert weight_block_size == [128, 128]

    *_, n, k = weight.shape

    print(
        f"requant_grouped_moe_weight {weight.shape=} {weight.dtype=} {weight_scale_inv.shape=} {weight_scale_inv.dtype=}"
    )

    weight_dequant = block_quant_dequant(
        weight,
        weight_scale_inv,
        weight_block_size,
        # TODO is it ok?
        torch.bfloat16,
    )

    assert n % 128 == 0
    assert k % 128 == 0
    # weight_dequant_flat = einops.rearrange(weight_dequant, "num_group n k -> (num_group n) k")
    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat)

    # def _unflatten(x):
    #     return einops.rearrange(
    #         x,
    #         "(num_group n_div_128) whatever_div_128 -> num_group n_div_128 whatever_div_128",
    #         num_group=num_group,
    #     )
    #
    # return _unflatten(out_w_flat), _unflatten(out_s_flat)

    return (
        out_w_flat.view(weight.shape),
        out_s_flat.view(weight_scale_inv.shape),
    )


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    # TODO: stronger tests
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

import torch
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)


def process_weights_after_loading(layer):

    w_q_name = "weight"
    w_s_name = "weight_scale"
    i_s_name = "input_scale"
    i_zp_name = "input_zero_point"
    azp_adj_name = "azp_adj"

    # WEIGHT
    # Cutlass kernels need transposed weight.
    weight = getattr(layer, w_q_name)
    replace_parameter(
        layer, w_q_name, torch.nn.Parameter(weight.t().data, requires_grad=False)
    )

    # WEIGHT SCALE
    # Cutlass kernels support only per-tensor and per-channel.
    # If we have a fused module (QKV, MLP) with per tensor scales (thus N
    # scales being passed to the kernel), convert to the per-channel case.
    is_fused_module = len(layer.logical_widths) > 1
    weight_scale = getattr(layer, w_s_name)
    # if is_fused_module and not False:
    weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
    replace_parameter(
        layer,
        w_s_name,
        torch.nn.Parameter(weight_scale.data.view(-1, 1), requires_grad=False),
    )

    # INPUT SCALE
    if True:
        input_scale = getattr(layer, i_s_name)

        if True:
            replace_parameter(
                layer,
                i_s_name,
                torch.nn.Parameter(
                    torch.full(
                        (1, 1),
                        input_scale.max(),
                        dtype=torch.float32,
                        device=layer.weight.device,
                    ),
                    requires_grad=False,
                ),
            )
            setattr(layer, i_zp_name, None)
        else:
            input_zero_point = getattr(layer, i_zp_name)

            # reconstruct the ranges
            int8_traits = torch.iinfo(torch.int8)
            azps = input_zero_point.to(dtype=torch.int32)
            range_max = (input_scale * (int8_traits.max - azps)).max()
            range_min = (input_scale * (int8_traits.min - azps)).min()

            scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
            replace_parameter(
                layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
            )

            # AZP loaded as int8 but used as int32
            azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
            replace_parameter(
                layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
            )

    else:
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)

    # azp_adj is the AZP adjustment term, used to account for weights.
    # It does not depend on scales or azp, so it is the same for
    # static and dynamic quantization.
    # For more details, see csrc/quantization/cutlass_w8a8/Epilogues.md
    # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md
    if not True:
        weight = getattr(layer, w_q_name)
        azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
        if True:
            # cutlass_w8a8 requires azp to be folded into azp_adj
            # in the per-tensor case
            azp_adj = getattr(layer, i_zp_name) * azp_adj
        setattr(layer, azp_adj_name, torch.nn.Parameter(azp_adj, requires_grad=False))
    else:
        setattr(layer, azp_adj_name, None)


def scaled_int8_quant(x, i_s, i_zp, symmetric):
    pass


def cutlass_scaled_mm(x_q, w_q, scale_a, scale_b, out_dtype, bias):
    pass

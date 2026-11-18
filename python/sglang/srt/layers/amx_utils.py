import logging

import torch
import transformers

from sglang.srt.utils import cpu_has_amx_support

logger = logging.getLogger(__name__)

from enum import IntEnum


class CPUQuantMethod(IntEnum):
    UNQUANT = 0
    INT8_W8A8 = 1
    FP8_W8A16 = 2
    INT4_W4A8 = 3
    MXFP4 = 4
    FP8_W8A8 = 5
    FP8_W8A8_FULL = 6


class CPUQuantAlgo(IntEnum):
    AWQ = 0
    GPTQ = 1


# mirrors CPUActMethod in sglang/kernels/aot/csrc/cpu/gemm.h
class CPUActMethod(IntEnum):
    SILU_AND_MUL = 0
    SWIGLU = 1
    GELU_AND_MUL = 2


_CPU_ACT_METHOD_BY_NAME = {
    None: CPUActMethod.SILU_AND_MUL,
    "silu": CPUActMethod.SILU_AND_MUL,
    "swiglu": CPUActMethod.SWIGLU,
    "gelu": CPUActMethod.GELU_AND_MUL,
}


def _cpu_act_method(activation: str | None) -> CPUActMethod:
    if activation not in _CPU_ACT_METHOD_BY_NAME:
        raise ValueError(f"Unsupported CPU MoE activation: {activation}")
    return _CPU_ACT_METHOD_BY_NAME[activation]


def cpp_wrapper_enabled() -> bool:
    # cpp_wrapper calls ops with list or str arguments back through Python, so
    # under it the scalar_args overloads are used instead
    from torch._inductor import config as inductor_config

    return inductor_config.cpp_wrapper


def call_fused_experts_cpu(
    cpp_wrapper: bool,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    inplace,
    moe_comp_method,
    w1_scale,
    w2_scale,
    w1_zero,
    w2_zero,
    a1_scale,
    block_size,
    w1_bias,
    w2_bias,
    alpha,
    limit,
    is_vnni,
    activation,
):
    if not cpp_wrapper:
        return torch.ops.sgl_kernel.fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace,
            moe_comp_method,
            w1_scale,
            w2_scale,
            w1_zero,
            w2_zero,
            a1_scale,
            block_size,
            w1_bias,
            w2_bias,
            alpha,
            limit,
            is_vnni,
            activation,
        )
    block_size_n, block_size_k = block_size or (0, 0)
    return torch.ops.sgl_kernel.fused_experts_cpu.scalar_args(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace,
        moe_comp_method,
        w1_scale,
        w2_scale,
        w1_zero,
        w2_zero,
        a1_scale,
        block_size_n,
        block_size_k,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
        _cpu_act_method(activation),
    )


def fast_preprocess_cpu(
    self,
    images: list["torch.Tensor"],
    do_resize: bool,
    size,
    interpolation,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean,
    image_std,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    disable_grouping,
    return_tensors,
    **kwargs,
):
    pixel_values, image_grid_thw = torch.ops.sgl_kernel.image_preprocess_cpu(
        images,
        True,
        do_resize,
        size["shortest_edge"],
        size["longest_edge"],
        "bicubic",
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        temporal_patch_size,
        merge_size,
        True,
        torch.bfloat16,
    )
    return transformers.image_processing_base.BatchFeature(
        data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        tensor_type=return_tensors,
    )


def amx_process_weight_after_loading(weight, is_conv=False):
    if weight.device != torch.device("cpu"):
        return weight
    if not cpu_has_amx_support():
        return weight
    if is_conv:
        if weight.dim() == 5:
            return torch.ops.sgl_kernel.conv3d_embed_weight_pack(weight)
        return torch.ops.sgl_kernel.causal_conv1d_weight_pack(
            weight.view(-1, weight.size(-1))
        )
    return torch.ops.sgl_kernel.convert_weight_packed(weight)


# TODO: currently gemm kernel has the below requirements:
# OC: OC % TILE_N == 0 or OC < TILE_N, where TILE_N = 16
# IC: IC % TILE_K == 0, where TILE_K = 32
def dim_is_supported(weight):
    TILE_N = 16
    TILE_K = 32
    ndim = weight.ndim
    OC = weight.size(1) if ndim == 3 else weight.size(0)
    IC = weight.size(2) if ndim == 3 else weight.size(1)
    is_oc_support = OC < TILE_N or OC % TILE_N == 0
    is_ic_support = IC % TILE_K == 0
    return is_oc_support and is_ic_support


def dtype_is_supported(weight):
    return weight.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.float8_e4m3fn,
    ]


def is_dim_conv_weight(weight):
    return (weight.dim() == 3 and weight.size(1) == 1) or weight.dim() == 5


def _init_amx_conv_state(conv_state):
    # CPU AMX layout for conv_state kernel optimization
    conv_state_cpu = []
    for conv_shape_t in conv_state:
        conv_shape_new = conv_shape_t.as_strided_(
            conv_shape_t.size(),
            (
                conv_shape_t.stride(0),
                conv_shape_t.stride(1),
                1,
                conv_shape_t.size(2),
            ),
        )
        conv_state_cpu.append(conv_shape_new)
    return conv_state_cpu


def _amx_process_weight_after_loading(
    module, weight_names, transpose_dims=None, qweight_packed_method=None
) -> None:
    # Pack weight for get better performance on CPU
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, f"Expects all weights to be on the same device"
    device = devices.pop()

    if transpose_dims:
        assert len(weight_names) == len(transpose_dims), (
            "len(weight_names) should be equal to len(transpose_dims)"
        )

    module.use_intel_amx_backend = (
        device == torch.device("cpu") and cpu_has_amx_support()
    )

    is_conv_weight = False
    if qweight_packed_method is None:
        for i, weight_name in enumerate(weight_names):
            weight_tensor = getattr(module, weight_name)

            if transpose_dims and transpose_dims[i]:
                weight_tensor = weight_tensor.transpose(*transpose_dims[i])
            is_conv_weight = is_dim_conv_weight(weight_tensor)
            # We don't pack weight or use intel amx backend if any weight of this module has unsupported dim.
            if (
                (not dim_is_supported(weight_tensor))
                or not dtype_is_supported(weight_tensor)
            ) and (not is_conv_weight):
                logger.warning(
                    f"Unsupported dimension or dtype for prepacking for weight '{weight_name}' with shape {weight_tensor.shape} and dtype {weight_tensor.dtype} in {module}. "
                    f"The derived (OC, IC) dimensions must be divisible by (16, 32). "
                )
                module.use_intel_amx_backend = False
                return

            packed_weight = torch.nn.Parameter(
                amx_process_weight_after_loading(weight_tensor, is_conv_weight),
                requires_grad=False,
            )
            packed_weight.__dict__ = weight_tensor.__dict__
            setattr(module, weight_name, packed_weight)
            if is_conv_weight and weight_tensor.dim() != 5:
                # need to use inplace copy for conv weight amx packing,
                # as its usage in radix_linear_attention will use the original conv weight.
                weight_tensor = weight_tensor.view(-1, weight_tensor.size(-1))
                weight_tensor.copy_(packed_weight)
    else:
        assert qweight_packed_method in ["awq", "gptq"]
        qweight_tensor = getattr(module, weight_names[0])
        qzeros_tensor = getattr(module, weight_names[1])
        scales_tensor = getattr(module, weight_names[2])
        qweight, qzeros, scales = torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
            qweight_tensor,
            qzeros_tensor,
            scales_tensor,
            CPUQuantAlgo.AWQ if qweight_packed_method == "awq" else CPUQuantAlgo.GPTQ,
        )
        packed_qweight = torch.nn.Parameter(
            qweight.detach(),
            requires_grad=False,
        )
        packed_qzeros = torch.nn.Parameter(
            qzeros.detach(),
            requires_grad=False,
        )
        packed_scales = torch.nn.Parameter(
            scales.detach(),
            requires_grad=False,
        )
        packed_qweight.__dict__ = qweight_tensor.__dict__
        packed_qzeros.__dict__ = qzeros_tensor.__dict__
        packed_scales.__dict__ = scales_tensor.__dict__
        setattr(module, weight_names[0], packed_qweight)
        setattr(module, weight_names[1], packed_qzeros)
        setattr(module, weight_names[2], packed_scales)
    if (
        module.use_intel_amx_backend
        and hasattr(module, "bias")
        and module.bias is not None
    ):
        if is_conv_weight and module.weight.data.dim() == 5:
            module.bias = torch.nn.Parameter(module.bias.data, requires_grad=False)
        else:
            module.bias = torch.nn.Parameter(
                module.bias.data.float(), requires_grad=False
            )


class PackWeightMethod:
    def __init__(self, weight_names, transpose_dims=None, use_fp8_w8a8=False):
        self.weight_names = weight_names
        self.transpose_dims = transpose_dims
        self.use_fp8_w8a8 = use_fp8_w8a8

    def process_weights_after_loading(self, module) -> None:
        if self.use_fp8_w8a8 and all(
            getattr(module, name).dtype == torch.float8_e4m3fn
            for name in self.weight_names
        ):
            if self.transpose_dims is not None and len(self.transpose_dims) != len(
                self.weight_names
            ):
                raise ValueError(
                    "Weight names and transpose dimensions must have equal lengths."
                )
            packed = {}
            for index, name in enumerate(self.weight_names):
                weight = getattr(module, name)
                if self.transpose_dims and self.transpose_dims[index]:
                    weight = weight.transpose(*self.transpose_dims[index])
                weight = weight.contiguous()
                if weight.ndim != 3 or weight.shape[1] % 32 or weight.shape[2] % 128:
                    raise ValueError(
                        "FP8 CPU MLA requires unpacked head weights with "
                        "output divisible by 32 and input divisible by 128."
                    )
                scales = (
                    module.w_scale.float()
                    .expand(weight.shape[1], weight.shape[2] // 128)
                    .contiguous()
                )
                heads = [
                    torch.ops.sgl_kernel.float8_linear_prepack_cpu(head, scales)
                    for head in weight
                ]
                packed[name] = (
                    torch.nn.Parameter(
                        torch.stack([head[0] for head in heads]), requires_grad=False
                    ),
                    torch.nn.Parameter(
                        torch.stack([head[1] for head in heads]), requires_grad=False
                    ),
                )
            for name, (weight, scales) in packed.items():
                setattr(module, name, weight)
                setattr(module, name + "_scale", scales)
            module.use_intel_amx_backend = True
            return
        _amx_process_weight_after_loading(
            module, self.weight_names, self.transpose_dims
        )

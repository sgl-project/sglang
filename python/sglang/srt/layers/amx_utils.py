import logging

import torch

from sglang.srt.utils import cpu_has_amx_support

logger = logging.getLogger(__name__)

from enum import IntEnum


class CPUQuantMethod(IntEnum):
    UNQUANT = 0
    INT8_W8A8 = 1
    FP8_W8A16 = 2
    INT4_W4A8 = 3


def amx_process_weight_after_loading(weight, is_conv=False):
    if weight.device != torch.device("cpu"):
        return weight
    if not cpu_has_amx_support():
        return weight
    if is_conv:
        return torch.ops.sgl_kernel.causal_conv1d_weight_pack(
            weight.view(-1, weight.size(-1))
        )
    else:
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
        torch.int8,
        torch.float8_e4m3fn,
    ]


def is_dim_conv_weight(weight):
    return weight.dim() == 3 and weight.size(1) == 1


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
    module, weight_names, transpose_dims=None
) -> None:
    # Pack weight for get better performance on CPU
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, f"Expects all weights to be on the same device"
    device = devices.pop()

    if transpose_dims:
        assert len(weight_names) == len(
            transpose_dims
        ), "len(weight_names) should be equal to len(transpose_dims)"

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
        if is_conv_weight:
            # need to use inplace copy for conv weight amx packing,
            # as its usage in radix_linear_attention will use the original conv weight.
            weight_tensor = weight_tensor.view(-1, weight_tensor.size(-1))
            weight_tensor.copy_(packed_weight)

    module.use_intel_amx_backend = (
        device == torch.device("cpu") and cpu_has_amx_support()
    )

    if (
        module.use_intel_amx_backend
        and hasattr(module, "bias")
        and module.bias is not None
    ):
        module.bias = torch.nn.Parameter(module.bias.data.float(), requires_grad=False)


class PackWeightMethod:
    def __init__(self, weight_names, transpose_dims=None):
        self.weight_names = weight_names
        self.transpose_dims = transpose_dims

    def process_weights_after_loading(self, module) -> None:
        _amx_process_weight_after_loading(
            module, self.weight_names, self.transpose_dims
        )

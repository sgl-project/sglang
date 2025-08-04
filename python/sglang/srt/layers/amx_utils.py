import logging

import torch

from sglang.srt.utils import cpu_has_amx_support, get_bool_env_var

logger = logging.getLogger(__name__)

import os

SGLANG_USE_CPU_INT4_W4A8 = get_bool_env_var("SGLANG_USE_CPU_INT4_W4A8")

def amx_process_weight_after_loading(weight):
    if weight.device != torch.device("cpu"):
        return weight
    if not cpu_has_amx_support():
        return weight

    return torch.ops.sgl_kernel.convert_weight_packed(weight)


# TODO: currently gemm kernel has the below requirements:
# OC % TILE_N == 0, where TILE_N = 16
# IC % TILE_K == 0, where TILE_K = 32
def dim_is_supported(weight):
    TILE_N = 16
    TILE_K = 32
    ndim = weight.ndim
    OC = weight.size(1) if ndim == 3 else weight.size(0)
    IC = weight.size(2) if ndim == 3 else weight.size(1)
    return OC % TILE_N == 0 and IC % TILE_K == 0


def _autoawq_to_int4pack(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, is_w4a8: bool
):
    """Convert AutoAWQ weight format to sgl-kernel's CPU int4

    Args:
        qweight: (*, K, N / 8), int32
        qzeros: (*, K / group_size, N / 8), int32
        scales: (*, K / group_size, N), bfloat16
    """
    # unpack from AutoAWQ format
    # https://github.com/casper-hansen/AutoAWQ/blob/23d584c2/awq/modules/triton/gemm.py#L73-L86
    bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
    qweight = (qweight.unsqueeze(-1) >> bitshifts) & 0xF
    qweight = qweight.flatten(-2).transpose(-1, -2).to(torch.uint8)

    qzeros = (qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    qzeros = qzeros.flatten(-2).to(torch.uint8)
    if is_w4a8:
        qzeros = qzeros.T.contiguous()
        scales = scales.T.contiguous()
        qweight, scales, qzeros, compensation = (
            torch.ops.sgl_kernel.convert_int4_weight_packed(qweight, scales, qzeros)
        )
        return qweight, qzeros, scales, compensation
    else:
        # TODO: unify below in convert_int4_weight_packed
        # convert to VNNI format: (*, N/BLOCK_N, K/2, BLOCK_N, 2)
        BLOCK_N = 32  # must match what's used in the kernel
        *dims, N, K = qweight.shape
        qweight = qweight.reshape(*dims, N // BLOCK_N, BLOCK_N, K // 2, 2)
        qweight = qweight.transpose(-3, -2)
        # bit packing
        BIT_COUNT = 32
        qweight = qweight.reshape(-1, BIT_COUNT * 2)
        qweight = (qweight[:, BIT_COUNT:] << 4) | qweight[:, :BIT_COUNT]
        qweight = qweight.reshape(*dims, N, K // 2)
        return qweight, qzeros, scales


def _amx_process_int4_packed_qweight_after_loading(
    module, weight_names, quant_method
) -> None:
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, f"Expects all weights to be on the same device"
    device = devices.pop()
    assert quant_method in ["awq"]  # TODO: add GPTQ, etc.
    module.use_intel_amx_backend = (
        device == torch.device("cpu") and cpu_has_amx_support()
    )
    if module.use_intel_amx_backend and quant_method == "awq":
        qweight_tensor = getattr(module, weight_names[0])
        qzeros_tensor = getattr(module, weight_names[1])
        scales_tensor = getattr(module, weight_names[2])
        prefix_list = weight_names[0].split("_")
        # MoE layers have prefix
        has_prefix = len(prefix_list) != 1
        use_w4a8 = SGLANG_USE_CPU_INT4_W4A8 and not has_prefix
        if use_w4a8:
            # TODO: support MoE layers for W4A8 path
            qweight, qzeros, scales, compensation = _autoawq_to_int4pack(
                qweight_tensor.data, qzeros_tensor.data, scales_tensor.data, use_w4a8
            )
            compensation = torch.nn.Parameter(compensation, requires_grad=False)
            setattr(
                module,
                "compensation" if not has_prefix else prefix_list[0] + "_compensation",
                compensation,
            )
        else:
            qweight, qzeros, scales = _autoawq_to_int4pack(
                qweight_tensor.data, qzeros_tensor.data, scales_tensor.data, use_w4a8
            )
        packed_qweight = torch.nn.Parameter(
            qweight,
            requires_grad=False,
        )
        packed_qzeros = torch.nn.Parameter(
            qzeros,
            requires_grad=False,
        )
        packed_scales = torch.nn.Parameter(
            scales,
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
        module.bias = torch.nn.Parameter(module.bias.data.float(), requires_grad=False)


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

        # We don't pack weight or use intel amx backend if any weight of this module has unsupported dim.
        if not dim_is_supported(weight_tensor):
            logger.warning(
                f"Unsupported dimension for prepacking for weight '{weight_name}' with shape {weight_tensor.shape} in {module}. "
                f"The derived (OC, IC) dimensions must be divisible by (16, 32). "
            )
            module.use_intel_amx_backend = False
            return

        packed_weight = torch.nn.Parameter(
            amx_process_weight_after_loading(weight_tensor),
            requires_grad=False,
        )
        packed_weight.__dict__ = weight_tensor.__dict__
        setattr(module, weight_name, packed_weight)

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

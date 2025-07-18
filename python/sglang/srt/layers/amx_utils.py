import logging

import torch

from sglang.srt.utils import cpu_has_amx_support

logger = logging.getLogger(__name__)

import os
SGLANG_USE_CPU_W4A8 = os.getenv("SGLANG_USE_CPU_W4A8", "0") == "1"

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

def _autoawq_to_int4pack(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor):
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

    # convert to VNNI format: (*, N/BLOCK_N, K/2, BLOCK_N, 2)
    BLOCK_N = 32  # must match what's used in the kernel
    *dims, N, K = qweight.shape
    qweight = qweight.reshape(*dims, N // BLOCK_N, BLOCK_N, K // 2, 2)
    qweight = qweight.transpose(-3, -2)

    # bit packing
    COUNT = 32
    qweight = qweight.reshape(-1, COUNT * 2)
    qweight = (qweight[:, COUNT:] << 4) | qweight[:, :COUNT]
    qweight = qweight.reshape(*dims, N, K // 2)

    qzeros = (qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    qzeros = qzeros.flatten(-2).to(torch.uint8)
    return qweight, qzeros, scales

# ref weight unpack from AutoAWQ https://github.com/AutoGPTQ/AutoGPTQ/blob/v0.7.1/auto_gptq/modeling/_utils.py#L516
def unpack_awq_weight(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        unpacked awq_qweight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4

    qzeros = awq_qzeros
    qweight = awq_qweight
    qweight = qweight.T.contiguous()

    scales = awq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    # infeatures = awq_qweight.shape[0]

    wf = torch.tensor(
        list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device
    ).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)

    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])

    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)

    return weight.contiguous(), zeros.contiguous()

def awq_reverse_reorder_int_tensor(int_tensor, bits: int):
    assert bits == 4

    int_tensor = int_tensor.T.contiguous()
    compress_ratio = 32 // bits
    assert int_tensor.shape[-1] % compress_ratio == 0

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    order_tensor = torch.tensor(
        order_map, dtype=torch.int32, device=int_tensor.device
    ).reshape(1, -1)
    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    order_tensor = order_tensor + torch.arange(
        0,
        int_tensor.shape[1],
        compress_ratio,
        dtype=torch.int32,
        device=int_tensor.device,
    ).reshape(-1, 1)
    order_tensor = order_tensor.reshape(-1)

    reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
    reverse_order_tensor = reverse_order_tensor[order_tensor]
    int_tensor = int_tensor[:, reverse_order_tensor]
    return int_tensor

def _autoawq_to_int4pack_w4a8(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int=4,
    group_size: int=128,
):
    t, zp = unpack_awq_weight(awq_qweight, awq_qzeros, awq_scales, bits, group_size)
    qweight_ = t.T.contiguous().to(torch.uint8)
    scales_ = awq_scales.T.contiguous()
    qzeros_ = zp.T.contiguous()
    qweight_, scales_, zp_ , comp = torch.ops.sgl_kernel.da8w4_linear_prepack_cpu(qweight_, scales_, qzeros_)
    return qweight_,  zp_, scales_, comp
 
def _amx_process_packed_qweight_after_loading(
    module, weight_names, quant_method
) -> None:
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, f"Expects all weights to be on the same device"
    device = devices.pop()
    assert quant_method in ["awq"]
    module.use_intel_amx_backend = (
        device == torch.device("cpu") and cpu_has_amx_support()
    )
    if module.use_intel_amx_backend and quant_method == "awq":
        assert weight_names ==  ["qweight", "qzeros", "scales"]
        qweight_tensor = getattr(module, weight_names[0])
        qzeros_tensor = getattr(module, weight_names[1])
        scales_tensor = getattr(module, weight_names[2])
        if SGLANG_USE_CPU_W4A8:
            qweight, qzeros, scales, compensation = _autoawq_to_int4pack_w4a8(
                qweight_tensor.data, qzeros_tensor.data, scales_tensor.data
            )
            compensation = torch.nn.Parameter(compensation, requires_grad=False)
            setattr(module, "compensation", compensation)
        else:
            qweight, qzeros, scales = _autoawq_to_int4pack(
                qweight_tensor.data, qzeros_tensor.data, scales_tensor.data
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

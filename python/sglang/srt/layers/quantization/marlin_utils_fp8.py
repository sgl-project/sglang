# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def fake_apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (size_n,)
    fake_output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    return fake_output


@register_custom_op(fake_impl=fake_apply_fp8_marlin_linear)
def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module, size_k_first: bool = True
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = pack_fp8_to_int32(layer.weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Permute scales
    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(layer.orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(layer.orig_dtype)
        del layer.weight_scale_inv

    group_size = -1 if weight_block_size is None else weight_block_size[1]

    # marlin kernel only support channel-wise and group-wise quantization
    # we need to convert the scales
    if weight_block_size is None:
        if scales.nelement() == 1:
            # tensor-wise quantization -> channel-wise quantization
            # (1, 1) =>(repeat)=> (1, size_n)
            scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
        elif scales.nelement() > 1 and scales.nelement() != part_size_n:
            assert part_size_n % scales.nelement() == 0
            s_size = scales.nelement()
            # tensor-wise quantization (for gate-up proj)
            #     -> channel-wise quantization
            # (1, s_size) =>(repeat)=> (1, size_n)
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(part_size_n // s_size, 1)
        else:
            # channel-wise quantization
            # (1, size_n)
            scales = scales.view(1, part_size_n)
    else:
        # block-wise quantization -> group-wise quantization
        # (size_k // block_size[1], ceil(size_n / block_size[0]))
        #  =>(repeat)=> (size_k // block_size[1], size_n)
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        scales = scales.repeat_interleave(block_n, 1)
        # size_n may not divisible by block_size[0]
        scales = scales[:, :part_size_n]

    marlin_scales = marlin_permute_scales(
        s=scales, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)

    # The dense FP8 Marlin wrapper adds bias after the kernel returns, so the
    # bias must remain in logical output-channel order. Only scales need the
    # Marlin tile permutation.
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        layer.bias = torch.nn.Parameter(layer.bias.detach(), requires_grad=False)


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor


def prepare_moe_mxfp8_layer_for_marlin(layer: torch.nn.Module) -> None:
    """Repack canonical E4M3 experts and unswizzled group-32 E8M0 scales.

    Gate/up rows remain in [gate; up] order. Unlike ordinary FP8 Marlin,
    E8M0 scales retain their exponent bytes rather than absorbing an FP8 bias.
    """
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        _permute_moe_fp4_scales_for_marlin,
        mxfp4_marlin_process_scales,
    )

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    num_experts, w13_size_n, hidden_size = w13.shape
    intermediate_size = w13_size_n // 2
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"MXFP8 Marlin needs float8_e4m3fn experts, got {w13.dtype} and {w2.dtype}"
        )
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError(
            f"MXFP8 Marlin w2 shape {tuple(w2.shape)} does not match "
            f"w13 shape {tuple(w13.shape)}"
        )
    if hidden_size % 128 or intermediate_size % 64:
        raise ValueError(
            "MXFP8 Marlin needs hidden_size % 128 == 0 and intermediate_size % 64 "
            f"== 0, got {hidden_size} and {intermediate_size}"
        )

    perm = torch.empty(0, dtype=torch.int32, device=w13.device)

    def repack(weight: torch.Tensor) -> torch.Tensor:
        _, size_n, size_k = weight.shape
        output = torch.empty(
            (num_experts, size_k // 16, size_n * 4),
            dtype=torch.int32,
            device=weight.device,
        )
        for i, expert in enumerate(weight):
            output[i] = gptq_marlin_repack(
                b_q_weight=pack_fp8_to_int32(expert, size_k_first=False).T.contiguous(),
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=8,
            )
        return output

    def permute_scales(scales: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        if scales.shape != (num_experts, size_n, size_k // 32):
            raise ValueError(
                f"MXFP8 Marlin scale shape {tuple(scales.shape)} does not match "
                f"[{num_experts}, {size_n}, {size_k // 32}]"
            )
        if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(f"MXFP8 Marlin needs E8M0 scale bytes, got {scales.dtype}")
        # BF16 represents E8M0 exactly and supports the shared scale permutation.
        scales = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
        return _permute_moe_fp4_scales_for_marlin(
            scales,
            num_experts=num_experts,
            size_n=size_n,
            size_k=size_k,
            group_size=32,
            process_scales=mxfp4_marlin_process_scales,
        )

    w13_scale = permute_scales(layer.w13_weight_scale_inv.data, w13_size_n, hidden_size)
    w2_scale = permute_scales(
        layer.w2_weight_scale_inv.data, hidden_size, intermediate_size
    )
    layer.w13_weight = torch.nn.Parameter(repack(w13), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(repack(w2), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    del layer.w13_weight_scale_inv
    del layer.w2_weight_scale_inv

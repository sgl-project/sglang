# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    check_moe_marlin_supports_layer,
    marlin_make_workspace,
    marlin_moe_permute_scales,
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


# Compensate for the destination exponent bias minus the E4M3 bias (7).
_FP8_SCALE_COMPENSATION = {torch.float16: 2.0**8, torch.bfloat16: 2.0**120}


def fp8_fused_exponent_bias_into_scales(
    scales: torch.Tensor, *, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = scales.dtype
    return (scales * _FP8_SCALE_COMPENSATION[output_dtype]).to(output_dtype)


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


def marlin_fp8_scales_to_bf16(
    scales: torch.Tensor, *, name: str = "scales"
) -> torch.Tensor:
    """Compensate before the BF16 cast to preserve tiny scales."""
    scales = scales.to(torch.float32)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError(f"FP8 Marlin {name} must be finite")
    limit = torch.finfo(torch.bfloat16).max / _FP8_SCALE_COMPENSATION[torch.bfloat16]
    largest = float(scales.abs().amax()) if scales.numel() else 0.0
    if largest > limit:
        raise ValueError(
            f"FP8 Marlin {name} exceed the BF16 range after exponent compensation: "
            f"largest absolute scale {largest:.4g} > {limit:.6g}"
        )
    return fp8_fused_exponent_bias_into_scales(scales, output_dtype=torch.bfloat16)


def marlin_moe_fp8_channel_scales(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, 1] per-output-row scales as Marlin's [E, 1, N] channel-wise groups."""
    return scales.reshape(scales.shape[0], 1, -1)


def marlin_moe_fp8_tensor_scales(scales: torch.Tensor, *, size_n: int) -> torch.Tensor:
    """Repeat each expert or shard scale over its own output channels."""
    scales = scales.to(torch.float32)
    if scales.ndim == 1:
        scales = scales[:, None]
    experts, shards = scales.shape
    return scales.repeat_interleave(size_n // shards, dim=1).reshape(experts, 1, size_n)


def marlin_moe_fp8_block_scales(
    scales: torch.Tensor, *, size_n: int, block_n: int
) -> torch.Tensor:
    """Expand block scales along N into Marlin [E, K / block_k, N] order."""
    scales = scales.transpose(1, 2).repeat_interleave(block_n, dim=2)
    return scales[:, :, :size_n].contiguous()


def marlin_moe_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, K / 32] E8M0 bytes as [E, K / 32, N] FP32 values 2 ** (e - 127)."""
    return scales.view(torch.float8_e8m0fnu).float().transpose(1, 2).contiguous()


def marlin_fp8_moe_group_size(
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> int:
    """Use multiple K groups so Marlin scales before the MMA,
    avoiding underflow from the E4M3 exponent shift.
    """
    present = [
        size_k // scales.shape[1]
        for scales, size_k in (
            (w13_scales, hidden_size),
            (w2_scales, intermediate_size),
        )
        if scales.shape[1] > 1
    ]
    group_size = min(present) if present else 128
    # The MoE kernel instantiates E4M3 weights for these group sizes only.
    if group_size not in (32, 128):
        raise NotImplementedError(
            f"the Marlin FP8 MoE kernels take 32- or 128-wide K groups, got {group_size}"
        )
    if group_size == 128 and not all(
        k % 128 == 0 and k >= 256 for k in (hidden_size, intermediate_size)
    ):
        group_size = 32
    return group_size


def marlin_fp8_scales_at_group(
    scales: torch.Tensor, *, size_k: int, group_size: int
) -> torch.Tensor:
    """Expand scales from [E, groups, N] to the kernel's K group width."""
    groups = size_k // group_size
    if scales.shape[1] == groups:
        return scales
    if groups % scales.shape[1] == 0:
        return scales.repeat_interleave(groups // scales.shape[1], dim=1)
    raise ValueError(
        f"{scales.shape[1]} K groups over {size_k} cannot be served with "
        f"{group_size}-wide groups"
    )


def check_lora_marlin_fp8_moe_layer(
    params_dtype: torch.dtype, *, with_bias: bool
) -> None:
    if params_dtype != torch.bfloat16:
        raise NotImplementedError(
            "--moe-runner-backend lora_marlin serves FP8 MoE experts with BF16 "
            f"activations only (--dtype bfloat16), got {params_dtype}"
        )
    if with_bias:
        raise NotImplementedError(
            "--moe-runner-backend lora_marlin serves FP8 MoE experts without expert "
            "biases, and this layer has them"
        )


def repack_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    layout: str,
    *,
    scales: str = "weight_scale",
    block_n: int | None = None,
) -> None:
    converted = []
    for name in ("w13", "w2"):
        scale = getattr(layer, f"{name}_{scales}")
        size_n = getattr(layer, f"{name}_weight").shape[1]
        match layout:
            case "block":
                converted.append(
                    marlin_moe_fp8_block_scales(scale, size_n=size_n, block_n=block_n)
                )
            case "mxfp8":
                converted.append(marlin_moe_mxfp8_scales(scale))
            case "channel":
                converted.append(marlin_moe_fp8_channel_scales(scale))
            case "tensor":
                converted.append(marlin_moe_fp8_tensor_scales(scale, size_n=size_n))
            case _:
                raise ValueError(f"unknown FP8 weight-scale layout {layout!r}")
    prepare_moe_fp8_layer_for_marlin(
        layer,
        w13_scales=converted[0],
        w2_scales=converted[1],
    )


def prepare_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    *,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
) -> None:
    """Repack E4M3 weights and FP32 [E, groups, N] scales for Marlin W8A16."""
    hidden_size = layer.w13_weight.shape[2]
    intermediate_size = layer.w2_weight.shape[2]
    group_size = marlin_fp8_moe_group_size(
        w13_scales,
        w2_scales,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    w13_scales = marlin_fp8_scales_at_group(
        w13_scales, size_k=hidden_size, group_size=group_size
    )
    w2_scales = marlin_fp8_scales_at_group(
        w2_scales, size_k=intermediate_size, group_size=group_size
    )
    # The kernel tiles N in 128s; without a gate, gate/up N is the intermediate size.
    if (
        not check_moe_marlin_supports_layer(layer, group_size)
        or layer.w13_weight.shape[1] % 128
    ):
        raise ValueError(
            "this FP8 MoE layer does not satisfy the Marlin constraints: "
            "hidden_size % 128 == 0, intermediate_size % "
            f"{max(64, group_size)} == 0 (% 128 without a gate), SiLU/SiTU or ReLU2 "
            "without router weights on the input"
        )
    stages = (("w13", w13_scales), ("w2", w2_scales))
    # Validate both scale tensors before modifying either expert weight.
    converted = {
        name: marlin_fp8_scales_to_bf16(scales, name=f"{name} scales")
        for name, scales in stages
    }
    for name, scales in converted.items():
        weight = getattr(layer, f"{name}_weight")
        _, size_n, size_k = weight.shape
        qweight = marlin_moe_fp8_weights(weight)
        marlin_scales = marlin_moe_permute_scales(
            s=scales, size_k=size_k, size_n=size_n, group_size=group_size
        )
        setattr(
            layer, f"{name}_weight", torch.nn.Parameter(qweight, requires_grad=False)
        )
        setattr(
            layer,
            f"{name}_weight_scale",
            torch.nn.Parameter(marlin_scales, requires_grad=False),
        )
    for stale in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        if hasattr(layer, stale):
            delattr(layer, stale)


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


def marlin_moe_fp8_weights(weight: torch.Tensor) -> torch.Tensor:
    """[E, N, K] E4M3 experts as Marlin tiles, int32 [E, K / 16, 4 * N]."""
    num_experts, size_n, size_k = weight.shape
    perm = torch.empty(0, dtype=torch.int32, device=weight.device)
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
    layer.w13_weight = torch.nn.Parameter(
        marlin_moe_fp8_weights(w13), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        marlin_moe_fp8_weights(w2), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    del layer.w13_weight_scale_inv
    del layer.w2_weight_scale_inv

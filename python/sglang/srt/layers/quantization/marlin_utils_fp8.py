# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    check_moe_marlin_supports_layer,
    marlin_make_workspace,
    marlin_moe_permute_scales,
    marlin_permute_bias,
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


def fp8_marlin_scale_compensation(dtype: torch.dtype) -> float:
    """The power of two folded into FP8 Marlin scales for this activation dtype.

    The kernel dequantizes an E4M3 byte by moving its bits into the activation
    dtype's exponent field, which yields the value times 2**-(bias_dtype -
    bias_e4m3): 2**-120 for BF16 and 2**-8 for FP16; the scale carries the
    inverse (see fp8_fused_exponent_bias_into_scales).
    """
    if dtype == torch.bfloat16:
        return 2.0**120
    if dtype == torch.float16:
        return 2.0**8
    raise ValueError(f"FP8 Marlin serves BF16 or FP16 activations, not {dtype}")


def marlin_fp8_scales_to_dtype(
    scales: torch.Tensor, dtype: torch.dtype, *, name: str = "scales"
) -> torch.Tensor:
    """FP32 scales as the activation-dtype scales the FP8 Marlin kernels read.

    The compensation is folded in FP32 and the result cast once, so a scale
    that only underflows the activation dtype before compensation (BF16: below
    2**-133) still reaches the kernel. Every admitted scale keeps the dtype's
    round-to-nearest precision (unit roundoff 2**-8 in BF16, 2**-11 in FP16),
    so the domain is checked, not clamped: every scale must be finite and
    non-negative; scale * compensation must not exceed the dtype's largest
    finite value (BF16: scale <= 255, FP16: scale <= 255.875; an MXFP8
    exponent byte of at most 134); and a nonzero compensated scale must be a
    normal number of the dtype (FP16: scale >= 2**-22, exponent byte 105;
    every nonzero FP32 scale is normal in BF16 after the fold, as
    2**-149 * 2**120 = 2**-29).
    """
    scales = scales.to(torch.float32)
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError(f"FP8 Marlin {name} must be finite and non-negative")
    compensation = fp8_marlin_scale_compensation(dtype)
    compensated = scales * compensation
    finfo = torch.finfo(dtype)
    largest = float(compensated.max()) if compensated.numel() else 0.0
    if largest > finfo.max:
        raise ValueError(
            f"FP8 Marlin {name} exceed the {dtype} range once the exponent bias "
            f"is folded in: the largest scale {float(scales.max()):.4g} becomes "
            f"{largest:.4g} > {finfo.max:.4g}; scales must stay at or below "
            f"{finfo.max / compensation:.6g}"
        )
    nonzero = compensated[scales != 0]
    if nonzero.numel() and float(nonzero.min()) < finfo.tiny:
        smallest = float(scales[scales != 0].min())
        raise ValueError(
            f"FP8 Marlin {name} fall below the {dtype} normal range once the "
            f"exponent bias is folded in: the smallest nonzero scale {smallest:.4g} "
            f"becomes {smallest * compensation:.4g} < {finfo.tiny:.4g}, a subnormal "
            f"that would lose precision; scales must be at least "
            f"{finfo.tiny / compensation:.4g}"
        )
    return compensated.to(dtype)


def marlin_moe_fp8_channel_scales(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, 1] per-output-row scales as Marlin's [E, 1, N] channel-wise groups."""
    return scales.reshape(scales.shape[0], 1, -1)


def marlin_moe_fp8_tensor_scales(scales: torch.Tensor, *, size_n: int) -> torch.Tensor:
    """Per-tensor scales as Marlin's [E, 1, N] channel-wise groups: one scale
    per expert ([E]), or one per output shard ([E, S], the loader's separate
    gate and up scales) repeated over that shard's N / S columns, so the shards
    keep their own scales instead of being requantized to a shared one."""
    scales = scales.to(torch.float32)
    if scales.ndim == 1:
        scales = scales[:, None]
    experts, shards = scales.shape
    if size_n % shards:
        raise ValueError(
            f"{size_n} output columns do not split into {shards} equal scale shards"
        )
    return scales.repeat_interleave(size_n // shards, dim=1).reshape(experts, 1, size_n)


def marlin_moe_fp8_block_scales(
    scales: torch.Tensor, *, size_n: int, block_n: int
) -> torch.Tensor:
    """[E, ceil(N / block_n), K / block_k] block scales as [E, K / block_k, N]
    group scales: the block's scale repeated over its block_n output columns."""
    scales = scales.transpose(1, 2).repeat_interleave(block_n, dim=2)
    return scales[:, :, :size_n].contiguous()


def marlin_moe_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, K / 32] E8M0 bytes as [E, K / 32, N] FP32 values 2 ** (e - 127)."""
    if scales.dtype != torch.float8_e8m0fnu:
        scales = scales.view(torch.float8_e8m0fnu)
    return scales.to(torch.float32).transpose(1, 2).contiguous()


def marlin_fp8_moe_group_size(
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> int:
    """The K group size both stages of an FP8 MoE layer are served with: the
    finest group the scales already have (32 for MXFP8, 128 for block scales),
    128 for channel-wise and per-tensor scales when both K extents are
    multiples of 128 holding at least two groups, else 32 (the 32-wide kernels
    take every K the layer admission allows, intermediate sizes of 320 or 448
    included).

    The kernel reads a single group over K as channel-wise and, for E4M3
    weights, then applies the scale to the FP32 accumulators after the K loop;
    the products it sums carry the 2**-120 dequantization shift, so small
    activations times small codes underflow FP32 before the scale can restore
    them (BF16 inputs of 2**-60 against codes of 1 give 0 instead of 2**-52).
    Grouped scales are applied to the weights before the MMA, so every stage is
    given at least two groups.
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
    """[E, groups, N] FP32 scales re-expressed with ``group_size``-wide K groups:
    one channel-wise group is repeated over K, a coarser group is repeated over
    the finer groups it covers."""
    groups = size_k // group_size
    if scales.shape[1] == groups:
        return scales
    if scales.shape[1] == 1:
        return scales.repeat(1, groups, 1)
    if groups % scales.shape[1] == 0:
        return scales.repeat_interleave(groups // scales.shape[1], dim=1)
    raise ValueError(
        f"{scales.shape[1]} K groups over {size_k} cannot be served with "
        f"{group_size}-wide groups"
    )


def _check_fp16_dequantized_range(layer, stages, converted) -> None:
    """Refuse a stage whose largest |code| * scale exceeds FP16's largest finite
    value, with the scale as the kernel holds it: the compensated scale rounded
    to FP16, divided back by the compensation. The grouped kernel forms that
    product in FP16 before the MMA, so a weight that fits FP16 before the scale's
    rounding can still become infinite (codes of 448 under 65504 / 448: the
    compensated scale rounds to 37440 and 1.75 * 37440 is infinity), and an
    infinite weight poisons a GEMM whose exact result would be finite."""
    compensation = fp8_marlin_scale_compensation(torch.float16)
    limit = torch.finfo(torch.float16).max
    for name, scales in stages:
        weight = getattr(layer, f"{name}_weight")
        num_experts, size_n, size_k = weight.shape
        groups = scales.shape[1]
        # [E, groups, N]: the per-weight scale after its rounding to FP16.
        effective = converted[name].float() / compensation
        for e in range(num_experts):
            codes = weight[e].float().abs().view(size_n, groups, size_k // groups)
            largest = (codes.amax(dim=2).T * effective[e]).max()
            if float(largest) > limit:
                raise ValueError(
                    f"FP8 Marlin {name} weights dequantize beyond the FP16 range: "
                    f"|code| * scale reaches {float(largest):.6g} > {limit} in expert "
                    f"{e}; the grouped kernel multiplies code * scale in FP16 before "
                    "the MMA, so every dequantized weight must stay at or below 65504"
                )


def repack_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    layout: str,
    *,
    scales: str = "weight_scale",
    block_n: int | None = None,
) -> None:
    """Repack a loaded FP8 MoE layer into Marlin W8A16 tiles from the weight
    scales its checkpoint stores in ``layer.w13_<scales>`` and
    ``layer.w2_<scales>``. ``layout`` names their granularity: "block" (one
    per ``block_n`` x K-group block), "mxfp8" (an E8M0 byte per 32 columns),
    "channel" (one per output row) or "tensor" (one per expert shard)."""
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
        params_dtype=layer.params_dtype,
    )


def prepare_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    *,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    params_dtype: torch.dtype,
) -> None:
    """Repack [E, N, K] e4m3 expert weights into Marlin tiles for W8A16.

    The scales are FP32 [E, groups, N], one row per K group of the stage (one
    row for channel-wise and per-tensor scales, which are then repeated over
    the groups the kernel is given; see marlin_fp8_moe_group_size); they become
    permuted scales in the activation dtype with the FP8 exponent bias folded
    in (marlin_fp8_scales_to_dtype states the admitted range), replacing
    ``*_weight_scale``; expert biases, if the layer has them, take the kernel's
    tile order and dtype; the loader-format ``*_weight_scale_inv`` parameters
    are released. The grouped kernel multiplies code * scale in the activation
    dtype before the MMA, so with FP16 activations every |code| times the scale
    as the kernel holds it (rounded to FP16) must stay at or below 65504
    (checked here; BF16's range never binds, as 448 * 255 is far below it).
    """
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
    if not check_moe_marlin_supports_layer(layer, group_size):
        raise ValueError(
            "this FP8 MoE layer does not satisfy the Marlin constraints: "
            "hidden_size % 128 == 0, intermediate_size % "
            f"{max(64, group_size)} == 0, SiLU/SiTU or ReLU2 without router "
            "weights on the input"
        )
    stages = (("w13", w13_scales), ("w2", w2_scales))
    # Both stages' scales are admitted and converted before either parameter
    # moves, so a rejected layer keeps its loader-format weights intact.
    converted = {
        name: marlin_fp8_scales_to_dtype(scales, params_dtype, name=f"{name} scales")
        for name, scales in stages
    }
    if params_dtype == torch.float16:
        _check_fp16_dequantized_range(layer, stages, converted)
    biases = {}
    for name in ("w13", "w2"):
        bias = getattr(layer, f"{name}_weight_bias", None)
        if bias is not None:
            # The kernel adds biases in its tile order and activation dtype.
            biases[name] = torch.stack(
                [marlin_permute_bias(row.to(params_dtype)) for row in bias.data]
            )
    device = layer.w13_weight.device
    perm = torch.empty(0, dtype=torch.int, device=device)
    for name, scales in stages:
        weight = getattr(layer, f"{name}_weight")
        num_experts, size_n, size_k = weight.shape
        group_size = size_k // scales.shape[1]
        # Marlin tiles 16 rows of K; 8-bit weights pack four per int32 column.
        qweight = torch.empty(
            (num_experts, size_k // 16, size_n * 4), dtype=torch.int32, device=device
        )
        for e in range(num_experts):
            packed = pack_fp8_to_int32(weight[e], size_k_first=False).T.contiguous()
            qweight[e] = gptq_marlin_repack(
                b_q_weight=packed,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=8,
            )
        marlin_scales = marlin_moe_permute_scales(
            s=converted[name], size_k=size_k, size_n=size_n, group_size=group_size
        )
        setattr(
            layer, f"{name}_weight", torch.nn.Parameter(qweight, requires_grad=False)
        )
        setattr(
            layer,
            f"{name}_weight_scale",
            torch.nn.Parameter(marlin_scales, requires_grad=False),
        )
    for name, bias in biases.items():
        setattr(
            layer, f"{name}_weight_bias", torch.nn.Parameter(bias, requires_grad=False)
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

# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    _sm70_marlin_v100_gemm_op,
    _sm70_marlin_v100_repack_ops,
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
_SM70_MXFP8_PACKED_LOGGED = False


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

    sm70_gemm = _sm70_marlin_v100_gemm_op()
    if sm70_gemm is not None:
        output = sm70_gemm(
            reshaped_x,
            None,
            weight,
            bias,
            weight_scale,
            None,
            None,
            None,
            None,
            None,
            workspace,
            scalar_types.float8_e4m3fn.id,
            reshaped_x.size(0),
            size_n,
            size_k,
            True,
            use_atomic_add,
            use_fp32_reduce,
            False,
        )
    else:
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

    sm70_gptq_repack, _ = _sm70_marlin_v100_repack_ops()
    if sm70_gptq_repack is not None:
        marlin_qweight = sm70_gptq_repack(
            qweight,
            perm,
            part_size_k,
            part_size_n,
            8,
        )
    else:
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

    if sm70_gptq_repack is not None:
        # marlin_v100's SM70 kernels consume logical N-contiguous metadata.
        # The stock SM80+ Marlin permutation corrupts group-128 FP8 scales.
        marlin_scales = scales.reshape(-1, part_size_n).contiguous()
    else:
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


# SM70 marlin_v100 FP8 W8A16 only instantiates group_size -1 (channel) or 128.
# Official DSV4.1-Flash dense MXFP8 is UE8M0 g32 (1x32 or 32x32). Requantizing
# g32 -> g128 would be a third scale format. Unpack to FP16 instead: dense is
# ~7 GiB stored, FP16 is ~14 GiB total / 8 ranks, and the conversion is exact
# for UE8M0 bytes in {0} ∪ [113, 142].
SM70_FP8_MARLIN_GROUP_SIZES = (-1, 128)
_MXFP8_UE8M0_BLOCK_SIZES = ((1, 32), (32, 32))


def sm70_mxfp8_ue8m0_to_fp16_scales(scales: torch.Tensor) -> torch.Tensor:
    """Exact UE8M0 → FP16 numerical scales (``2**(e-127)``).

    Used if a future MXFP8 layout is expressible as Marlin FP8 group -1/128.
    Out-of-range exponents that cannot be represented in FP16 fail loud.
    """
    from sglang.srt.layers.quantization.marlin_utils import (
        SM70_MXFP4_UE8M0_FP16_EXACT_MAX,
        SM70_MXFP4_UE8M0_FP16_EXACT_MIN,
        sm70_mxfp4_ue8m0_to_uint8,
    )

    raw = sm70_mxfp4_ue8m0_to_uint8(scales)
    in_range = (raw == 0) | (
        (raw >= SM70_MXFP4_UE8M0_FP16_EXACT_MIN)
        & (raw <= SM70_MXFP4_UE8M0_FP16_EXACT_MAX)
    )
    if not bool(in_range.all()):
        bad = raw[~in_range]
        raise RuntimeError(
            "SM70 MXFP8→FP16 scale conversion is exact only for UE8M0 bytes "
            f"in {{0}} ∪ [{SM70_MXFP4_UE8M0_FP16_EXACT_MIN}, "
            f"{SM70_MXFP4_UE8M0_FP16_EXACT_MAX}]. "
            f"Found {int(bad.min())}..{int(bad.max())}."
        )
    return raw.view(torch.float8_e8m0fnu).to(torch.float16)


def dequant_mxfp8_ue8m0_to_fp16(
    weight: torch.Tensor,
    scales: torch.Tensor,
    weight_block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Unpack MXFP8 e4m3fn + UE8M0 g32 to ``out_dtype`` (fp16 on SM70).

    ``weight_block_size`` is ``(1, 32)`` (per-row K-groups) or ``(32, 32)``
    (2D blocks). Scales are converted in fp32 (exact for every UE8M0
    exponent); the e4m3×scale product is then cast to ``out_dtype``.
    """
    block = tuple(int(x) for x in weight_block_size)
    if block not in _MXFP8_UE8M0_BLOCK_SIZES:
        raise RuntimeError(
            "Dense MXFP8 is UE8M0 g32 with weight_block_size [1, 32] or "
            f"[32, 32]; got {list(block)}. Refusing a third block size."
        )
    if out_dtype not in (torch.float16, torch.float32):
        out_dtype = torch.float16
    n, k = weight.shape
    bn, bk = block
    # UE8M0 → fp32 is exact for every exponent. Official DSV4.1-Flash dense
    # has bytes 109–112 (2^-18..2^-15), below fp16 min normal (byte 113).
    # Multiplying e4m3 × scale in fp32 then casting the product keeps those
    # blocks; storing the scale itself as fp16 would flush them to 0.
    from sglang.srt.layers.quantization.marlin_utils import (
        sm70_mxfp4_ue8m0_to_uint8,
    )

    raw = sm70_mxfp4_ue8m0_to_uint8(scales)
    sf = raw.view(torch.float8_e8m0fnu).float()
    # Allocate the fp16 result first, then dequant in row stripes so we never
    # hold a full fp32 copy (~2x) plus the fp16 out at once. Prefill on 32 GiB
    # V100 otherwise OOMs the `.to(fp16)` after a 2048-token extend.
    if weight.is_cuda:
        free_b, _ = torch.cuda.mem_get_info()
        need_b = n * k * (2 if out_dtype == torch.float16 else 4) + 64 * 1024 * 1024
        if free_b < need_b:
            torch.cuda.empty_cache()
    out = torch.empty((n, k), dtype=out_dtype, device=weight.device)
    row_bytes = k * 4
    stripe = max(bn, min(n, (16 * 1024 * 1024) // max(row_bytes, 1)))
    if bn > 1:
        stripe = max(bn, (stripe // bn) * bn)
    if bn == 1:
        if sf.numel() != n * (k // bk):
            raise RuntimeError(
                f"MXFP8 1x32 scale numel {sf.numel()} != N*(K/32)={n * (k // bk)} "
                f"for weight {tuple(weight.shape)}"
            )
        sf_nk = sf.reshape(n, k // bk)
        for i0 in range(0, n, stripe):
            i1 = min(n, i0 + stripe)
            ns = i1 - i0
            wf = weight[i0:i1].float()
            chunk = (wf.view(ns, k // bk, bk) * sf_nk[i0:i1].unsqueeze(-1)).reshape(
                ns, k
            )
            out[i0:i1] = chunk.to(out_dtype)
    else:
        pn = ((n + bn - 1) // bn) * bn
        pk = ((k + bk - 1) // bk) * bk
        if sf.numel() != (pn // bn) * (pk // bk):
            raise RuntimeError(
                f"MXFP8 32x32 scale numel {sf.numel()} != "
                f"(N/32)*(K/32)={(pn // bn) * (pk // bk)} for weight "
                f"{tuple(weight.shape)}"
            )
        sf_tiles = sf.reshape(pn // bn, pk // bk)
        for i0 in range(0, pn, stripe):
            i1 = min(pn, i0 + stripe)
            n_pad = i1 - i0
            wf = weight.new_zeros((n_pad, pk), dtype=torch.float32)
            src_n = min(n, i1) - min(n, i0)
            if src_n > 0:
                wf[:src_n, :k] = weight[min(n, i0) : min(n, i1)].float()
            t0 = i0 // bn
            t1 = i1 // bn
            chunk = (
                wf.view(t1 - t0, bn, pk // bk, bk)
                * sf_tiles[t0:t1, None, :, None]
            ).reshape(n_pad, pk)
            dst0 = min(n, i0)
            dst1 = min(n, i1)
            if dst1 > dst0:
                out[dst0:dst1] = chunk[: dst1 - dst0, :k].to(out_dtype)
    return out


def expand_mxfp8_ue8m0_row_scales(
    scales: torch.Tensor,
    *,
    size_n: int,
    size_k: int,
    weight_block_size: tuple[int, int],
) -> torch.Tensor:
    """Return UE8M0 bytes as uint8 ``[N, K/32]``.

    ``[1, 32]`` scales already have one byte per output row and K-group.
    ``[32, 32]`` tiles share a scale across 32 rows; those bytes are
    repeated along N so the decode GEMV is a uniform row lookup.
    """
    from sglang.srt.layers.quantization.marlin_utils import (
        sm70_mxfp4_ue8m0_to_uint8,
    )

    block = tuple(int(x) for x in weight_block_size)
    if block not in _MXFP8_UE8M0_BLOCK_SIZES:
        raise RuntimeError(
            "Dense MXFP8 is UE8M0 g32 with weight_block_size [1, 32] or "
            f"[32, 32]; got {list(block)}. Refusing a third block size."
        )
    if size_k % 32 != 0:
        raise RuntimeError(f"MXFP8 K must be a multiple of 32, got {size_k}")
    raw = sm70_mxfp4_ue8m0_to_uint8(scales).contiguous().reshape(-1)
    n_groups = size_k // 32
    bn, _bk = block
    if bn == 1:
        if raw.numel() != size_n * n_groups:
            raise RuntimeError(
                f"MXFP8 1x32 scale numel {raw.numel()} != N*(K/32)="
                f"{size_n * n_groups} for weight ({size_n}, {size_k})"
            )
        return raw.view(size_n, n_groups).contiguous()
    pn = ((size_n + 31) // 32) * 32
    n_tiles = pn // 32
    k_tiles = n_groups
    if raw.numel() != n_tiles * k_tiles:
        raise RuntimeError(
            f"MXFP8 32x32 scale numel {raw.numel()} != (N/32)*(K/32)="
            f"{n_tiles * k_tiles} for weight ({size_n}, {size_k})"
        )
    return raw.view(n_tiles, k_tiles).repeat_interleave(32, dim=0)[:size_n].contiguous()


def sm70_mxfp8_layer_skips_marlin(layer: torch.nn.Module) -> bool:
    """True when SM70 MXFP8 must not fall through into Marlin/TurboMind pack."""
    return bool(
        getattr(layer, "_sm70_mxfp8_w8a16", False)
        or getattr(layer, "_sm70_mxfp8_dequant_fp16", False)
    )


def _unpack_mxfp8_layer_to_fp16(
    layer: torch.nn.Module,
    scale_param: torch.Tensor,
    block: tuple[int, int],
    orig: torch.dtype,
) -> None:
    fp16_w = dequant_mxfp8_ue8m0_to_fp16(
        layer.weight.data, scale_param.data, block, out_dtype=orig
    )
    if fp16_w.numel() == 0 or not bool(torch.isfinite(fp16_w).all()):
        raise RuntimeError(
            "SM70 MXFP8 unpack produced an empty or non-finite weight "
            f"{tuple(fp16_w.shape)} dtype={fp16_w.dtype}"
        )
    max_abs = float(fp16_w.abs().max())
    if max_abs == 0.0:
        raise RuntimeError(
            "SM70 MXFP8 unpack produced an all-zero weight "
            f"{tuple(fp16_w.shape)} dtype={fp16_w.dtype}. UE8M0 scales were "
            "likely copied into uint8 with a numeric cast instead of a "
            "bit-preserving view (copy_with_check)."
        )
    layer.weight = torch.nn.Parameter(fp16_w.contiguous(), requires_grad=False)
    if hasattr(layer, "weight_scale_inv"):
        del layer.weight_scale_inv
    if hasattr(layer, "weight_scale"):
        delattr(layer, "weight_scale")
    layer._sm70_mxfp8_dequant_fp16 = True
    logger.warning(
        "SM70: unpacking dense MXFP8 UE8M0 g32 to FP16 (SGLANG_DSV41_MXFP8_W8A16=0; "
        "marlin_v100 FP8 W8A16 has no group-32)."
    )


def prepare_mxfp8_layer_for_sm70_marlin(layer: torch.nn.Module) -> None:
    """Keep dense MXFP8 e4m3+UE8M0 packed, or unpack to FP16 if the kill-switch is off.

    marlin_v100's FP8 kernel accepts group_size in {-1, 128} only. Official
    DSV4.1-Flash dense is 32x32 UE8M0. Default: leave e4m3 in HBM and expand
    scales to ``[N, K/32]`` uint8 for the SM70 decode GEMV. ``SGLANG_DSV41_MXFP8_W8A16=0``
    restores the previous persistent FP16 unpack.
    """
    from sglang.srt.environ import envs

    global _SM70_MXFP8_PACKED_LOGGED

    weight_block_size = getattr(layer, "weight_block_size", None)
    if weight_block_size is None:
        weight_block_size = [32, 32]
    block = tuple(int(x) for x in weight_block_size)
    scale_param = getattr(layer, "weight_scale_inv", None)
    if scale_param is None:
        scale_param = getattr(layer, "weight_scale", None)
    if scale_param is None:
        raise ValueError("MXFP8 unpack requires weight_scale / weight_scale_inv.")
    orig = getattr(layer, "orig_dtype", torch.float16)
    if orig is None or orig == torch.bfloat16:
        orig = torch.float16
    if not envs.SGLANG_DSV41_MXFP8_W8A16.get():
        _unpack_mxfp8_layer_to_fp16(layer, scale_param, block, orig)
        return

    n, k = int(layer.weight.shape[0]), int(layer.weight.shape[1])
    row_scales = expand_mxfp8_ue8m0_row_scales(
        scale_param.data, size_n=n, size_k=k, weight_block_size=block
    )
    if int(row_scales.max()) == 0:
        raise RuntimeError(
            "SM70 MXFP8 packed scales are all-zero for weight "
            f"{tuple(layer.weight.shape)}. UE8M0 scales were likely copied "
            "into uint8 with a numeric cast instead of a bit-preserving "
            "view (copy_with_check)."
        )
    layer.weight = torch.nn.Parameter(
        layer.weight.data.contiguous(), requires_grad=False
    )
    scale_out = torch.nn.Parameter(row_scales, requires_grad=False)
    if hasattr(layer, "weight_scale_inv"):
        layer.weight_scale_inv = scale_out
        if hasattr(layer, "weight_scale"):
            delattr(layer, "weight_scale")
    else:
        layer.weight_scale = scale_out
    layer.weight_block_size = [1, 32]
    layer._sm70_mxfp8_w8a16 = True
    if not _SM70_MXFP8_PACKED_LOGGED:
        logger.info(
            "SM70: keeping dense MXFP8 e4m3+UE8M0 packed (marlin_v100 FP8 W8A16 "
            "has no group-32). Decode M<=4 uses JIT GEMV; prefill dequants to a "
            "transient FP16 workspace."
        )
        _SM70_MXFP8_PACKED_LOGGED = True


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

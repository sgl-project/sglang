from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_mla_kv_pack_quantize_fp8_module(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    qk_nope: int,
    qk_rope: int,
    v_head: int,
    use_pdl: bool,
) -> Module:
    args = make_cpp_args(in_dtype, out_dtype, qk_nope, qk_rope, v_head, use_pdl)
    tag = (
        f"mla_kv_pack_quantize_fp8_{qk_nope}_{qk_rope}_{v_head}_"
        f"{str(in_dtype).split('.')[-1]}_{str(out_dtype).split('.')[-1]}"
    )
    return load_jit(
        tag,
        *args,
        cuda_files=["elementwise/mla_kv_pack_quantize_fp8.cuh"],
        cuda_wrappers=[
            (
                "mla_kv_pack_quantize_fp8",
                f"MlaKVPackQuantizeFp8Kernel<{args}>::run",
            )
        ],
    )


def _pick_dispatch(s_total: int) -> Tuple[int, int, int]:
    """Return (vec_n, block_s, num_warps).

    - block_s=0 (sentinel): "small / fat-warp" kernel. One CTA = 1 warp per
      (token, head), straight-line code, no inner loop. Wins at bs <= 16 where
      the loop kernel's per-CTA bookkeeping dominates.
    - vec_n=8 (128-bit BF16/FP16 loads): wins at small s where per-CTA work
      is light and the extra cost of 256-bit ld/st instructions isn't paid back.
    - vec_n=16 (256-bit BF16/FP16 loads): wins at large s where the kernel is
      memory bandwidth-bound and fewer instructions help.

    Tuned by sweep on GB300 (148 SMs, BF16 → FP8 e4m3, DSv3 dims).
    """
    if s_total <= 14:
        # Fat-warp variant: one CTA = 1 warp per (token, head), straight-line.
        # vec_n=4 fills the warp for the 32-vec K_nope/V phases (128/4=32 vecs)
        # and uses 16 lanes for the 16-vec K_pe phase. Wins through bs~14 where
        # SM occupancy from one-CTA-per-(token,head) outweighs the loop kernel's
        # ability to keep many threads per CTA busy.
        return 4, 0, 1
    if s_total <= 192:
        return 8, 8, 4
    if s_total <= 256:
        # Marginal win over the bs<=1536 band at exactly bs=256 in the 8-layer
        # cudagraph bench (1.82us vs 1.86us). vec_n=8 + larger block_s keeps
        # register pressure low and warp occupancy high.
        return 8, 32, 8
    if s_total <= 1536:
        return 16, 16, 4
    # Bandwidth-bound regime. Sweep on GB300 (148 SMs, BF16 → FP8 e4m3, DSv3
    # dims) shows block_s=64 with the batched-load fast path keeps enough
    # transactions in flight at bs >= 2048.
    if s_total <= 8192:
        return 16, 64, 8
    return 16, 64, 16


def mla_kv_pack_quantize_fp8(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    k_scale_inv: float = 1.0,
    v_scale_inv: float = 1.0,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: Optional[bool] = None,
    vec_n: int = 0,
    block_s: int = 0,
    num_warps: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused cat(k_nope, broadcast k_pe) + per-tensor FP8 quantize for K, plus
    per-tensor FP8 quantize for V. Single-kernel JIT replacement for the Triton
    reference in ``tokenspeed_mla.mla_kv_pack_quantize_fp8``.

    Args:
        k_nope: BF16/FP16 ``[s, h, qk_nope]``. May be a strided view.
        k_pe:   BF16/FP16 ``[s, 1, qk_rope]`` or ``[s, qk_rope]``. Shared across heads.
        v:      BF16/FP16 ``[s, h, v_head]``. May be a strided view.
        k_scale_inv: scalar multiplier applied before the FP8 cast on K.
        v_scale_inv: scalar multiplier applied before the FP8 cast on V.
        k_out, v_out: optional pre-allocated FP8 outputs.
        fp8_dtype: ``torch.float8_e4m3fn`` (default) or ``torch.float8_e5m2``.
        enable_pdl: opt into PDL. Auto-detected when ``None``.
        block_s, num_warps: dispatch tuning knobs (0 = auto from ``s``).

    Returns:
        ``(k_fp8 [s, h, qk_nope + qk_rope], v_fp8 [s, h, v_head])``.
    """
    assert k_nope.dtype in (torch.bfloat16, torch.float16), (
        f"k_nope must be bf16/fp16, got {k_nope.dtype}"
    )
    assert k_pe.dtype == k_nope.dtype and v.dtype == k_nope.dtype, (
        "k_nope, k_pe, v must share dtype"
    )
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    s, num_heads, qk_nope = k_nope.shape
    qk_rope = k_pe.shape[-1]
    v_head = v.shape[-1]

    assert v.shape[0] == s and v.shape[1] == num_heads, (
        f"v shape {tuple(v.shape)} mismatches k_nope {tuple(k_nope.shape)}"
    )
    assert k_pe.shape[0] == s, (
        f"k_pe first dim {k_pe.shape[0]} mismatches k_nope first dim {s}"
    )
    assert k_nope.stride(-1) == 1, "k_nope must have stride-1 inner dim"
    assert v.stride(-1) == 1, "v must have stride-1 inner dim"
    assert k_pe.stride(-1) == 1, "k_pe must have stride-1 inner dim"

    if k_pe.dim() == 3:
        assert k_pe.shape[1] == 1, f"k_pe head dim must be 1, got {k_pe.shape[1]}"
        k_pe_2d = k_pe.squeeze(1)
    else:
        k_pe_2d = k_pe

    if k_out is None:
        k_out = torch.empty(
            (s, num_heads, qk_nope + qk_rope), dtype=fp8_dtype, device=k_nope.device
        )
    if v_out is None:
        v_out = torch.empty((s, num_heads, v_head), dtype=fp8_dtype, device=v.device)

    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()

    if vec_n <= 0 or block_s <= 0 or num_warps <= 0:
        auto_vn, auto_bs, auto_nw = _pick_dispatch(s)
        if vec_n <= 0:
            vec_n = auto_vn
        if block_s <= 0:
            block_s = auto_bs
        if num_warps <= 0:
            num_warps = auto_nw

    module = _jit_mla_kv_pack_quantize_fp8_module(
        k_nope.dtype, fp8_dtype, qk_nope, qk_rope, v_head, enable_pdl
    )
    module.mla_kv_pack_quantize_fp8(
        k_out,
        v_out,
        k_nope,
        k_pe_2d,
        v,
        float(k_scale_inv),
        float(v_scale_inv),
        vec_n,
        block_s,
        num_warps,
    )
    return k_out, v_out

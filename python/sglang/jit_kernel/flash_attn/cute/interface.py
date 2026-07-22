# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor

from sglang.jit_kernel.flash_attn.cute.cache_utils import get_jit_cache
from sglang.jit_kernel.flash_attn.cute.testing import is_fake_mode
from sglang.jit_kernel.utils import is_arch_support_pdl

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from sglang.jit_kernel.flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from sglang.jit_kernel.flash_attn.cute import fa_logging, utils
from sglang.jit_kernel.flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    get_sparse_q_block_size,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from sglang.jit_kernel.flash_attn.cute.cu_blocks_kernels import CuSeqlensToBlocksKernel
from sglang.jit_kernel.flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from sglang.jit_kernel.flash_attn.cute.flash_fwd_combine import (
    FlashAttentionForwardCombine,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd_mla_sm100 import (
    FlashAttentionMLAForwardSm100,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from sglang.jit_kernel.flash_attn.cute.flash_fwd_sm100 import (
    DescaleTensors,
    FlashAttentionForwardSm100,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from sglang.jit_kernel.flash_attn.cute.shearing_bias import ShearingBias

# SM100 head_dim=256 2CTA kernel imports
from sglang.jit_kernel.flash_attn.cute.sm100_hd256_2cta_fmha_forward import (
    BlackwellFusedMultiHeadAttentionForward,
)
from sglang.jit_kernel.flash_attn.cute.utils import AuxData


def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(
    head_dim: int, head_dim_v: int, compute_capability: int, alignment: int
) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_deepseek_mla_absorbed_shape = (
        head_dim == 64 or head_dim == head_dim_v
    ) and head_dim_v == 512
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert (
            is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0
        ), (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (
            (
                is_standard_range
                or is_deepseek_shape
                or is_deepseek_mla_absorbed_shape
                or is_dedicate_kernel_shape
            )
            and head_dim % alignment == 0
            and head_dim_v % alignment == 0
        ), (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(
    head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None
):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert (
        t.shape == expected_shape
    ), f"{name} shape {t.shape} != expected {expected_shape}"
    assert (
        t.dtype == expected_dtype
    ), f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert (
        t.device == expected_device
    ), f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


_shear_bias_workspace: dict = {}


def _round_up_to_tile(size: int, tile_size: int) -> int:
    """Return the smallest whole-tile buffer capacity that holds ``size`` rows."""
    assert tile_size > 0
    return (size + tile_size - 1) // tile_size * tile_size


def _shear_bias_empty(shape, dtype, device):
    # Grow-only per-device workspace: the sheared-bias staging tensor is large
    # (total_q x num_head x rel_extent_padded) and call shapes vary, so per-call
    # torch.empty fragments the caching allocator until GPU memory is exhausted.
    # Contents never persist across calls (written by the shear kernel, read by
    # the fwd kernel within the same call); assumes attention calls on a device
    # are serialized. Bypassed under graph capture (a capture-pool pointer must
    # not leak into eager use) and fake mode (a fake tensor must not be cached).
    if is_fake_mode() or torch.cuda.is_current_stream_capturing():
        return torch.empty(shape, dtype=dtype, device=device)
    nbytes = math.prod(shape) * dtype.itemsize
    buf = _shear_bias_workspace.get(device)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _shear_bias_workspace[device] = buf
    return buf[:nbytes].view(dtype).view(shape)


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1
    # Avoid ZeroDivisionError when batch_size or seqlen_q is 0. The empty-Q
    # early-exit in _flash_attn_fwd handles correctness for those shapes; this
    # guard just keeps the heuristic safe if called in other contexts.
    if total_mblocks == 0:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(
    causal, window_size_left, window_size_right, mask_mod=None
):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left + window_size_right < 0
    ):
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _group_tile_bias(qhead_per_kvhead_packgqa=1):
    return 128


def _flash_attn_fwd(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    qk_sf_vec_size: Optional[int] = None,
    v_sf_vec_size: Optional[int] = None,
    rel_bias_prep_cache: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        aux_scalars: Runtime scalar captures used by score_mod or mask_mod.
    """
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    q, k, v, qv = [maybe_contiguous(t) for t in (q, k, v, qv)]
    assert q is not None or qv is not None
    assert v is not None
    q_descale, k_descale, v_descale = [
        maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)
    ]
    q_shape = q.shape if q is not None else qv.shape
    num_head, head_dim = q_shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
        # SFQ can be laid out in the interleaved BlockScaledBasicChunk atom when
        # the whole q tensor is dense (no varlen / seqused).
        q_sf_interleaved = seqused_q is None
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q_shape[0]
        q_sf_interleaved = False
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert (
            page_table.stride(-1) == 1
        ), "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = v.shape[:2]
        seqlen_k = num_pages * page_size
        # Paged KV: SF is interleaved (TMA-loadable) only when page_size == 128
        # so a page maps exactly onto the SF atom's 128-row tile.
        kv_sf_interleaved = page_size == 128
    else:
        num_pages, page_size = None, None
        seqlen_k = v.shape[-3]
        kv_sf_interleaved = True
    num_head_kv = v.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k is None or k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (
        batch_size,
    ), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (
        batch_size,
    ), "seqused_k must have shape (batch_size,)"
    # MXFP8 block-scaled attention.
    #   qk_blockscaled   (sfq given): Q/K fp8 e4m3 with per-32 UE8M0 scales; QK^T
    #                 runs as tcgen05 mxf8f6f4 with scales fed from TMEM.
    #   v_blockscaled (sfv given): V stored fp8 e4m3, dequantized to bf16
    #                 in-kernel by the correction warp; PV MMA stays bf16.
    # Paged KV with qk_blockscaled QK requires v_blockscaled.
    qk_blockscaled = sfq is not None
    v_blockscaled = sfv is not None
    if page_table is not None and qk_blockscaled:
        assert v_blockscaled, "paged KV with qk_blockscaled requires v_blockscaled"
    if v_blockscaled:
        assert v.dtype in [torch.float8_e4m3fn], "v_blockscaled V must be float8_e4m3fn"
        assert sfv.dtype == torch.float8_e8m0fnu, "sfv must be float8_e8m0fnu"
        assert (
            v_sf_vec_size is not None
        ), "v_sf_vec_size must be provided for v_blockscaled"
    if qk_blockscaled:
        assert sfk is not None, "sfq and sfk must both be provided for qk_blockscaled"
        assert (
            qk_sf_vec_size is not None
        ), "qk_sf_vec_size must be provided for qk_blockscaled"
        assert q is not None and q.dtype in [
            torch.float8_e4m3fn
        ], "qk_blockscaled Q must be float8_e4m3fn"
        assert q.dtype == k.dtype, "qk_blockscaled Q and K must have the same dtype"
        assert sfq.dtype == torch.float8_e8m0fnu, "sfq must be float8_e8m0fnu"
        assert sfk.dtype == torch.float8_e8m0fnu, "sfk must be float8_e8m0fnu"
        if not v_blockscaled:
            assert v.dtype in [
                torch.float16,
                torch.bfloat16,
            ], "qk_blockscaled V must be float16 or bfloat16"
    else:
        assert sfk is None, "sfq and sfk must both be provided for qk_blockscaled"
        assert v.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ], "inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2"

    # qk_blockscaled: Q/K are fp8 (same dtype); V may be bf16 (no v_blockscaled) or
    # fp8 (v_blockscaled) -- exclude V from the cross-dtype check either way.
    # v_blockscaled without qk_blockscaled: Q/K bf16, V fp8 -- exclude V.
    if qk_blockscaled:
        input_tensors = {"q": q, "k": k, "qv": qv}
    elif v_blockscaled:
        input_tensors = {"q": q, "qv": qv}
    else:
        input_tensors = {"q": q, "k": k, "v": v, "qv": qv}
    present = {name: t for name, t in input_tensors.items() if t is not None}
    names = list(present.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            assert (
                present[a].dtype == present[b].dtype
            ), f"{a}.dtype {present[a].dtype} != {b}.dtype {present[b].dtype}"

    q_dtype = q.dtype if q is not None else qv.dtype

    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert (
                t.dtype == torch.int32
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert (
                t.stride(0) == 1
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                qv,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [
        8,
        9,
        10,
        11,
        12,
    ], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // v.element_size()
    if arch // 10 not in [8, 12]:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim)
            if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if pack_gqa:
        # pack_gqa reshapes SFQ's head/token layout, which the interleaved atom
        # can't express; fall back to the dense (non-interleaved) SFQ path.
        q_sf_interleaved = False

    is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
    if is_fp8 and requires_grad:
        raise NotImplementedError(
            "FA4 CuTe FP8 backward is not supported yet (forward-only)."
        )
    # qk_blockscaled (fp8 Q/K, bf16 V): output follows V's dtype. v_blockscaled
    # (fp8 V dequanted in-kernel): output is bf16.
    if qk_blockscaled:
        out_torch_dtype = torch.bfloat16 if v_blockscaled else v.dtype
    else:
        out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
    device = v.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )

    if qv is None:
        lse_shape = (
            (batch_size, num_head, seqlen_q)
            if cu_seqlens_q is None
            else (num_head, total_q)
        )
    else:
        # num_head contiguous better for MQA in MLA absorbed
        lse_shape = (
            (batch_size, seqlen_q, num_head)
            if cu_seqlens_q is None
            else (total_q, num_head)
        )

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=device,
        )
    else:
        _validate_tensor(
            out,
            "out",
            (*q_batch_seqlen_shape, num_head, head_dim_v),
            out_torch_dtype,
            device,
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse

    if is_fp8:
        for t, name in (
            (q_descale, "q_descale"),
            (k_descale, "k_descale"),
            (v_descale, "v_descale"),
        ):
            if t is not None:
                _validate_tensor(
                    t, name, (batch_size, num_head_kv), torch.float32, device
                )
    else:
        assert (
            q_descale is None and k_descale is None and v_descale is None
        ), "q_descale/k_descale/v_descale are only supported for FP8 inputs"

    dtype = torch2cute_dtype_map[q_dtype]
    if is_fp8:
        assert (
            arch // 10 == 10
        ), "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default(is_fwd=True)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
    if arch // 10 in [8, 12]:
        num_threads = 128

    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    if tile_mn is None:
        if arch // 10 == 12:
            # SM120 tile sizes tuned for 99 KB SMEM capacity:
            # D<=64:  128x128 → 48 KB (good occupancy)
            # D>64:   128x64  → 64 KB (128x128 would use 96 KB, hurting occupancy)
            if head_dim <= 64:
                fwd_cfg = FwdConfig(128, 128, True, True)
            else:
                fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = _tile_size_fwd_sm90(
                head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q
            )
    else:
        fwd_cfg = FwdConfig(
            tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap
        )
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if arch // 10 in [10, 11]:
        # q_stage=2 hangs on sm100 for qk_blockscaled; force q_stage=1 there.
        q_stage = 1 if qk_blockscaled else (2 if seqlen_q_packgqa > tile_m else 1)
    else:
        q_stage = 1

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = (
        max_seqlen_k
        if not local
        else max(
            0,
            min(
                max_seqlen_k,
                (window_size_right or max_seqlen_k)
                + (window_size_left or max_seqlen_k)
                + 1
                + tile_m,
            ),
        )
    )
    num_m_blocks = (
        seqlen_q_packgqa + m_block_size_effective - 1
    ) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = (
        132
        if is_fake_mode()
        else torch.cuda.get_device_properties(device).multi_processor_count
    )
    if num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if (
            num_n_blocks >= 64
            and head_dim_v != 512
            and (page_table is None or page_size == 64)
        ):
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        lse_partial = torch.empty(
            num_splits, *lse_shape, dtype=torch.float32, device=device
        )

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
        and not qk_blockscaled
    )

    # hd=256 2CTA forward uses dedicated kernel (Blackwell family)
    use_dedicated_hd256_kernel = (
        arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    )
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError(
                "Custom user-provided score_mod is not supported on SM8x architectures."
            )

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = (
        requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal
    )

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if cu_seqlens_q is not None:
            assert (
                block_sparse_tensors.cu_total_m_blocks is not None
            ), "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    aux_scalar_metadata = (
        tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None
    )

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert q is None or qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim_v == 512
        assert q is None or head_dim == 64
        assert not local, "local not yet supported with qv"
        assert (
            q_descale is None and k_descale is None and v_descale is None
        ), "q_descale/k_descale/v_descale are not yet supported with qv"
        assert tile_n == 128

        assert not is_split_kv, "split kv not supported with qv"
        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None

        if page_table is not None:
            assert (
                gather_kv_indices is None
            ), "paged KV + topk sparsity not yet supported together"

        qv = maybe_contiguous(qv)

        gather_kv_length = 2048  # dummy value
        sparse_kv = gather_kv_indices is not None
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == qv.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 128 == 0
            if min_seqlen_k is None or causal:
                disable_sparse_kv_bitmask = False
            else:
                # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
                seqlen_k_boundary = min_seqlen_k
                disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length
        # to be used for sparse backward
        p = row_max = None
    else:
        assert gather_kv_indices is None, "gather_kv_indices is only supported with qv"
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None
        p = row_max = None

    # rel_bias -> sheared bias (Inkling relative attention). Produces `bias`, the column-aligned
    # bias the SM100 kernel adds to pre-softmax scores via its dedicated TMA pipeline.
    rel_extent = 0
    rel_extent_padded = 0
    bias = None
    tile_bias = tile_m
    cu_total_m_blocks_bias = None
    blocks_to_batch_idx = None
    if rel_bias is not None:
        assert arch // 10 in [
            9,
            10,
            11,
        ], "rel_bias (sheared bias) is only supported on SM9x/10x"
        qhead_per_kvhead_packgqa = qhead_per_kvhead if pack_gqa else 1
        rel_extent = rel_bias.shape[-1]
        rel_extent_padded = rel_extent + 256
        assert rel_extent % 128 == 0
        assert tile_m == 128 and tile_n == 128
        assert (
            causal
            or window_size_left is None
            or (
                window_size_right is not None
                and window_size_left + window_size_right + 1 == rel_extent
            )
        ), "for relative bias, require causal or window length == rel_extent"
        tile_bias = (
            (seqlen_q_packgqa + 7) // 8 * 8 if seqlen_q_packgqa < tile_m else tile_m
        )
        if cu_seqlens_q is None:
            bias_seqlen_q_rounded = _round_up_to_tile(seqlen_q, tile_m)
            assert rel_bias.shape == (batch_size, seqlen_q, num_head, rel_extent)
            bias = _shear_bias_empty(
                (batch_size, bias_seqlen_q_rounded, num_head, rel_extent_padded),
                rel_bias.dtype,
                device,
            )
        else:
            assert rel_bias.shape == (total_q, num_head, rel_extent)
            bias_total_q_rounded = _round_up_to_tile(total_q, tile_m)
            bias = _shear_bias_empty(
                (bias_total_q_rounded, num_head, rel_extent_padded),
                rel_bias.dtype,
                device,
            )

        rows_per_cta = 4
        group_tile_bias = _group_tile_bias(qhead_per_kvhead_packgqa)
        # Decode and target verification fit each sequence in one packed-Q block.
        # In that case the batch index is already a scheduler coordinate, so the
        # prefix-sum/block-map preparation kernel is pure launch overhead.
        max_m_blocks_leq_one = seqlen_q_packgqa <= group_tile_bias
        use_pdl = is_arch_support_pdl()
        bias_max_seqlen_q = max_seqlen_q if max_seqlen_q is not None else seqlen_q
        bias_max_seqlen_k = max_seqlen_k if max_seqlen_k is not None else seqlen_k

        # Block-packed scheduling for the shear kernel (varlen only).
        use_prepare_bias_kernel = (
            cu_seqlens_q is not None and not max_m_blocks_leq_one and batch_size <= 1024
        )
        if use_prepare_bias_kernel:
            prep_cache_key = (
                group_tile_bias,
                qhead_per_kvhead_packgqa,
                cu_seqlens_q.data_ptr(),
            )
            cached_prep = (
                rel_bias_prep_cache.get(prep_cache_key)
                if rel_bias_prep_cache is not None
                else None
            )
            if cached_prep is not None:
                cu_total_m_blocks_bias, blocks_to_batch_idx = cached_prep
            else:
                cu_total_m_blocks_bias = torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                total_group_blocks_max = (
                    total_q * qhead_per_kvhead_packgqa
                    + batch_size * (group_tile_bias - 1)
                ) // group_tile_bias
                blocks_to_batch_idx = torch.empty(
                    total_group_blocks_max, dtype=torch.int32, device=device
                )
                compile_key_prepare = (group_tile_bias, qhead_per_kvhead_packgqa)
                if (
                    compile_key_prepare
                    not in _flash_attn_fwd.compile_cache_prepare_shear_bias
                ):
                    (
                        cu_total_m_blocks_bias_tensor,
                        cu_seqlens_q_tensor,
                        blocks_to_batch_idx_tensor,
                    ) = [
                        to_cute_tensor(t, assumed_align=4, leading_dim=0)
                        for t in (
                            cu_total_m_blocks_bias,
                            cu_seqlens_q,
                            blocks_to_batch_idx,
                        )
                    ]
                    _flash_attn_fwd.compile_cache_prepare_shear_bias[
                        compile_key_prepare
                    ] = cute.compile(
                        CuSeqlensToBlocksKernel(
                            tile=group_tile_bias,
                            seqlen_multiple=qhead_per_kvhead_packgqa,
                            use_pdl=use_pdl,
                        ),
                        cu_total_m_blocks_bias_tensor,
                        cu_seqlens_q_tensor,
                        blocks_to_batch_idx_tensor,
                        current_stream,
                        options="--enable-tvm-ffi",
                    )
                if not is_fake_mode():
                    _flash_attn_fwd.compile_cache_prepare_shear_bias[
                        compile_key_prepare
                    ](
                        cu_total_m_blocks_bias,
                        cu_seqlens_q,
                        blocks_to_batch_idx,
                    )
                if rel_bias_prep_cache is not None:
                    rel_bias_prep_cache[prep_cache_key] = (
                        cu_total_m_blocks_bias,
                        blocks_to_batch_idx,
                    )

        shear_compile_key = (
            rel_bias.dtype,
            rel_extent,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            pack_gqa,
            qhead_per_kvhead,
            rows_per_cta,
            group_tile_bias,
            max_m_blocks_leq_one,
            cu_total_m_blocks_bias is not None,
            blocks_to_batch_idx is not None,
        )
        if shear_compile_key not in _flash_attn_fwd.compile_cache_shear_bias:
            (
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                cu_total_m_blocks_bias_tensor,
                blocks_to_batch_idx_tensor,
            ) = [
                (
                    to_cute_tensor(t, assumed_align=4, leading_dim=0)
                    if t is not None
                    else None
                )
                for t in (
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqused_q,
                    seqused_k,
                    cu_total_m_blocks_bias,
                    blocks_to_batch_idx,
                )
            ]
            _flash_attn_fwd.compile_cache_shear_bias[shear_compile_key] = cute.compile(
                ShearingBias(
                    rel_extent,
                    is_causal=causal,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    rows_per_cta=rows_per_cta,
                    tile_m=group_tile_bias,
                    max_m_blocks_leq_one=max_m_blocks_leq_one,
                    use_pdl=use_pdl,
                ),
                to_cute_tensor(rel_bias),
                to_cute_tensor(bias),
                bias_max_seqlen_q,
                bias_max_seqlen_k,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                cu_total_m_blocks_bias_tensor,
                blocks_to_batch_idx_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )
        if not is_fake_mode():
            _flash_attn_fwd.compile_cache_shear_bias[shear_compile_key](
                rel_bias,
                bias,
                bias_max_seqlen_q,
                bias_max_seqlen_k,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                cu_total_m_blocks_bias,
                blocks_to_batch_idx,
                window_size_left,
                window_size_right,
            )
        if os.environ.get("BIAS_PREP_ONLY", "0") == "1":
            # Benchmark hook: time just the shear-prep kernels, skip the attention kernel.
            return out, lse

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        aux_scalar_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None
        or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        page_size not in [None, tile_n],  # paged KV non-TMA
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        q is not None,
        qv is not None,
        p is not None,
        row_max is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        bias is not None,
        tile_bias,
        rel_extent,
        qk_blockscaled,
        qk_sf_vec_size,
        v_blockscaled,
        v_sf_vec_size,
        q_sf_interleaved,
        kv_sf_interleaved,
        sfq.ndim if sfq is not None else None,
        sfk.ndim if sfk is not None else None,
        sfv.ndim if sfv is not None else None,
        fa_logging.get_fa_log_level(),
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t)
            for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        bias_tensor = to_cute_tensor(bias) if bias is not None else None
        if qk_blockscaled:
            sfq_tensor = to_cute_tensor(sfq)
            sfk_tensor = to_cute_tensor(sfk)
        else:
            sfq_tensor = None
            sfk_tensor = None
        sfv_tensor = to_cute_tensor(sfv) if v_blockscaled else None
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        q_descale_tensor = (
            to_cute_tensor(q_descale, assumed_align=4, leading_dim=1)
            if q_descale is not None
            else None
        )
        k_descale_tensor = (
            to_cute_tensor(k_descale, assumed_align=4, leading_dim=1)
            if k_descale is not None
            else None
        )
        v_descale_tensor = (
            to_cute_tensor(v_descale, assumed_align=4, leading_dim=1)
            if v_descale is not None
            else None
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor,
                k_descale=k_descale_tensor,
                v_descale=v_descale_tensor,
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(
                normalized_block_sparse_tensors
            )

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        qv_tensor = to_cute_tensor(qv) if qv is not None else None
        gather_kv_indices_tensor = (
            to_cute_tensor(gather_kv_indices) if gather_kv_indices is not None else None
        )
        p_tensor = to_cute_tensor(p) if p is not None else None
        row_max_tensor = to_cute_tensor(row_max) if row_max is not None else None

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=page_size not in [None, tile_n],
                has_bias=bias is not None,
                bias_block_size=tile_bias,
                rel_extent_padded=rel_extent_padded,
            )
        elif arch // 10 in [10, 11]:
            if qv is not None:
                paged_kv_cpasync = page_table is not None and page_size != tile_n
                has_qk = q is not None
                fa_fwd = FlashAttentionMLAForwardSm100(
                    is_causal=causal,
                    use_cpasync_load_KV=sparse_kv or paged_kv_cpasync,
                    topk_length=gather_kv_length,
                    is_topk_gather=sparse_kv,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    nheads_kv=num_head_kv,
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    disable_bitmask=disable_sparse_kv_bitmask,
                    has_qk=has_qk,
                )
            else:
                if use_dedicated_hd256_kernel:
                    # hd=256 2CTA forward: check for currently unsupported features
                    assert (
                        softcap is None
                    ), "SM100 forward with head_dim=256 does not support softcap"
                    assert (
                        not use_block_sparsity
                    ), "SM100 forward with head_dim=256 does not support block sparsity"
                    assert (
                        learnable_sink is None
                    ), "SM100 forward with head_dim=256 does not support learnable_sink"
                    assert (
                        seqused_q is None and seqused_k is None
                    ), "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
                    if page_table is not None:
                        assert max_seqlen_k % page_size == 0, (
                            f"SM100 hd256 2CTA paged KV requires max_seqlen_k divisible by "
                            f"page_size ({page_size}), got max_seqlen_k={max_seqlen_k}"
                        )
                        assert page_table.shape[1] == max_seqlen_k // page_size, (
                            f"SM100 hd256 2CTA paged KV requires page_table.shape[1] == "
                            f"max_seqlen_k // page_size ({max_seqlen_k} // {page_size} = "
                            f"{max_seqlen_k // page_size}), got {page_table.shape[1]}; "
                            f"pass page_table[:, :{max_seqlen_k // page_size}] to slice to "
                            f"the actual sequence length"
                        )
                        assert page_table.stride(0) == page_table.shape[1], (
                            f"SM100 hd256 2CTA paged KV requires a fully contiguous page_table "
                            f"(stride(0)={page_table.stride(0)} must equal "
                            f"shape[1]={page_table.shape[1]})"
                        )
                    # pack_gqa is an auto-selected optimization; disable it for hd256 kernel
                    pack_gqa = False

                flash_fwd_obj_cls = (
                    BlackwellFusedMultiHeadAttentionForward
                    if use_dedicated_hd256_kernel
                    else FlashAttentionForwardSm100
                )

                fa_fwd = flash_fwd_obj_cls(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    is_persistent=not causal
                    and not local
                    and cu_seqlens_q is None
                    and seqused_q is None
                    and not is_split_kv,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, tile_n],
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    has_bias=bias is not None,
                    bias_block_size=tile_bias,
                    rel_extent_padded=rel_extent_padded,
                    # hd256 class doesn't take these kwargs (qk_blockscaled excludes hd256)
                    **(
                        {}
                        if use_dedicated_hd256_kernel
                        else dict(
                            qk_blockscaled=qk_blockscaled,
                            v_dequant=v_blockscaled,
                            q_sf_interleaved=q_sf_interleaved,
                            kv_sf_interleaved=kv_sf_interleaved,
                        )
                    ),
                )
        elif arch // 10 == 12:
            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
            assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                qv_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                p_tensor,
                row_max_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                gather_kv_indices_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )
        else:
            compile_args = [
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
            ]
            if arch // 10 in [10, 11]:
                compile_args.append(descale_tensors_tensor)
            compile_args.extend(
                [
                    sparse_tensors,
                    AuxData(cute_aux_tensors, aux_scalars),
                ]
            )
            if arch // 10 in [9, 10, 11]:
                compile_args.append(bias_tensor)  # mBias
            if arch // 10 in [10, 11]:
                if not use_dedicated_hd256_kernel:
                    compile_args.extend(
                        [
                            sfq_tensor,  # mSFQ
                            sfk_tensor,  # mSFK
                            sfv_tensor,  # mSFV
                            qk_sf_vec_size,
                            v_sf_vec_size,
                        ]
                    )
            compile_args.append(current_stream)
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                *compile_args, options="--enable-tvm-ffi"
            )

    if not is_fake_mode():
        q_call, k_call, v_call, qv_call = [
            t.detach() if t is not None else None for t in (q, k, v, qv)
        ]
        if is_fp8 or qk_blockscaled:
            # need uint8 workaround until we pin torch >= 2.11.0 where fp8 export
            # is supported. Under qk_blockscaled/v_blockscaled some tensors stay
            # bf16 (e.g. bf16 V), so view only the actual fp8 tensors.
            q_call, k_call, v_call, qv_call = [
                (
                    t.view(torch.uint8)
                    if t is not None
                    and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                    else t
                )
                for t in (q_call, k_call, v_call, qv_call)
            ]
        # SF tensors are e8m0fnu; the compile-time cute tensors keep that dtype,
        # so pass them through unchanged (matches to_cute_tensor(sfq/sfk/sfv)).
        sfq_call, sfk_call, sfv_call = sfq, sfk, sfv
        descale_tensors = (
            DescaleTensors(
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
            )
            if q_descale is not None or k_descale is not None or v_descale is not None
            else None
        )
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q_call,
                qv_call,
                k_call,
                v_call,
                out.detach(),
                lse,
                softmax_scale,
                p,
                row_max,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left,
                window_size_right,
            )
        else:
            call_args = [
                q_call,
                k_call,
                v_call,
                out.detach() if not is_split_kv else out_partial,
                lse_partial if is_split_kv else lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
                learnable_sink,
            ]
            if arch // 10 in [10, 11]:
                call_args.append(descale_tensors)
            call_args.extend(
                [
                    (
                        (
                            normalized_block_sparse_tensors.mask_block_cnt,
                            normalized_block_sparse_tensors.mask_block_idx,
                            normalized_block_sparse_tensors.full_block_cnt,
                            normalized_block_sparse_tensors.full_block_idx,
                            normalized_block_sparse_tensors.cu_total_m_blocks,
                            normalized_block_sparse_tensors.cu_block_idx_offsets,
                            normalized_block_sparse_tensors.dq_write_order,
                            normalized_block_sparse_tensors.dq_write_order_full,
                        )
                        if normalized_block_sparse_tensors is not None
                        else None
                    ),
                    AuxData(aux_tensors, aux_scalars),
                ]
            )
            if arch // 10 in [9, 10, 11]:
                call_args.append(bias)  # mBias
            if arch // 10 in [10, 11]:
                if not use_dedicated_hd256_kernel:
                    # qk_sf_vec_size / v_sf_vec_size are Constexpr (baked at
                    # compile time), so only the SF tensors go on the call.
                    call_args.extend(
                        [
                            sfq_call,  # mSFQ (None unless qk_blockscaled)
                            sfk_call,  # mSFK (None unless qk_blockscaled)
                            sfv_call,  # mSFV (None unless v_blockscaled)
                        ]
                    )
            _flash_attn_fwd.compile_cache[compile_key](*call_args)
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")
_flash_attn_fwd.compile_cache_shear_bias = get_jit_cache("fwd_shear_bias")
_flash_attn_fwd.compile_cache_prepare_shear_bias = get_jit_cache(
    "fwd_prepare_shear_bias"
)


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
        block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # specialize MLA attention formula
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # by setting q, k to None
            qv = q if qv is None else qv
            q = k = None
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(q, k, v, out, lse, *(aux_tensors or ()))
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.block_sparse_tensors_bwd = block_sparse_tensors_bwd
        ctx.set_materialize_grads(False)
        return out, lse


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        min_seqlen_k: Optional[int] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        block_sparse_tensors: Optional[list] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        rel_bias: Optional[torch.Tensor] = None,
        sfq: Optional[torch.Tensor] = None,
        sfk: Optional[torch.Tensor] = None,
        sfv: Optional[torch.Tensor] = None,
        qk_sf_vec_size: Optional[int] = None,
        v_sf_vec_size: Optional[int] = None,
        rel_bias_prep_cache: Optional[dict] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # specialize MLA attention formula
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # by setting q, k to None
            qv = q if qv is None else qv
            q = k = None
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=min_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
            rel_bias=rel_bias,
            sfq=sfq,
            sfk=sfk,
            sfv=sfv,
            qk_sf_vec_size=qk_sf_vec_size,
            v_sf_vec_size=v_sf_vec_size,
            rel_bias_prep_cache=rel_bias_prep_cache,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            *(aux_tensors or ()),
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.return_lse = return_lse
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.set_materialize_grads(False)
        return out, lse


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
        return_lse,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    qk_sf_vec_size: Optional[int] = None,
    v_sf_vec_size: Optional[int] = None,
    rel_bias_prep_cache: Optional[dict] = None,
    return_lse: bool = False,
):
    """
    Tensor arguments:
        q:  (total_q, nheads,   hdim)   or (batch, seqlen_q, nheads,   hdim)
        k:  (total_k, nheads_k, hdim)   or (batch, seqlen_k, nheads_k, hdim)
        v:  (total_k, nheads_k, hdim_v) or (batch, seqlen_k, nheads_k, hdim_v)
        qv: (total_q, nheads,   hdim_v) or (batch, seqlen_q, nheads,   hdim_v)
        cu_seqlens_q: (batch + 1)       or seqused_q: (batch)
        cu_seqlens_k: (batch + 1)       or seqused_k: (batch)
        gather_kv_indices: (total_q, gather_kv_length) or
                           (batch, seqlen_q, gather_kv_length)
        page_table: (batch, max_num_pages_per_seq)

    Return:
       out: (total_q, nheads, hdim) or (batch, seqlen_q, nheads, hdim)
       lse: (nheads, total_q)       or (batch, nheads, seqlen_q) if not has_qv (standard)
            (total_q, nheads)       or (batch, seqlen_q, nheads) if has_qv

    Explanation of some optional arguments & decisions:

    qv: we write the MLA weight absorbed formula as
        O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
        where Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache.

    lse return shape: with Qv, MQA with nheads at least divisible by 4 is typical,
        so we arrange for nheads as the contiguous mode for better vectorization.

    gather_kv_indices: used for topk sparsity with MLA absorption kernel.

    min_seqlen_k: for varlen, specifies the minimum kv sequence length for any batch.
        Used with gather_kv_indices to determine if we need oob masking.
    """
    # Default the SF vector size to 32 (the block-scaled granularity) only for
    # MXFP8 (e8m0-scaled) inputs -- other block-scaled dtypes use a different
    # vec size, so gate on the e8m0 scale dtype rather than mere presence.
    if qk_sf_vec_size is None and sfq is not None and sfq.dtype == torch.float8_e8m0fnu:
        qk_sf_vec_size = 32
    if v_sf_vec_size is None and sfv is not None and sfv.dtype == torch.float8_e8m0fnu:
        v_sf_vec_size = 32
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        gather_kv_indices,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        block_sparse_tensors,
        aux_tensors,
        aux_scalars,
        q_descale,
        k_descale,
        v_descale,
        rel_bias,
        sfq,
        sfk,
        sfv,
        qk_sf_vec_size,
        v_sf_vec_size,
        rel_bias_prep_cache,
        return_lse,
    )


def _compile_fwd_combine(
    dtype,
    dtype_partial,
    head_dim,
    tile_m,
    k_block_size,
    log_max_splits,
    has_cu_seqlens,
    has_seqused,
    has_lse,
    has_varlen_batch_idx,
    *,
    use_pdl,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
        use_pdl=use_pdl,
    )
    if not fa_combine.can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(
            dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div
        )
        mLSE_partial = fake_tensor(
            Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1
        )
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = (
            fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0)
            if has_lse
            else None
        )
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(
            dtype_partial,
            (num_splits, batch, seqlen, nheads, head_dim),
            divisibility=div,
        )
        mLSE_partial = fake_tensor(
            Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2
        )
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = (
            fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1)
            if has_lse
            else None
        )
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = (
        fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    )
    mSeqused = (
        fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    )
    mNumSplitsDynamic = None  # Not parametrized in compile_key
    mVarlenBatchIdx = (
        fake_tensor(Int32, (batch_for_1d,), divisibility=1)
        if has_varlen_batch_idx
        else None
    )
    mSemaphore = None  # Not parametrized in compile_key

    return cute.compile(
        fa_combine,
        mO_partial,
        mLSE_partial,
        mO,
        mLSE,
        mCuSeqlens,
        mSeqused,
        mNumSplitsDynamic,
        mVarlenBatchIdx,
        mSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    assert out_partial.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "out_partial must be fp16, bf16, or fp32"
    if not is_fake_mode():
        assert (
            out_partial.is_cuda and lse_partial.is_cuda
        ), "tensors must be on CUDA device"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    # Device architecture is invariant for the lifetime of this server/JIT
    # cache, so PDL does not belong in the compile key.
    use_pdl = is_arch_support_pdl()
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key, use_pdl=use_pdl
        )
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        varlen_batch_idx: Optional mapping from virtual batch index to real batch index
            (int32 tensor of shape (batch_size,)). Used by persistent tile schedulers
            that reorder batch processing for load balancing.
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype
    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(
                total_q, num_heads, head_size, dtype=out_dtype, device=device
            )
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )
    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device)
        else:
            lse = torch.empty(
                batch_size, num_heads, seqlen, dtype=torch.float32, device=device
            )
        lse = lse.transpose(-1, -2)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        varlen_batch_idx=varlen_batch_idx,
    )
    return out, lse

# SPDX-License-Identifier: Apache-2.0
"""Online FP8 weights for sm120 (RTX PRO 6000 Blackwell).

Single switch SGLANG_SM120_ONLINE_MXFP8, propagated to ``online_fp8_enabled``
by ``initialize_bf16_gemm_config`` on SM120. What it arms, all rowwise fp8
(one fp32 scale per output row, halving bytes on bandwidth-bound GEMVs):
  - HyperConnection mix pair: created on meta, the checkpoint shard is
    quantized on CPU and the Parameter swapped at load (attach_rowwise_ingest).
  - lm_head: quantized and swapped at the model's post_load_weights, read
    through the fp8 branch in LogitsProcessor. Its scale rides the Parameter
    because the NEXTN draft shares the target lm_head object via
    set_embed_and_head and never revisits the target module.
  - Decode-size lm_head GEMVs before that swap: the lazy fp8 copy cached next
    to the bf16 original (maybe_sm120_fp8_lm_head).
The large projections are covered by the model-side hook in qwen4_exp.py
(MXFP8 from load time). Replacement is per-module, all-or-nothing, and final:
reader asserts fire if any code still expects a bf16 original. Kernels are
split-free: an atomic + fp32->bf16 epilogue costs more than the split-K gains
on these shapes.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

# Decode row counts (bs <= 8, verify <= 16, eager draft slack).
_MAX_M = 32

# Set by initialize_bf16_gemm_config from SGLANG_SM120_ONLINE_MXFP8.
online_fp8_enabled = False


@triton.jit
def _fp8w_gemv_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Weight-only FP8: w is float8_e4m3fn with per-output-row fp32 scales."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    n_mask = rn < N
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    k_end = tl.minimum(k_start + k_per, K)
    for k0 in tl.range(k_start, k_end, BLOCK_K, num_stages=NUM_STAGES):
        rk = k0 + tl.arange(0, BLOCK_K)
        k_mask = rk < k_end
        xb = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(xb, tl.trans(wb), out_dtype=tl.float32)
    scale = tl.load(scale_ptr + rn, mask=n_mask, other=0.0)
    acc = acc * scale[None, :]
    if OUT_BF16:
        tl.store(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
            sem="relaxed",
        )


# fp8 weight-only cache: weight key -> (fp8 weight, per-row fp32 scale).
# Populated during eager warmup, before CUDA graph capture. A capture-time miss falls back to the caller's
# bf16 path, so no allocation ever lands inside a graph.
_fp8w_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_fp8w_cache_bytes = 0
_FP8W_CACHE_MAX_BYTES = 6 * 1024**3


def _quantize_rowwise_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = w.shape[0]
    w_q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    # Chunk rows: a full-tensor fp32 temp of the 1.27 GB lm_head would need ~5 GB that a
    # 0.96-mem-fraction server does not have.
    step = max(1, (64 * 1024 * 1024) // max(1, w.shape[1] * 4))
    for i in range(0, n, step):
        blk = w[i : i + step].float()
        amax = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        s = amax / 448.0
        w_q[i : i + step] = (blk / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        scale[i : i + step] = s.squeeze(1)
    return w_q, scale


def _get_fp8_weight(weight: torch.Tensor):
    global _fp8w_cache_bytes
    if weight.dtype != torch.bfloat16:
        return None
    # data_ptr alone can be reused by the allocator after a free, and an in-place weight update
    # (copy_) keeps the ptr. Shape + _version make the key safe for both.
    key = (weight.data_ptr(), weight.shape[0], weight.shape[1], weight._version)
    hit = _fp8w_cache.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return None
    if _fp8w_cache_bytes + weight.numel() + weight.shape[0] * 4 > _FP8W_CACHE_MAX_BYTES:
        return None
    entry = _quantize_rowwise_fp8(weight)
    _fp8w_cache[key] = entry
    _fp8w_cache_bytes += entry[0].numel() + entry[1].numel() * 4
    return entry


def maybe_sm120_fp8_lm_head(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    FP8 weight-only lm_head GEMV for decode-size row counts, or None to let the caller keep
    the cuBLAS bf16 path (which reads 2x the bytes).

    Only reached while the bf16 original is still resident: after post_load_weights replaces
    the head, the fp8 branch in LogitsProcessor calls sm120_fp8_lm_head_logits directly.
    """
    if not online_fp8_enabled:
        return None
    if hidden_states.dim() != 2 or hidden_states.shape[0] > _MAX_M:
        return None
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    entry = _get_fp8_weight(weight)
    if entry is None:
        return None
    w_q, scale = entry
    m, k = hidden_states.shape
    n = w_q.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=hidden_states.device)
    _fp8w_gemv_kernel[(triton.cdiv(n, 32), 1)](
        hidden_states,
        w_q,
        scale,
        out,
        m,
        n,
        k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w_q.stride(0),
        w_q.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=32,
        BLOCK_K=128,
        BLOCK_M=max(16, triton.next_power_of_2(m)),
        SPLIT_K=1,
        NUM_STAGES=4,
        OUT_BF16=True,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Load-time replacement
# ---------------------------------------------------------------------------
# After replacement there is no bf16 fallback: the reader asserts fire instead
# of reading freed memory.

_SCALE_ATTR = "_sm120_rowwise_scale"


def rowwise_scale_of(weight: torch.Tensor):
    return getattr(weight, _SCALE_ATTR, None)


def dequant_rowwise_weight(
    weight: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """bf16 view of a replaced weight, for the prefill torch.compile paths."""
    scale = rowwise_scale_of(weight)
    assert scale is not None, "sm120 fp8 weight lost its rowwise scale"
    return (weight.to(torch.float32) * scale[:, None]).to(dtype)


_rowwise_load_stats = {"weights": 0, "bytes_avoided": 0}


def rowwise_mix_enabled(params_dtype: torch.dtype) -> bool:
    """True when HyperConnection mix projections should be born rowwise fp8.
    Also false when CUDA is unavailable."""
    return (
        online_fp8_enabled
        and params_dtype == torch.bfloat16
        and torch.cuda.is_available()
    )


def _rowwise_ingest_weight(module, param, loaded_weight, *args, **kwargs) -> None:
    """
    weight_loader for a meta-born rowwise fp8 weight, bound to the module at attach.
    The checkpoint shard is quantized on CPU with the same math as _quantize_rowwise_fp8 (the bf16-to-e4m3 cast is round-to-nearest-even on either host).

    Replaces the whole Parameter because set_data cannot change dtype in place.

    The draft head tie runs after load, so the new object is the one every reader sees.
    """
    assert param.shape == loaded_weight.shape, (
        f"sm120 online FP8: checkpoint shard {tuple(loaded_weight.shape)} does not "
        f"fit the rowwise meta weight {tuple(param.shape)}"
    )
    assert (
        loaded_weight.dtype == torch.bfloat16
    ), f"sm120 online FP8: rowwise ingest expects bf16, got {loaded_weight.dtype}"
    cpu_w = loaded_weight if loaded_weight.device.type == "cpu" else loaded_weight.cpu()
    w_q, scale = _quantize_rowwise_fp8(cpu_w)
    device = torch.cuda.current_device()
    new_param = torch.nn.Parameter(w_q.to(device), requires_grad=False)
    setattr(new_param, _SCALE_ATTR, scale.to(device))
    module.weight = new_param
    _rowwise_load_stats["weights"] += 1
    _rowwise_load_stats["bytes_avoided"] += (
        loaded_weight.numel() * loaded_weight.element_size()
    )


def attach_rowwise_ingest(linears) -> int:
    """Point each Linear's weight at the rowwise ingest loader.

    Requires weights created with device="meta" and rejects a materialized
    tensor, so a creation-site slip cannot leave a half-meta module behind.
    """
    attached = 0
    for lin in linears:
        weight = getattr(lin, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
            return 0
        if weight.device.type != "meta":
            raise RuntimeError(
                "sm120 online FP8: rowwise attach expects a meta-born weight, "
                f"got a {weight.device} tensor (creation site did not request rowwise)"
            )
        if weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"sm120 online FP8: rowwise attach expects bf16 dtype, got {weight.dtype}"
            )
        weight.weight_loader = functools.partial(_rowwise_ingest_weight, lin)
        attached += 1
    return attached


def replace_linears_with_fp8_copies(linears) -> int:
    """
    Swap each linear's bf16 weight for its rowwise fp8 quantization and return
    the freed bf16 bytes. All-or-nothing: any ineligible weight or an in-flight graph capture
    leaves every module untouched.
    """
    originals = []
    for lin in linears:
        weight = getattr(lin, "weight", None)
        if (
            not isinstance(weight, torch.nn.Parameter)
            or weight.dim() != 2
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
        ):
            return 0
        originals.append(weight.data)
    if torch.cuda.is_current_stream_capturing():
        return 0
    freed = 0
    for lin, original in zip(linears, originals):
        w_q, scale = _quantize_rowwise_fp8(original)
        param = torch.nn.Parameter(w_q, requires_grad=False)
        setattr(param, _SCALE_ATTR, scale)
        lin.weight = param
        freed += original.numel() * original.element_size()
    return freed


# The bf16 original is gone after replacement, so prefill-row logits dequantize
# row blocks at a time. The block caps the fp32 temp.
_DEQUANT_ROWS = 8192


def sm120_fp8_lm_head_logits(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Logits from a replaced (rowwise fp8) lm_head weight."""
    scale = rowwise_scale_of(weight)
    assert scale is not None, (
        "sm120 online FP8: fp8 lm_head weight without its rowwise scale "
        "(replacement attaches it to the Parameter)"
    )
    m, k = hidden_states.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=hidden_states.device)
    if m <= _MAX_M:
        _fp8w_gemv_kernel[(triton.cdiv(n, 32), 1)](
            hidden_states,
            weight,
            scale,
            out,
            m,
            n,
            k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=32,
            BLOCK_K=128,
            BLOCK_M=max(16, triton.next_power_of_2(m)),
            SPLIT_K=1,
            NUM_STAGES=4,
            OUT_BF16=True,
            num_warps=4,
        )
        return out
    block = max(1, min(_DEQUANT_ROWS, (64 * 1024 * 1024) // max(1, k * 4)))
    for i in range(0, n, block):
        w_bf = (
            weight[i : i + block].to(torch.float32) * scale[i : i + block, None]
        ).to(torch.bfloat16)
        out[:, i : i + w_bf.shape[0]] = hidden_states @ w_bf.t()
    return out

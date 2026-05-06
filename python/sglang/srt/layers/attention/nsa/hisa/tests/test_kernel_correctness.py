"""Correctness tests for the hierarchical indexer sub-kernels and the
end-to-end pipeline.

This file exists to pin the **current semantics** of every tilelang kernel in
``sglang/srt/layers/attention/nsa/hisa/custom_ops.py`` that the hierarchical
indexer depends on, so that future kernel optimizations must preserve the same
numerical output.

Layout
------
1. **PyTorch reference implementations** — plain, readable torch code that
   computes what each tilelang kernel claims to compute. These are the "spec".
2. **Per-kernel allclose tests** — run the tilelang kernel + the ref on the
   same inputs and assert that they match (within tolerance appropriate to
   the mixed bf16/fp8/fp32 math).
3. **End-to-end recall@k tests** — run the full hierarchical pipeline
   (``fp8_native_hierarchy_mqa_logits_tilelang_legacy`` / ``..._paged_mqa_logits``) alongside
   the DeepGEMM baseline (``fp8_mqa_logits`` / ``fp8_paged_mqa_logits``) and
   verify that the hierarchical top-k recovers ≥95% of the baseline top-k.

Usage
-----
The repo's conda env `new_vllm` loads a broken `libtorch_python.so` unless
`LD_LIBRARY_PATH` is cleared first::

    unset LD_LIBRARY_PATH && \\
    python /data/a_ccr/sglang/python/sglang/srt/layers/attention/nsa/hisa/tests/test_kernel_correctness.py

Runs all tests; exits non-zero if any fails. Uses pytest if available,
otherwise a tiny local runner.
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass

import torch

# Kernels under test (imported directly from source; any future edit will be
# picked up automatically).
from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    batch_pool_mqa_attn_return_logits_interface,
    pool_mqa_attn_return_logits_interface,
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    fp8_native_hierarchy_mqa_logits_tilelang_legacy,
    fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy,
    fp8_native_paged_block_sparse_mqa_attn_return_logits_interface,
    fp8_native_paged_mean_pooling_interface,
)

# Baseline (DeepGEMM) — used only in the end-to-end recall@k test.
import deep_gemm


DEVICE = torch.device("cuda")
torch.manual_seed(0)

# (k_block_size, block_topk) pairs — always k_block_size * block_topk == 8192.
BLOCK_CONFIGS = [(64, 128), (128, 64), (256, 32)]


# =============================================================================
# Numerical helpers
# =============================================================================

def _maxabsdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a-b| treating +inf-+inf and -inf--inf as zero."""
    m = torch.isfinite(a) & torch.isfinite(b)
    if not m.any():
        return 0.0
    return float((a[m] - b[m]).abs().max().item())


def _inf_pattern_matches(a: torch.Tensor, b: torch.Tensor) -> bool:
    """+inf/-inf positions must agree exactly (these are masks, not math)."""
    return bool(
        torch.equal(torch.isposinf(a), torch.isposinf(b))
        and torch.equal(torch.isneginf(a), torch.isneginf(b))
    )


def _assert_close_with_inf(
    got: torch.Tensor,
    ref: torch.Tensor,
    rtol: float,
    atol: float,
    name: str,
) -> None:
    """Assert got≈ref, where both may contain +inf/-inf at identical positions."""
    assert got.shape == ref.shape, f"{name}: shape mismatch {got.shape} vs {ref.shape}"
    assert _inf_pattern_matches(got, ref), (
        f"{name}: +inf/-inf mask pattern differs\n"
        f"  kernel +inf: {int(torch.isposinf(got).sum())}, ref +inf: {int(torch.isposinf(ref).sum())}\n"
        f"  kernel -inf: {int(torch.isneginf(got).sum())}, ref -inf: {int(torch.isneginf(ref).sum())}"
    )
    m = torch.isfinite(got) & torch.isfinite(ref)
    if not m.any():
        return
    diff = (got[m] - ref[m]).abs()
    denom = ref[m].abs().clamp(min=1.0)
    relerr = (diff / denom).max().item()
    abserr = diff.max().item()
    assert abserr <= atol or relerr <= rtol, (
        f"{name}: max abs diff = {abserr:.4g} (atol={atol}), "
        f"max rel diff = {relerr:.4g} (rtol={rtol})"
    )


# =============================================================================
# Synthetic input builders
# =============================================================================

@dataclass
class PrefillCase:
    M: int                    # query length (= kv length here)
    H: int = 64               # indexer heads
    D: int = 128              # head dim (= quant block size)
    k_block_size: int = 128
    block_topk: int = 64
    topk_tokens: int = 2048


@dataclass
class DecodeCase:
    B: int
    ctx_len: int              # all batches get same context length (simplest)
    H: int = 64
    D: int = 128
    paged_block_size: int = 64
    k_block_size: int = 128
    block_topk: int = 64
    topk_tokens: int = 2048


def _make_prefill_kv(case: PrefillCase, seed: int = 1):
    """Create the (k_fp8, k_scale) tensors in both layouts needed by the two
    consumers: tilelang wants [N,4] uint8 (viewed as float32 per token), the
    baseline mqa_logits wants [N] float32."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    N = case.M
    # FP8 values — use small-magnitude bf16 to avoid saturating fp8.
    k_bf16 = torch.randn(N, case.D, generator=g, device=DEVICE, dtype=torch.bfloat16)
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)
    # Per-token scales (positive).
    k_scale_f32 = (0.1 + 0.01 * torch.rand(N, generator=g, device=DEVICE, dtype=torch.float32)).contiguous()
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(N, 4)
    return k_fp8, k_scale_f32, k_scale_uint8


def _make_prefill_q(case: PrefillCase, seed: int = 2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_bf16 = torch.randn(case.M, case.H, case.D, generator=g, device=DEVICE, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    return q_fp8


def _make_prefill_weights(case: PrefillCase, seed: int = 3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return 0.1 * torch.randn(case.M, case.H, generator=g, device=DEVICE, dtype=torch.float32)


def _make_prefill_cu_seqlen(case: PrefillCase):
    # Full-causal: q attends to k[0..m]
    cu_ks = torch.zeros(case.M, device=DEVICE, dtype=torch.int32)
    cu_ke = (torch.arange(case.M, device=DEVICE, dtype=torch.int32) + 1)
    return cu_ks, cu_ke


def _make_decode_inputs(case: DecodeCase, seed: int = 1):
    """Return kv_cache_uint8, q_fp8, weights, context_lens, block_tables, num_sms.

    **Important layout note.** The vLLM API type for the paged KV cache is
    ``[num_blocks, block_size, 1, D+4] uint8``, and its docstring claims each
    row carries the per-token fp8 (first ``D`` bytes) + per-token fp32 scale
    (last 4 bytes). **That docstring is wrong** — both the DeepGEMM baseline
    and the hisa kernels actually interpret each block as a single flat blob:
    the first ``block_size * D`` bytes are **all** fp8 values (laid out as
    ``[block_size, D]`` row-major), and the last ``block_size * 4`` bytes are
    the ``block_size`` fp32 scales, one per token. Writing the cache with the
    "documented" per-token interleaved layout yields numerically-garbage
    logits. We therefore allocate the cache flat, populate it in the split
    layout, and only at the end reshape it to the declared 4D shape.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)

    max_blocks_per_seq = (case.ctx_len + case.paged_block_size - 1) // case.paged_block_size
    total_blocks = max_blocks_per_seq * case.B + 8  # slack

    # Allocate in the kernel's actual layout: per block, [all fp8 | all scales].
    D = case.D
    pbs = case.paged_block_size
    bytes_per_block = pbs * (D + 4)
    kv_cache_flat = torch.empty(total_blocks, bytes_per_block, device=DEVICE, dtype=torch.uint8)

    # FP8 part: [total_blocks, block_size, D]
    k_bf16 = torch.randn(total_blocks, pbs, D, generator=g, device=DEVICE, dtype=torch.bfloat16)
    k_fp8 = k_bf16.to(torch.float8_e4m3fn).contiguous()
    kv_cache_flat[:, : pbs * D] = k_fp8.view(torch.uint8).reshape(total_blocks, pbs * D)

    # Scale part: [total_blocks, block_size] fp32
    scales = (0.1 + 0.01 * torch.rand(total_blocks, pbs, generator=g, device=DEVICE,
                                      dtype=torch.float32)).contiguous()
    kv_cache_flat[:, pbs * D:] = scales.view(torch.uint8).reshape(total_blocks, pbs * 4)

    # Present the cache under the API-declared 4D shape. The underlying bytes
    # are already in the split layout the kernel expects.
    kv_cache = kv_cache_flat.view(total_blocks, pbs, 1, D + 4)

    q_bf16 = torch.randn(case.B, 1, case.H, case.D, generator=g,
                         device=DEVICE, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    weights = 0.1 * torch.randn(case.B, case.H, generator=g, device=DEVICE, dtype=torch.float32)

    context_lens = torch.full((case.B,), case.ctx_len, device=DEVICE, dtype=torch.int32)
    block_tables = torch.arange(max_blocks_per_seq * case.B, device=DEVICE, dtype=torch.int32)\
        .reshape(case.B, max_blocks_per_seq)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    return kv_cache, q_fp8, weights, context_lens, block_tables, num_sms


# =============================================================================
# PyTorch reference implementations
# =============================================================================

def ref_fp8_block_mean_pooling(
    k_fp8: torch.Tensor, k_scale_f32: torch.Tensor, k_block_size: int,
) -> torch.Tensor:
    """Ref for `fp8_native_block_mean_pooling_interface` — returns the f32 mean.

    Dequantize with per-token scale, group into pooling blocks of size
    `k_block_size`, mean pool (dividing by the actual number of valid tokens
    in the block — the last block may be shorter).  Returned in float32 so
    the caller can compare against fp8*scale re-quantization.
    """
    N, D = k_fp8.shape
    dequant_f32 = k_fp8.float() * k_scale_f32[:, None]  # [N, D]
    num_blocks = (N + k_block_size - 1) // k_block_size
    out = torch.empty(num_blocks, D, device=k_fp8.device, dtype=torch.float32)
    for b in range(num_blocks):
        s = b * k_block_size
        e = min(s + k_block_size, N)
        out[b] = dequant_f32[s:e].sum(dim=0) / (e - s)
    return out


def ref_pool_mqa(
    q_bf16: torch.Tensor,            # [M, H, D]
    blocked_k_bf16: torch.Tensor,    # [Nb, D]
    weights_bf16: torch.Tensor,      # [M, H]
    cu_ks_blocked: torch.Tensor,     # [M] int32
    cu_ke_blocked: torch.Tensor,     # [M] int32
    clean_logits: bool = True,
    force_maintain: bool = True,
) -> torch.Tensor:
    """Ref for `pool_mqa_attn_return_logits_interface`."""
    M, H, D = q_bf16.shape
    Nb = blocked_k_bf16.shape[0]
    # fp32 accumulation matches kernel (bf16 GEMM -> fp32 accum).
    q_f = q_bf16.float()
    k_f = blocked_k_bf16.float()
    w_f = weights_bf16.float()
    # score[m, n, h] = q[m, h] . k[n]
    s = torch.einsum("mhd,nd->mnh", q_f, k_f)       # [M, Nb, H]
    logits = (s.clamp(min=0) * w_f[:, None, :]).sum(dim=-1)   # [M, Nb]

    if clean_logits:
        n = torch.arange(Nb, device=q_bf16.device)[None, :]
        mask_out = (n < cu_ks_blocked.long()[:, None]) | (n >= cu_ke_blocked.long()[:, None])
        logits = logits.masked_fill(mask_out, float("-inf"))
    if force_maintain:
        m_idx = torch.arange(M, device=q_bf16.device)
        logits[m_idx, cu_ks_blocked.long()] = float("inf")
        logits[m_idx, (cu_ke_blocked - 1).clamp(min=0).long()] = float("inf")
    return logits


def ref_fp8_block_sparse_mqa(
    q_fp8: torch.Tensor,              # [M, H, D]
    k_fp8: torch.Tensor,              # [N, D]
    k_scale_f32: torch.Tensor,        # [N]
    topk_block_index: torch.Tensor,   # [M, topk] int32
    kv_block_size: int,
    weights_f32: torch.Tensor,        # [M, H]
    cu_seqlen_ks: torch.Tensor,       # [M] int32
    cu_seqlen_ke: torch.Tensor,       # [M] int32
) -> torch.Tensor:
    """Ref for `fp8_native_block_sparse_mqa_attn_return_logits_interface`."""
    M, H, D = q_fp8.shape
    N = k_fp8.shape[0]
    topk = topk_block_index.shape[1]

    # Absolute k index for every (m, t, i).
    block_starts = topk_block_index.long() * kv_block_size               # [M, topk]
    pos_in_block = torch.arange(kv_block_size, device=q_fp8.device)
    k_abs = block_starts[..., None] + pos_in_block[None, None, :]        # [M, topk, B]
    k_safe = k_abs.clamp(0, N - 1)                                        # in-range gather

    q_f = q_fp8.float()
    k_f = k_fp8.float() * k_scale_f32[:, None]                            # dequant once
    gathered_k = k_f[k_safe.flatten()].reshape(M, topk, kv_block_size, D)

    # score[m, t, i, h] = q[m, h] . k[k_abs]
    s = torch.einsum("mhd,mtid->mtih", q_f, gathered_k)                   # [M, topk, B, H]
    logits = (s.clamp(min=0) * weights_f32[:, None, None, :]).sum(dim=-1)  # [M, topk, B]

    # Mask: k_abs must be in [cu_ks, cu_ke) AND a valid k row.
    in_range = (
        (k_abs >= cu_seqlen_ks.long()[:, None, None])
        & (k_abs < cu_seqlen_ke.long()[:, None, None])
        & (k_abs < N)
    )
    logits = logits.masked_fill(~in_range, float("-inf"))

    return logits.reshape(M, topk * kv_block_size)


def ref_fp8_paged_mean_pooling(
    max_num_pooling_blocks: int,
    kv_cache_uint8: torch.Tensor,     # [num_blocks, block_size, 1, D+4]
    context_lens: torch.Tensor,       # [B] int32
    block_tables: torch.Tensor,       # [B, max_blocks] int32
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ref for `fp8_native_paged_mean_pooling_interface`.

    Returns (blocked_k [B, max_num_pooling_blocks, D] bf16, num_pooling_blocks [B] int32).
    """
    num_blocks, block_size, _, D_plus_4 = kv_cache_uint8.shape
    D = D_plus_4 - 4
    B = context_lens.shape[0]

    # Split cache -> fp8 values + fp32 scales, in the REAL layout: per block
    # the first block_size*D bytes are all fp8, the remaining block_size*4
    # bytes are the fp32 scales.
    flat_u8 = kv_cache_uint8.reshape(num_blocks, block_size * D_plus_4)
    fp8_vals = flat_u8[:, : block_size * D].contiguous()\
        .view(torch.float8_e4m3fn).reshape(num_blocks, block_size, D)
    scales = flat_u8[:, block_size * D:].contiguous()\
        .view(torch.float32).reshape(num_blocks, block_size)
    dequant = fp8_vals.float() * scales[:, :, None]                          # [B_, bs, D]

    out = torch.zeros(B, max_num_pooling_blocks, D, device=kv_cache_uint8.device, dtype=torch.float32)
    for b in range(B):
        seqlen = int(context_lens[b].item())
        nblocks = (seqlen + k_block_size - 1) // k_block_size
        for n in range(nblocks):
            s = n * k_block_size
            e = min(s + k_block_size, seqlen)
            cur = e - s
            if cur <= 0:
                continue
            # Collect all paged blocks that cover [s, e).
            paged_start = s // block_size
            paged_end = (e + block_size - 1) // block_size
            phys_ids = block_tables[b, paged_start:paged_end].long()
            tokens = dequant[phys_ids].reshape(-1, D)
            # Trim to exact [s, e) window.
            offset = s - paged_start * block_size
            tokens = tokens[offset:offset + cur]
            out[b, n] = tokens.sum(dim=0) / cur

    num_pooling_blocks = ((context_lens + k_block_size - 1) // k_block_size).to(torch.int32)
    return out.to(torch.bfloat16), num_pooling_blocks


def ref_batch_pool_mqa(
    q_bf16: torch.Tensor,          # [B, 1, H, D]
    blocked_kv_bf16: torch.Tensor, # [B, Nb, D]
    weights_bf16: torch.Tensor,    # [B, 1, H] or [B, H]
    context_lens: torch.Tensor,    # [B] int32
    clean_logits: bool = True,
    force_maintain: bool = True,
) -> torch.Tensor:
    """Ref for `batch_pool_mqa_attn_return_logits_interface`. Returns [B, 1, Nb]."""
    B, _, H, D = q_bf16.shape
    Nb = blocked_kv_bf16.shape[1]
    if weights_bf16.ndim == 3:
        w = weights_bf16.squeeze(1)  # [B, H]
    else:
        w = weights_bf16

    q_f = q_bf16.float().squeeze(1)     # [B, H, D]
    k_f = blocked_kv_bf16.float()       # [B, Nb, D]
    w_f = w.float()                     # [B, H]

    s = torch.einsum("bhd,bnd->bnh", q_f, k_f)             # [B, Nb, H]
    logits = (s.clamp(min=0) * w_f[:, None, :]).sum(dim=-1) # [B, Nb]

    # The kernel already writes -inf for positions >= context_lens[b] (see the
    # "else" branch), and clean_logits repeats that. So apply the same mask.
    if clean_logits:
        n = torch.arange(Nb, device=q_bf16.device)[None, :]
        mask_out = n >= context_lens.long()[:, None]
        logits = logits.masked_fill(mask_out, float("-inf"))
    if force_maintain:
        ctx = context_lens.long().clamp(min=0, max=Nb)
        last = (ctx - 1).clamp(min=0, max=Nb - 1)
        logits[:, 0] = float("inf")
        logits.scatter_(dim=1, index=last.unsqueeze(1), value=float("inf"))

    return logits.unsqueeze(1)  # [B, 1, Nb]


def ref_fp8_paged_block_sparse_mqa(
    q_fp8: torch.Tensor,           # [B, seq, H, D]
    kv_cache_uint8: torch.Tensor,  # [num_blocks, block_size, 1, D+4]
    topk_block_index: torch.Tensor, # [B, seq, topk] int32
    kv_block_size: int,
    weights_f32: torch.Tensor,     # [B, seq, H] or [B*seq, H]
    context_lens: torch.Tensor,    # [B] int32
    block_tables: torch.Tensor,    # [B, max_blocks] int32
) -> torch.Tensor:
    """Ref for `fp8_native_paged_block_sparse_mqa_attn_return_logits_interface`.
    Returns [B, seq, topk*kv_block_size] fp32."""
    B, seq, H, D = q_fp8.shape
    num_blocks, paged_block_size, _, D_plus_4 = kv_cache_uint8.shape
    topk = topk_block_index.shape[-1]
    if weights_f32.ndim == 2:
        weights_f32 = weights_f32.view(B, seq, H)
    max_blocks = block_tables.shape[1]

    # Dequantize the whole cache once. Real layout per block is
    # [all fp8 (block_size*D bytes) | all scales (block_size*4 bytes)].
    D_plus_4 = D + 4
    flat_u8 = kv_cache_uint8.reshape(num_blocks, paged_block_size * D_plus_4)
    fp8_vals = flat_u8[:, : paged_block_size * D].contiguous()\
        .view(torch.float8_e4m3fn).reshape(num_blocks, paged_block_size, D).float()
    scales = flat_u8[:, paged_block_size * D:].contiguous()\
        .view(torch.float32).reshape(num_blocks, paged_block_size)
    dequant = fp8_vals * scales[:, :, None]  # [num_blocks, paged_block_size, D]

    q_f = q_fp8.float()   # [B, seq, H, D]
    out = torch.full((B, seq, topk * kv_block_size),
                     float("-inf"), device=q_fp8.device, dtype=torch.float32)

    for b in range(B):
        ctx = int(context_lens[b].item())
        for s_i in range(seq):
            for t in range(topk):
                blk_id = int(topk_block_index[b, s_i, t].item())
                k_start = blk_id * kv_block_size
                # Gather kv_block_size tokens.
                for i in range(kv_block_size):
                    k_abs = k_start + i
                    p = k_abs // paged_block_size
                    if p < 0 or p >= max_blocks or k_abs < 0 or k_abs >= ctx:
                        continue
                    phys = int(block_tables[b, p].item())
                    pos_in = k_abs - p * paged_block_size
                    score = (q_f[b, s_i] * dequant[phys, pos_in][None, :]).sum(dim=-1)  # [H]
                    val = (score.clamp(min=0) * weights_f32[b, s_i]).sum()
                    out[b, s_i, t * kv_block_size + i] = val
    return out


# =============================================================================
# Per-kernel tests
# =============================================================================

def test_fp8_block_mean_pooling():
    case = PrefillCase(M=1024, k_block_size=128)
    k_fp8, k_scale_f32, _ = _make_prefill_kv(case)

    got_fp8, got_scale = fp8_native_block_mean_pooling_interface(k_fp8, k_scale_f32, case.k_block_size)
    got = got_fp8.float() * got_scale[:, None]
    ref = ref_fp8_block_mean_pooling(k_fp8, k_scale_f32, case.k_block_size)

    # fp8 re-quantization: ~1/256 rel error on top of bf16-level precision.
    _assert_close_with_inf(got, ref, rtol=5e-2, atol=5e-3,
                           name="fp8_block_mean_pooling")


def test_fp8_block_mean_pooling_ragged_tail():
    # Last block has 17/128 valid tokens — exercises the ragged-tail path.
    case = PrefillCase(M=128 * 3 + 17, k_block_size=128)
    k_fp8, k_scale_f32, _ = _make_prefill_kv(case)

    got_fp8, got_scale = fp8_native_block_mean_pooling_interface(k_fp8, k_scale_f32, case.k_block_size)
    got = got_fp8.float() * got_scale[:, None]
    ref = ref_fp8_block_mean_pooling(k_fp8, k_scale_f32, case.k_block_size)

    _assert_close_with_inf(got, ref, rtol=5e-2, atol=5e-3,
                           name="fp8_block_mean_pooling_ragged")


def test_pool_mqa():
    case = PrefillCase(M=512, k_block_size=128)
    k_fp8, k_scale_f32, _ = _make_prefill_kv(case)
    q_fp8 = _make_prefill_q(case)
    weights_f32 = _make_prefill_weights(case)
    cu_ks, cu_ke = _make_prefill_cu_seqlen(case)

    blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_interface(k_fp8, k_scale_f32, case.k_block_size)
    # Reconstruct bf16 blocked_k for the bf16 reference kernel comparison.
    blocked_k = (blocked_k_fp8.float() * blocked_k_scale[:, None]).to(torch.bfloat16)
    q_bf16 = q_fp8.float().bfloat16()
    w_bf16 = weights_f32.bfloat16()
    cu_ks_blk = cu_ks // case.k_block_size
    cu_ke_blk = (cu_ke + case.k_block_size - 1) // case.k_block_size

    got = pool_mqa_attn_return_logits_interface(
        q=q_bf16, blocked_kv=blocked_k, kv_block_size=case.k_block_size,
        weights=w_bf16, cu_seqlen_blocked_ks=cu_ks_blk, cu_seqlen_blocked_ke=cu_ke_blk,
    )
    ref = ref_pool_mqa(q_bf16, blocked_k, w_bf16, cu_ks_blk, cu_ke_blk)

    # bf16 gemm with fp32 accumulate + H=64 ReLU-weighted sum.
    _assert_close_with_inf(got, ref, rtol=5e-2, atol=5e-2, name="pool_mqa")


def test_fp8_block_sparse_mqa():
    for k_block_size, block_topk in BLOCK_CONFIGS:
        case = PrefillCase(M=1024, k_block_size=k_block_size, block_topk=block_topk)
        k_fp8, k_scale_f32, _ = _make_prefill_kv(case)
        q_fp8 = _make_prefill_q(case)
        weights_f32 = _make_prefill_weights(case)
        cu_ks, cu_ke = _make_prefill_cu_seqlen(case)

        num_k_blocks = (case.M + case.k_block_size - 1) // case.k_block_size
        g = torch.Generator(device="cuda").manual_seed(42)
        topk = min(case.block_topk, num_k_blocks)
        topk_block_index = torch.stack([
            torch.randperm(num_k_blocks, generator=g, device=DEVICE)[:topk]
            for _ in range(case.M)
        ]).to(torch.int64)

        got = fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q_fp8, k=k_fp8, k_scale=k_scale_f32,
            topk_block_index=topk_block_index, kv_block_size=case.k_block_size,
            weights=weights_f32, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )
        ref = ref_fp8_block_sparse_mqa(
            q_fp8, k_fp8, k_scale_f32, topk_block_index, case.k_block_size,
            weights_f32, cu_ks, cu_ke,
        )
        _assert_close_with_inf(got, ref, rtol=1e-1, atol=2e-1,
                               name=f"fp8_block_sparse_mqa[k={k_block_size}]")


def test_fp8_paged_mean_pooling():
    case = DecodeCase(B=2, ctx_len=512, paged_block_size=64, k_block_size=128)
    kv_cache, _, _, context_lens, block_tables, _ = _make_decode_inputs(case)

    max_num_pooling_blocks = (case.ctx_len + case.k_block_size - 1) // case.k_block_size

    got_k_fp8, got_scale, got_n = fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks, kv_cache, context_lens, block_tables, case.k_block_size,
    )
    got_k = got_k_fp8.float() * got_scale[:, :, None]
    ref_k, ref_n = ref_fp8_paged_mean_pooling(
        max_num_pooling_blocks, kv_cache, context_lens, block_tables, case.k_block_size,
    )

    assert torch.equal(got_n.to(torch.int32), ref_n.to(torch.int32)), \
        f"num_pooling_blocks mismatch: {got_n} vs {ref_n}"
    _assert_close_with_inf(got_k, ref_k.float(), rtol=5e-2, atol=5e-3,
                           name="fp8_paged_mean_pooling.blocked_k")


def test_fp8_paged_mean_pooling_ragged_tail():
    case = DecodeCase(B=3, ctx_len=128 * 2 + 33, paged_block_size=64, k_block_size=128)
    kv_cache, _, _, context_lens, block_tables, _ = _make_decode_inputs(case)
    max_num_pooling_blocks = (case.ctx_len + case.k_block_size - 1) // case.k_block_size

    got_k_fp8, got_scale, got_n = fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks, kv_cache, context_lens, block_tables, case.k_block_size,
    )
    got_k = got_k_fp8.float() * got_scale[:, :, None]
    ref_k, ref_n = ref_fp8_paged_mean_pooling(
        max_num_pooling_blocks, kv_cache, context_lens, block_tables, case.k_block_size,
    )

    assert torch.equal(got_n.to(torch.int32), ref_n.to(torch.int32))
    _assert_close_with_inf(got_k, ref_k.float(), rtol=5e-2, atol=5e-3,
                           name="fp8_paged_mean_pooling_ragged")


def test_batch_pool_mqa():
    case = DecodeCase(B=4, ctx_len=512, paged_block_size=64, k_block_size=128)
    kv_cache, q_fp8, weights_f32, context_lens, block_tables, _ = _make_decode_inputs(case)

    max_num_pooling_blocks = (case.ctx_len + case.k_block_size - 1) // case.k_block_size
    blocked_k_fp8, blocked_k_scale, num_pooling_blocks = fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks, kv_cache, context_lens, block_tables, case.k_block_size,
    )
    blocked_k = (blocked_k_fp8.float() * blocked_k_scale[:, :, None]).to(torch.bfloat16)
    q_bf16 = q_fp8.bfloat16()  # [B,1,H,D]
    w_bf16 = weights_f32.unsqueeze(1).bfloat16()  # [B,1,H]

    got = batch_pool_mqa_attn_return_logits_interface(
        q=q_bf16, blocked_kv=blocked_k, kv_block_size=case.k_block_size,
        weights=w_bf16, context_lens=num_pooling_blocks,
    )
    ref = ref_batch_pool_mqa(q_bf16, blocked_k, w_bf16, num_pooling_blocks)

    _assert_close_with_inf(got, ref, rtol=5e-2, atol=5e-2, name="batch_pool_mqa")


def test_fp8_paged_block_sparse_mqa():
    for k_block_size, block_topk in BLOCK_CONFIGS:
        case = DecodeCase(B=2, ctx_len=512, paged_block_size=64,
                          k_block_size=k_block_size, block_topk=block_topk)
        kv_cache, q_fp8, weights_f32, context_lens, block_tables, _ = _make_decode_inputs(case)

        num_k_blocks = (case.ctx_len + case.k_block_size - 1) // case.k_block_size
        g = torch.Generator(device="cuda").manual_seed(11)
        topk = min(case.block_topk, num_k_blocks)
        topk_block_index = torch.stack([
            torch.stack([
                torch.randperm(num_k_blocks, generator=g, device=DEVICE)[:topk]
                for _ in range(1)
            ])
            for _ in range(case.B)
        ]).to(torch.int64)    # [B, 1, topk] (torch.topk's native dtype)

        weights_bs = weights_f32.unsqueeze(1)  # [B, 1, H]

        got = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
            q_fp8=q_fp8, kv_cache_fp8=kv_cache, topk_block_index=topk_block_index,
            kv_block_size=case.k_block_size, weights=weights_bs,
            context_lens=context_lens, block_tables=block_tables,
        )
        ref = ref_fp8_paged_block_sparse_mqa(
            q_fp8, kv_cache, topk_block_index, case.k_block_size,
            weights_bs, context_lens, block_tables,
        )
        _assert_close_with_inf(got, ref, rtol=1e-1, atol=2e-1,
                               name=f"fp8_paged_block_sparse_mqa[k={k_block_size}]")


# =============================================================================
# End-to-end recall@k tests
# =============================================================================

def _recall_at_k(baseline_indices: torch.Tensor, hier_indices: torch.Tensor,
                 valid_mask: torch.Tensor | None = None) -> float:
    """Recall = mean over rows of |B ∩ H| / |B|, where B is the baseline top-k
    token index set and H is the hierarchy's top-k. If `valid_mask` is given
    (a [M] bool saying which rows have enough valid tokens to be meaningful),
    only those rows count.
    """
    assert baseline_indices.shape == hier_indices.shape
    rows = baseline_indices.shape[0]
    per_row_recall = []
    for r in range(rows):
        if valid_mask is not None and not bool(valid_mask[r]):
            continue
        b_set = set(int(x) for x in baseline_indices[r].tolist() if int(x) >= 0)
        h_set = set(int(x) for x in hier_indices[r].tolist() if int(x) >= 0)
        if len(b_set) == 0:
            continue
        per_row_recall.append(len(b_set & h_set) / len(b_set))
    return float(sum(per_row_recall) / max(1, len(per_row_recall)))


def test_e2e_prefill_recall():
    """Baseline dense top-k vs. hierarchy top-k on a causal prefill chunk."""
    for k_block_size, block_topk in BLOCK_CONFIGS:
        case = PrefillCase(M=4096, k_block_size=k_block_size, block_topk=block_topk, topk_tokens=2048)
        k_fp8, k_scale_f32, k_scale_uint8 = _make_prefill_kv(case)
        q_fp8 = _make_prefill_q(case)
        weights_f32 = _make_prefill_weights(case)
        cu_ks, cu_ke = _make_prefill_cu_seqlen(case)

        baseline_logits = deep_gemm.fp8_mqa_logits(
            q_fp8, (k_fp8, k_scale_f32), weights_f32, cu_ks, cu_ke, clean_logits=True,
        )
        k = min(case.topk_tokens, baseline_logits.shape[-1])
        baseline_topk = torch.topk(baseline_logits, k=k, dim=-1).indices

        block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
            q_fp8, (k_fp8, k_scale_uint8), weights_f32, cu_ks, cu_ke,
            case.k_block_size, case.block_topk,
        )
        k2 = min(case.topk_tokens, block_sparse_logits.shape[-1])
        relevant = torch.topk(block_sparse_logits, k=k2, dim=-1).indices
        abs_blocks = torch.gather(topk_block_indices.long(), dim=-1,
                                  index=(relevant // case.k_block_size))
        hier_topk = abs_blocks * case.k_block_size + (relevant % case.k_block_size)

        valid_mask = (cu_ke - cu_ks) >= case.topk_tokens
        recall = _recall_at_k(baseline_topk, hier_topk, valid_mask=valid_mask)
        print(f"    prefill [k={k_block_size:3d},topk={block_topk:3d}] "
              f"recall@{case.topk_tokens} = {recall:.4f}  "
              f"(eval on {int(valid_mask.sum())}/{case.M} rows)")
        assert recall >= 0.95, (
            f"prefill[k={k_block_size}] recall@{case.topk_tokens} = {recall:.4f} < 0.95")


def _make_decode_inputs_correlated(case: DecodeCase, block_scale: float = 1.0, seed: int = 1):
    """Build a paged KV-cache where K has block-structured correlation.

    Real attention K-caches exhibit temporal locality: tokens within the same
    semantic span share similar representations. We simulate this by adding a
    per-pooling-block "topic" vector to every token in that block. With a
    large-enough ``block_scale``, mean-pooling recovers the dominant topic, so
    the hierarchy can reliably pick the blocks most relevant to Q.

    Returns the same 6-tuple as ``_make_decode_inputs``.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    B, ctx_len = case.B, case.ctx_len
    pbs, D = case.paged_block_size, case.D

    max_blocks_per_seq = (ctx_len + pbs - 1) // pbs
    total_blocks = max_blocks_per_seq * B + 8

    # One "topic" per pooling block (k_block_size tokens share the same topic).
    num_topic_blocks = (ctx_len + case.k_block_size - 1) // case.k_block_size
    topics = torch.randn(num_topic_blocks, D, generator=g,
                         device=DEVICE, dtype=torch.bfloat16) * block_scale

    # Per-token noise added on top of the topic.
    k_noise = torch.randn(total_blocks, pbs, D, generator=g,
                          device=DEVICE, dtype=torch.bfloat16) * 0.3

    # Add topic: token t in batch b lives at physical block
    # b*max_blocks_per_seq + t//pbs, position t%pbs.  Vectorized: topic index
    # only depends on token position within the sequence (same across B).
    token_idx = torch.arange(ctx_len, device=DEVICE)
    topic_idx = (token_idx // case.k_block_size).view(max_blocks_per_seq, pbs)
    topic_add = topics[topic_idx]  # [max_blocks_per_seq, pbs, D]
    used = k_noise[: B * max_blocks_per_seq].view(B, max_blocks_per_seq, pbs, D)
    used += topic_add.unsqueeze(0)

    k_fp8 = k_noise.to(torch.float8_e4m3fn).contiguous()

    # Build cache in the split layout: all fp8 first, then all scales per block.
    kv_cache_flat = torch.empty(total_blocks, pbs * (D + 4), device=DEVICE, dtype=torch.uint8)
    kv_cache_flat[:, : pbs * D] = k_fp8.view(torch.uint8).reshape(total_blocks, pbs * D)
    scales = (0.1 + 0.01 * torch.rand(total_blocks, pbs, generator=g,
                                      device=DEVICE, dtype=torch.float32)).contiguous()
    kv_cache_flat[:, pbs * D:] = scales.view(torch.uint8).reshape(total_blocks, pbs * 4)
    kv_cache = kv_cache_flat.view(total_blocks, pbs, 1, D + 4)

    # Q: random direction + slight tilt toward a few sampled topics so it has
    # block-structure signal to exploit.
    q_bf16 = torch.randn(B, 1, case.H, D, generator=g,
                         device=DEVICE, dtype=torch.bfloat16) * 0.5
    for b in range(B):
        pick = torch.randint(0, num_topic_blocks, (10,), generator=g, device=DEVICE)
        q_bf16[b, 0] += topics[pick].sum(dim=0).unsqueeze(0) * 0.1
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    weights = 0.1 * torch.randn(B, case.H, generator=g,
                                device=DEVICE, dtype=torch.float32)
    context_lens = torch.full((B,), ctx_len, device=DEVICE, dtype=torch.int32)
    block_tables = torch.arange(max_blocks_per_seq * B, device=DEVICE,
                                dtype=torch.int32).reshape(B, max_blocks_per_seq)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    return kv_cache, q_fp8, weights, context_lens, block_tables, num_sms


def test_e2e_decode_recall():
    """Baseline paged dense top-k vs. hierarchy paged top-k.

    Uses block-correlated K so that mean-pooling has real signal to exploit
    (same as real attention locality). With uniformly-random K, mean-pooling
    has no discriminative power and recall is bounded by block coverage (~50%
    for block_topk=64/128 blocks), which is not a useful regression floor.
    """
    for k_block_size, block_topk in BLOCK_CONFIGS:
        case = DecodeCase(B=4, ctx_len=16384, paged_block_size=64,
                          k_block_size=k_block_size, block_topk=block_topk, topk_tokens=2048)
        kv_cache, q_fp8, weights_f32, context_lens, block_tables, num_sms = \
            _make_decode_inputs_correlated(case, block_scale=1.0)

        sched = deep_gemm.get_paged_mqa_logits_metadata(context_lens, case.paged_block_size, num_sms)
        baseline_logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8, kv_cache, weights_f32, context_lens, block_tables, sched,
            max_context_len=case.ctx_len, clean_logits=True,
        )
        k = min(case.topk_tokens, baseline_logits.shape[-1])
        baseline_topk = torch.topk(baseline_logits, k=k, dim=-1).indices

        block_sparse_logits, topk_block_indices = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
            q_fp8, kv_cache, weights_f32, context_lens, block_tables, sched,
            max_model_len=case.ctx_len, max_seq_len=case.ctx_len,
            k_block_size=case.k_block_size, block_topk=case.block_topk,
        )
        block_sparse_logits = block_sparse_logits.squeeze(1)
        topk_block_indices = topk_block_indices.squeeze(1)

        k2 = min(case.topk_tokens, block_sparse_logits.shape[-1])
        relevant = torch.topk(block_sparse_logits, k=k2, dim=-1).indices
        abs_blocks = torch.gather(topk_block_indices.long(), dim=-1,
                                  index=(relevant // case.k_block_size))
        hier_topk = abs_blocks * case.k_block_size + (relevant % case.k_block_size)

        valid_mask = context_lens >= case.topk_tokens
        recall = _recall_at_k(baseline_topk, hier_topk, valid_mask=valid_mask)
        print(f"    decode  [k={k_block_size:3d},topk={block_topk:3d}] "
              f"recall@{case.topk_tokens} = {recall:.4f}  "
              f"(eval on {int(valid_mask.sum())}/{case.B} rows, ctx_len={case.ctx_len})")
        assert recall >= 0.95, (
            f"decode[k={k_block_size}] recall@{case.topk_tokens} = {recall:.4f} < 0.95")


# =============================================================================
# Large-scale + determinism regression tests
# =============================================================================

def _make_prefill_inputs_correlated(case: PrefillCase, block_scale: float = 1.0, seed: int = 1):
    """Correlated-K prefill inputs.

    Mirrors ``_make_decode_inputs_correlated``: add a per-pooling-block topic
    vector to every token so mean-pooling has real signal.  Needed for the
    large-seq recall test where topk_tokens << seq_len.

    Returns the same tuple as ``_make_prefill_kv`` + q_fp8 + weights + cu_ks/ke.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    N, D = case.M, case.D

    num_topic_blocks = (N + case.k_block_size - 1) // case.k_block_size
    topics = torch.randn(num_topic_blocks, D, generator=g,
                         device=DEVICE, dtype=torch.bfloat16) * block_scale

    k_noise = torch.randn(N, D, generator=g, device=DEVICE, dtype=torch.bfloat16) * 0.3
    token_idx = torch.arange(N, device=DEVICE)
    topic_add = topics[token_idx // case.k_block_size]   # [N, D]
    k_bf16 = k_noise + topic_add
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)

    k_scale_f32 = (0.1 + 0.01 * torch.rand(N, generator=g, device=DEVICE, dtype=torch.float32)).contiguous()
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(N, 4)

    # Q slightly correlated with topics (mirror decode setup).
    q_bf16 = torch.randn(case.M, case.H, D, generator=g, device=DEVICE, dtype=torch.bfloat16) * 0.5
    pick = torch.randint(0, num_topic_blocks, (10,), generator=g, device=DEVICE)
    q_bf16 += topics[pick].sum(dim=0)[None, None, :] * 0.1
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    weights_f32 = 0.1 * torch.randn(case.M, case.H, generator=g, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(case.M, device=DEVICE, dtype=torch.int32)
    cu_ke = (torch.arange(case.M, device=DEVICE, dtype=torch.int32) + 1)
    return k_fp8, k_scale_f32, k_scale_uint8, q_fp8, weights_f32, cu_ks, cu_ke


def test_e2e_prefill_recall_large():
    """Production-scale prefill recall (seq=32768, k=128).

    Uses correlated K (topic-per-block) so mean-pooling has real signal,
    mirroring real-attention locality.  Recall must be >= 0.97 — a tighter
    floor than the default e2e test (0.95).
    """
    case = PrefillCase(M=32768, k_block_size=128, block_topk=64, topk_tokens=2048)
    k_fp8, k_scale_f32, k_scale_uint8, q_fp8, weights_f32, cu_ks, cu_ke = \
        _make_prefill_inputs_correlated(case, block_scale=1.0)

    baseline_logits = deep_gemm.fp8_mqa_logits(
        q_fp8, (k_fp8, k_scale_f32), weights_f32, cu_ks, cu_ke, clean_logits=True,
    )
    k = min(case.topk_tokens, baseline_logits.shape[-1])
    baseline_topk = torch.topk(baseline_logits, k=k, dim=-1).indices

    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        q_fp8, (k_fp8, k_scale_uint8), weights_f32, cu_ks, cu_ke,
        case.k_block_size, case.block_topk,
    )
    k2 = min(case.topk_tokens, block_sparse_logits.shape[-1])
    relevant = torch.topk(block_sparse_logits, k=k2, dim=-1).indices
    abs_blocks = torch.gather(topk_block_indices.long(), dim=-1,
                              index=(relevant // case.k_block_size))
    hier_topk = abs_blocks * case.k_block_size + (relevant % case.k_block_size)

    valid_mask = (cu_ke - cu_ks) >= case.topk_tokens
    recall = _recall_at_k(baseline_topk, hier_topk, valid_mask=valid_mask)
    print(f"    prefill_large[M={case.M},k=128] recall@{case.topk_tokens} = {recall:.4f}")
    assert recall >= 0.97, (
        f"prefill_large recall@{case.topk_tokens} = {recall:.4f} < 0.97")


def test_e2e_decode_recall_large():
    """Production-scale decode recall (B=32, ctx=32768, k=128)."""
    case = DecodeCase(B=32, ctx_len=32768, paged_block_size=64,
                      k_block_size=128, block_topk=64, topk_tokens=2048)
    kv_cache, q_fp8, weights_f32, context_lens, block_tables, num_sms = \
        _make_decode_inputs_correlated(case, block_scale=1.0)

    sched = deep_gemm.get_paged_mqa_logits_metadata(context_lens, case.paged_block_size, num_sms)
    baseline_logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8, kv_cache, weights_f32, context_lens, block_tables, sched,
        max_context_len=case.ctx_len, clean_logits=True,
    )
    k = min(case.topk_tokens, baseline_logits.shape[-1])
    baseline_topk = torch.topk(baseline_logits, k=k, dim=-1).indices

    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
        q_fp8, kv_cache, weights_f32, context_lens, block_tables, sched,
        max_model_len=case.ctx_len, max_seq_len=case.ctx_len,
        k_block_size=case.k_block_size, block_topk=case.block_topk,
    )
    block_sparse_logits = block_sparse_logits.squeeze(1)
    topk_block_indices = topk_block_indices.squeeze(1)

    k2 = min(case.topk_tokens, block_sparse_logits.shape[-1])
    relevant = torch.topk(block_sparse_logits, k=k2, dim=-1).indices
    abs_blocks = torch.gather(topk_block_indices.long(), dim=-1,
                              index=(relevant // case.k_block_size))
    hier_topk = abs_blocks * case.k_block_size + (relevant % case.k_block_size)

    valid_mask = context_lens >= case.topk_tokens
    recall = _recall_at_k(baseline_topk, hier_topk, valid_mask=valid_mask)
    print(f"    decode_large[B={case.B},ctx={case.ctx_len},k=128] "
          f"recall@{case.topk_tokens} = {recall:.4f}")
    assert recall >= 0.97, (
        f"decode_large recall@{case.topk_tokens} = {recall:.4f} < 0.97")


def test_determinism():
    """Same input across two runs -> bit-equal output (no race conditions)."""
    # --- Prefill: run hierarchy twice on identical inputs ---
    case = PrefillCase(M=2048, k_block_size=128, block_topk=64)
    k_fp8, k_scale_f32, k_scale_uint8 = _make_prefill_kv(case)
    q_fp8 = _make_prefill_q(case)
    weights_f32 = _make_prefill_weights(case)
    cu_ks, cu_ke = _make_prefill_cu_seqlen(case)

    logits1, idx1 = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        q_fp8, (k_fp8, k_scale_uint8), weights_f32, cu_ks, cu_ke,
        case.k_block_size, case.block_topk,
    )
    torch.cuda.synchronize()
    logits2, idx2 = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        q_fp8, (k_fp8, k_scale_uint8), weights_f32, cu_ks, cu_ke,
        case.k_block_size, case.block_topk,
    )
    torch.cuda.synchronize()
    # NaN-safe equality (any -inf masked positions should match exactly too).
    both_nan = torch.isnan(logits1) & torch.isnan(logits2)
    diff = (logits1 != logits2) & ~both_nan
    assert not diff.any(), (
        f"prefill logits not deterministic: {diff.sum().item()} / "
        f"{diff.numel()} elements differ (max abs diff: "
        f"{(logits1 - logits2).abs().nan_to_num(0).max().item()})")
    assert torch.equal(idx1, idx2), "prefill topk_block_indices not deterministic"

    # --- Decode: same check on paged hierarchy ---
    case_d = DecodeCase(B=4, ctx_len=4096, paged_block_size=64,
                        k_block_size=128, block_topk=64)
    kv_cache, q_fp8_d, weights_f32_d, context_lens, block_tables, num_sms = \
        _make_decode_inputs_correlated(case_d, block_scale=1.0)
    sched = deep_gemm.get_paged_mqa_logits_metadata(context_lens, case_d.paged_block_size, num_sms)

    logits1d, idx1d = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
        q_fp8_d, kv_cache, weights_f32_d, context_lens, block_tables, sched,
        max_model_len=case_d.ctx_len, max_seq_len=case_d.ctx_len,
        k_block_size=case_d.k_block_size, block_topk=case_d.block_topk,
    )
    torch.cuda.synchronize()
    logits2d, idx2d = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
        q_fp8_d, kv_cache, weights_f32_d, context_lens, block_tables, sched,
        max_model_len=case_d.ctx_len, max_seq_len=case_d.ctx_len,
        k_block_size=case_d.k_block_size, block_topk=case_d.block_topk,
    )
    torch.cuda.synchronize()
    both_nan_d = torch.isnan(logits1d) & torch.isnan(logits2d)
    diff_d = (logits1d != logits2d) & ~both_nan_d
    assert not diff_d.any(), (
        f"decode logits not deterministic: {diff_d.sum().item()} / "
        f"{diff_d.numel()} elements differ")
    assert torch.equal(idx1d, idx2d), "decode topk_block_indices not deterministic"


# =============================================================================
# Chunked prefill (mirrors vLLM's chunked prefill: Q=CHUNK_Q per iteration, K
# grows cumulatively) — verifies that the hierarchical indexer produces the
# same candidate-token set whether prefill runs in one shot or is split into
# chunks that feed growing K.
# =============================================================================

def _tokens_from_blocks(topk: torch.Tensor, k_block_size: int) -> torch.Tensor:
    """Given [M, block_topk] pool-block indices, expand to the sorted set of
    [M, block_topk * k_block_size] candidate token indices. Sorted along the
    last dim so that top-k permutations (from sorted=False in torch.topk)
    don't cause spurious mismatches."""
    base = topk.long().unsqueeze(-1) * k_block_size                  # [M, topk, 1]
    offs = torch.arange(k_block_size, device=topk.device).long()     # [kb]
    tokens = (base + offs).reshape(topk.shape[0], -1)                # [M, topk*kb]
    return tokens.sort(dim=-1).values


def test_chunked_prefill_consistency():
    """Run hisa prefill in two modes and compare the candidate-token set:

      * one-shot: single call with Q = total_seq_len, K = full context
      * chunked:  CHUNK_Q queries per iteration, K grows cumulatively to the
                  end of the current chunk (mirrors vLLM's chunked prefill)

    Both modes must pick the *same* set of candidate tokens for every query,
    because (a) pool blocks are fully aligned to k_block_size at chunk
    boundaries (so fp8 pool values are deterministic across modes), and
    (b) per-query visible K is identical in both modes.
    """
    CHUNK_Q = 8192
    NUM_CHUNKS = 2
    TOTAL = CHUNK_Q * NUM_CHUNKS
    case = PrefillCase(M=TOTAL, k_block_size=128, block_topk=64)
    k_fp8, k_scale_f32, k_scale_uint8 = _make_prefill_kv(case)
    q_fp8 = _make_prefill_q(case)
    weights_f32 = _make_prefill_weights(case)
    cu_ks, cu_ke = _make_prefill_cu_seqlen(case)  # single-seq causal

    # One-shot.
    _, ref_topk = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        q_fp8, (k_fp8, k_scale_uint8), weights_f32,
        cu_ks, cu_ke, case.k_block_size, case.block_topk,
    )
    ref_tokens = _tokens_from_blocks(ref_topk, case.k_block_size)

    # Chunked: each iter has Q=CHUNK_Q, K grows by CHUNK_Q.
    got_topk_list = []
    for i in range(NUM_CHUNKS):
        qs, qe = i * CHUNK_Q, (i + 1) * CHUNK_Q
        k_end = qe  # K grows to end of current chunk
        _, topk_c = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
            q_fp8[qs:qe],
            (k_fp8[:k_end], k_scale_uint8[:k_end]),
            weights_f32[qs:qe],
            cu_ks[qs:qe], cu_ke[qs:qe],
            case.k_block_size, case.block_topk,
        )
        got_topk_list.append(topk_c)
    got_topk = torch.cat(got_topk_list, dim=0)
    got_tokens = _tokens_from_blocks(got_topk, case.k_block_size)

    assert got_tokens.shape == ref_tokens.shape, (
        f"shape mismatch: chunked {got_tokens.shape} vs one-shot {ref_tokens.shape}"
    )
    # Allow a small fraction of per-query rows to differ (topk tie-breaking
    # between runs can shuffle blocks with near-identical logits). Require
    # most rows to match exactly.
    row_match = (got_tokens == ref_tokens).all(dim=-1)
    frac = row_match.float().mean().item()
    print(f"    chunked_prefill_consistency: row-exact match = {frac:.4f}  "
          f"({int(row_match.sum())}/{TOTAL} queries)")
    assert frac >= 0.95, (
        f"chunked vs one-shot candidate-token sets differ on {TOTAL - int(row_match.sum())} "
        f"/{TOTAL} queries (row-exact frac={frac:.4f})"
    )


# =============================================================================
# Test runner
# =============================================================================

ALL_TESTS = [
    ("fp8_block_mean_pooling",           test_fp8_block_mean_pooling),
    ("fp8_block_mean_pooling_ragged",    test_fp8_block_mean_pooling_ragged_tail),
    ("pool_mqa",                         test_pool_mqa),
    ("fp8_block_sparse_mqa",             test_fp8_block_sparse_mqa),
    ("fp8_paged_mean_pooling",           test_fp8_paged_mean_pooling),
    ("fp8_paged_mean_pooling_ragged",    test_fp8_paged_mean_pooling_ragged_tail),
    ("batch_pool_mqa",                   test_batch_pool_mqa),
    ("fp8_paged_block_sparse_mqa",       test_fp8_paged_block_sparse_mqa),
    ("chunked_prefill_consistency",      test_chunked_prefill_consistency),
    ("e2e_prefill_recall",                test_e2e_prefill_recall),
    ("e2e_decode_recall",                 test_e2e_decode_recall),
    ("e2e_prefill_recall_large",          test_e2e_prefill_recall_large),
    ("e2e_decode_recall_large",           test_e2e_decode_recall_large),
    ("determinism",                       test_determinism),
]


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.init()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_pass = n_fail = 0
    for name, fn in ALL_TESTS:
        try:
            print(f"[RUN ] {name}")
            fn()
            print(f"[PASS] {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"[FAIL] {name}:\n  {e}")
            n_fail += 1
        except Exception as e:
            print(f"[ERR ] {name}: {e}")
            traceback.print_exc()
            n_fail += 1
        torch.cuda.synchronize()

    print(f"\n{n_pass} passed, {n_fail} failed (of {len(ALL_TESTS)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

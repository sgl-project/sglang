# SPDX-License-Identifier: Apache-2.0
"""Pure-torch fallback for the DSA indexer's paged-MQA-logits kernel.

`sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged` scores
paged KV against the query via a paged-MQA-logits kernel before top-k
selection. The existing backends (DeepGEMM, CuTe DSL, aiter) each have a hard
arch/platform requirement: DeepGEMM's compiled kernel does not cover SM120/121
(consumer Blackwell; upstream DeepGEMM declined support, see
https://github.com/deepseek-ai/DeepGEMM/pull/318), CuTe DSL is gated to SM100
and its `_setup_mma` uses `tcgen05.MmaF8F6F4Op`, a datacenter-Blackwell-only
tensor-core op that does not exist on SM120/121, and aiter is ROCm-only. That
leaves every CUDA arch outside SM90/SM100/ROCm with no working backend at all.

This module is a vectorized (no `.item()`, no data-dependent control flow, so
CUDA-graph-capture-safe) torch implementation with no arch requirement,
selected via `--dsa-paged-mqa-logits-backend torch` (opt-in only; not chosen
by "auto", so archs where DeepGEMM/CuTe DSL already work are unaffected).

It follows the same layout and math as the sibling SM120 fallback added for
the DeepSeek-V4 indexer in `sglang.srt.layers.attention.dsv4.indexer` (see
`fp8_paged_mqa_logits_torch_sm120`, PR #24692) and discards the DeepGEMM
schedule metadata the same way (`deep_gemm_metadata` is accepted for call-site
signature compatibility only) -- the torch path does no SM-tiled scheduling,
so callers may pass `None`. It accumulates the KV x Q dot product in float32
rather than bfloat16; this is a deliberate accuracy/robustness choice for a
fallback path that has no faster alternative to cross-check against on its
target hardware, not a numerical-equivalence claim against the sibling
kernel.

Includes an optional fused Triton fast path (`triton_paged_mqa_logits.py`),
dispatched below after the shape asserts. Under CUDA graph capture, the
scan width is fixed at the full page-table capacity regardless of the
request's true sequence length; this pure-torch kernel pays that full width
on every step (gather + dequant + matmul over the whole capture width), while
the Triton kernel early-exits per (request, KV-page) block on the true
`seq_lens` value read at replay time. See `SGLANG_DSA_INDEXER_TRITON`
(`sglang.srt.environ`) to control it; it is bit-exact with this module's pure
implementation and is on by default.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

try:
    from sglang.srt.layers.attention.dsa.triton_paged_mqa_logits import (
        fp8_paged_mqa_logits_triton_dsa,
    )

    _TRITON_LOGITS_AVAILABLE = True
except ImportError:  # pragma: no cover - triton is a standard SGLang CUDA dep
    fp8_paged_mqa_logits_triton_dsa = None
    _TRITON_LOGITS_AVAILABLE = False
    logger.warning(
        "DSA indexer: failed to import the Triton paged-MQA-logits fast path; "
        "falling back to the pure-torch kernel (dsa_paged_mqa_logits_backend="
        "'torch' still works, just slower under CUDA graph capture)."
    )


def fp8_paged_mqa_logits_torch_dsa(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """CUDA-graph-compatible FP8 paged MQA logits, pure torch (no DeepGEMM/CuTe DSL).

    `deep_gemm_metadata` is accepted for call-site signature compatibility
    with the DeepGEMM/CuTe DSL backends but unused: this path does no SM-tiled
    scheduling, so it has no notion of a schedule to consume. Callers may pass
    `None`.

    Dispatches to the fused Triton kernel (`SGLANG_DSA_INDEXER_TRITON`,
    default on) right after the shape asserts below, so every layout
    guarantee this function checks also holds for the Triton path.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    assert (
        head_dim == 128
    ), f"torch paged-MQA-logits fallback hardcodes DSA indexer head_dim=128, got {head_dim}"
    assert block_size == 64, (
        "torch paged-MQA-logits fallback hardcodes the DSA cache page layout "
        f"(block_size=64), got {block_size}"
    )
    assert q_fp8.shape == (
        batch_size,
        1,
        num_heads,
        head_dim,
    ), f"expected q_fp8 shape {(batch_size, 1, num_heads, head_dim)}, got {q_fp8.shape}"
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4), (
        f"expected kvcache_fp8 trailing shape {(block_size, 1, head_dim + 4)}, "
        f"got {kvcache_fp8.shape[1:]}"
    )
    assert weight.shape == (
        batch_size,
        num_heads,
    ), f"expected weight shape {(batch_size, num_heads)}, got {weight.shape}"
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (
        batch_size,
    ), f"expected seq_lens shape {(batch_size,)} after squeeze, got {seq_lens.shape}"
    assert (
        page_table.shape[0] == batch_size
    ), f"expected page_table batch dim {batch_size}, got {page_table.shape[0]}"
    assert (
        clean_logits is False
    ), "clean_logits=True is not supported; cleaning happens in topk_transform"

    if (
        _TRITON_LOGITS_AVAILABLE
        and envs.SGLANG_DSA_INDEXER_TRITON.get()
        and num_heads >= 16  # tl.dot's minimum inner dimension
    ):
        return fp8_paged_mqa_logits_triton_dsa(
            q_fp8, kvcache_fp8, weight, seq_lens, page_table, None, max_seq_len
        )

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_padded_seq = max_pages * block_size

    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    SCALE_OFFSET = block_size * head_dim

    # page_table entries beyond a request's allocated page count are, in
    # SGLang's current DSA req_to_token pool, always zero-initialized (never
    # negative). This kernel nonetheless clamps defensively to a valid
    # non-negative page id before gathering: the codebase's established
    # convention for paged-gather kernels elsewhere (e.g. the sibling SM120
    # FlashMLA decode kernel added by the same precedent PR) IS to represent
    # "no page here" as -1 and clamp before use. A negative page id here would
    # otherwise silently wrap around to a valid-looking but wrong row via
    # Python/torch negative indexing rather than raising -- clamping makes
    # that failure mode inert. Whatever garbage is gathered for a clamped
    # (originally invalid) page is masked out below via `invalid_mask`, so
    # the clamp never affects correctness of valid output positions.
    page_ids = page_table[:, :max_pages].clamp(min=0)
    kvcache_gathered = kvcache_flat[page_ids]

    kv_value_raw = kvcache_gathered[..., :SCALE_OFFSET]
    kv_scale_raw = kvcache_gathered[..., SCALE_OFFSET:]

    kv_value = kv_value_raw.contiguous().view(dtype=FP8_DTYPE).to(torch.float32)
    kv_value = kv_value.view(batch_size, max_padded_seq, head_dim)

    kv_scale = kv_scale_raw.contiguous().view(dtype=torch.float32)
    kv_scale = kv_scale.view(batch_size, max_padded_seq)

    q = q_fp8[:, 0].to(torch.float32)

    score = torch.bmm(kv_value, q.transpose(1, 2))

    score = F.relu(score)
    score = score * weight.unsqueeze(1)
    score = score.sum(dim=2)

    score = score * kv_scale

    out_width = min(max_padded_seq, max_seq_len)
    logits = score.new_full((batch_size, max_seq_len), float("-inf"))
    logits[:, :out_width] = score[:, :out_width]

    positions = torch.arange(max_seq_len, device=device)
    invalid_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    logits.masked_fill_(invalid_mask, float("-inf"))

    return logits

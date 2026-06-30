# Copyright 2025-2026 SGLang Team
"""Torch reference + backend selection for the *ragged* FP8 MQA logits used by
the DSA indexer prefill path (the ``deep_gemm.fp8_mqa_logits`` kernel).

On SM120 (Blackwell desktop / RTX PRO 6000) ``deep_gemm`` is unavailable
(no tcgen05/TMEM), so the ragged prefill index logits fall back to the pure-torch
reference below. It is correctness-first (fp32, not fused) and matches the DSA
"lightning indexer" scoring used by the proven DeepSeek-V4 paged torch reference
(``dsv4/indexer.py``):

    logit(i, j) = ( sum_h weights[i, h] * ReLU( q[i, h] . k[j] ) ) * k_scale[j]

where ``q_fp8`` is unit-scale fp8 (its per-token quant scale, the softmax scale,
and the per-head gate are all folded into ``weights`` upstream), and ``k_scale``
is the per-key fp8 dequant scale.

The function signature matches ``deep_gemm.fp8_mqa_logits`` so it is a drop-in
replacement. With ``clean_logits=False`` (the path used by the indexer) no masking
is applied here; the downstream ``topk_transform`` masks invalid positions via
``ks`` / lengths, exactly as it does for the deep_gemm output.
"""

from typing import Callable, Tuple

import torch

from sglang.srt.utils import is_sm120_supported


def fp8_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kv_fp8: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Pure-torch reference for ``deep_gemm.fp8_mqa_logits``.

    Args:
        q_fp8: ``[num_q, num_heads, head_dim]`` fp8 query (unit scale).
        kv_fp8: ``(k_fp8 [num_k, head_dim] fp8, k_scale [num_k] fp32)``.
        weights: ``[num_q, num_heads]`` fp32 per-(token, head) gates (q-scale and
            softmax-scale folded in).
        ks, ke: ``[num_q]`` int32 causal key range ``[ks, ke)`` per query token.
        clean_logits: if True, positions outside ``[ks, ke)`` are set to ``-inf``.
            The indexer calls with False and relies on downstream masking.

    Returns:
        ``[num_q, num_k]`` fp32 logits.
    """
    k_fp8, k_scale = kv_fp8
    q = q_fp8.to(torch.float32)  # [Q, H, D]
    k = k_fp8.to(torch.float32)  # [K, D]
    w = weights.to(torch.float32)  # [Q, H]
    num_heads = q.shape[1]

    # Per-head ReLU(q . k), weighted and summed over heads. Looping over heads
    # avoids materializing the [Q, K, H] score tensor.
    logits = q.new_zeros((q.shape[0], k.shape[0]), dtype=torch.float32)
    for h in range(num_heads):
        score_h = torch.matmul(q[:, h, :], k.t())  # [Q, K] raw fp8 dot
        score_h = torch.relu(score_h)
        logits += score_h * w[:, h].unsqueeze(1)

    # Per-key fp8 dequant scale (k_scale > 0, so applying it after ReLU is exact).
    logits = logits * k_scale.to(torch.float32).unsqueeze(0)  # [Q, K]

    if clean_logits:
        num_k = logits.shape[1]
        pos = torch.arange(num_k, device=logits.device).unsqueeze(0)
        valid = (pos >= ks.unsqueeze(1)) & (pos < ke.unsqueeze(1))
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def select_fp8_mqa_logits_fn() -> Tuple[Callable, str]:
    """Return ``(fn, backend_name)`` for the ragged FP8 MQA logits.

    ``backend_name`` is ``"torch"`` on SM120 (deep_gemm unavailable) and
    ``"deep_gemm"`` otherwise. Both share ``deep_gemm.fp8_mqa_logits``'s
    signature, so callers can substitute directly.
    """
    if is_sm120_supported():
        return fp8_mqa_logits_torch, "torch"

    import deep_gemm

    return deep_gemm.fp8_mqa_logits, "deep_gemm"

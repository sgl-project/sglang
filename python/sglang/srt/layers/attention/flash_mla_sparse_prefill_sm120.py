"""SM120 FlashMLA sparse *prefill* dispatch.

Stock ``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` is SM90a/SM100f-only and raises on
SM120; that path is hit when a prefill batches > ``_LARGE_INDEXER_QUERY_THRESHOLD`` (=11673)
query tokens. Mirrors ``flash_mla_sm120.py`` for the prefill op, selected by
``SGLANG_SM120_SPARSE_PREFILL``:

- ``hmma``      -- custom SM120 sparse-prefill kernel (``deepseek_v4_kernel.ops.sparse_prefill_fwd``),
                   a drop-in for ``flash_mla_sparse_fwd``.
- ``sglkernel`` -- stock kernel (SM90a/SM100f; crashes on SM120).
- ``torch``     -- chunked pure-PyTorch reference (correct, slow).

Default is *safe*: SM120 → ``hmma`` if installed else ``torch`` (never ``sglkernel``); off
SM120 → ``sglkernel``. An explicit ``sglkernel`` on SM120 is downgraded to ``torch`` so the
server never crashes on the > 11673-token path.

Contract: ``flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink, topk_length)
-> (out, max_logits, lse)``. Callers use only ``out`` (``o, _, _ = ...``).
"""

import logging
import os

import torch

from sglang.srt.utils.common import (
    is_deepseek_v4_kernel_available,
    is_sm120_supported,
)

logger = logging.getLogger(__name__)

_is_sm120 = is_sm120_supported()


def _resolve_sm120_sparse_prefill_backend() -> str:
    """Resolve ``SGLANG_SM120_SPARSE_PREFILL`` once at import.

    Unset -> safe per-arch default (``hmma``/``torch`` on SM120, ``sglkernel`` elsewhere).
    An explicit value that can't run on this device is downgraded — never to one that crashes.
    """
    raw = os.environ.get("SGLANG_SM120_SPARSE_PREFILL")
    if raw is None:
        if not _is_sm120:
            return "sglkernel"
        if is_deepseek_v4_kernel_available():
            return "hmma"
        logger.info(
            "SM120 sparse-prefill: deepseek_v4_kernel not installed and stock "
            "sgl_kernel is SM90a/SM100f-only; using the pure-PyTorch reference. "
            "Install deepseek_v4_kernel or set SGLANG_SM120_SPARSE_PREFILL=hmma "
            "for the custom kernel."
        )
        return "torch"

    backend = raw.lower()
    if backend not in ("hmma", "sglkernel", "torch"):
        raise ValueError(
            "SGLANG_SM120_SPARSE_PREFILL must be 'hmma', 'sglkernel', or "
            f"'torch' (got {raw!r})."
        )
    if backend == "hmma" and not is_deepseek_v4_kernel_available():
        fallback = "torch" if _is_sm120 else "sglkernel"
        logger.info(
            "SGLANG_SM120_SPARSE_PREFILL=hmma but deepseek_v4_kernel is not "
            "installed; using %r.",
            fallback,
        )
        return fallback
    if backend == "sglkernel" and _is_sm120:
        logger.warning(
            "SGLANG_SM120_SPARSE_PREFILL=sglkernel on SM120, but the stock "
            "sparse-prefill kernel only supports SM90a/SM100f and would crash; "
            "using the pure-PyTorch reference instead."
        )
        return "torch"
    return backend


_sm120_sparse_prefill_backend = _resolve_sm120_sparse_prefill_backend()


def _normalize_kv_indices(kv: torch.Tensor, indices: torch.Tensor):
    """Accept the [s_kv,1,d] / [s_q,1,topk] (h_kv=1) shapes the callers pass and
    the flat [s_kv,d] / [s_q,topk] shapes, returning the flat forms."""
    if kv.dim() == 3:
        # [s_kv, h_kv=1, d_qk] -> [s_kv, d_qk]
        kv = kv.squeeze(1)
    if indices.dim() == 3:
        # [s_q, h_kv=1, topk] -> [s_q, topk]
        indices = indices.squeeze(1)
    return kv, indices


def _torch_sparse_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink=None,
    topk_length=None,
    row_chunk: int = 1024,
):
    """Chunked pure-PyTorch reference mirroring DeepSeek's ``ref_sparse_attn_fwd``.

    Chunked over the query dimension so the dense ``[chunk,topk,d_qk]`` gather
    stays bounded (a single-shot gather at s_q=16384, topk=2048 is ~64 GiB).
    Natural-log ``max_logits`` / ``lse``; lonely query -> out 0 / mx -inf /
    lse +inf, matching the HMMA kernel and the stock op.
    """
    kv, indices = _normalize_kv_indices(kv, indices)
    s_q, h_q, d_qk = q.shape
    s_kv = kv.size(0)
    topk = indices.size(-1)
    dev = q.device

    out = torch.empty(s_q, h_q, d_v, dtype=torch.bfloat16, device=dev)
    max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=dev)
    lse = torch.empty(s_q, h_q, dtype=torch.float32, device=dev)

    sink = attn_sink.float() if attn_sink is not None else None
    ar_topk = torch.arange(topk, device=dev)
    kv_f = kv.float()

    for start in range(0, s_q, row_chunk):
        end = min(start + row_chunk, s_q)
        n = end - start
        idx = indices[start:end].clone()
        if topk_length is not None:
            len_mask = ar_topk.unsqueeze(0).broadcast_to(n, topk) >= topk_length[
                start:end
            ].unsqueeze(1)
            idx[len_mask] = -1
        invalid = (idx < 0) | (idx >= s_kv)  # [n, topk]
        idx_safe = torch.where(invalid, torch.zeros_like(idx), idx)

        gathered = kv_f.index_select(0, idx_safe.flatten().long()).reshape(
            n, topk, d_qk
        )
        qf = q[start:end].float()
        P = (qf @ gathered.transpose(1, 2)) * sm_scale  # [n, h_q, topk]
        P.masked_fill_(invalid.unsqueeze(1), float("-inf"))

        orig_lse = torch.logsumexp(P, dim=-1)  # [n, h_q]
        mx = P.max(dim=-1).values  # [n, h_q]

        if sink is not None:
            lse_for_o = torch.logsumexp(
                torch.stack([orig_lse, sink.broadcast_to(n, h_q)], dim=0),
                dim=0,
            )
        else:
            lse_for_o = orig_lse.clone()
        lse_for_o[lse_for_o == float("-inf")] = float("+inf")  # -> O row 0
        s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
        o = s_for_o @ gathered[..., :d_v]  # [n, h_q, d_v]

        lonely = orig_lse == float("-inf")
        orig_lse[lonely] = float("+inf")

        out[start:end] = o.to(torch.bfloat16)
        max_logits[start:end] = mx
        lse[start:end] = orig_lse

    return out, max_logits, lse


def flash_mla_sparse_fwd_sm120(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink=None,
    topk_length=None,
):
    """SM120 sparse-prefill entry point; dispatches to the resolved backend.

    Drop-in for ``sgl_kernel.flash_mla.flash_mla_sparse_fwd``. Returns
    ``(out, max_logits, lse)``.
    """
    if _sm120_sparse_prefill_backend == "hmma":
        from deepseek_v4_kernel.ops import sparse_prefill_fwd

        return sparse_prefill_fwd(
            q,
            kv,
            indices,
            float(sm_scale),
            int(d_v),
            attn_sink,
            topk_length,
        )

    if _sm120_sparse_prefill_backend == "sglkernel":
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        return flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

    return _torch_sparse_prefill_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

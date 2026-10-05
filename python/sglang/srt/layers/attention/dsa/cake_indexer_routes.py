"""``dsa_indexer`` Cake route helpers for the DeepSeek-V3.2 lightning indexer.

``SGLANG_CAKE_ROUTES=dsa_indexer`` lets the two DeepGEMM call sites of
``dsa_indexer.Indexer`` take the Cake FlashInfer entries that mirror the
DeepGEMM signatures (``sglang.kernels.ops.attention.cake``):

* ragged prefill (``_get_topk_ragged``): ``deep_gemm.fp8_mqa_logits(q, (kv,
  kv_scales), weights, ks, ke, clean_logits=False)`` ->
  ``flashinfer.dense_mqa.fp8_mqa_logits`` on the same tensors;
* paged decode / target-verify / draft-extend-v2 (``_get_topk_paged``,
  inside the engine's ``_chunked_fp8_paged_mqa_logits``):
  ``deep_gemm.fp8_paged_mqa_logits(q, kv_cache, weights, context_lens,
  block_table, schedule_meta, max_len, clean_logits=False)`` ->
  ``flashinfer.paged_mqa.fp8_paged_mqa_logits`` on the same tensors with
  ``schedule_meta=None``: the Cake paged program derives its schedule
  in-kernel, so the route is ONE launch and the engine's DeepGEMM schedule
  buffer (and FlashInfer's placeholder ``get_paged_mqa_logits_metadata``) is
  not needed on the Cake path.

The returned logits keep the engine's contract (f32, ``[Q, K]`` /
``[B * next_n, max_len]`` views with unit column stride and a row stride that
is a multiple of 4), so ``_mask_init_and_local_tokens`` and ``topk_transform``
run unchanged. Chunking, ``q_offset`` handling and padding restoration stay in
the engine; one Cake call is made per engine call.

A route is taken only when the adapter's ``supports_*`` admission accepts the
exact tensors about to be passed (device cc 10.0 / 10.3, FlashInfer modules
present, dtypes / layouts, 32 or 64 heads, page 64, and the shipped FlashInfer
catalog carrying the program for the point); otherwise the stock DeepGEMM call
runs unchanged. A FlashInfer host-side ``ValueError`` is cached per (site,
reason) as a rejection; a shape first seen inside CUDA-graph capture falls back
for that graph (the JIT module is built by an eager warm-up forward). The
first "taken" and the first "fallback" per site and reason are logged once.

Importing this module loads no FlashInfer or DeepGEMM code.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled

logger = logging.getLogger(__name__)

CAKE_ROUTE_DSA_INDEXER = "dsa_indexer"
_CAKE_LOG_PREFIX = "[cake-route]"
SITE_RAGGED = "fp8_mqa_logits"
SITE_PAGED = "fp8_paged_mqa_logits"

# (site, event, key) triples already logged; keeps the per-call path free of I/O.
_cake_route_logged: set = set()
# (site, reason kind) pairs FlashInfer rejected on the host; never retried.
_cake_route_rejected: set = set()
# (site, shape signature) pairs that ran eagerly at least once (JIT built).
_cake_route_primed: set = set()


def _reason_kind(detail: str) -> str:
    """Digit-normalised prefix of a fallback detail (the text before the tensor
    dump), so one line is emitted per distinct reason, not per shape."""
    return re.sub(r"\d+", "N", detail.split(":", 1)[0])[:64]


def _log_cake_route_once(site: str, event: str, detail: str) -> None:
    key = (site, event, _reason_kind(detail) if event == "fallback" else "")
    if key in _cake_route_logged:
        return
    _cake_route_logged.add(key)
    route = f"{CAKE_ROUTE_DSA_INDEXER}/{site}"
    if event == "taken":
        logger.info("%s %s: Cake kernel selected (%s)", _CAKE_LOG_PREFIX, route, detail)
    else:
        logger.info(
            "%s %s: fallback to the default kernel (%s)",
            _CAKE_LOG_PREFIX,
            route,
            detail,
        )


def reset_cake_route_state_for_tests() -> None:
    _cake_route_logged.clear()
    _cake_route_rejected.clear()
    _cake_route_primed.clear()


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _tensor_summary(**tensors: Optional[torch.Tensor]) -> str:
    parts = []
    for name, t in tensors.items():
        if t is None:
            parts.append(f"{name}=None")
        else:
            parts.append(
                f"{name}={tuple(t.shape)}/{str(t.dtype).removeprefix('torch.')}"
            )
    return " ".join(parts)


@functools.lru_cache(maxsize=None)
def _cake_ragged_kernels() -> Tuple[Callable, Callable]:
    """Lazy (admission, forwarder) of the ragged site."""
    from sglang.kernels.cake_kernels.attention_sparse import supports_fp8_mqa_logits
    from sglang.kernels.ops.attention.cake import cake_fp8_mqa_logits

    return supports_fp8_mqa_logits, cake_fp8_mqa_logits


@functools.lru_cache(maxsize=None)
def _cake_paged_kernels() -> Tuple[Callable, Callable]:
    """Lazy (admission, logits forwarder) of the paged site (no metadata launch: one kernel per call)."""
    from sglang.kernels.cake_kernels.attention_sparse import (
        supports_fp8_paged_mqa_logits,
    )
    from sglang.kernels.ops.attention.cake import cake_fp8_paged_mqa_logits

    return supports_fp8_paged_mqa_logits, cake_fp8_paged_mqa_logits


def _capture_gate(site: str, signature: tuple) -> bool:
    """True when the call may proceed: eager calls prime the (site, shape) key;
    a key first seen inside CUDA-graph capture falls back for that graph."""
    key = (site, signature)
    if _is_capturing():
        if key in _cake_route_primed:
            return True
        _log_cake_route_once(
            site,
            "fallback",
            f"shape first seen inside CUDA-graph capture: {signature}",
        )
        return False
    _cake_route_primed.add(key)
    return True


def cake_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    num_sms: Optional[int],
) -> Optional[torch.Tensor]:
    """Cake ``fp8_mqa_logits`` on the engine's ragged-prefill tensors, or ``None``.

    ``num_sms`` is the CTA budget DeepGEMM would use (``deep_gemm.get_num_sms()``
    at call time, lowered by the pipeline-parallel reservation).
    """
    if not cake_route_enabled(CAKE_ROUTE_DSA_INDEXER):
        return None
    kv, kv_scales = kv_fp8
    supports, forward = _cake_ragged_kernels()
    summary = _tensor_summary(q=q_fp8, kv=kv, kv_scales=kv_scales, w=weights, ks=ks)
    if not supports(q_fp8, kv, kv_scales, weights, ks, ke):
        _log_cake_route_once(
            SITE_RAGGED, "fallback", f"adapter admission rejected: {summary}"
        )
        return None
    signature = (tuple(q_fp8.shape), int(kv.shape[0]), num_sms)
    if not _capture_gate(SITE_RAGGED, signature):
        return None
    try:
        logits = forward(
            q_fp8,
            (kv, kv_scales),
            weights,
            ks,
            ke,
            clean_logits=False,
            max_seqlen_k=0,
            sm_count=num_sms,
        )
    except ValueError as error:
        kind = (SITE_RAGGED, _reason_kind(str(error)))
        if kind in _cake_route_rejected:
            return None
        _cake_route_rejected.add(kind)
        _log_cake_route_once(
            SITE_RAGGED, "fallback", f"FlashInfer host rejection: {error}: {summary}"
        )
        return None
    _log_cake_route_once(SITE_RAGGED, "taken", summary)
    return logits


def cake_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    *,
    block_kv: int,
    num_sms: Optional[int],
) -> Optional[torch.Tensor]:
    """Cake ``fp8_paged_mqa_logits`` on the engine's paged tensors (one launch), or ``None``.

    Mirrors one ``deep_gemm.fp8_paged_mqa_logits`` call of the engine's chunk
    loop: ``q [B, next_n, H, 128]``, fused ``kv_cache [pages, 64, 1, 132]``,
    ``weights [B * next_n, H]``, 2-D ``context_lens [B, next_n]``,
    ``block_table [B, S]`` (any row stride), ``clean_logits=False``. No
    schedule-metadata launch: the Cake program derives its walk in-kernel
    (``schedule_meta=None``; ``num_sms`` is the CTA budget).
    """
    if not cake_route_enabled(CAKE_ROUTE_DSA_INDEXER):
        return None
    supports, forward = _cake_paged_kernels()
    summary = _tensor_summary(
        q=q, kv_cache=kv_cache, w=weights, ctx=context_lens, bt=block_table
    )
    if block_kv != 64 or not supports(
        q, kv_cache, weights, context_lens, block_table, int(max_context_len)
    ):
        _log_cake_route_once(
            SITE_PAGED,
            "fallback",
            f"adapter admission rejected: page {block_kv} max_context_len {int(max_context_len)} {summary}",
        )
        return None
    signature = (tuple(q.shape), int(max_context_len), num_sms)
    if not _capture_gate(SITE_PAGED, signature):
        return None
    try:
        logits = forward(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            None,
            max_context_len,
            clean_logits=False,
            sm_count=num_sms,
        )
    except ValueError as error:
        kind = (SITE_PAGED, _reason_kind(str(error)))
        if kind in _cake_route_rejected:
            return None
        _cake_route_rejected.add(kind)
        _log_cake_route_once(
            SITE_PAGED, "fallback", f"FlashInfer host rejection: {error}: {summary}"
        )
        return None
    _log_cake_route_once(SITE_PAGED, "taken", summary)
    return logits

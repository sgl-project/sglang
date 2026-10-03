"""``dsv4_sparse_mla_decode`` Cake route helpers for the DeepSeek-V4 backends.

``SGLANG_CAKE_ROUTES=dsv4_sparse_mla_decode`` lets two DeepSeek-V4 sparse MLA
call sites take a Cake KernelSpec (``sglang.kernels.ops.attention.cake``)
instead of the engine's default kernel:

* :class:`CakeDsv4TrtllmRoute` — ``--dsv4-attn-backend trtllm`` on SM100 /
  SM103 (``deepseek_v4_trtllm_backend.py``).  The stock call is FlashInfer's
  ``trtllm_batch_decode_sparse_mla_dsv4``; the Cake route is the same entry
  with ``backend="cake"`` on the same uniform-FP8 pools, metadata tables and
  workspace.  Covers decode, target-verify / draft-extend-v2 (table rows per
  draft token) and the varlen prefill (``cum_seq_lens_q`` / ``max_q_len``).
* :class:`CakeDsv41MixedDecodeRoute` — the default backend on SM120 / SM121
  with a DeepSeek-V4.1 pool (``deepseek_v4_backend.py``).  The stock call is
  ``flash_mla_with_kvcache_sm120`` on the 528-byte FP8 main cache and the
  288-byte FP4 compressed cache; the Cake mixed-cache decode reads exactly
  those layouts.  Covers the decode-kernel token range (decode, target-verify,
  draft-extend, small extend chunks); larger extends keep the stock prefill
  kernel (the Cake family is decode-only).

Not covered (stock path kept, see the backend docstrings): the SM100 / SM103
default backend (packed FlashMLA caches of 584 / 528 bytes per token, while
the Cake SM100 route reads uniform 512-wide pools of the query dtype) and the
SM120 NVFP4 route (no 384-byte NVFP4 DeepSeek-V4 pool exists in SGLang).

A route is taken only when the adapter's ``supports_*`` admission accepts the
exact tensors the engine is about to pass; otherwise the stock call runs
unchanged.  Workspaces / scratch are prepared eagerly (warm-up forward) and a
shape first seen inside CUDA-graph capture falls back for that graph.  The
first "taken" and the first "fallback" per site are logged once.

Importing this module loads no FlashInfer code.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled

logger = logging.getLogger(__name__)

CAKE_ROUTE_DSV4_SPARSE_MLA_DECODE = "dsv4_sparse_mla_decode"
_CAKE_LOG_PREFIX = "[cake-route]"
_DSV4_HEAD_DIM = 512

# (site, event) pairs already logged; keeps the per-call path free of I/O.
_cake_route_logged: set = set()


def _reason_kind(detail: str) -> str:
    """Digit-normalised prefix of a fallback detail (the text before the tensor
    dump), so one line is emitted per distinct reason, not per shape."""
    return re.sub(r"\d+", "N", detail.split(":", 1)[0])[:64]


def _log_cake_route_once(
    site: str, event: str, detail: str, key_extra: str = ""
) -> None:
    key = (
        site,
        event,
        _reason_kind(detail) if event == "fallback" else key_extra,
    )
    if key in _cake_route_logged:
        return
    _cake_route_logged.add(key)
    route = f"{CAKE_ROUTE_DSV4_SPARSE_MLA_DECODE}/{site}"
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
    _dsv4_ab_mode.cache_clear()


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


_DSV4_AB_ENV = "SGLANG_CAKE_DEBUG_DSV4_AB"


@functools.lru_cache(maxsize=None)
def _dsv4_ab_mode() -> str:
    """Debug A/B of the SM100 Cake route against the stock kernel (eager only).

    ``SGLANG_CAKE_DEBUG_DSV4_AB=log`` runs the stock kernel on the same tensors
    after every Cake call, logs the difference and keeps the Cake output;
    ``=stock`` logs the difference and returns the stock output (the model
    stays on the default numerics).  Unset / ``0`` disables it.  Calls inside
    CUDA-graph capture are never compared; use ``--disable-cuda-graph``.
    """
    value = os.environ.get(_DSV4_AB_ENV, "").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return ""
    return "stock" if value in ("stock", "2") else "log"


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
def _cake_sm100_kernels() -> Tuple[Callable, Callable, Callable, Callable]:
    """Lazy (admission, forwarder, workspace_bytes, workspace_reset) for SM100/103."""
    from sglang.kernels.cake_kernels.attention_mla import (
        cake_dsv4_workspace_reset,
        get_cake_dsv4_workspace_bytes,
        supports_trtllm_batch_decode_sparse_mla_dsv4,
    )
    from sglang.kernels.ops.attention.cake import (
        cake_trtllm_batch_decode_sparse_mla_dsv4,
    )

    return (
        supports_trtllm_batch_decode_sparse_mla_dsv4,
        cake_trtllm_batch_decode_sparse_mla_dsv4,
        get_cake_dsv4_workspace_bytes,
        cake_dsv4_workspace_reset,
    )


@functools.lru_cache(maxsize=None)
def _cake_sm120_dsv41_kernels() -> Tuple[Callable, Callable, Callable]:
    """Lazy (admission, forwarder, num_chunks) for the SM120/121 DSv4.1 decode."""
    from sglang.kernels.cake_kernels.attention_mla_sm120_dsv41 import (
        sparse_mla_sm120_dsv41_mixed_num_chunks,
        supports_sparse_mla_sm120_dsv41_mixed_decode,
    )
    from sglang.kernels.ops.attention.cake import (
        cake_sparse_mla_sm120_dsv41_mixed_decode,
    )

    return (
        supports_sparse_mla_sm120_dsv41_mixed_decode,
        cake_sparse_mla_sm120_dsv41_mixed_decode,
        sparse_mla_sm120_dsv41_mixed_num_chunks,
    )


class CakeDsv4TrtllmRoute:
    """SM100/103 ``trtllm_batch_decode_sparse_mla_dsv4(backend="cake")`` route.

    ``run()`` returns the attention output or ``None`` ("use the stock call").
    The Cake kernels carve the caller's ``workspace_buffer`` deterministically;
    the first eager call zeroes its split-merge counters once
    (``cake_dsv4_workspace_reset``) so later CUDA-graph captures replay without
    host state.  A workspace first seen while capturing falls back for that
    graph.
    """

    SITE = "trtllm"

    def __init__(self) -> None:
        self._primed_workspaces: set = set()
        # (rows, heads, sparse_topk, dtype) -> required workspace bytes.
        self._workspace_bytes: Dict[tuple, int] = {}
        # (sparse_topk, rows) -> number of debug A/B comparisons so far.
        self._ab_counts: Dict[Tuple[int, int], int] = {}

    def _required_workspace_bytes(
        self,
        get_bytes: Callable[..., int],
        *,
        rows: int,
        heads: int,
        sparse_topk: int,
        dtype: torch.dtype,
    ) -> Optional[int]:
        key = (rows, heads, sparse_topk, dtype)
        needed = self._workspace_bytes.get(key)
        if needed is None:
            try:
                needed = int(get_bytes(rows, heads, sparse_topk, dtype))
            except ValueError:
                # sparse_topk outside the kernel's table contract.
                needed = -1
            self._workspace_bytes[key] = needed
        return None if needed < 0 else needed

    def run(
        self,
        *,
        query: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sparse_indices: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        sparse_topk_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bmm1_scale: float,
        bmm2_scale: float,
        sinks: torch.Tensor,
        kv_layout: str = "HND",
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not cake_route_enabled(CAKE_ROUTE_DSV4_SPARSE_MLA_DECODE):
            return None
        supports, cake_decode, get_bytes, reset = _cake_sm100_kernels()
        detail = _tensor_summary(
            q=query,
            swa=swa_kv_cache,
            compressed=compressed_kv_cache,
            indices=sparse_indices,
            lens=sparse_topk_lens,
            seq_lens=seq_lens,
            cum_q=cum_seq_lens_q,
        )
        if not supports(
            query,
            swa_kv_cache,
            kv_cache_format="fp8",
            compressed_kv_cache=compressed_kv_cache,
            enable_pdl=None,
        ):
            _log_cake_route_once(
                self.SITE, "fallback", f"adapter admission rejected: {detail}"
            )
            return None
        needed = self._required_workspace_bytes(
            get_bytes,
            rows=int(sparse_indices.shape[0]),
            heads=int(query.shape[-2]),
            sparse_topk=int(sparse_indices.shape[1]),
            dtype=query.dtype,
        )
        available = workspace_buffer.numel() * workspace_buffer.element_size()
        if needed is None or needed > available:
            _log_cake_route_once(
                self.SITE,
                "fallback",
                f"workspace needs {needed} bytes, engine buffer has {available}: "
                f"{detail}",
            )
            return None
        ws_key = workspace_buffer.data_ptr()
        if ws_key not in self._primed_workspaces:
            if _is_capturing():
                _log_cake_route_once(
                    self.SITE,
                    "fallback",
                    "workspace counters not primed before CUDA-graph capture "
                    f"(no eager warm-up call reached this site): {detail}",
                )
                return None
            reset(workspace_buffer)
            self._primed_workspaces.add(ws_key)
        out = cake_decode(
            query,
            swa_kv_cache,
            workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            kv_layout=kv_layout,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )
        _log_cake_route_once(
            self.SITE,
            "taken",
            detail,
            key_extra=f"topk{int(sparse_indices.shape[1])}",
        )
        if _dsv4_ab_mode() and not _is_capturing():
            out = self._debug_compare_with_stock(
                out,
                reset,
                query=query,
                swa_kv_cache=swa_kv_cache,
                workspace_buffer=workspace_buffer,
                sparse_indices=sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=seq_lens,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=sinks,
                kv_layout=kv_layout,
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=max_q_len,
            )
        return out

    def _debug_compare_with_stock(
        self,
        cake_out: torch.Tensor,
        reset: Callable[[torch.Tensor], None],
        *,
        query: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sparse_indices: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        sparse_topk_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
        sinks: torch.Tensor,
        kv_layout: str,
        cum_seq_lens_q: Optional[torch.Tensor],
        max_q_len: Optional[int],
    ) -> torch.Tensor:
        """Run the stock kernel on the same tensors and log the difference.

        Debug only (``SGLANG_CAKE_DEBUG_DSV4_AB``); synchronises the stream.
        The stock kernel shares the workspace, so the Cake split-merge counters
        are re-zeroed afterwards.
        """
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4 as stock

        ref = stock(
            query=query,
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            kv_layout=kv_layout,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )
        reset(workspace_buffer)
        rows = int(sparse_indices.shape[0])
        heads = int(query.shape[-2])
        a = cake_out.reshape(-1, heads, _DSV4_HEAD_DIM)[:rows].float()
        b = ref.reshape(-1, heads, _DSV4_HEAD_DIM)[:rows].float()
        diff = (a - b).abs()
        bad = (diff > 0.1 + 0.1 * b.abs()) | ~torch.isfinite(a)
        key = (int(sparse_indices.shape[1]), rows)
        n = self._ab_counts.get(key, 0)
        self._ab_counts[key] = n + 1
        if n < 4 or n % 500 == 0:
            lens = sparse_topk_lens[:rows]
            sl = seq_lens[:rows]
            logger.info(
                "%s dsv4 A/B topk=%d rows=%d heads=%d call=%d: max|d|=%.4g "
                "mean|d|=%.4g frac>fp8tol=%.4g nonfinite=%d ref_max=%.4g "
                "lens[min,max]=(%d,%d) seq_lens[min,max]=(%d,%d) neg_idx=%d",
                _CAKE_LOG_PREFIX,
                key[0],
                rows,
                heads,
                n,
                float(diff.max()),
                float(diff.mean()),
                float(bad.float().mean()),
                int((~torch.isfinite(a)).sum()),
                float(b.abs().max()),
                int(lens.min()),
                int(lens.max()),
                int(sl.min()),
                int(sl.max()),
                int((sparse_indices[:rows] < 0).sum()),
            )
        return ref if _dsv4_ab_mode() == "stock" else cake_out


class CakeDsv41MixedDecodeRoute:
    """SM120/121 DeepSeek-V4.1 mixed-cache Cake decode route.

    ``run()`` returns ``o`` shaped like the stock ``flash_mla_with_kvcache_sm120``
    result (``[T, 1, H, 512]``) or ``None`` ("use the stock call").  The Cake
    entry is allocation-free: ``output`` / ``out_lse`` are allocated per call
    like the stock kernel's outputs, while the split-merge scratch
    (``mid_out`` / ``mid_lse``) comes from a grow-only arena sized for every
    split plan of the shape (``num_chunks`` upper bound).  The arena grows only
    outside CUDA-graph capture; retired storage is kept alive so captured
    graphs that reference it stay valid.
    """

    SITE = "sm120_dsv41"

    def __init__(self) -> None:
        self._mid_out: Optional[torch.Tensor] = None
        self._mid_lse: Optional[torch.Tensor] = None
        self._retired: List[torch.Tensor] = []
        # (topk, extra_topk) -> num_chunks (host-only FlashInfer query).
        self._num_chunks: Dict[Tuple[int, int], int] = {}

    def _scratch(
        self, rows: int, chunks: int, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        n_out = rows * chunks * _DSV4_HEAD_DIM
        n_lse = rows * chunks
        if (
            self._mid_out is None
            or self._mid_out.numel() < n_out
            or self._mid_lse.numel() < n_lse
            or self._mid_out.device != device
        ):
            if _is_capturing():
                return None
            if self._mid_out is not None:
                self._retired.extend((self._mid_out, self._mid_lse))
            self._mid_out = torch.empty(n_out, dtype=torch.bfloat16, device=device)
            self._mid_lse = torch.empty(n_lse, dtype=torch.float32, device=device)
        return self._mid_out[:n_out], self._mid_lse[:n_lse]

    def run(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        attn_sink: torch.Tensor,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        sm_scale: float,
        head_dim_v: int,
    ) -> Optional[torch.Tensor]:
        if not cake_route_enabled(CAKE_ROUTE_DSV4_SPARSE_MLA_DECODE):
            return None
        supports, cake_decode, num_chunks = _cake_sm120_dsv41_kernels()
        detail = _tensor_summary(
            q=q,
            k_cache=k_cache,
            indices=indices,
            extra_k_cache=extra_k_cache,
            extra_indices=extra_indices,
        )
        # The engine passes q as [T, 1, H, 512]; the Cake entry takes [T, H, 512].
        q3 = q.squeeze(1) if q.ndim == 4 else q
        if head_dim_v != _DSV4_HEAD_DIM or not supports(
            q3,
            k_cache,
            indices,
            extra_kv_cache=extra_k_cache,
            extra_indices=extra_indices,
            compute_precision="bf16",
        ):
            _log_cake_route_once(
                self.SITE,
                "fallback",
                f"adapter admission rejected (head_dim_v={head_dim_v}): {detail}",
            )
            return None
        num_tokens, num_heads = int(q3.shape[0]), int(q3.shape[1])
        topk = int(indices.shape[-1])
        extra_topk = int(extra_indices.shape[-1]) if extra_indices is not None else 0
        chunks_key = (topk, extra_topk)
        chunks = self._num_chunks.get(chunks_key)
        if chunks is None:
            chunks = max(1, int(num_chunks(topk, extra_topk)))
            self._num_chunks[chunks_key] = chunks
        scratch = self._scratch(num_tokens * num_heads, chunks, q3.device)
        if scratch is None:
            _log_cake_route_once(
                self.SITE,
                "fallback",
                "split-merge scratch for this shape was first needed inside "
                f"CUDA-graph capture (no eager warm-up call): {detail}",
            )
            return None
        mid_out, mid_lse = scratch
        output = torch.empty(
            (num_tokens, num_heads, _DSV4_HEAD_DIM),
            dtype=torch.bfloat16,
            device=q3.device,
        )
        out_lse = torch.empty(
            (num_tokens, num_heads), dtype=torch.float32, device=q3.device
        )
        cake_decode(
            q3,
            k_cache,
            indices,
            output,
            out_lse,
            sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_k_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            mid_out=mid_out.view(num_tokens, num_heads, chunks, _DSV4_HEAD_DIM),
            mid_lse=mid_lse.view(num_tokens, num_heads, chunks),
            compute_precision="bf16",
        )
        _log_cake_route_once(self.SITE, "taken", detail)
        return output.unsqueeze(1) if q.ndim == 4 else output

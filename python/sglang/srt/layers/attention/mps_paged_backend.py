"""Block paged attention decode for Torch MPS on Apple silicon.

``TorchNativeAttnBackend`` decodes with a Python loop over requests, gathering
each request's KV out of the pool before every SDPA call. This backend keeps
that behavior for prefill and for every case the Metal kernel cannot serve, and
replaces the decode loop with one block-table paged attention launch.

No new KV pool is needed. SGLang's NHD buffer is already block shaped:
``[size + page_size, head_num, head_dim]`` with a page-aligned ``size``, so it
views as ``[num_blocks, page_size, head_num, head_dim]`` without a copy. The
paged allocator hands out ``page_id * page_size + arange(page_size)``, so a
request's block table is exactly its per-page token slots divided by the page
size.

Opt in with ``--attention-backend mps_paged``. Set
``SGLANG_MPS_DISABLE_PAGED_KERNEL=1`` to force the SDPA decode path while
keeping everything else identical, which isolates the kernel when benchmarking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# The Metal kernel stages per-head state in threadgroup memory sized for this
# bound; see csrc/metal/paged_attention.metal.
_MAX_HEAD_DIM = 256

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class MpsPagedAttnBackend(TorchNativeAttnBackend):
    """Torch MPS attention with a block paged decode kernel."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.page_size = getattr(model_runner.token_to_kv_pool, "page_size", 1)
        self._kernel = None
        self._fallback_reason: Optional[str] = None

    def _load_kernel(self):
        if self._kernel is None:
            from sglang.srt.hardware_backend.mlx.ops.paged_attention import (
                block_paged_attention_decode,
            )

            self._kernel = block_paged_attention_decode
        return self._kernel

    def _fallback(self, reason: str) -> None:
        """Log the first time a batch shape sends decode back to SDPA."""
        if self._fallback_reason != reason:
            self._fallback_reason = reason
            logger.info("mps_paged: using the SDPA decode path because %s", reason)

    def _unsupported_reason(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        is_cross_attn: bool,
        sliding_window_size: Optional[int],
    ) -> Optional[str]:
        """Return why the kernel cannot serve this batch, or None if it can."""
        if envs.SGLANG_MPS_DISABLE_PAGED_KERNEL.get():
            return "SGLANG_MPS_DISABLE_PAGED_KERNEL is set"
        if is_cross_attn or encoder_lens is not None:
            return "cross attention is not supported by the kernel"
        if sliding_window_size is not None and sliding_window_size > -1:
            return "sliding window attention is not supported by the kernel"
        if query.device.type != "mps":
            return f"the query is on {query.device}, not MPS"
        if query.dtype not in _SUPPORTED_DTYPES:
            return f"dtype {query.dtype} is not supported by the kernel"
        # Quantized pools hand back a uint8 store dtype, and non-NHD layouts
        # (HND, vectorized_5d) do not view as [blocks, page, head, dim].
        if k_cache.dtype != query.dtype or v_cache.dtype != query.dtype:
            return "the KV pool dtype does not match the query dtype"
        if k_cache.ndim != 3 or v_cache.ndim != 3:
            return "the KV pool is not in the NHD layout"
        if k_cache.shape[-1] != v_cache.shape[-1]:
            return "asymmetric K/V head dims are not supported by the kernel"
        if k_cache.shape[-1] > _MAX_HEAD_DIM:
            return f"head_dim {k_cache.shape[-1]} exceeds the kernel bound of {_MAX_HEAD_DIM}"
        if query.shape[0] != seq_lens.shape[0]:
            return "decode emits more than one query token per request"
        if query.shape[1] % k_cache.shape[1] != 0:
            return "the query head count is not divisible by the KV head count"
        if k_cache.shape[0] % self.page_size != 0:
            return "the KV pool size is not page aligned"
        if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
            return "the KV pool is not contiguous"
        if int(seq_lens.min()) <= 0:
            return "a request has no visible KV tokens"
        return None

    def _build_block_tables(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Map each request's pages to physical block IDs, padding with -1."""
        page_size = self.page_size
        max_blocks = (max_seq_len + page_size - 1) // page_size
        first_slot_cols = (
            torch.arange(max_blocks, device=req_to_token.device) * page_size
        )

        slots = req_to_token[req_pool_indices.to(torch.long)][:, first_slot_cols]
        block_tables = torch.div(slots, page_size, rounding_mode="floor")
        in_range = first_slot_cols.unsqueeze(0) < seq_lens.unsqueeze(1)
        return torch.where(
            in_range, block_tables, torch.full_like(block_tables, -1)
        ).to(torch.int32)

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        reason = self._unsupported_reason(
            query,
            k_cache,
            v_cache,
            seq_lens,
            encoder_lens,
            is_cross_attn,
            sliding_window_size,
        )
        if reason is not None:
            self._fallback(reason)
            return super()._run_sdpa_forward_decode(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                is_cross_attn=is_cross_attn,
                sliding_window_size=sliding_window_size,
            )

        try:
            kernel = self._load_kernel()
        except ImportError as exc:
            self._fallback(f"the Metal kernel is unavailable ({exc})")
            return super()._run_sdpa_forward_decode(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                is_cross_attn=is_cross_attn,
                sliding_window_size=sliding_window_size,
            )

        page_size = self.page_size
        num_kv_heads, head_dim = k_cache.shape[1], k_cache.shape[2]
        num_qo_heads = query.shape[1]
        seq_lens_i32 = seq_lens.to(torch.int32)
        max_seq_len = int(seq_lens.max())

        block_tables = self._build_block_tables(
            req_to_token, req_pool_indices, seq_lens_i32, max_seq_len
        )
        k_blocks = k_cache.view(-1, page_size, num_kv_heads, head_dim)
        v_blocks = v_cache.view(-1, page_size, num_kv_heads, head_dim)

        attn_out = kernel(
            query.contiguous(),
            k_blocks,
            v_blocks,
            block_tables.contiguous(),
            seq_lens_i32.contiguous(),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=page_size,
            sm_scale=float(scaling) if scaling is not None else head_dim**-0.5,
        )
        output.copy_(attn_out)
        return output

    def support_triton(self):
        return False

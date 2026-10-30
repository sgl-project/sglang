"""MiniMax MSA's external Q8K8 backend with fixed decode buffers."""

from __future__ import annotations

import inspect
import math

import torch

WORKSPACE_BYTES = 256 * 1024 * 1024
_PAGE_SIZE = 128
_HEAD_DIM = 128
_TOPK = 16


class FlashInferQ8K8Decode:
    """Consume fused gather/dequant output without a copy or host-side planning.

    The compact payload is [query, kv_head, topk, token, dim]. A strided HND
    view makes each selected slot a physical page; block tables are static,
    while sparse lengths are refreshed in place before each forward/graph.
    TopK follows the indexer's contract: an unordered history prefix, then
    the local page as the last live slot, followed by a -1 padding suffix.
    """

    def __init__(
        self,
        compact_k: torch.Tensor,
        compact_v: torch.Tensor,
        *,
        num_q_heads: int,
        sm_scale: float,
        workspace_buffer: torch.Tensor | None = None,
    ) -> None:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        required = {"enable_block_sparse_attention", "multi_ctas_kv_counter_buffer"}
        if not required.issubset(
            inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
        ):
            raise RuntimeError(
                "MSA-v1 Q8K8 decode requires FlashInfer TRTLLM-GEN block-sparse support"
            )
        self._run = trtllm_batch_decode_with_kv_cache
        if compact_k.ndim != 5:
            raise ValueError("compact K/V must be five-dimensional")
        query_rows, kv_heads, topk, page_size, head_dim = compact_k.shape
        if min(query_rows, kv_heads, num_q_heads) <= 0:
            raise ValueError("query and head counts must be positive")
        if not math.isfinite(sm_scale) or sm_scale <= 0:
            raise ValueError("sm_scale must be finite and positive")
        if (topk, page_size, head_dim) != (_TOPK, _PAGE_SIZE, _HEAD_DIM):
            raise ValueError(
                "compact K/V must have shape [query_rows, kv_heads, 16, 128, 128]"
            )
        if num_q_heads % kv_heads or num_q_heads // kv_heads not in (1, 2, 4, 8, 16):
            raise ValueError(
                "MSA-v1 Q8K8 requires 1, 2, 4, 8, or 16 local Q heads per KV head"
            )
        for cache in (compact_k, compact_v):
            if cache.shape != compact_k.shape or cache.dtype != torch.float8_e4m3fn:
                raise ValueError("compact K/V must have matching FP8 shapes")
            if (
                not cache.is_cuda
                or not cache.is_contiguous()
                or cache.device != compact_k.device
            ):
                raise ValueError(
                    "compact K/V must be contiguous on the same CUDA device"
                )

        options = {"device": compact_k.device}
        page_stride = _PAGE_SIZE * _HEAD_DIM
        # For query q/head h/slot t, page=q*H*16+t and head stride=16 pages.
        # The final head's final page must stay within the original allocation.
        cache_pages = (query_rows - 1) * kv_heads * _TOPK + _TOPK
        cache_shape = (cache_pages, kv_heads, _PAGE_SIZE, _HEAD_DIM)
        cache_stride = (page_stride, _TOPK * page_stride, _HEAD_DIM, 1)
        self.kv_cache = tuple(
            cache.as_strided(cache_shape, cache_stride)
            for cache in (compact_k, compact_v)
        )
        query_base = torch.arange(query_rows, dtype=torch.int32, **options) * (
            kv_heads * _TOPK
        )
        slots = torch.arange(_TOPK, dtype=torch.int32, **options)
        self.block_tables = (
            (query_base[:, None] + slots).expand(kv_heads, -1, -1).contiguous()
        )
        self.seq_lens = torch.zeros(kv_heads, query_rows, dtype=torch.int32, **options)
        self.output = torch.empty(
            query_rows, num_q_heads, _HEAD_DIM, dtype=torch.bfloat16, **options
        )
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                WORKSPACE_BYTES, dtype=torch.uint8, **options
            )
        if (
            workspace_buffer.dtype != torch.uint8
            or workspace_buffer.device != compact_k.device
            or not workspace_buffer.is_contiguous()
            or workspace_buffer.numel() < WORKSPACE_BYTES
        ):
            raise ValueError(
                "FlashInfer workspace must be at least 256 MiB of contiguous device uint8"
            )
        self.workspace = workspace_buffer
        sm_count = torch.cuda.get_device_properties(
            compact_k.device
        ).multi_processor_count
        counter_bytes = ((max(query_rows * num_q_heads, sm_count) + 7) // 8) * 8 * 4
        self.counter = torch.zeros(counter_bytes, dtype=torch.uint8, **options)
        self.sm_scale = float(sm_scale)

    def update_lengths(self, query_seq_lens: torch.Tensor) -> None:
        # FlashInfer leaves zero-length rows unwritten. The gather kernel zeros
        # padded rows, so one zero K/V token produces an exact zero output.
        skipped_pages = torch.div(query_seq_lens - 1, _PAGE_SIZE, rounding_mode="floor")
        skipped_pages = (skipped_pages - (_TOPK - 1)).clamp_min_(0)
        self.seq_lens.copy_((query_seq_lens - skipped_pages * _PAGE_SIZE).clamp_min_(1))

    def run(self, q: torch.Tensor) -> torch.Tensor:
        return self._run(
            q,
            self.kv_cache,
            self.workspace,
            self.block_tables,
            self.seq_lens,
            _TOPK * _PAGE_SIZE,
            bmm1_scale=self.sm_scale,
            bmm2_scale=1.0,
            out=self.output,
            out_dtype=torch.bfloat16,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            enable_pdl=True,
            multi_ctas_kv_counter_buffer=self.counter,
            enable_block_sparse_attention=True,
        )

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class HiSparseDemandInputs:
    """Inputs owned by the HiSparse MTP Demand adapter."""

    host_kv: torch.Tensor
    host_locs: torch.Tensor
    device_locs: torch.Tensor
    cache_tags: torch.Tensor
    decode_calls: torch.Tensor
    num_real_reqs: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    mtp_committed_lens: torch.Tensor
    cache_rows: int

    def run(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        indices: torch.Tensor,
        tile_scheduler_metadata: torch.Tensor,
        num_splits: torch.Tensor,
        head_dim_v: int,
        softmax_scale: float,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        causal: bool,
        is_fp8_kvcache: bool,
        attn_sink: Optional[torch.Tensor],
        extra_k_cache: Optional[torch.Tensor],
        extra_indices_in_kvcache: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert block_table is not None
        assert cache_seqlens is not None
        assert attn_sink is None
        assert extra_k_cache is None
        assert extra_indices_in_kvcache is None
        assert topk_length is None
        assert extra_topk_length is None
        assert not causal
        assert is_fp8_kvcache
        assert self.host_locs.dtype == torch.int32 and self.host_locs.shape == (
            indices.shape[0],
            indices.shape[-1],
        )
        assert q.shape[1] == 1
        assert indices.shape[-1] == 2048
        assert self.cache_rows == 4096
        assert self.cache_tags.dtype == torch.int64
        assert self.cache_tags.shape[1] == self.cache_rows
        assert self.decode_calls.dtype == torch.int32
        assert self.decode_calls.shape == (self.cache_tags.shape[0],)
        assert self.num_real_reqs.dtype == torch.int32
        assert self.num_real_reqs.shape == (1,)
        assert self.device_locs.dim() == 2
        assert self.device_locs.shape == (
            self.cache_tags.shape[0],
            self.cache_rows + 6,
        )
        assert self.mtp_committed_lens.dtype == torch.int32
        assert self.mtp_committed_lens.shape == (indices.shape[0],)

        out, softmax_lse, _, _ = (
            torch.ops.sgl_kernel.sparse_decode_hisparse_demand_fwd.default(
                q,
                k_cache,
                indices,
                topk_length,
                attn_sink,
                tile_scheduler_metadata,
                num_splits,
                head_dim_v,
                softmax_scale,
                self.host_kv,
                self.host_locs,
                self.device_locs,
                self.cache_tags,
                self.decode_calls,
                self.num_real_reqs,
                self.req_pool_indices,
                self.seq_lens,
                self.mtp_committed_lens,
                self.cache_rows,
            )
        )
        return out, softmax_lse

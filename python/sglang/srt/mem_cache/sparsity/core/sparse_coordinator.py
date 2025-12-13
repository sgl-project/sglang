import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import nvtx
import torch

from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor
from sglang.srt.mem_cache.sparsity.core.sparse_kvcache_manager import (
    SparseKVCacheManager,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RequestTrackers:
    """State tracker for sparse attention requests."""

    def __init__(
        self,
        max_pool_size: int,
        device: torch.device,
        num_layers: int,
        min_sparse_prompt_len: int,
        max_context_len: int,
    ):
        self.device = device
        self.num_layers = num_layers
        self.top_k = min_sparse_prompt_len

        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )

        # Trackers for hierarchical NSA
        # TODO: Refactor and reduce memory usage
        self.full_host_indices = torch.full(
            (max_pool_size, max_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.curr_device_indices = torch.full(
            (max_pool_size, self.top_k + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.should_load_device_indices = torch.full(
            (max_pool_size, self.top_k),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.should_load_host_indices = torch.full(
            (max_pool_size, self.top_k),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.prev_top_k_result = torch.full(
            (max_pool_size, num_layers, self.top_k),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.prev_device_indices = torch.full(
            (max_pool_size, num_layers, self.top_k + 1),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def register(self, idx: int, prompt_len: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.last_constructed_page[idx] = 0
        self.full_host_indices[idx].fill_(-1)
        self.curr_device_indices[idx].fill_(-1)
        self.should_load_device_indices[idx].fill_(-1)
        self.should_load_host_indices[idx].fill_(-1)
        self.prev_top_k_result[idx].fill_(-1)
        self.prev_device_indices[idx].fill_(-1)

    def clear(self, idx: int) -> torch.Tensor:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0
        host_indices = self.full_host_indices[idx][self.full_host_indices[idx] != -1]
        self.full_host_indices[idx].fill_(-1)
        self.curr_device_indices[idx].fill_(-1)
        self.should_load_device_indices[idx].fill_(-1)
        self.should_load_host_indices[idx].fill_(-1)
        self.prev_top_k_result[idx].fill_(-1)
        self.prev_device_indices[idx].fill_(-1)

        return host_indices


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    backend: str = "flashattention"
    algorithm: str = "fake_random"
    page_size: int = 64
    sparse_ratio: float = 0.5
    min_sparse_prompt_len: int = 2048


class SparseCoordinator:
    """Coordinator for sparse attention pipeline."""

    def __init__(
        self,
        config: SparseConfig,
        algorithm: BaseSparseAlgorithm,
        backend_adaptor: Optional[BackendAdaptor],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        sparse_kv_cache_manager: SparseKVCacheManager,
        start_layer: int,
        end_layer: int,
        device: torch.device,
    ):
        self.config = config
        self.algorithm = algorithm
        self.backend_adaptor = backend_adaptor
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.sparse_kv_cache_manager = sparse_kv_cache_manager
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.page_size = config.page_size

        max_pool_size = req_to_token_pool.req_to_token.shape[0]
        self.states = RequestTrackers(
            max_pool_size,
            device,
            end_layer - start_layer + 1,
            self.config.min_sparse_prompt_len,
            self.req_to_token_pool.max_context_len,
        )
        if self.sparse_kv_cache_manager is not None:
            self.sparse_kv_cache_manager.req_states = self.states

        # Initialize algorithm representation pool and context
        self.algorithm.initialize_representation_pool(
            start_layer,
            end_layer,
            self.token_to_kv_pool,
            self.req_to_token_pool,
            self.states,
        )

        logger.info(
            f"SparseCoordinator initialized: algorithm={type(algorithm).__name__}"
        )

    def on_request_begin(self, req: "Req") -> None:
        """Handle request begin."""
        if req.req_pool_idx is not None:
            self.states.register(req.req_pool_idx, len(req.origin_input_ids))

    @nvtx.annotate("SparseCoordinator.on_request_prefill_end", color="yellow")
    def on_request_prefill_end(self, req: "Req") -> None:
        """Handle prefill end."""
        if (
            req.req_pool_idx is None
            or self.sparse_kv_cache_manager is None
            or len(req.origin_input_ids) < self.config.min_sparse_prompt_len
        ):
            return

        # Offload full KV cache for prefill
        self.sparse_kv_cache_manager.offload_prefill_full_kv_cache(req)
        self.sparse_kv_cache_manager.check_prefill_offload_progress()

        # Store previous device indices
        indices_len = self.config.min_sparse_prompt_len
        self.states.prev_device_indices[req.req_pool_idx][:, :-1] = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, :indices_len]
        )

    def on_request_end(self, req: "Req") -> None:
        """Handle request end."""
        if req.req_pool_idx is None or self.sparse_kv_cache_manager is None:
            return

        self.sparse_kv_cache_manager.check_sparse_offload_progress()
        host_indices = self.states.clear(req.req_pool_idx)

        if host_indices.numel() > 0:
            # Free host indices
            self.sparse_kv_cache_manager.host_mem_pool.free(host_indices.cpu())
            req_seqlen = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            assert (
                len(host_indices) == req_seqlen
            ), f"Host indices mismatch: {len(host_indices)} != {req_seqlen}"
            logger.info(
                f"Request {req.rid} ended, released {len(host_indices)} host indices"
            )

    @nvtx.annotate("SparseCoordinator.forward_begin", color="yellow")
    def forward_begin(self, forward_batch: "ForwardBatch") -> None:
        """Handle forward begin."""
        if self._should_check_offload(forward_batch):
            self.sparse_kv_cache_manager.check_sparse_offload_progress()

    @nvtx.annotate("SparseCoordinator.forward_end", color="yellow")
    def forward_end(self, forward_batch: "ForwardBatch") -> None:
        """Handle forward end."""
        if not self._should_check_offload(forward_batch):
            return

        offload_mask = (
            self.states.prompt_lens[forward_batch.req_pool_indices]
            >= self.config.min_sparse_prompt_len
        )
        if offload_mask.any():
            self.sparse_kv_cache_manager.offload_sparse_decode_req_tokens(
                forward_batch.req_pool_indices[offload_mask],
                forward_batch.out_cache_loc[offload_mask],
                forward_batch.seq_lens[offload_mask] - 1,
            )

    def _should_check_offload(self, forward_batch: "ForwardBatch") -> bool:
        return (
            forward_batch.req_pool_indices.numel() > 0
            and forward_batch.forward_mode.is_decode_or_idle()
            and self.sparse_kv_cache_manager is not None
        )

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> Optional[Any]:
        """Handle attention begin."""
        if layer.layer_id == self.start_layer:
            self.backend_adaptor.save_original_metadata(attn_metadata)

        return self._handle_sparse_retrieve(
            query, layer, forward_batch, attn_metadata, **kwargs
        )

    def attention_end(
        self,
        output: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> None:
        """Handle attention end."""
        layer_id = layer.layer_id

        # Maybe construct representations
        self.algorithm.construct_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

        # Maybe update representations
        self.algorithm.update_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

    def _handle_sparse_retrieve(
        self,
        query: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        req_pool_indices = forward_batch.req_pool_indices
        # Compute Topk
        sparse_mask = self._compute_sparse_mask(req_pool_indices)
        selected_indices, valid_lengths = self.algorithm.retrieve_topk(
            queries=query,
            layer_id=layer.layer_id,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Adapt Attention Metadata
        return self.backend_adaptor.adapt_for_attn_metadata(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=attn_metadata,
            forward_batch=forward_batch,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_size=self.page_size,
            layer_id=layer.layer_id,
        )

    def _compute_sparse_mask(self, req_pool_indices):
        mask = (
            self.states.prompt_lens[req_pool_indices]
            >= self.config.min_sparse_prompt_len
        )

        if not isinstance(self.algorithm, DeepSeekNSAAlgorithm):
            mask &= self.states.repr_constructed[req_pool_indices]
        return mask

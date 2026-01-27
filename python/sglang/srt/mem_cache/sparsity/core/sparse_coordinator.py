from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor
from sglang.srt.mem_cache.sparsity.core.sparse_kvcache_manager import (
    SparseKVCacheManager,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RequestTrackers:
    """State tracker for sparse attention requests."""

    def __init__(
        self,
        max_pool_size: int,
        device: torch.device,
        num_layers: int,
        topk_tokens_cnt: int,
        device_buffer_cnt: int,
        max_context_len: int,
        page_size: int,
    ):
        self.device = device
        self.num_layers = num_layers
        self.topk_tokens_cnt = topk_tokens_cnt
        self.device_buffer_cnt = device_buffer_cnt
        self.page_size = page_size

        # Request state tracker
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )
        self.hierarchical_sparse_enabled = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )

        # Trackers for diff kernel
        self.curr_device_indices = torch.full(
            (max_pool_size, self.device_buffer_cnt // self.page_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.last_device_indices = torch.full(
            (max_pool_size, num_layers, self.device_buffer_cnt // self.page_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.last_top_k_result = torch.full(
            (max_pool_size, num_layers, self.device_buffer_cnt // self.page_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_to_tokens_host = torch.full(
            (max_pool_size, max_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def _reset_state(self, idx: int) -> None:
        """Reset all tensor states for a request slot."""
        self.req_to_tokens_host[idx].fill_(-1)
        self.curr_device_indices[idx].fill_(-1)
        self.last_top_k_result[idx].fill_(-1)
        self.last_device_indices[idx].fill_(-1)
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0
        self.hierarchical_sparse_enabled[idx] = False

    def register(self, idx: int, prompt_len: int) -> None:
        self._reset_state(idx)
        self.prompt_lens[idx] = prompt_len

    def clear(self, idx: int) -> torch.Tensor:
        host_indices = self.req_to_tokens_host[idx][self.req_to_tokens_host[idx] != -1]
        self._reset_state(idx)
        return host_indices

    def init_topk_indices(self, req_pool_idx: int, req_to_token_pool) -> None:
        # Store device indices
        if self.page_size > 1:
            num_pages = self.device_buffer_cnt // self.page_size
            page_starts = torch.arange(num_pages, device=self.device) * self.page_size
            page_indices = (
                req_to_token_pool.req_to_token[req_pool_idx, page_starts]
                // self.page_size
            )
            self.last_device_indices[req_pool_idx] = page_indices
            self.last_top_k_result[req_pool_idx] = torch.arange(
                num_pages, device=self.device
            )
        else:
            indices_len = self.device_buffer_cnt
            self.last_device_indices[req_pool_idx] = req_to_token_pool.req_to_token[
                req_pool_idx, :indices_len
            ]
            self.last_top_k_result[req_pool_idx] = torch.arange(
                indices_len, device=self.device
            )
        self.hierarchical_sparse_enabled[req_pool_idx] = True


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    backend: str
    algorithm: str
    page_size: int = 64
    topk_tokens_cnt: int = 2048  # Top-k tokens selected by sparse algorithm
    device_buffer_cnt: int = 2048  # Device buffer size for LRU management
    sparse_extra_config: dict = field(
        default_factory=dict
    )  # Algorithm-specific config, parsed by each algorithm


class SparseCoordinator:
    """
    Coordinator for sparse attention with retrievable KV cache compression.

    This coordinator framework is designed for decode-phase retrievable algorithms
    (e.g., Quest, PQCache, SnapKV) that dynamically select important KV cache entries
    based on current queries. It manages the lifecycle of sparse attention including
    representation construction, sparse retrieval, and token offloading.

    Request Lifecycle and API Calls:
        1. Request Start:
           - on_request_begin(req) -> Register request and initialize state

        2. Prefill Phase:
           - attention_end(...)    -> Construct representations

        3. Decode Phase:
           - forward_begin(batch)  -> Wait for pending KVCache offloading
           - attention_begin(...)  -> Identify important KV, load offloaded KVCache, adapt attention metadata
           - attention_end(...)    -> Construct/update representations
           - forward_end(batch)    -> Trigger KVCache offloading

        4. Request End:
           - on_request_end(req) -> Clean up state and resources
    """

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

        self.states = RequestTrackers(
            req_to_token_pool.req_to_token.shape[0],
            device,
            end_layer - start_layer + 1,
            self.config.topk_tokens_cnt,
            self.config.device_buffer_cnt,
            self.req_to_token_pool.max_context_len,
            1 if self.algorithm.topk_mode() == "token" else self.page_size,
        )
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
            f"SparseCoordinator initialized with sparse algorithm={type(algorithm).__name__}"
        )

    def on_request_begin(self, req: "Req") -> None:
        """
        Handle request begin event. Called when a new request is created.

        Registers the request in the state tracker to enable sparse attention processing.
        """
        if req.req_pool_idx is not None:
            self.states.register(req.req_pool_idx, len(req.origin_input_ids))

            # In pd-disaggregation mode, decode node should Re-Construct representations
            self._maybe_construct_representations(
                layer_ids=list(range(self.start_layer, self.end_layer)),
                req_pool_indices=torch.tensor([req.req_pool_idx], device=self.device),
                seq_lens=torch.tensor([len(req.origin_input_ids)], device=self.device),
            )

    def trigger_async_offload_prompt_cache(self, req: "Req") -> None:
        """Trigger async offload of prompt cache."""
        if (
            req.req_pool_idx is None
            or not self.should_enable_hierarchical_sparse(
                self.states.prompt_lens[req.req_pool_idx]
            ).item()
        ):
            return

        self.sparse_kv_cache_manager.offload_prompt_kvcache(req)

    def check_prompt_offload_completion(self, tree_cache, blocking=False):
        """
        Check if the prompt KV cache offload has completed.
        If completed, update the request state and return the requests.
        """
        reqs = (
            self.sparse_kv_cache_manager.poll_prompt_offload_completion()
            if not blocking
            else self.sparse_kv_cache_manager.block_poll_prompt_offload_completion()
        )
        for req in reqs:
            assert (
                not req.hierarchical_sparse_enabled
            ), "Request should not be offloaded and truncated again"

            # Truncate KV cache after prompt is offloaded
            self._maybe_truncate_kv_cache_after_prompt_offloaded(
                req, self.req_to_token_pool, tree_cache
            )

            if req.hierarchical_sparse_enabled:
                self.states.init_topk_indices(req.req_pool_idx, self.req_to_token_pool)
        return reqs

    def on_request_end(self, req: "Req") -> None:
        """
        Handle request end event. Called when a request is completed or aborted.
        Cleans up request-specific state and releases resources.
        """
        if req.req_pool_idx is None:
            return

        self.sparse_kv_cache_manager.poll_decode_offload_completion()
        host_indices = self.states.clear(req.req_pool_idx)

        if host_indices.numel() > 0:
            # Free host indices
            self.sparse_kv_cache_manager.mem_pool_host.free(host_indices.cpu())
            req_seqlen = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            assert (
                len(host_indices) == req_seqlen
            ), f"Host indices mismatch: {len(host_indices)} != {req_seqlen}"

    def forward_begin(self, forward_batch: "ForwardBatch") -> None:
        """
        Handle forward pass begin event. Called before each forward pass starts.

        Wait for pending KVCache offloading operations to complete before forward pass.
        Ensures memory consistency for subsequent sparse attention operations.
        """
        if self._should_check_offload(forward_batch):
            self.sparse_kv_cache_manager.poll_decode_offload_completion()
            req_pool_indices = forward_batch.req_pool_indices
            if req_pool_indices.numel() > 0:
                sparse_mask = self._compute_sparse_mask(req_pool_indices)
                forward_batch._sparse_mask = sparse_mask
                forward_batch._sparse_mask_i32 = sparse_mask.to(torch.int32)
                forward_batch._sparse_any = bool(sparse_mask.any().item())
                forward_batch._sparse_all = bool(sparse_mask.all().item())
                forward_batch._req_pool_indices_i32 = req_pool_indices.to(torch.int32)
                forward_batch._seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)

    def forward_end(self, forward_batch: "ForwardBatch") -> None:
        """
        Handle forward pass end event. Called after each forward pass completes.

        Trigger async KVCache offloading operations.
        """
        if not self._should_check_offload(forward_batch):
            return

        offload_mask = self.should_enable_hierarchical_sparse(
            self.states.prompt_lens[forward_batch.req_pool_indices]
        )
        if offload_mask.any():
            self.sparse_kv_cache_manager.offload_decode_token_kvcache(
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

    def should_hook_attention(self) -> bool:
        """Check if attention hooks should be enabled for sparse attention."""
        return not isinstance(self.algorithm, DeepSeekNSAAlgorithm)

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
        """
        Handle attention begin event. Called before each attention pass starts.

        Identify important KV entries via sparse algorithm, load offloaded KVCache if needed,
        and adapt attention metadata for the attention backend.
        """
        if attn_metadata is None:
            return

        if not forward_batch.forward_mode.is_decode_or_idle():
            return attn_metadata

        sparse_any = getattr(forward_batch, "_sparse_any", None)
        if sparse_any is None:
            sparse_any = bool(
                self._compute_sparse_mask(forward_batch.req_pool_indices).any().item()
            )
            forward_batch._sparse_any = sparse_any

        if not sparse_any:
            return attn_metadata

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
        """
        Handle attention end event. Called after each attention pass completes.

        Maybe construct and update sparse representations.
        """
        layer_id = layer.layer_id

        # Maybe construct representations
        self._maybe_construct_representations(
            layer_ids=[layer_id],
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
        )

        # Maybe update representations
        self.algorithm.update_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

    def _maybe_construct_representations(
        self, layer_ids: List[int], req_pool_indices, seq_lens
    ) -> None:
        for layer_id in layer_ids:
            self.algorithm.construct_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
                forward_batch=None,
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
        sparse_mask = getattr(forward_batch, "_sparse_mask", None)
        if sparse_mask is None or sparse_mask.shape[0] != req_pool_indices.shape[0]:
            sparse_mask = self._compute_sparse_mask(req_pool_indices)
            forward_batch._sparse_mask = sparse_mask
            forward_batch._sparse_mask_i32 = sparse_mask.to(torch.int32)

        selected_indices, valid_lengths = self.algorithm.retrieve_topk(
            queries=query,
            layer_id=layer.layer_id,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            sparse_mask_i32=getattr(forward_batch, "_sparse_mask_i32", None),
            req_pool_indices_i32=getattr(forward_batch, "_req_pool_indices_i32", None),
            seq_lens_i32=getattr(forward_batch, "_seq_lens_i32", None),
            forward_batch=forward_batch,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Adapt Attention Metadata
        result = self.backend_adaptor.adapt_for_attn_metadata(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=attn_metadata,
            forward_batch=forward_batch,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_size=self.page_size,
            layer_id=layer.layer_id,
        )
        return result

    def _maybe_truncate_kv_cache_after_prompt_offloaded(
        self, req: "Req", req_to_token_pool, tree_cache
    ):
        """
        Truncate device KV cache after prompt is offloaded to host.

        Two strategies based on page_size:
        - page_size > 1: Keep prefix + last page, free middle (page-based caching)
        - page_size == 1: Keep first target_len tokens, free rest (token-wise caching)
        """
        if req.is_chunked > 0 or req.finished() or req.hierarchical_sparse_enabled:
            return

        kv_keep_len = self.get_hierarchical_sparse_truncated_len()
        allocated_len = req.kv_allocated_len
        page_size = tree_cache.page_size

        if allocated_len < kv_keep_len:
            return

        if allocated_len == kv_keep_len:
            req.hierarchical_sparse_enabled = True
            return

        if page_size > 1 and self.algorithm.topk_mode() == "page":
            # Page-based truncation: align and keep prefix + last page
            if allocated_len % page_size != 0:
                slots_needed = page_size - (allocated_len % page_size)
                last_slot = req_to_token_pool.req_to_token[
                    req.req_pool_idx, allocated_len - 1
                ]

                remaining_slots = torch.arange(
                    last_slot + 1,
                    last_slot + 1 + slots_needed,
                    dtype=torch.int32,
                )

                req_to_token_pool.req_to_token[
                    req.req_pool_idx, allocated_len : allocated_len + slots_needed
                ] = remaining_slots
                allocated_len += slots_needed
                req.kv_allocated_len = allocated_len

            keep_prefix_len = kv_keep_len - page_size
            last_page_start = allocated_len - page_size

            free_indices = req_to_token_pool.req_to_token[
                req.req_pool_idx, keep_prefix_len:last_page_start
            ]
            if len(free_indices) > 0:
                tree_cache.token_to_kv_pool_allocator.free(free_indices)

            last_page_slots = req_to_token_pool.req_to_token[
                req.req_pool_idx, last_page_start:allocated_len
            ].clone()
            req_to_token_pool.req_to_token[
                req.req_pool_idx, keep_prefix_len:kv_keep_len
            ] = last_page_slots

            logger.info(
                f"Page-based truncated req {req.req_pool_idx}: allocated={allocated_len} -> {kv_keep_len}, freed [{keep_prefix_len}:{last_page_start}]"
            )
        elif self.algorithm.topk_mode() == "token":
            # Token-wise truncation: keep first kv_keep_len tokens
            free_start = kv_keep_len
            free_indices = req_to_token_pool.req_to_token[
                req.req_pool_idx, free_start:allocated_len
            ]

            if len(free_indices) > 0:
                tree_cache.token_to_kv_pool_allocator.free(free_indices)
        else:
            raise ValueError(f"Invalid topk mode: {self.algorithm.topk_mode()}")

        req.kv_committed_len = kv_keep_len
        req.kv_allocated_len = kv_keep_len
        req.prefix_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_keep_len
        ].to(dtype=torch.int64, copy=True)
        req.hierarchical_sparse_enabled = True

    def alloc_for_hierarchical_sparse_decode(
        self, batch: "ScheduleBatch", token_per_req: int, alloc_tokens_func
    ) -> torch.Tensor:
        """
        Allocate KV cache for hierarchical sparse decode batch and write to req_to_token_pool.
        """
        bs = batch.seq_lens.shape[0]
        seq_lens_next = batch.seq_lens + token_per_req
        page_size = batch.tree_cache.page_size
        req_pool_indices = batch.req_pool_indices

        if batch.model_config.is_encoder_decoder:
            locs = batch.encoder_lens + batch.seq_lens
        else:
            locs = batch.seq_lens.clone()

        # Find truncated and non-truncated requests
        kv_truncated_len = self.get_hierarchical_sparse_truncated_len()
        hierarchical_sparse_masks = self.states.hierarchical_sparse_enabled[
            req_pool_indices
        ]
        out_cache_loc = torch.empty(bs, dtype=torch.int32, device=batch.device)

        # Handle truncated requests: reuse a pre-allocated rolling page per request
        if hierarchical_sparse_masks.any():
            truncated_indices = hierarchical_sparse_masks.nonzero(as_tuple=True)[0]
            if self.algorithm.topk_mode() == "page":
                # Map logical decode positions onto the reserved rolling page in a round-robin way
                decode_offsets = locs[truncated_indices]
                rolling_positions = (kv_truncated_len - page_size) + (
                    decode_offsets % page_size
                )
            elif self.algorithm.topk_mode() == "token":
                # Token-wise, Using the fixed last location
                rolling_positions = kv_truncated_len - 1

            out_cache_loc[truncated_indices] = batch.req_to_token_pool.req_to_token[
                batch.req_pool_indices[truncated_indices],
                rolling_positions,
            ]

        # Handle non-truncated requests: allocate normally
        if (~hierarchical_sparse_masks).any():
            non_truncated_indices = (~hierarchical_sparse_masks).nonzero(as_tuple=True)[
                0
            ]
            if batch.tree_cache.page_size == 1:
                non_truncated_out = alloc_tokens_func(
                    batch.tree_cache, len(non_truncated_indices) * token_per_req
                )
            else:
                non_truncated_last_loc = batch.req_to_token_pool.req_to_token[
                    batch.req_pool_indices[non_truncated_indices],
                    batch.seq_lens[non_truncated_indices] - 1,
                ]
                non_truncated_out = alloc_tokens_func(
                    tree_cache=batch.tree_cache,
                    seq_lens=seq_lens_next[non_truncated_indices],
                    seq_lens_cpu=batch.seq_lens_cpu[non_truncated_indices.cpu()]
                    + token_per_req,
                    last_loc=non_truncated_last_loc,
                    token_per_req=token_per_req,
                )

            out_cache_loc[non_truncated_indices] = non_truncated_out.to(torch.int32)
            batch.req_to_token_pool.write(
                (
                    batch.req_pool_indices[non_truncated_indices],
                    locs[non_truncated_indices],
                ),
                out_cache_loc[non_truncated_indices],
            )
        return out_cache_loc

    def get_hierarchical_sparse_truncated_len(self) -> Optional[int]:
        return self.states.device_buffer_cnt

    def should_enable_hierarchical_sparse(
        self, prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        """Get mask indicating which requests should enable hierarchical sparse."""
        return prompt_lens >= self.states.device_buffer_cnt

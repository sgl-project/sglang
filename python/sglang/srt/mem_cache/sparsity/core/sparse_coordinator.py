import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor

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

        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )

        # TODO: Add more trackers for hierarchical KVCache management

    def register(self, idx: int, prompt_len: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.last_constructed_page[idx] = 0

    def clear(self, idx: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    backend: str
    algorithm: str
    page_size: int = 64
    min_sparse_prompt_len: int = 2048
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
        start_layer: int,
        end_layer: int,
        device: torch.device,
    ):
        self.config = config
        self.algorithm = algorithm
        self.backend_adaptor = backend_adaptor
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.page_size = config.page_size

        self.states = RequestTrackers(
            req_to_token_pool.req_to_token.shape[0],
            device,
            end_layer - start_layer + 1,
            self.config.min_sparse_prompt_len,
            self.req_to_token_pool.max_context_len,
        )

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

    def on_request_end(self, req: "Req") -> None:
        """
        Handle request end event. Called when a request is completed or aborted.
        Cleans up request-specific state and releases resources.
        """
        if req.req_pool_idx is None:
            return

        self.states.clear(req.req_pool_idx)

        # TODO: Implement request end handling
        # - Release host indices if any were allocated for offloading

    def forward_begin(self, forward_batch: "ForwardBatch") -> None:
        """
        Handle forward pass begin event. Called before each forward pass starts.

        Wait for pending KVCache offloading operations to complete before forward pass.
        Ensures memory consistency for subsequent sparse attention operations.
        """
        # TODO: Implement forward begin handling
        # - Check if there are pending offloading operations
        pass

    def forward_end(self, forward_batch: "ForwardBatch") -> None:
        """
        Handle forward pass end event. Called after each forward pass completes.

        Trigger async KVCache offloading operations.
        """
        # TODO: Implement forward end handling
        # - Identify tokens to offload
        # - Trigger async offloading operations
        pass

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

        return mask

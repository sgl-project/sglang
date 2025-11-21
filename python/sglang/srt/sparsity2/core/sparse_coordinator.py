import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.sparsity2.algorithms.base_algorithm import (
    BaseSparseAlgorithm,
    SparseMode,
)
from sglang.srt.sparsity2.backend.backend_adaptor import BackendAdaptor
from sglang.srt.sparsity2.core.representation_pool import RepresentationPool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RequestTrackers:
    """Tensor-based state tracker for batch requests."""

    def __init__(self, max_pool_size: int, device: torch.device, num_layers: int):
        self.device = torch.device
        self.num_layers = num_layers
        self.top_k = 2048  # hardcode
        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.decode_steps = torch.zeros(max_pool_size, dtype=torch.int32, device=device)
        self.last_extracted_token = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )

        self.full_host_indices = [
            torch.tensor([], dtype=torch.int64, device=device)
            for _ in range(max_pool_size)
        ]
        self.prev_top_k_result = torch.zeros(
            [max_pool_size, num_layers, self.top_k], dtype=torch.int64, device=device
        )
        self.prev_device_indices = torch.zeros(
            [max_pool_size, num_layers, self.top_k], dtype=torch.int64, device=device
        )
        self.extra_2_token = torch.zeros(
            [max_pool_size, 2], dtype=torch.int64, device=device
        )

    def register(self, idx: int, prompt_len: int):
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.decode_steps[idx] = 0
        self.last_extracted_token[idx] = 0
        self.full_host_indices[idx] = torch.tensor(
            [], dtype=torch.int64, device=self.device
        )

    def clear(self, idx: int):
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.decode_steps[idx] = 0
        self.last_extracted_token[idx] = 0
        host_indices = self.full_host_indices[idx]
        return host_indices


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    backend: str = "flashattention"
    algorithm: str = "fake_random"
    page_size: int = 64
    sparse_ratio: float = 0.9
    min_sparse_prompt_len: int = 200


class SparseCoordinator:
    """
    Orchestrate the sparse attention pipeline.
    Manages request lifecycle, triggers algorithm phases, coordinates pool and backend.
    """

    def __init__(
        self,
        config: SparseConfig,
        algorithm: BaseSparseAlgorithm,
        backend_adaptor: Optional[BackendAdaptor],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        decode_offload_manager: Any,
        start_layer: int,
        end_layer: int,
        device: torch.device,
        total_num_pages: int,
    ):
        self.algorithm = algorithm
        self.backend_adaptor = backend_adaptor
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.page_size = config.page_size
        self.config = config
        self.decode_offload_manager = decode_offload_manager

        max_pool_size = req_to_token_pool.req_to_token.shape[0]
        self.states = RequestTrackers(
            max_pool_size=max_pool_size,
            device=device,
            num_layers=end_layer - start_layer,
        )
        self.decode_offload_manager.req_states = self.states

        self.repr_pool = RepresentationPool(
            total_num_pages=total_num_pages,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )
        self._register_representation_storage()

        if self.backend_adaptor is not None:
            self._backend_adaptor_method = (
                self.backend_adaptor.adapt_for_page_wise
                if algorithm.get_sparse_mode() == SparseMode.PAGE_WISE
                else self.backend_adaptor.adapt_for_token_wise
            )
        logger.info(
            f"SparseCoordinatorV2 initialized: algorithm={type(algorithm).__name__}, "
            f"mode={algorithm.get_sparse_mode().value}, "
            f"storage_slots={self.repr_pool.get_storage_names() if self.repr_pool else []}"
        )

    def _register_representation_storage(self):
        """Register algorithm's storage requirements to pool."""
        storage_specs = self.algorithm.get_representation_storage_shape(
            self.token_to_kv_pool
        )
        for storage_name, (shape, dtype) in storage_specs.items():
            self.repr_pool.register_storage(storage_name, shape, dtype)
        self.algorithm.set_pools(self.repr_pool, self.req_to_token_pool)

    def request_begin(self, req: "Req") -> None:
        if req.req_pool_idx is not None:
            prompt_len = len(req.origin_input_ids)
            self.states.register(req.req_pool_idx, prompt_len)
            logger.info(f"Request {req.rid} started with prompt length {prompt_len}")

    def request_end(self, req: "Req") -> None:
        if req.req_pool_idx is not None:
            host_indices = self.states.clear(req.req_pool_idx)
            self.decode_offload_manager.decode_host_mem_pool.free(host_indices)
            logger.info(f"Request {req.rid} ended")

    def offload_last_token_kv_cache(self, forward_batch: "ForwardBatch"):
        return self.decode_offload_manager.offload_sparse_req_tokens(
            forward_batch.req_pool_indices
        )

    def check_last_token_offload_progress(self, forward_batch: "ForwardBatch"):
        return self.decode_offload_manager.check_sparse_offload_progress(
            forward_batch.req_pool_indices
        )

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Any,
        **kwargs,
    ) -> Optional[Any]:
        """
        Apply sparsity during decode.
        """
        if not forward_batch.forward_mode.is_decode_or_idle():
            return None
        if forward_batch.spec_info is not None:
            return None

        # Save original metadata in first layer
        if layer.layer_id == self.start_layer and self.backend_adaptor is not None:
            self.backend_adaptor.save_original_metadata(attn_metadata)

        return self._handle_sparse_retrieve(query, layer, forward_batch, attn_metadata)

    def attention_end(
        self,
        output: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> None:
        """
        Extract representations after attention computation
        TODO:
            1. Support Asynchronous representation extraction
        """

        layer_id = layer.layer_id
        req_pool_indices = forward_batch.req_pool_indices

        # Construct representations if needed
        self._maybe_construct_representations(layer_id, forward_batch)

        # Incremental update representations in decode step if needed
        if (
            layer_id == self.start_layer
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            self.states.decode_steps[req_pool_indices] += 1
        # TODO: Prefill requires updating meta information in req states.

        self._maybe_incremental_udpate_representations(layer_id, forward_batch)

    def _handle_sparse_retrieve(
        self,
        query: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Any,
    ) -> None:
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices

        # Only retrieve for requests that have constructed representations and are in decode mode
        sparse_mask = (
            self.states.repr_constructed[req_pool_indices]
            & (
                self.states.prompt_lens[req_pool_indices]
                >= self.config.min_sparse_prompt_len
            )
            & forward_batch.forward_mode.is_decode()
        )

        if not sparse_mask.any():
            return None

        q_flat = query.view(bs, -1)
        kv_cache_lens = attn_metadata.cache_seqlens_int32

        # Retrieve topk indices for each request
        selected_indices, valid_lengths = self.algorithm.retrieve_topk(
            queries=q_flat,
            layer_id=layer.layer_id,
            req_pool_indices=req_pool_indices,
            seq_lens=kv_cache_lens,
            sparse_mask=sparse_mask,
        )

        # prefetch data from cpu
        curr_device_indices = self.decode_offload_manager.transform_sparse_top_k_cache(
            req_pool_indices=req_pool_indices,
            req_states=self.states,
            top_k_result=selected_indices,
            layer_id=layer.layer_id,
            valid_lengths=valid_lengths,
        )

        # Adapt metadata for sparse attention
        if self.backend_adaptor is not None:
            self._backend_adaptor_method(
                selected_indices=selected_indices,
                valid_lengths=valid_lengths,
                sparse_mask=sparse_mask,
                current_metadata=attn_metadata,
                forward_batch=forward_batch,
                req_to_token=self.req_to_token_pool.req_to_token,
                page_size=self.page_size,
                layer_id=layer.layer_id,
            )

    def _maybe_construct_representations(
        self, layer_id: int, forward_batch: "ForwardBatch"
    ) -> None:
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens

        construct_mask = self.algorithm.should_construct_representations(
            forward_batch=forward_batch,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            repr_constructed=self.states.repr_constructed,
            prompt_lens=self.states.prompt_lens,
        )

        if construct_mask.any():
            success_mask = self.algorithm.construct_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                construct_mask=construct_mask,
                k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            )

            if layer_id == self.end_layer - 1:
                construct_indices = req_pool_indices[construct_mask & success_mask]
                self.states.repr_constructed[construct_indices] = True
                updated_seq_lens = seq_lens[construct_mask & success_mask]
                self.states.last_extracted_token[construct_indices] = (
                    updated_seq_lens // self.page_size * self.page_size
                )

    def _maybe_incremental_udpate_representations(
        self, layer_id: int, forward_batch: "ForwardBatch"
    ):
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        is_decode = forward_batch.forward_mode.is_decode_or_idle()

        update_mask = self.algorithm.should_update_represetations(
            forward_batch=forward_batch,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            is_decode=is_decode,
            repr_constructed=self.states.repr_constructed,
            decode_steps=self.states.decode_steps,
        )

        if update_mask.any():
            success_mask = self.algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                update_mask=update_mask,
                k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
                last_extracted_tokens=self.states.last_extracted_token,
            )

            if layer_id == self.end_layer - 1 and success_mask.any():
                update_indices = req_pool_indices[success_mask]
                updated_seq_lens = seq_lens[success_mask]
                self.states.last_extracted_token[update_indices] = (
                    updated_seq_lens // self.page_size * self.page_size
                )
                self.states.decode_steps[update_indices] = 0

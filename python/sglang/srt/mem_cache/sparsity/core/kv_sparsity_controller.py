"""Lifecycle controller for post-hoc KV-cache sparsity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.sparsity.backend.visibility_adaptor import MetadataAdaptor
from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import (
    DecisionScope,
    KVStateAction,
    RequestIdentity,
    SelectionContext,
    SelectionResult,
)
from sglang.srt.mem_cache.sparsity.policies.base import SparsityPolicy

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class KVSparsityController:
    """Coordinate policy selection and backend metadata adaptation.

    The initial controller accepts visibility-only policies and never mutates KV
    allocation ownership. Request state, when required by a policy, is keyed by
    both pool slot and ``ReqToTokenPool.req_generation``.
    """

    def __init__(
        self,
        config: KVSparsityConfig,
        policy: SparsityPolicy,
        adaptor: MetadataAdaptor,
        req_to_token_pool: ReqToTokenPool,
        start_layer: int,
        end_layer: int,
    ):
        if policy.capabilities.state_action != KVStateAction.VISIBILITY_ONLY:
            raise ValueError("the initial KV sparsity controller is visibility-only")
        if policy.capabilities.scope == DecisionScope.REQUEST:
            raise ValueError("request-scoped selection caching is not implemented yet")
        self.config = config
        self.policy = policy
        self.adaptor = adaptor
        self.req_to_token_pool = req_to_token_pool
        self.start_layer = max(start_layer, config.start_layer)
        configured_end = end_layer if config.end_layer == -1 else config.end_layer
        self.end_layer = min(end_layer, configured_end)
        if self.end_layer <= self.start_layer:
            raise ValueError("configured sparse layer range is empty")

        # Keep the metadata object itself, not only id(metadata): Python may
        # reuse an id immediately after the previous eager metadata is freed.
        self._metadata: Any = None
        self._selection_cache: Optional[SelectionResult] = None
        self._request_generations: Optional[torch.Tensor] = None
        self._request_identities: dict[int, RequestIdentity] = {}
        self._dense_metadata_captured = False

    def applies_to_layer(self, layer_id: int) -> bool:
        return self.start_layer <= layer_id < self.end_layer

    def bind_attention_backend(self, attention_backend: Any) -> None:
        self.adaptor.bind_attention_backend(attention_backend)

    def _sync_request_identities(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        if not self.policy.capabilities.requires_request_state:
            return None

        pool_indices_cpu = forward_batch.req_pool_indices.detach().to("cpu").long()
        generations = self.req_to_token_pool.req_generation[pool_indices_cpu]
        for pool_index, generation in zip(
            pool_indices_cpu.tolist(), generations.tolist()
        ):
            identity = RequestIdentity(pool_index=pool_index, generation=generation)
            previous = self._request_identities.get(pool_index)
            if previous == identity:
                continue
            if previous is not None:
                self.policy.on_request_end(previous)
            self.policy.on_request_begin(identity)
            self._request_identities[pool_index] = identity
        return generations

    def _make_context(
        self,
        *,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        output: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: Any,
    ) -> SelectionContext:
        return SelectionContext(
            query=query,
            key=key,
            value=value,
            output=output,
            layer_id=layer.layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_batch=forward_batch,
            attention_layer=layer,
            request_generations=self._request_generations,
            metadata=metadata,
        )

    def _begin_forward(
        self,
        metadata: Any,
        forward_batch: ForwardBatch,
        context: SelectionContext,
    ) -> None:
        self._metadata = metadata
        self._selection_cache = None
        self._request_generations = self._sync_request_identities(forward_batch)
        context = self._make_context(
            query=context.query,
            key=context.key,
            value=context.value,
            output=None,
            layer=context.attention_layer,
            forward_batch=forward_batch,
            metadata=metadata,
        )
        self.policy.begin_forward(context)
        self._dense_metadata_captured = forward_batch.forward_mode.is_decode()
        if self._dense_metadata_captured:
            self.adaptor.capture_dense_metadata(metadata)

    def before_attention(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: Any,
    ) -> Any:
        if not self.applies_to_layer(layer.layer_id):
            return metadata
        context = self._make_context(
            query=query,
            key=key,
            value=value,
            output=None,
            layer=layer,
            forward_batch=forward_batch,
            metadata=metadata,
        )
        if metadata is not self._metadata:
            self._begin_forward(metadata, forward_batch, context)

        context = self._make_context(
            query=query,
            key=key,
            value=value,
            output=None,
            layer=layer,
            forward_batch=forward_batch,
            metadata=metadata,
        )
        if not forward_batch.forward_mode.is_decode():
            return metadata
        scope = self.policy.capabilities.scope
        if scope == DecisionScope.LAYER or self._selection_cache is None:
            result = self.policy.select(context)
            if scope != DecisionScope.LAYER:
                self._selection_cache = result
        else:
            # FA3 metadata is shared across layers and remains sparse until the
            # configured final layer. A STEP decision therefore needs one
            # adaptor rewrite, not one rewrite per layer.
            return metadata
        return self.adaptor.apply(result, metadata, forward_batch)

    def after_attention(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        output: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: Any,
    ) -> None:
        if not self.applies_to_layer(layer.layer_id):
            return
        context = self._make_context(
            query=query,
            key=key,
            value=value,
            output=output,
            layer=layer,
            forward_batch=forward_batch,
            metadata=metadata,
        )
        self.policy.on_attention_complete(context)
        if self._dense_metadata_captured and layer.layer_id == self.end_layer - 1:
            self.adaptor.restore_dense_metadata(metadata)
            self._dense_metadata_captured = False

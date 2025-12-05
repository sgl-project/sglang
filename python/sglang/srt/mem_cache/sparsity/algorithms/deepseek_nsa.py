import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import nvtx
import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithm,
    SparseMode,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class DeepSeekNSAAlgorithm(BaseSparseAlgorithm):
    """Sparse attention algorithm for DeepSeek NSA."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.index_topk = getattr(config, "index_topk", 2048)
        logger.info(f"DeepSeekNSAAlgorithm initialized: index_topk={self.index_topk}")

    def get_sparse_mode(self) -> SparseMode:
        return SparseMode.ORIGINAL_WISE

    def get_representation_storage_shape(self, token_to_kv_pool) -> Dict[str, tuple]:
        return {}

    @nvtx.annotate("DeepSeekNSAAlgorithm.retrieve_topk", color="green")
    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> tuple:
        indexer: Optional["Indexer"] = kwargs.get("indexer")
        forward_batch: Optional["ForwardBatch"] = kwargs.get("forward_batch")
        x, q_lora, positions = (
            kwargs.get("x"),
            kwargs.get("q_lora"),
            kwargs.get("positions"),
        )

        if any(v is None for v in [indexer, x, q_lora, positions, forward_batch]):
            raise ValueError("Required: indexer, x, q_lora, positions, forward_batch")

        try:
            # Using the nsa's original indexer to get the topk indices.
            topk_indices = indexer(
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
            )

            if topk_indices is None:
                return self._empty_result(queries.shape[0], queries.device)

            return topk_indices, None
        except Exception as e:
            logger.error(f"Layer {layer_id} NSA indexer failed: {e}", exc_info=True)
            return self._empty_result(queries.shape[0], queries.device)

    def _empty_result(self, batch_size: int, device: torch.device) -> tuple:
        selected_indices = torch.full(
            (batch_size, self.index_topk), -1, dtype=torch.int32, device=device
        )
        valid_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        return selected_indices, valid_lengths

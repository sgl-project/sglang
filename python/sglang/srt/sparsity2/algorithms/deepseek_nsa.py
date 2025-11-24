import logging
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.sparsity2.algorithms.base_algorithm import (
    BaseSparseAlgorithm,
    SparseMode,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class DeepSeekNSAAlgorithm(BaseSparseAlgorithm):
    """
    Token-wise sparse algorithm for DeepSeek NSA.
    Directly delegates TopK computation to the NSA Indexer.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.index_topk = getattr(config, "index_topk", 2048)
        logger.info(
            f"DeepSeekNSAAlgorithm initialized with index_topk={self.index_topk}"
        )

    def get_sparse_mode(self) -> SparseMode:
        return SparseMode.ORIGINAL_WISE

    def get_representation_storage_shape(self, token_to_kv_pool) -> Dict[str, tuple]:
        """NSA doesn't need separate representation storage."""
        return {}

    # def should_construct_representations(
    #     self,
    #     forward_batch: "ForwardBatch",
    #     layer_id: int,
    #     req_pool_indices: torch.Tensor,
    #     seq_lens: torch.Tensor,
    #     repr_constructed: torch.Tensor,
    #     prompt_lens: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Check if the representations pool should be constructed for the given requests.
    #     Returns:
    #         [bs] bool mask - True only when:
    #             1. repr_constructed[req_pool_indices] == False
    #             2. seq_lens >= prompt_lens[req_pool_indices]
    #     """
    #     mask = ~repr_constructed[req_pool_indices] & (
    #         seq_lens >= prompt_lens[req_pool_indices]
    #     )
    #     return mask

    # def construct_representations(
    #     self,
    #     layer_id: int,
    #     req_pool_indices: torch.Tensor,
    #     seq_lens: torch.Tensor,
    #     construct_mask: torch.Tensor,
    #     k_buffer: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     NSA doesn't need explicit representation construction.
    #     Returns:
    #         [bs] bool success mask
    #     """
    #     return construct_mask.clone()

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Retrieve topk indices using NSA Indexer.

        Args:
            **kwargs: Should contain:
                - indexer: NSA Indexer instance
                - forward_batch: ForwardBatch
                - x: hidden_states
                - q_lora: q_lora tensor
                - positions: position tensor

        Returns:
            selected_indices: [bs, index_topk] topk token indices
            valid_lengths: [bs] actual valid token count per request
        """
        indexer: Optional["Indexer"] = kwargs.get("indexer")
        forward_batch: Optional["ForwardBatch"] = kwargs.get("forward_batch")
        x = kwargs.get("x")
        q_lora = kwargs.get("q_lora")
        positions = kwargs.get("positions")

        if any(v is None for v in [indexer, x, q_lora, positions, forward_batch]):
            raise ValueError(
                "Required parameters: indexer, x, q_lora, positions, forward_batch"
            )

        try:
            # Call NSA indexer to compute topk indices
            topk_indices = indexer(
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
            )

            if topk_indices is None:
                if layer_id == 0:
                    logger.info("NSA indexer returned None (likely short sequence)")
                return self._empty_result(queries.shape[0], queries.device)

            valid_lengths = (topk_indices != -1).sum(dim=1).to(torch.int32)
            if layer_id == 0:
                logger.info(
                    f"NSA topk computed: shape={topk_indices.shape}, "
                    f"valid_lengths={valid_lengths.tolist()[:10]}..."
                )

            return topk_indices, valid_lengths

        except Exception as e:
            logger.error(
                f"Layer {layer_id}: NSA indexer call failed: {e}", exc_info=True
            )
            return self._empty_result(queries.shape[0], queries.device)

    def _empty_result(self, batch_size: int, device: torch.device) -> tuple:
        """Return empty topk result."""
        selected_indices = torch.full(
            (batch_size, self.index_topk), -1, dtype=torch.int32, device=device
        )
        valid_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        return selected_indices, valid_lengths

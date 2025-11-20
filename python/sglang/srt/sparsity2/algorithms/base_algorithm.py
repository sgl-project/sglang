from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING
import logging

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sparsity2.core.sparse_coordinator import RequestState

logger = logging.getLogger(__name__)


class SparseMode(Enum):
    """Sparse attention granularity mode."""

    PAGE_WISE = "page_wise"
    TOKEN_WISE = "token_wise"


class BaseSparseAlgorithm(ABC):
    """Abstract base class for sparse attention algorithms."""

    def __init__(self, config, device: torch.device, **kwargs):
        self.config = config
        self.device = device
        self.repr_pool = None
        self.req_to_token = None

    def set_pools(self, repr_pool, req_to_token_pool):
        """Set representation pool for zero-copy access."""
        self.repr_pool = repr_pool
        self.req_to_token_pool = req_to_token_pool

    @abstractmethod
    def get_sparse_mode(self) -> SparseMode:
        """Return the sparsity granularity mode."""
        pass

    @abstractmethod
    def get_representation_storage_shape(self, token_to_kv_pool) -> Dict[str, tuple]:
        """
        Register storage slots needed by this algorithm.

        Returns:
            Dict mapping storage_name to (shape, dtype).
            Example: {
                "page_repr": ((num_heads * head_dim,), torch.float32),
                "centroids": ((n_subvec, n_centroids, subvec_dim), torch.float32),
                "token_codes": ((page_size, n_subvec), torch.long)
            }
        """
        pass

    def should_construct_representations(
        self,
        forward_batch: "ForwardBatch",
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        repr_constructed: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if the representations pool should be constructed for the given requests.
        Args:
            forward_batch: Forward batch info
            layer_id: Current layer
            req_pool_indices: [bs]
            seq_lens: [bs]
            repr_constructed: [max_pool_size] global state
            prompt_lens: [max_pool_size] global state

        Returns:
            [bs] bool mask
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        construct_mask: torch.Tensor,
        k_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct representations for the given requests.
        Args:
            layer_id: Current layer
            req_pool_indices: [bs]
            seq_lens: [bs]
            construct_mask: [bs] bool
            k_buffer: KV cache buffer

        Returns:
            [bs] bool success mask
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)

    @abstractmethod
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
        Retrieve topk indices for the given requests.
        Args:
            queries: [bs, repr_dim]
            layer_id: layer index
            req_pool_indices: [bs]
            seq_lens: [bs] sequence lengths
            sparse_mask: [bs] bool, which requests need to be retrieved

        Returns:
            selected_indices: [bs, max_selected] padded with -1
            valid_lengths: [bs] actual selected count per request
        """
        pass

    def should_update_represetations(
        self,
        forward_batch: Any,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        is_decode: bool,
        repr_constructed: torch.Tensor,
        decode_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if the representations pool should be updated for the given requests.

        Args:
            forward_batch: Forward batch info
            layer_id: Current layer
            req_pool_indices: [bs]
            is_decode: Whether in decode mode
            repr_constructed: [max_pool_size] global state
            decode_steps: [max_pool_size] global state

        Returns:
            [bs] bool mask
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        update_mask: torch.Tensor,
        k_buffer: torch.Tensor,
        last_extracted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update representations for the given requests.

        Args:
            layer_id: Current layer
            req_pool_indices: [bs]
            seq_lens: [bs]
            update_mask: [bs] bool
            k_buffer: KV cache buffer
            req_to_token: Mapping tensor
            last_extracted_tokens: [max_pool_size] global state

        Returns:
            success_mask: [bs] bool, which requests successfully updated
        """
        return torch.zeros_like(req_pool_indices, dtype=torch.bool)


class FakeRandomSparseAlgorithm(BaseSparseAlgorithm):
    """
    Fake random TOKEN_WISE sparse algorithm for testing.
    Strategy: Keep recent N tokens + randomly select 80% from historical tokens.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.sparse_ratio = 1
        self.num_recent_tokens = 256
        self.min_sparse_prompt_len = 128
        logger.info(
            f"FakeRandomSparseAlgorithm initialized: sparse_ratio={self.sparse_ratio}, "
            f"num_recent_tokens={self.num_recent_tokens}"
        )

    def get_sparse_mode(self) -> SparseMode:
        return SparseMode.TOKEN_WISE

    def get_representation_storage_shape(self, token_to_kv_pool) -> Dict[str, tuple]:
        return {}

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        pass

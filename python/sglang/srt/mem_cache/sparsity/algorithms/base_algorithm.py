from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseSparseAlgorithm(ABC):
    """
    Abstract base class for sparse attention algorithms.

    This class provides a unified interface for implementing various retrievable KVCache
    compression algorithms. Token-wise sparsity is treated as page-wise with page_size=1.

    References:
        - ChunkKV: https://arxiv.org/abs/2502.00299
        - Quest: https://arxiv.org/pdf/2406.10774
        - PQCache: https://arxiv.org/abs/2407.12820
        - SnapKV: https://arxiv.org/pdf/2404.14469
        - Look-ahead QCache: https://arxiv.org/pdf/2505.20334
        - and more...
    """

    def __init__(self, config, device: torch.device, **kwargs):
        self.config = config
        self.device = device
        self.req_to_token_pool = None
        self.states = None

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ):
        """
        Initialize algorithm-specific representation pool and set context.

        Called once during SparseCoordinator initialization. Algorithms allocate
        their own representation tensors and store references to context.

        Algorithm-specific implementations:
            - ChunkKV: Allocate chunk scores [num_chunks, 1] for tracking semantic chunk importance
            - Quest: Allocate page representations [num_pages, repr_dim] via key pooling
            - PQCache: Allocate centroids [n_subvec, n_centroids, subvec_dim] and token codes [num_tokens, n_subvec]
            - SnapKV: Allocate voting scores [num_tokens] and selected positions mask for retention strategy
            - Look-ahead QCache: Allocate importance scores [num_tokens], eviction mask, and optional pseudo query cache [cache_size, hidden_dim]
        """
        pass

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """
        Construct initial representations during prefill phase.

        Called at every layer during forward pass. Algorithm internally decides
        whether to perform construction.
        Typically only constructs once per request during prefill/extend phase.

        Algorithm-specific implementations:
            - ChunkKV: Compute chunk importance scores via aggregated key L2 norms within semantic chunks
            - Quest: Compute page representations via mean pooling of keys within each page
            - PQCache: Run K-means clustering to generate centroids and assign each token to nearest centroid
            - SnapKV: Select observation window (recent tokens), compute attention weights, aggregate via voting to identify important prefix positions, apply 1D pooling to preserve context
            - Look-ahead QCache: Generate pseudo lookahead query (e.g., mean of last k queries), compute KV importance scores, mark low-importance KVs for eviction
        """
        pass

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """
        Incrementally update representations during decode phase.

        Called at every layer during forward pass. Algorithm internally decides
        whether to update based on:
        - self.states.repr_constructed[req_id]: Whether initial construction done
        - self.states.last_constructed_page[req_id]: Last constructed page index
        - Current seq_lens: To detect new tokens/pages

        Algorithm-specific implementations:
            - ChunkKV: Incrementally compute importance scores for newly generated chunks during decode
            - Quest: Incrementally compute representations for newly generated pages during decode
            - PQCache: Assign new tokens to existing centroids (no centroid update during decode)
            - SnapKV: Optional: periodically re-run voting with sliding observation window (typically static after prefill)
            - Look-ahead QCache: Periodically regenerate pseudo queries and re-evaluate importance scores to adapt to generation dynamics
        """
        pass

    @abstractmethod
    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Retrieve top-k important KV indices for sparse attention.

        Called before attention computation at each layer. Uses current query
        and pre-computed representations to select the most important subset
        of KV cache for attention computation.

        Args:
            queries: [bs, num_heads, head_dim] Current query vectors
            layer_id: Current layer index
            req_pool_indices: [bs] Request pool indices
            sparse_mask: [bs] bool, which requests need sparse attention
            attn_metadata: Attention metadata (contains seq_lens, etc.)
            **kwargs: Algorithm-specific arguments

        Returns:
            selected_indices: [bs, max_selected] Selected page/token indices, padded with -1
            valid_lengths: [bs] Actual number of selected indices per request

        Note:
            - Indices are logical positions that will be mapped to physical KV cache by BackendAdaptor

        Algorithm-specific implementations:
            - ChunkKV: Select top-k chunks based on pre-computed importance scores with layer-wise index reuse
            - Quest: Compute query-page similarity using current query and stored page representations, select top-k pages
            - PQCache: Calculate query-centroid similarity, use centroid scores to rank tokens, select top-k tokens
            - SnapKV: Return union of voted important prefix positions (with clustered neighbors) and observation window tokens
            - Look-ahead QCache: Return KVs not marked for eviction (eviction based on pseudo query importance evaluation)
        """
        pass


class BaseSparseAlgorithmImpl(BaseSparseAlgorithm):
    """
    Implementation base class for sparse attention algorithms.

    Provides common infrastructure for algorithms that operate at page/chunk granularity
    (token-wise is simply page_size=1):
    - Generic construct/update flow with state tracking
    - TopK retrieval with recent page retention (can be overridden)

    Subclasses need to implement:
    - _initialize_representation_pools(): Initialize algorithm-specific representation pools
    - _compute_page_representations(): Compute page scores/representations
    - _retrieve_page_scores(): Retrieve page scores for TopK selection

    Subclasses can also override any method for specialized behavior
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.sparsity_ratio = config.sparse_extra_config.get("sparsity_ratio", 0.7)
        self.num_recent_pages = config.sparse_extra_config.get("num_recent_pages", 4)
        self.page_size = config.page_size

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.states = states

        total_num_tokens = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
        total_num_pages = (total_num_tokens + self.page_size - 1) // self.page_size

        # Initialize algorithm-specific representation pools
        self._initialize_representation_pools(start_layer, end_layer, total_num_pages)

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:

        if not forward_batch.forward_mode.is_extend():
            return

        num_pages = seq_lens // self.page_size
        valid_mask = (
            ~self.states.repr_constructed[req_pool_indices]
            & (seq_lens >= self.states.prompt_lens[req_pool_indices])
            & (num_pages > 0)
        )

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            0,
            num_pages[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.repr_constructed[success_indices] = True
            self.states.last_constructed_page[success_indices] = num_pages[valid_mask]

    def update_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_decode_or_idle():
            return

        start_page = self.states.last_constructed_page[req_pool_indices]
        end_page = seq_lens // self.page_size
        valid_mask = self.states.repr_constructed[req_pool_indices] & (
            start_page < end_page
        )

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            end_page[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.last_constructed_page[success_indices] = end_page[valid_mask]

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Default TopK retrieval: score-based selection + recent pages.
        Subclasses can override for query-dependent retrieval.

        TODO:
            1. Using triton kernel to speed up this function
            2. Support CUDA Graph
        """
        bs, device = queries.shape[0], queries.device

        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for TopK retrieval"
            )
        if bs == 0:
            return (
                torch.empty((0, 1), dtype=torch.int32, device=device),
                torch.zeros(0, dtype=torch.int32, device=device),
            )

        seq_lens = seq_lens_source.seq_lens.to(device=device, dtype=torch.int64)
        sparse_mask_bool = sparse_mask.to(device=device, dtype=torch.bool)
        req_pool_indices = req_pool_indices.to(device=device, dtype=torch.int64)

        num_recent = max(int(self.num_recent_pages), 0)
        num_pages_all = (seq_lens + self.page_size - 1) // self.page_size
        active_mask = sparse_mask_bool & (num_pages_all > num_recent)

        recent_start_all = (num_pages_all - num_recent).clamp(min=0)
        history_pages_all = recent_start_all.clamp(min=1)
        k_all = (
            (history_pages_all.to(torch.float64) * self.sparsity_ratio)
            .to(torch.int64)
            .clamp(min=1)
        )
        k_all = torch.minimum(k_all, history_pages_all)

        active_num_pages_masked = torch.where(
            active_mask, num_pages_all, torch.zeros_like(num_pages_all)
        )
        active_k_masked = torch.where(active_mask, k_all, torch.zeros_like(k_all))
        max_pages, max_k = torch.stack(
            [active_num_pages_masked.max(), active_k_masked.max()]
        ).tolist()

        if max_pages <= 0:
            return (
                torch.full((bs, 1), -1, dtype=torch.int32, device=device),
                torch.zeros(bs, dtype=torch.int32, device=device),
            )

        active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        n_active = active_rows.shape[0]
        active_req_pool_indices = req_pool_indices[active_rows]
        active_queries = queries[active_rows]
        recent_start = recent_start_all[active_rows]
        k_per_req = k_all[active_rows]

        req_to_token = self.req_to_token_pool.req_to_token
        max_req_tokens = req_to_token.shape[1]

        page_idx = torch.arange(max_pages, device=device, dtype=torch.int64).unsqueeze(
            0
        )
        token_pos = (page_idx * self.page_size).clamp(0, max_req_tokens - 1)
        page_start_token = req_to_token[active_req_pool_indices.unsqueeze(1), token_pos]
        phys_pages = page_start_token // self.page_size

        scores = self._retrieve_page_scores(
            layer_id,
            phys_pages,
            active_req_pool_indices,
            active_queries,
        ).clone()

        if scores.is_floating_point():
            scores = scores.clamp(min=torch.finfo(scores.dtype).min)
        scores.masked_fill_(page_idx >= recent_start.unsqueeze(1), float("-inf"))

        topk_idx = torch.topk(scores, k=max_k, dim=1, sorted=True)[1]
        k_col_idx = torch.arange(max_k, device=device, dtype=torch.int64).unsqueeze(0)
        k_valid_mask = k_col_idx < k_per_req.unsqueeze(1)
        topk_sorted = torch.where(
            k_valid_mask, topk_idx, torch.full_like(topk_idx, max_pages + 1)
        ).sort(dim=1)[0]

        max_len = max(max_k + num_recent, 1)
        active_out = torch.full(
            (n_active, max_len), -1, dtype=torch.int32, device=device
        )
        active_out[:, :max_k] = torch.where(
            k_valid_mask,
            topk_sorted.to(torch.int32),
            torch.full_like(topk_sorted, -1, dtype=torch.int32),
        )

        if num_recent > 0:
            recent_offsets = torch.arange(
                num_recent, device=device, dtype=torch.int64
            ).unsqueeze(0)
            recent_pages = (recent_start.unsqueeze(1) + recent_offsets).to(torch.int32)
            recent_cols = k_per_req.unsqueeze(1) + recent_offsets
            active_out.scatter_(1, recent_cols, recent_pages)

        out_indices = torch.full((bs, max_len), -1, dtype=torch.int32, device=device)
        out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)
        out_indices[active_rows] = active_out
        out_lengths[active_rows] = (k_per_req + num_recent).to(torch.int32)

        return out_indices, out_lengths

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        """Initialize algorithm-specific representation pools for all layers."""
        raise NotImplementedError

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        """Compute and store page representations for given page range."""
        raise NotImplementedError

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve page scores for TopK selection."""
        raise NotImplementedError

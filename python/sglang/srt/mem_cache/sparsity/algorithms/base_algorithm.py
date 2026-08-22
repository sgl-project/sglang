from abc import ABC, abstractmethod
from ctypes import c_float
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def load_optional_quest_kernel(module_name: str):
    """Load an installed Quest kernel; an absent module selects the fallback."""
    try:
        spec = find_spec(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == module_name or module_name.startswith(f"{exc.name}.")
        ):
            return None
        raise
    if spec is None:
        return None
    return import_module(module_name)


def _float32_scaled_count(count: int, ratio: float) -> int:
    """Match a device float32 multiply followed by integer truncation."""
    count_f32 = c_float(count).value
    ratio_f32 = c_float(ratio).value
    return int(c_float(count_f32 * ratio_f32).value)


@dataclass
class _TopKPlan:
    """Temporary tensors needed by one batched page-selection call."""

    max_num_pages: int
    physical_pages: torch.Tensor
    valid_page_mask: torch.Tensor
    active_mask: torch.Tensor
    recent_start: torch.Tensor
    history_page_mask: torch.Tensor
    k_per_req: torch.Tensor
    max_k: int
    score_order_required: bool
    recent_idx: torch.Tensor
    recent_valid: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    fixed_capacity: bool


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

    def begin_forward(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
        fixed_capacity: bool | int = False,
    ) -> None:
        """Prepare optional state shared by sparse layers in one forward."""

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
        self._topk_plan: _TopKPlan | None = None
        self._topk_plan_forward_batch = None

    def begin_forward(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
        fixed_capacity: bool | int = False,
    ) -> None:
        self._topk_plan = None
        self._topk_plan_forward_batch = None
        if not fixed_capacity or self.req_to_token_pool is None:
            return
        self._topk_plan = self._build_topk_plan(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
            fixed_capacity=fixed_capacity,
        )
        self._topk_plan_forward_batch = forward_batch

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
        """Select important history pages and retain the recent window."""
        bs, device = queries.shape[0], queries.device

        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for TopK retrieval"
            )
        plan = self._topk_plan
        if (
            bs == 1
            and self._get_num_pages_cpu(seq_lens_source, 1) is not None
            and (plan is None or not plan.fixed_capacity)
        ):
            return self._retrieve_topk_single(
                queries,
                layer_id,
                req_pool_indices,
                sparse_mask,
                seq_lens_source,
            )

        if (
            plan is None
            or self._topk_plan_forward_batch is not seq_lens_source
            or plan.req_pool_indices.numel() != bs
            or plan.seq_lens.device != device
        ):
            plan = self._build_topk_plan(
                seq_lens_source,
                req_pool_indices,
                sparse_mask,
                device,
            )
        if plan.max_num_pages <= self.num_recent_pages or plan.max_k <= 0:
            return self._empty_retrieval(bs, device)

        scores = self._retrieve_page_scores_batched(layer_id, queries, plan)
        topk_scores, topk_idx = torch.topk(
            scores,
            k=plan.max_k,
            dim=1,
            sorted=plan.score_order_required,
        )

        return self._finalize_topk_with_recent(topk_scores, topk_idx, plan, **kwargs)

    def _finalize_topk_with_recent(
        self,
        topk_scores: torch.Tensor,
        topk_idx: torch.Tensor,
        plan: _TopKPlan,
        **kwargs,
    ) -> tuple:
        """Finalize a batched selection, using optional kernels when available."""
        if kwargs.get("allow_prepared_metadata", False):
            direct_result = self._try_finalize_to_flashattention_metadata(
                topk_scores, topk_idx, plan, kwargs.get("attn_metadata")
            )
            if direct_result is not None:
                return direct_result

        kernel_module = None
        if topk_scores.is_cuda and torch.version.hip is None:
            kernel_module = load_optional_quest_kernel(
                "sglang.srt.mem_cache.sparsity.kernels.quest_finalize"
            )
        combined_width = topk_scores.shape[1] + plan.recent_idx.shape[1]
        kernel_inputs = (
            topk_scores,
            topk_idx,
            plan.k_per_req,
            plan.recent_idx,
            plan.recent_valid,
        )
        if (
            kernel_module is not None
            and combined_width <= kernel_module.QUEST_FINALIZE_MAX_WIDTH
            and all(tensor.is_contiguous() for tensor in kernel_inputs)
        ):
            return kernel_module.quest_finalize_selected_pages(
                topk_scores,
                topk_idx,
                plan.k_per_req,
                plan.recent_idx,
                plan.recent_valid,
            )

        device = topk_scores.device
        topk_idx = topk_idx.to(torch.long)
        topk_rank = torch.arange(plan.max_k, device=device, dtype=torch.long)
        topk_valid = (
            topk_rank.unsqueeze(0) < plan.k_per_req.unsqueeze(1)
        ) & torch.isfinite(topk_scores)
        combined_idx = torch.cat([topk_idx, plan.recent_idx], dim=1)
        combined_valid = torch.cat([topk_valid, plan.recent_valid], dim=1)
        return self._finalize_selected_pages(
            combined_idx, combined_valid, plan.max_num_pages
        )

    def _try_finalize_to_flashattention_metadata(
        self,
        topk_scores: torch.Tensor,
        topk_idx: torch.Tensor,
        plan: _TopKPlan,
        attn_metadata,
    ) -> tuple | None:
        if (
            attn_metadata is None
            or not topk_scores.is_cuda
            or torch.version.hip is not None
            or topk_scores.dtype != torch.float32
        ):
            return None
        required_attrs = ("page_table", "cache_seqlens_int32", "cu_seqlens_k")
        if not all(hasattr(attn_metadata, attr) for attr in required_attrs):
            return None

        kernel_module = load_optional_quest_kernel(
            "sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata"
        )
        if kernel_module is None:
            return None

        combined_width = topk_scores.shape[1] + plan.recent_idx.shape[1]
        kernel_inputs = (
            topk_idx,
            plan.k_per_req,
            plan.recent_idx,
            plan.recent_valid,
            plan.active_mask,
            plan.seq_lens,
            plan.req_pool_indices,
            self.req_to_token_pool.req_to_token,
        )
        metadata_tensors = (
            attn_metadata.page_table,
            attn_metadata.cache_seqlens_int32,
            attn_metadata.cu_seqlens_k,
        )
        if (
            combined_width > kernel_module.QUEST_DIRECT_METADATA_MAX_WIDTH
            or attn_metadata.page_table.shape[0] != topk_scores.shape[0]
            or attn_metadata.page_table.shape[1] < combined_width
            or any(
                not tensor.is_cuda or tensor.device != topk_scores.device
                for tensor in (*kernel_inputs, *metadata_tensors)
            )
            or not all(
                tensor.is_contiguous()
                for tensor in (
                    topk_scores,
                    topk_idx,
                    plan.k_per_req,
                    plan.recent_idx,
                    plan.recent_valid,
                )
            )
        ):
            return None

        valid_lengths = torch.empty(
            topk_scores.shape[0], dtype=torch.int32, device=topk_scores.device
        )
        kernel_module.quest_finalize_to_flashattention_metadata_(
            topk_scores=topk_scores,
            topk_indices=topk_idx,
            k_per_req=plan.k_per_req,
            recent_indices=plan.recent_idx,
            recent_valid=plan.recent_valid,
            valid_lengths=valid_lengths,
            sparse_mask=plan.active_mask,
            seq_lens=plan.seq_lens,
            req_pool_indices=plan.req_pool_indices,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_table=attn_metadata.page_table,
            cache_seqlens_int32=attn_metadata.cache_seqlens_int32,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            page_size=self.page_size,
            update_lengths=True,
        )
        return attn_metadata.page_table[:, :combined_width], valid_lengths, True

    def _retrieve_topk_single(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> tuple:
        """Keep the low-overhead historical path for single-request decode."""
        device = queries.device
        sparse_mask_cpu = self._get_bool_mask_cpu(
            sparse_mask, 1, forward_batch=forward_batch
        )
        if sparse_mask_cpu is not None and not sparse_mask_cpu[0]:
            return self._empty_retrieval(1, device)

        num_pages_cpu = self._get_num_pages_cpu(forward_batch, 1)
        assert num_pages_cpu is not None
        num_pages = num_pages_cpu[0]
        if num_pages <= self.num_recent_pages:
            return self._empty_retrieval(1, device)

        req_to_token = self.req_to_token_pool.req_to_token
        page_idx = torch.arange(num_pages, device=device, dtype=torch.long)
        page_starts = (page_idx * self.page_size).clamp(0, req_to_token.shape[1] - 1)
        single_req_pool_indices = req_pool_indices.to(device=device, dtype=torch.long)
        physical_pages = (
            req_to_token[
                single_req_pool_indices,
                page_starts.unsqueeze(0),
            ].to(torch.long)
            // self.page_size
        )
        scores = self._retrieve_page_scores(
            layer_id,
            physical_pages,
            single_req_pool_indices,
            queries,
        )

        recent_start = num_pages - self.num_recent_pages
        history_pages = max(recent_start, 1)
        k = min(
            max(_float32_scaled_count(history_pages, self.sparsity_ratio), 1),
            history_pages,
        )
        topk_idx = torch.topk(
            scores[:, :recent_start], k=k, dim=1, sorted=False
        ).indices.squeeze(0)
        recent_idx = torch.arange(
            recent_start, num_pages, device=device, dtype=torch.long
        )
        selected = torch.cat([topk_idx, recent_idx]).sort()[0].to(torch.int32)
        active_mask = sparse_mask.to(device=device, dtype=torch.bool).reshape(1)
        out_indices = torch.where(
            active_mask.unsqueeze(1),
            selected.unsqueeze(0),
            torch.full((1, selected.numel()), -1, dtype=torch.int32, device=device),
        )
        lengths = torch.where(
            active_mask,
            torch.full((1,), selected.numel(), dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        return out_indices, lengths

    def _build_topk_plan(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
        *,
        fixed_capacity: bool | int = False,
    ) -> _TopKPlan:
        """Vectorize ragged request metadata without retaining forward state."""
        batch_size = req_pool_indices.numel()
        seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)
        req_pool_indices = req_pool_indices.to(device=device, dtype=torch.long)
        sparse_mask_source = sparse_mask
        sparse_mask = sparse_mask.to(device=device, dtype=torch.bool)
        num_pages = (seq_lens + self.page_size - 1) // self.page_size

        num_pages_cpu = (
            None
            if fixed_capacity
            else self._get_num_pages_cpu(forward_batch, batch_size)
        )
        sparse_mask_cpu = (
            self._get_bool_mask_cpu(
                sparse_mask_source,
                batch_size,
                forward_batch=forward_batch,
            )
            if num_pages_cpu is not None
            else None
        )
        if fixed_capacity:
            pool_max_pages = getattr(self, "cuda_graph_max_num_pages", None)
            if pool_max_pages is None:
                max_context_len = getattr(
                    self.req_to_token_pool,
                    "max_context_len",
                    self.req_to_token_pool.req_to_token.shape[1],
                )
                pool_max_pages = max(
                    (max_context_len + self.page_size - 1) // self.page_size, 1
                )
            max_num_pages = (
                pool_max_pages
                if isinstance(fixed_capacity, bool)
                else min(fixed_capacity, pool_max_pages)
            )
        else:
            max_num_pages = (
                max(num_pages_cpu, default=0)
                if num_pages_cpu is not None
                else int(num_pages.max().item()) if batch_size > 0 else 0
            )
        page_idx = torch.arange(max_num_pages, device=device, dtype=torch.long)
        valid_page_mask = page_idx.unsqueeze(0) < num_pages.unsqueeze(1)

        req_to_token = self.req_to_token_pool.req_to_token
        if max_num_pages > 0:
            page_starts = (page_idx * self.page_size).clamp(
                0, req_to_token.shape[1] - 1
            )
            physical_pages = (
                req_to_token[
                    req_pool_indices[:, None],
                    page_starts[None, :],
                ].to(torch.long)
                // self.page_size
            )
            physical_pages = torch.where(
                valid_page_mask,
                physical_pages,
                torch.full_like(physical_pages, -1),
            )
        else:
            physical_pages = torch.empty(
                (batch_size, 0), device=device, dtype=torch.long
            )

        active_mask = sparse_mask & (num_pages > self.num_recent_pages)
        recent_start = (num_pages - self.num_recent_pages).clamp(min=0)
        history_page_mask = page_idx.unsqueeze(0) < recent_start.unsqueeze(1)
        history_pages = recent_start.clamp(min=1)
        k_per_req = (history_pages.to(torch.float32) * self.sparsity_ratio).to(
            torch.int32
        )
        k_per_req = torch.maximum(k_per_req, torch.ones_like(k_per_req))
        k_per_req = torch.minimum(k_per_req, history_pages.to(torch.int32))
        k_per_req = torch.where(active_mask, k_per_req, torch.zeros_like(k_per_req))

        if fixed_capacity:
            history_capacity = max(max_num_pages - self.num_recent_pages, 0)
            max_k = (
                min(
                    max(
                        _float32_scaled_count(history_capacity, self.sparsity_ratio),
                        1,
                    ),
                    history_capacity,
                )
                if history_capacity > 0
                else 0
            )
            score_order_required = True
        elif num_pages_cpu is not None:
            k_per_req_cpu = []
            for row, count in enumerate(num_pages_cpu):
                # Without a host mask, size conservatively. Device k_per_req
                # remains authoritative and zeros inactive rows.
                is_sparse = (
                    sparse_mask_cpu[row] if sparse_mask_cpu is not None else True
                )
                history_count = max(count - self.num_recent_pages, 0)
                k_per_req_cpu.append(
                    min(
                        max(
                            _float32_scaled_count(
                                max(history_count, 1), self.sparsity_ratio
                            ),
                            1,
                        ),
                        history_count,
                    )
                    if is_sparse and history_count > 0
                    else 0
                )
            max_k = max(k_per_req_cpu, default=0)
            positive_k = {value for value in k_per_req_cpu if value > 0}
            score_order_required = len(positive_k) > 1
        else:
            max_k = int(k_per_req.max().item()) if batch_size > 0 else 0
            score_order_required = True

        recent_offsets = torch.arange(
            self.num_recent_pages, device=device, dtype=torch.long
        )
        recent_idx = recent_start.unsqueeze(1) + recent_offsets.unsqueeze(0)
        recent_valid = active_mask.unsqueeze(1) & (recent_idx < num_pages.unsqueeze(1))
        return _TopKPlan(
            max_num_pages=max_num_pages,
            physical_pages=physical_pages,
            valid_page_mask=valid_page_mask,
            active_mask=active_mask,
            recent_start=recent_start,
            history_page_mask=history_page_mask,
            k_per_req=k_per_req,
            max_k=max_k,
            score_order_required=score_order_required,
            recent_idx=recent_idx,
            recent_valid=recent_valid,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            fixed_capacity=bool(fixed_capacity),
        )

    @staticmethod
    def _get_bool_mask_cpu(
        mask: torch.Tensor,
        batch_size: int,
        *,
        forward_batch=None,
    ) -> list[bool] | None:
        if not torch.is_tensor(mask) or mask.numel() != batch_size:
            raise ValueError("sparse_mask must contain one value per request")

        host_mask = getattr(forward_batch, "sparse_mask_cpu", None)
        if host_mask is None:
            if mask.device.type != "cpu":
                return None
            host_mask = mask

        if torch.is_tensor(host_mask):
            if host_mask.device.type != "cpu" or host_mask.numel() != batch_size:
                return None
            values = host_mask.detach().reshape(-1).tolist()
        else:
            try:
                values = list(host_mask)
            except TypeError:
                return None
            if len(values) != batch_size:
                return None
        return [bool(value) for value in values]

    def _get_num_pages_cpu(
        self, forward_batch: "ForwardBatch", batch_size: int
    ) -> list[int] | None:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return None
        if torch.is_tensor(seq_lens_cpu):
            if seq_lens_cpu.device.type != "cpu" or seq_lens_cpu.numel() != batch_size:
                return None
            values = seq_lens_cpu.reshape(-1).tolist()
        else:
            try:
                values = list(seq_lens_cpu)
            except TypeError:
                return None
            if len(values) != batch_size:
                return None
        return [
            max((int(value) + self.page_size - 1) // self.page_size, 0)
            for value in values
        ]

    def should_update_representations(self, forward_batch: "ForwardBatch") -> bool:
        """Skip only proven single-token decodes that cannot complete a page."""
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return True
        if torch.is_tensor(seq_lens_cpu):
            if seq_lens_cpu.device.type != "cpu":
                return True
            values = seq_lens_cpu.reshape(-1).tolist()
        else:
            try:
                values = list(seq_lens_cpu)
            except TypeError:
                return True

        batch_size = len(values)
        if self._may_process_multiple_decode_tokens(forward_batch, batch_size):
            return True
        return any(
            int(seq_len) > 0 and int(seq_len) % self.page_size == 0
            for seq_len in values
        )

    @staticmethod
    def _may_process_multiple_decode_tokens(forward_batch, batch_size: int) -> bool:
        """Detect known multi-token/speculative layouts and keep the safe path."""
        if getattr(forward_batch, "spec_info", None) is not None:
            return True

        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            try:
                if any(int(length) != 1 for length in extend_seq_lens_cpu):
                    return True
            except TypeError:
                return True

        num_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if num_tokens is not None and int(num_tokens) != batch_size:
            return True

        positions = getattr(forward_batch, "positions", None)
        if torch.is_tensor(positions) and positions.numel() != batch_size:
            return True

        return False

    def _retrieve_page_scores_batched(
        self, layer_id: int, queries: torch.Tensor, plan: _TopKPlan
    ) -> torch.Tensor:
        scores = self._retrieve_page_scores(
            layer_id,
            plan.physical_pages,
            plan.req_pool_indices,
            queries,
        )
        score_mask = (
            plan.active_mask.unsqueeze(1)
            & plan.valid_page_mask
            & plan.history_page_mask
        )
        return torch.where(score_mask, scores, torch.full_like(scores, float("-inf")))

    @staticmethod
    def _empty_retrieval(batch_size: int, device: torch.device) -> tuple:
        return (
            torch.full((batch_size, 1), -1, dtype=torch.int32, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    @staticmethod
    def _finalize_selected_pages(
        combined_idx: torch.Tensor,
        combined_valid: torch.Tensor,
        sentinel: int,
    ) -> tuple:
        sortable_idx = torch.where(
            combined_valid, combined_idx, torch.full_like(combined_idx, sentinel)
        )
        sorted_idx = torch.sort(sortable_idx, dim=1)[0].to(torch.int32)
        out_indices = torch.where(
            sorted_idx == sentinel,
            torch.full_like(sorted_idx, -1),
            sorted_idx,
        )
        return out_indices, combined_valid.sum(dim=1).to(torch.int32)

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

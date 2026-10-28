import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind, KVLocPlan
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

logger = logging.getLogger(__name__)


@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor
    # Keep this dataclass argument required despite SpecInput's None default;
    # otherwise the required positions field would follow a defaulted field.
    custom_mask: torch.Tensor = field()
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    retrieve_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    # Stacked per-step draft proposal distribution q, shape (bs, num_steps,
    # vocab); only set under rejection sampling. Consumed by the verify kernel.
    draft_probs: torch.Tensor = None
    prepared_out_cache_loc: Optional[torch.Tensor] = None
    prepared_mrope_positions: Optional[torch.Tensor] = None
    # The iteration's plan when the draft planned the verify window
    # (`prepared_out_cache_loc`); verify and draft extend take it.
    kv_loc_plan: Optional[KVLocPlan] = None

    # Shape info for padding
    num_tokens_per_req: int = -1  # -1 auto-fills from draft_token_num.

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num
        self.num_tokens_for_logprob_per_req = self.draft_token_num

    @property
    def max_tree_depth(self) -> int:
        """Longest root-to-leaf chain of the verify tree, incl. the root;
        bounds the accept_index row width. EAGLE trees are depth-bounded by
        the draft loop. Algorithms with other tree shapes override this."""
        return self.spec_steps + 1

    @property
    def tree_topk(self) -> int:
        """Branching factor passed to the tree-verify kernels; -1 means an
        irregular tree (no fixed per-level branching)."""
        return self.topk

    @classmethod
    def create_idle_input(
        cls, topk: int, spec_steps: int, num_verify_tokens: int, device: str
    ):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device=device),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device=device),
            positions=torch.empty((0,), dtype=torch.int64, device=device),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
        )

    def generate_attn_arg_prefill(
        self,
        *,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        translator: KVIndexTranslator,
        plan: KVLocPlan,
        kind: IdSpaceKind = IdSpaceKind.FULL,
    ):
        """CSR verify args. ``paged_kernel_lens`` excludes the verify tokens and
        is widened here; the translator packs the read ids straight into
        ``kv_indices``."""
        device = req_pool_indices.device
        batch_size = req_pool_indices.numel()
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        total_tokens = paged_kernel_lens_sum + self.draft_token_num * batch_size
        kv_indices = torch.empty(total_tokens, dtype=torch.int32, device=device)
        translator.pack_read_stream(
            plan,
            req_pool_indices=req_pool_indices,
            seq_lens=paged_kernel_lens,
            indptr=cum_kv_seq_len,
            out=kv_indices,
            kind=kind,
            # The translator tiles the gather over token blocks, sized from
            # the length sum (an upper bound on the longest live row) while
            # it is still a host int.
            num_token_blocks=None,
            kv_lens_sum=paged_kernel_lens_sum,
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask


@dataclass
class EagleDraftInput(SpecInput):
    # For idle stubs use `create_idle_input`, not the bare ctor: it gives
    # `topk_p` / `topk_index` / `hidden_states` / `bonus_tokens` zero rows.

    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # Draft proposal q from draft-extend, only set under rejection sampling:
    # (b, vocab) single-layer; (b, num_steps, vocab) multi-layer chain.
    draft_probs: torch.Tensor = None
    # shape: (b, hidden_size) - one hidden per req, consumed by `draft` forward.
    # None when the spec algorithm's draft doesn't read hidden_states
    # (e.g., STANDALONE — vanilla LLM draft).
    hidden_states: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Survives across draft steps: spec_info is shared by reference across the
    # per-step forwards (each runs on a copied ForwardBatch, dropping writebacks).
    dsa_topk_indices: Optional[torch.Tensor] = None

    # Per-req bonus token (the "+1" target prediction at end of each accept
    # chain); the worker copies it here post-extend for next iter's draft.
    bonus_tokens: torch.Tensor = None

    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None
    # The window-clipped counterpart of the pair above, for the draft's
    # sliding-window layers; None on a pool with no window side.
    window_kv_indptr: torch.Tensor = None
    window_kv_indices: torch.Tensor = None

    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # V2 overlap worker only: req_pool_indices used as buf slot keys.
    future_indices: Optional[torch.Tensor] = None
    future_dsa_topk_indices_available: bool = False

    # False when this input is missing model-required state for graph replay.
    # DP attention synchronizes the bit across ranks before the worker runs.
    cuda_graph_compatible: bool = True

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def pad_batch(
        self,
        pad_tensor_to_size: Callable[..., torch.Tensor],
        batch_size: int,
    ) -> None:
        if self.draft_probs is not None:
            self.draft_probs = pad_tensor_to_size(self.draft_probs, batch_size)

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
        vocab_size: int = 0,
    ):
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            draft_probs=(
                torch.empty((0, vocab_size), device=device, dtype=torch.float32)
                if get_spec().speculative_use_rejection_sampling
                else None
            ),
            capture_hidden_mode=capture_hidden_mode,
        )


@dataclass
class EagleDraftExtendInput(SpecInput):
    """Inputs to the draft-extend forward (the fill-draft-kvcache pass after
    target prefill / verify).

    Installed on `batch.spec_info` by the worker's `_draft_extend_for_*`
    (and synthetically by draft-extend cuda-graph capture), then replaced
    with a fresh `EagleDraftInput` for the next iter's draft.
    """

    # Target-model hidden states for the draft-extend forward; None when the
    # draft doesn't read hidden_states (e.g., STANDALONE). Shape: decode
    # (bs * num_draft_tokens, hidden), prefill (extend_num_tokens, hidden).
    hidden_states: Optional[torch.Tensor] = None

    # Per-req accept counts. `num_accept_tokens = num_correct_drafts + 1`.
    # Both kept for cuda-graph buffer indexing.
    num_correct_drafts: torch.Tensor = None
    num_accept_tokens: torch.Tensor = None
    num_front_tokens: int = 0
    # CPU view, read by attention backends during the extend forward.
    num_accept_tokens_cpu: List[int] = None

    # Per-req batch-state slices for the draft-extend forward:
    #   - input_ids:        accept tokens flat over surviving reqs
    #   - seq_lens / _cpu:  per-req sequence length (post-accept)
    #   - req_pool_indices: per-req kv-pool slot
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    req_pool_indices: torch.Tensor = None

    #   - positions: shape `[total_accepted]`.
    #   - bonus_tokens: shape `[bs]`; read post-extend to populate next iter's
    #     `EagleDraftInput.bonus_tokens`.
    positions: Optional[torch.Tensor] = None
    bonus_tokens: Optional[torch.Tensor] = None

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = 1

    dsa_seed_topk_capture: Optional[torch.Tensor] = None
    dsa_seed_topk_select: Optional[torch.Tensor] = None

    # Flat per-req index of each request's last accepted window row
    # (i * window + front + num_correct_drafts[i]). When set, the logits
    # processor runs lm_head and LAST hidden capture only on these rows; FULL
    # hidden capture remains unpruned.
    select_index: Optional[torch.Tensor] = None

    # None for draft-extend's idle batch; attention backends fall back to
    # rebuilding plain metadata from seq_lens when this is None.
    kv_indptr: torch.Tensor = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)

    def pad_batch(
        self,
        pad_tensor_to_size: Callable[..., torch.Tensor],
        batch_size: int,
    ) -> None:
        if self.num_correct_drafts is not None:
            self.num_correct_drafts = pad_tensor_to_size(
                self.num_correct_drafts, batch_size
            )
            self.num_accept_tokens = pad_tensor_to_size(
                self.num_accept_tokens, batch_size
            )

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
    ) -> "EagleDraftExtendInput":
        return cls(
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            num_correct_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens_cpu=[],
            input_ids=torch.empty((0,), device=device, dtype=torch.long),
            seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def generate_attn_arg_prefill(
        self,
        *,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: Optional[int],
        translator: KVIndexTranslator,
        plan: KVLocPlan,
        kind: IdSpaceKind = IdSpaceKind.FULL,
    ):
        """Draft-extend CSR args. ``paged_kernel_lens`` already includes the
        draft-extend window."""
        device = req_pool_indices.device
        bs = self.num_correct_drafts.numel()
        # Constant num_tokens_per_req qo layout (required for cuda-graph capture).
        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.num_tokens_per_req,
            step=self.num_tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # Captured while the length sum is still a host int (None -> the
        # translator sizes from its table width).
        host_kv_lens_sum = paged_kernel_lens_sum

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        translator.pack_read_stream(
            plan,
            req_pool_indices=req_pool_indices,
            seq_lens=paged_kernel_lens,
            indptr=cum_kv_seq_len,
            out=kv_indices,
            kind=kind,
            num_token_blocks=None,
            kv_lens_sum=host_kv_lens_sum,
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None

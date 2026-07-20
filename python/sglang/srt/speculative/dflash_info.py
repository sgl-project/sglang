from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash.

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it.
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # EAGLE-style tree verify metadata (only set when --speculative-dflash-tree-verify
    # and topk > 1). When present and `positions is None`, prepare_for_verify
    # builds the tree mask / positions / retrieve buffers via the EAGLE kernel.
    tree_parent_list: Optional[torch.Tensor] = None  # Full topk-tree parent list
    tree_selected_index: Optional[torch.Tensor] = None  # Selected indices in full tree
    retrieve_index: Optional[torch.Tensor] = None
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_req: int = -1

    # Reusable tree-mask / position buffers (EAGLE-style buffer reuse). When set,
    # build_tree_kernel_efficient fills these in place instead of allocating fresh
    # per-step buffers.
    tree_mask_buf: Optional[torch.Tensor] = None
    position_buf: Optional[torch.Tensor] = None
    ragged_verify_layout: Optional[RaggedVerifyLayout] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)
        self.num_tokens_for_logprob_per_req = int(self.draft_token_num)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ) -> tuple[ForwardBatch, bool]:
        """Prepare a DFLASH verify forward batch for overlap scheduling.

        The caller computes and stores `batch.out_cache_loc` before this
        method is called. This helper only packages the verify forward and pre-initializes either CUDA-graph replay
        metadata or eager attention metadata so the actual forward can run with
        `skip_attn_backend_init=True`.
        """
        batch.input_ids = self.draft_token

        # EAGLE-style tree verify: build the tree mask / positions / retrieve
        # buffers from the pruned-tree metadata before constructing the forward
        # batch. `generate_attn_arg_prefill` consumes `self.custom_mask`, so the
        # verify forward picks up the tree mask automatically.
        is_tree_verify = (
            self.positions is None
            and self.tree_parent_list is not None
            and self.tree_parent_list.numel() > 0
            and self.tree_selected_index is not None
            and self.tree_selected_index.numel() > 0
        )
        if is_tree_verify and not batch.forward_mode.is_idle():
            from sglang.srt.speculative.eagle_utils import (
                TreeMaskMode,
                build_tree_kernel_efficient,
            )

            bs = batch.batch_size()
            depth = int((self.tree_parent_list.size(1) - 1) / int(self.topk)) + 1
            # The tree mask buffer is sized from sum(seq_lens); the caller may have
            # temporarily set batch.seq_lens_sum to an over-allocated planning value,
            # so recompute the true sum here for a correctly sized FULL_MASK.
            # When a preallocated tree_mask_buf is supplied (EAGLE-style reuse),
            # the kernel ignores seq_lens_sum and writes into the (over-sized)
            # buffer directly, saving a per-step alloc + memset (and the D2H sync).
            if self.tree_mask_buf is not None:
                true_seq_lens_sum = 0
            else:
                true_seq_lens_sum = int(batch.seq_lens.sum().item())
            (
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                draft_tokens,
            ) = build_tree_kernel_efficient(
                bonus_tokens=self.draft_token.view(bs, self.draft_token_num)[:, 0],
                parent_list=self.tree_parent_list,
                top_scores_index=self.tree_selected_index,
                draft_tokens=self.draft_token.view(bs, self.draft_token_num)[:, 1:],
                seq_lens=batch.seq_lens,
                seq_lens_sum=true_seq_lens_sum,
                topk=int(self.topk),
                spec_steps=depth,
                num_verify_tokens=self.draft_token_num,
                tree_mask_mode=TreeMaskMode.FULL_MASK,
                tree_mask_buf=self.tree_mask_buf,
                position_buf=self.position_buf,
            )
            batch.input_ids = draft_tokens
            self.draft_token = draft_tokens
            self.custom_mask = tree_mask
            self.positions = positions
            self.retrieve_index = retrieve_index
            self.retrieve_next_token = retrieve_next_token
            self.retrieve_next_sibling = retrieve_next_sibling

        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        verify_forward_batch = ForwardBatch.init_new(
            batch,
            target_worker.model_runner,
            capture_hidden_mode=self.capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )

        # The DFLASH verify graph is captured mask-capable for tree verify (see the
        # get_spec_info mask patch / v1 fork), so tree verify replays the graph with
        # its per-step custom mask. No eager fallback needed.
        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        layout = self.ragged_verify_layout

        if layout is None:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            verify_lens = self.draft_token_num
            kv_indices_extra = self.draft_token_num * bs
        else:
            qo_indptr = layout.qo_indptr_device
            verify_lens = layout.verify_lens
            kv_indices_extra = layout.total_verify_tokens

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + verify_lens
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + kv_indices_extra,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

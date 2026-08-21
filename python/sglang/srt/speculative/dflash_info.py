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
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_is_npu = is_npu()


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
    # Beam width kept per draft depth: 1 = linear chain, > 1 = tree verify. Required (no
    # default) so every construction site states which shape it is building -- attention
    # backends gate tree metadata on `topk > 1` and a forgotten site would silently verify a
    # tree as a chain. DSPARK reuses this class and always passes 1.
    topk: int
    # Longest root-to-leaf chain, i.e. the DFLASH draft block width. Bounds `accept_index`,
    # whose outer accept loop silently truncates the accepted path if it is too narrow.
    # Defaults to `draft_token_num`, which is exact for any chain (topk == 1); tree callers
    # must pass it, since there the verify width 1 + (block_size - 1) * topk is wider.
    block_size: Optional[int] = None
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it.
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    # Left-child / right-sibling encoding of the draft tree, plus the flat node index, in the
    # layout `reconstruct_indices_from_tree_mask` writes and `verify_tree_greedy` walks. None
    # for chain verify. Declared here rather than alongside their producer because the GDN
    # backend reads `spec_info.retrieve_next_token` as a bare attribute as soon as topk > 1.
    retrieve_index: Optional[torch.Tensor] = None
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_req: int = -1

    ragged_verify_layout: Optional[RaggedVerifyLayout] = None
    # Committed/live lengths before the verify caller temporarily expands
    # batch.seq_lens_cpu to the target-attention KV lengths.
    live_seq_lens_cpu: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)
        self.num_tokens_for_logprob_per_req = int(self.draft_token_num)
        if self.block_size is None:
            if self.topk > 1:
                raise ValueError(
                    "DFlashVerifyInput with topk > 1 must pass block_size: the verify width "
                    "1 + (block_size - 1) * topk no longer equals the longest root-to-leaf "
                    f"chain, so accept_index cannot be sized from draft_token_num. Got "
                    f"topk={self.topk}, draft_token_num={self.draft_token_num}."
                )
            self.block_size = int(self.draft_token_num)

    @property
    def max_tree_depth(self) -> int:
        # Longest root-to-leaf chain, NOT the node count: a fixed-width beam over the
        # selector lattice keeps W nodes per depth but every path is still block_size long.
        return int(self.block_size)

    @property
    def tree_topk(self) -> int:
        # Per-parent fanout. The beam picks its W nodes per depth globally across parents, so
        # a parent can end up with anywhere from 0 to W children: irregular, hence -1. topk
        # == 1 keeps reporting 1 so the chain path's callers see today's value.
        return -1 if self.topk > 1 else int(self.topk)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ) -> tuple[ForwardBatch, bool]:
        """Prepare a DFLASH verify forward batch for overlap scheduling.

        The caller computes and stores `batch.out_cache_loc` before this
        method is called. GPU keeps the original pre-planning path. NPU leaves
        attention/graph metadata initialization to ModelRunner because DP/EP
        padding can still change the compressor's runtime shapes.
        """
        from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

        batch.input_ids = self.draft_token
        batch.spec_info = self
        if _is_npu and not batch.forward_mode.is_idle():
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                maybe_build_dsv4_verify_bundle,
            )

            batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
                batch,
                self.draft_token_num,
                live_seq_lens_cpu=self.live_seq_lens_cpu,
            )

        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        if not batch.forward_mode.is_idle():
            # Rebuild mamba track indices (lazy: gather the positions planned
            # by mamba_lazy_spec_prepare) and clear the stale extend-time mask
            # before init_new snapshots them into the verify ForwardBatch.
            # Same hook eagle/ngram/dspark run before TARGET_VERIFY.
            prepare_mamba_track_for_verify(batch)
        verify_forward_batch = ForwardBatch.init_new(
            batch,
            target_worker.model_runner,
            capture_hidden_mode=self.capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )

        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if _is_npu:
            # Do not pre-plan target verify on NPU. DP/EP padding can change
            # the compressor's logical batch without changing ForwardBatch's
            # stale-plan shape fields. Let ModelRunner select graph/eager and
            # initialize metadata after final batch preparation.
            return verify_forward_batch, can_run_cuda_graph
        elif can_run_cuda_graph:
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
        if layout is not None and layout.bs != bs:
            # Graph replay pads the batch to the captured slots; match it.
            layout = layout.padded_to_bucket(padded_bs=bs)

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

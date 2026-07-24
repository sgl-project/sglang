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

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_req: int = -1

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
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )

        # TARGET_VERIFY is an extend-mode forward (init_new copies
        # batch.extend_lens / batch.prefix_lens onto the ForwardBatch). The
        # verify block is `draft_token_num` new tokens per request on top of
        # the committed prefix, so populate those two fields explicitly —
        # otherwise init_new inherits stale prefill values and backends that
        # key on extend_seq_lens (KVarN's fused verify path) read garbage.
        # Mirrors eagle_worker_common.prepare_for_draft_extend.
        bs = len(batch.seq_lens)
        gpu_only = batch.seq_lens_cpu is None
        draft_token_num = int(self.draft_token_num)
        saved_prefix_lens = batch.prefix_lens
        saved_extend_lens = batch.extend_lens
        saved_seq_lens = batch.seq_lens
        saved_seq_lens_sum = batch.seq_lens_sum
        try:
            if gpu_only:
                batch.prefix_lens = batch.seq_lens.to(torch.int32)
                batch.extend_lens = torch.full(
                    (bs,),
                    draft_token_num,
                    dtype=torch.int32,
                    device=batch.seq_lens.device,
                )
                # KVarN's fused verify path computes `committed = seq_lens -
                # extend_lens`, so seq_lens must be the *post-verify* length.
                batch.seq_lens = batch.seq_lens + draft_token_num
                batch.seq_lens_sum = int(batch.seq_lens.sum().item())
            else:
                batch.prefix_lens = batch.seq_lens_cpu.tolist()
                batch.extend_lens = [draft_token_num] * bs
                # The caller already bumped seq_lens_cpu to committed + block;
                # bump the GPU mirror to match for backends that read it.
                batch.seq_lens = batch.seq_lens + draft_token_num
                batch.seq_lens_sum = int(batch.seq_lens.sum().item())
            verify_forward_batch = ForwardBatch.init_new(
                batch,
                target_worker.model_runner,
                capture_hidden_mode=self.capture_hidden_mode,
                return_hidden_states_before_norm=False,
            )
        finally:
            batch.prefix_lens = saved_prefix_lens
            batch.extend_lens = saved_extend_lens
            batch.seq_lens = saved_seq_lens
            batch.seq_lens_sum = saved_seq_lens_sum

        # init_new treats TARGET_VERIFY like decode for position purposes
        # (forward_batch_info.py: line 833 `is_target_verify()` branch) and
        # therefore never copies batch.extend_lens -> ret.extend_seq_lens.
        # Backends that key on extend_seq_lens (KVarN's fused verify path)
        # see None. Populate the verify geometry explicitly: every request
        # contributes `draft_token_num` new query tokens on top of its
        # committed prefix, and the post-verify seq_len is committed + block.
        device = verify_forward_batch.seq_lens.device
        verify_extend_lens = torch.full(
            (bs,), draft_token_num, dtype=torch.int32, device=device
        )
        # `verify_forward_batch.seq_lens` was built from the (bumped)
        # batch.seq_lens, so it already holds committed + draft_token_num.
        verify_prefix_lens = (
            verify_forward_batch.seq_lens.to(torch.int32) - verify_extend_lens
        )
        verify_forward_batch.extend_seq_lens = verify_extend_lens
        verify_forward_batch.extend_prefix_lens = verify_prefix_lens
        if not gpu_only:
            verify_forward_batch.extend_seq_lens_cpu = [draft_token_num] * bs
            verify_forward_batch.extend_prefix_lens_cpu = (
                verify_prefix_lens.cpu().tolist()
            )

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

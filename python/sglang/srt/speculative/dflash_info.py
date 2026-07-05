from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


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
    # Gates tree metadata in attention backends that key on `topk > 1`. The linear
    # chain path leaves this at 1; `from_tree` sets it to tree_width for tree mode (#29524).
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it.
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    @classmethod
    def from_tree(
        cls,
        *,
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        custom_mask: torch.Tensor,
        tree_width: int,
        num_nodes: int,
        **kwargs,
    ) -> DFlashVerifyInput:
        """Build a tree-mode verify input (#29524): ``topk = tree_width`` and a
        tree-causal ``custom_mask``. Field semantics are unchanged from the linear
        chain case; only the previously-fixed ``topk``/``custom_mask`` are populated."""
        return cls(
            draft_token=draft_token,
            positions=positions,
            draft_token_num=num_nodes,
            topk=tree_width,
            custom_mask=custom_mask,
            **kwargs,
        )

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

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
        batch.capture_hidden_mode = self.capture_hidden_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

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

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
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


def build_dflash_tree_verify_input(
    *,
    root_tokens: list[int],
    topk_tokens: list[list[list[int]]],
    topk_logprobs: list[list[list[float]]],
    tree_width: int,
    budget: int,
    construction: str,
    base_positions: list[int],
    kv_lens: list[int],
    device,
) -> Tuple[DFlashVerifyInput, dict]:
    """Bridge from per-request draft-head top-W output to a tree-mode verify input
    (#29524, JetSpec). Builds one tree per request, materializes the verify buffers +
    tree-causal mask, and packs them into a ``DFlashVerifyInput.from_tree``.

    This is the documented integration seam that the (deferred, GPU-validated) worker
    decode loop will call once the live verify-loop integration lands. The host-side
    pieces it composes (``build_tree`` / ``tree_to_verify_buffers`` /
    ``build_tree_custom_mask``) are unit-tested in
    ``test/registered/spec/dflash/test_dflash_tree.py``.

    Args:
        root_tokens: per-request root token (bonus / last verified), length ``bs``.
        topk_tokens / topk_logprobs: per request, shape ``(depth_count, W)``.
        tree_width / budget / construction: tree drafting knobs.
        base_positions: per-request absolute position of the root (= seq_len).
        kv_lens: per-request committed-prefix length, for the mask context part.

    Returns ``(verify_input, buffers)`` where ``buffers`` is the dict from
    ``tree_to_verify_buffers``.
    """
    from sglang.srt.speculative.dflash_tree import (
        build_tree,
        build_tree_custom_mask,
        tree_to_verify_buffers,
    )

    trees = [
        build_tree(
            root_tokens[b],
            topk_tokens[b],
            topk_logprobs[b],
            tree_width=tree_width,
            budget=budget,
            construction=construction,
        )
        for b in range(len(root_tokens))
    ]
    buffers = tree_to_verify_buffers(trees, budget, base_positions, device=device)
    custom_mask = build_tree_custom_mask(trees, budget, kv_lens, device=device)
    verify_input = DFlashVerifyInput.from_tree(
        draft_token=buffers["draft_token"].reshape(-1),
        positions=buffers["positions"].reshape(-1),
        custom_mask=custom_mask,
        tree_width=tree_width,
        num_nodes=budget,
    )
    return verify_input, buffers

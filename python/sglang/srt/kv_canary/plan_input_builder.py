from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanInput:
    """Pre-staged input to launch_canary_plan_kernels for the per-forward path.

    All tensors live on device.

    Fields:
        fb_req_pool_indices: Per-row ReqToTokenPool row index, shape [bs_capacity], int64.
            0 = padding sentinel. Pre-allocated static buffer that the runner writes into each
            step (see Static-buffer contract below).
        fb_prefix_lens: Per-req prefix length already written before this step, shape
            [bs_capacity], int64. Extend → extend_prefix_lens; decode → seq_lens - 1.
        fb_extend_seq_lens: Per-req tokens being written this step, shape [bs_capacity], int64.
            Extend length or all-ones for decode.

    Allocated up front by CanaryRunner. ForwardBatch token/position/slot tensors may arrive as int32
    or int64 at the boundary; canary canonicalizes them to int64 internally.

    **Static-buffer contract (cuda-graph correctness)**: the PlanInput's tensors are allocated once
    during CanaryRunner.__init__ (sized for the worst-case per-forward batch). The per-forward
    builder MUTATES those tensors in place each step via ``.copy_()`` / ``.fill_()`` / index-assign
    on the default stream. The captured cuda-graph reads them by address; therefore: (a) never
    reallocate, (b) all writes complete on the default stream before the captured region runs (i.e.
    the writes happen in ``CanaryRunner.before_forward`` which the caller invokes in
    ``ModelRunner.forward`` BEFORE ``graph_runner.replay()`` or ``model.forward()``).
    """

    fb_req_pool_indices: torch.Tensor
    fb_prefix_lens: torch.Tensor
    fb_extend_seq_lens: torch.Tensor

    def zero_(self) -> None:
        self.fb_req_pool_indices.zero_()
        self.fb_prefix_lens.zero_()
        self.fb_extend_seq_lens.zero_()

    @classmethod
    def allocate(
        cls,
        *,
        bs_capacity: int,
        device: torch.device,
    ) -> "PlanInput":
        return cls(
            fb_req_pool_indices=torch.zeros(
                bs_capacity, dtype=torch.int64, device=device
            ),
            fb_prefix_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            fb_extend_seq_lens=torch.zeros(
                bs_capacity, dtype=torch.int64, device=device
            ),
        )


def fill_plan_input_per_forward(
    *,
    forward_batch: "ForwardBatch",
    plan_input_out: PlanInput,
) -> int:
    """Builder for the per-forward (head + tail) caller. MUTATES plan_input_out's static buffers
    in place — does NOT return a new PlanInput (see PlanInput Static-buffer contract).

    - plan_input_out.fb_req_pool_indices[:bs] ← forward_batch.req_pool_indices; rows beyond bs
      are zeroed (padding sentinel).
    - plan_input_out.fb_prefix_lens[:bs] / fb_extend_seq_lens[:bs] are dispatched per
      forward_mode (see the per-mode block below). Rows beyond bs are zeroed (padding skipped
      via req_pool_indices sentinel; the lens value there is irrelevant).

    Returns the current bs (valid prefix length of the per-forward buffers).
    """
    req_pool_indices = forward_batch.req_pool_indices
    bs = int(req_pool_indices.shape[0])
    capacity = int(plan_input_out.fb_req_pool_indices.shape[0])
    if bs > capacity:
        raise RuntimeError(
            f"kv-canary: per-forward batch size {bs} exceeds static capacity {capacity}; "
            "raise the buffer size in CanaryRunner.__init__"
        )

    plan_input_out.zero_()
    plan_input_out.fb_req_pool_indices[:bs].copy_(req_pool_indices)

    _extract_prefix_lens_and_extend_seq_lens(
        forward_batch=forward_batch,
        out_prefix_lens=plan_input_out.fb_prefix_lens[:bs],
        out_extend_seq_lens=plan_input_out.fb_extend_seq_lens[:bs],
        bs=bs,
    )

    return bs


def _extract_prefix_lens_and_extend_seq_lens(
    *,
    forward_batch: "ForwardBatch",
    out_prefix_lens: torch.Tensor,
    out_extend_seq_lens: torch.Tensor,
    bs: int,
) -> None:
    # TODO: once ForwardMode is refactored upstream so every mode ships a canonical
    # (prefix_lens, extend_seq_lens) pair on forward_batch, collapse this back to a single
    # unconditional copy.
    forward_mode = forward_batch.forward_mode
    spec_info = forward_batch.spec_info
    if forward_mode.is_decode_or_idle():
        # Evidence: ForwardBatch.init_new leaves extend_* fields unset for decode/idle, while
        # attention backends treat decode as one query token whose cache length is seq_lens.
        # Therefore the covered span is prefix seq_lens - 1 plus one extend token.
        out_prefix_lens.copy_((forward_batch.seq_lens[:bs] - 1).to(torch.int64))
        out_extend_seq_lens.fill_(1)
    elif forward_mode.is_target_verify():
        # Evidence: EagleVerifyInputV2Mixin.prepare_for_v2_verify assigns out_cache_loc in
        # [seq_lens, seq_lens + draft_token_num) without bumping seq_lens. The target-verify
        # branch in TRTLLMHAAttnBackend.init_forward_metadata uses seq_lens as the prefix and
        # tokens_per_req as the query length, so mirror that as seq_lens plus draft_token_num.
        out_prefix_lens.copy_(forward_batch.seq_lens[:bs].to(torch.int64))
        out_extend_seq_lens.fill_(int(spec_info.draft_token_num))
    elif forward_mode.is_draft_extend_v2():
        # Evidence: EagleDraftInputV2Mixin.prepare_for_extend_to_fill_draft_kvcache bumps
        # seq_lens by num_draft_tokens. FlashAttentionBackend.init_forward_metadata reads the
        # draft-extend-v2 query length from spec_info.extend_seq_lens_tensor when available.
        # CUDA-graph replay passes extend_seq_lens but omits extend_prefix_lens, so derive the
        # prefix as seq_lens - extend_seq_lens.
        extend_seq_lens = forward_batch.extend_seq_lens[:bs].to(torch.int64)
        out_extend_seq_lens.copy_(extend_seq_lens)
        out_prefix_lens.copy_(
            forward_batch.seq_lens[:bs].to(torch.int64) - extend_seq_lens
        )
    elif forward_mode.is_extend():
        # Evidence: ForwardBatch.init_new copies batch.prefix_lens and batch.extend_lens into
        # extend_prefix_lens / extend_seq_lens for non-decode, non-idle modes, matching regular
        # extend metadata builders that consume those tensors directly.
        out_prefix_lens.copy_(forward_batch.extend_prefix_lens[:bs].to(torch.int64))
        out_extend_seq_lens.copy_(forward_batch.extend_seq_lens[:bs].to(torch.int64))
    else:
        raise NotImplementedError(
            f"Unsupported forward mode for kv-canary: {forward_mode}"
        )

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
        req_pool_indices: Per-row ReqToTokenPool row index, shape [bs_capacity], int64.
            0 = padding sentinel.
        prefix_lens: Per-req prefix length already written before this step, shape
            [bs_capacity], int64. Extend → extend_prefix_lens; decode → seq_lens - 1.
        extend_seq_lens: Per-req tokens being written this step, shape [bs_capacity], int64.
            Extend length or all-ones for decode.
        req_to_verify_expected_tokens_valid_lens: Per-req snapshot length on the verify-token pool,
            shape [bs_capacity], int64. Equals
            ``len(req.origin_input_ids) + len(req.output_ids)`` at the moment ``ForwardBatch``
            was built. The plan kernel uses ``valid_lens[req_id]`` as the upper bound on
            ``sot_pos`` when gathering the expected token; everything past the snapshot
            (e.g. EAGLE draft / verify positions, or stale residue from a longer recycled
            slot owner) returns the ``-1`` sentinel and the verify kernel skips the check.
            Set only when ``CanaryConfig.enable_verify_token_assert`` is on.

    Allocated fresh per forward by :class:`SingleForwardManager`. The boundary
    ForwardBatch token/position/slot tensors must already be int64
    contiguous (upstream phase-1 hook is responsible).
    """

    req_pool_indices: torch.Tensor
    prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    req_to_verify_expected_tokens_valid_lens: torch.Tensor

    def zero_(self) -> None:
        self.req_pool_indices.zero_()
        self.prefix_lens.zero_()
        self.extend_seq_lens.zero_()
        self.req_to_verify_expected_tokens_valid_lens.zero_()

    @classmethod
    def allocate(
        cls,
        *,
        bs_capacity: int,
        device: torch.device,
    ) -> PlanInput:
        return cls(
            req_pool_indices=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            prefix_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            extend_seq_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            req_to_verify_expected_tokens_valid_lens=torch.zeros(
                bs_capacity, dtype=torch.int64, device=device
            ),
        )

    def fill_from_forward_batch(self, *, forward_batch: ForwardBatch) -> None:
        req_pool_indices = forward_batch.req_pool_indices
        bs = int(req_pool_indices.shape[0])
        capacity = int(self.req_pool_indices.shape[0])
        if bs > capacity:
            raise RuntimeError(
                f"kv-canary: per-forward batch size {bs} exceeds static capacity {capacity}; "
                "raise the buffer size in CanaryLaunchCapacities"
            )

        self.zero_()
        self.req_pool_indices[:bs].copy_(req_pool_indices)

        _extract_prefix_lens_and_extend_seq_lens(
            forward_batch=forward_batch,
            out_prefix_lens=self.prefix_lens[:bs],
            out_extend_seq_lens=self.extend_seq_lens[:bs],
            bs=bs,
        )

        req_all_ids_lens = forward_batch.req_all_ids_lens
        if req_all_ids_lens is not None:
            self.req_to_verify_expected_tokens_valid_lens[:bs].copy_(
                req_all_ids_lens.to(torch.int64), non_blocking=True
            )


def _extract_prefix_lens_and_extend_seq_lens(
    *,
    forward_batch: ForwardBatch,
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
        # Anchor on ``positions`` (canonical write position) — eagle draft leaves seq_lens
        # pre-bump so deriving prefix_lens from seq_lens is off-by-one. Padding tail (positions
        # shorter than bs under cuda-graph padding) keeps whatever stale data it had; the offsets
        # kernel masks those rows via ``is_active`` before using prefix_lens.
        positions = forward_batch.positions
        out_prefix_lens[: positions.shape[0]].copy_(positions.to(torch.int64))
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

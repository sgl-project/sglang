"""Cuda-graph-safe real-model token-id validator wrappers.

The host-side populate of the source-of-truth pool / valid_lens lives in
``SingleForwardManager`` (it runs in the outside-graph hook); this module
exposes the kernel-launch entry point plus the host-side
``mode_offset`` helper that the manager calls from inside the graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.kv_canary.gather_expected_tokens import (
    launch_gather_expected_tokens_kernel,
)
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.plan_input import PlanInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def fill_expected_inputs_from_reqs(
    *,
    forward_batch: "ForwardBatch",
    out_expected_inputs: ExpectedInputs,
    plan_input: PlanInput,
    pool: torch.Tensor,
    valid_lens: torch.Tensor,
) -> None:
    """Launch the gather kernel into ``out_expected_inputs.tokens[:num_tokens]``
    and copy positions. Pure device path; safe inside cuda-graph capture.

    The caller (``SingleForwardManager.pre_ops_outside_graph``) is responsible
    for refilling ``pool`` / ``valid_lens`` host-side before this runs.
    """
    positions = forward_batch.positions
    input_ids = forward_batch.input_ids
    num_tokens = int(input_ids.shape[0])
    if num_tokens == 0:
        return

    mode_offset = _logical_pos_offset(forward_batch=forward_batch)
    launch_gather_expected_tokens_kernel(
        req_pool_indices=plan_input.req_pool_indices,
        prefix_lens=plan_input.prefix_lens,
        extend_seq_lens=plan_input.extend_seq_lens,
        pool=pool,
        valid_lens=valid_lens,
        input_ids=input_ids.to(torch.int64),
        mode_offset=mode_offset,
        out_expected_tokens=out_expected_inputs.tokens,
        num_tokens=num_tokens,
    )
    out_expected_inputs.positions[:num_tokens].copy_(positions.to(torch.int64))


def _logical_pos_offset(*, forward_batch: "ForwardBatch") -> int:
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    forward_mode = forward_batch.forward_mode
    if forward_mode.is_draft_extend(include_v2=True):
        return 1
    # EAGLE draft prefill keeps forward_mode == EXTEND but rotates input_ids in place
    # so slot p stores K/V for token p+1 (see eagle_worker_v2._draft_extend_for_prefill).
    if forward_mode.is_extend() and isinstance(
        forward_batch.spec_info, EagleDraftInput
    ):
        return 1
    return 0

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Request admission for UBTree's greedy and target-sampling paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def validate_ubtree_request(req: Req) -> str | None:
    if req.custom_logit_processor:
        return "UBTREE does not support custom_logit_processor."
    # Acceptance consumes captured target hidden states internally. Returning
    # per-token hidden states or sampling masks needs separate output plumbing.
    if req.return_hidden_states:
        return "UBTREE does not support return_hidden_states yet."
    if req.return_sampling_mask:
        return "UBTREE does not support return_sampling_mask yet."
    if req.return_logprob:
        return "UBTREE does not support return_logprob yet."

    sampling_params = req.sampling_params
    # SamplingParams normalizes temperature=0 to temperature=1, top_k=1.
    # Use the same greedy criterion as SamplingBatchInfo.
    if sampling_params.top_k != 1:
        if sampling_params.sampling_seed is not None:
            return "UBTREE does not support sampling_seed for non-greedy requests yet."
        exec_cfg = get_exec()
        if exec_cfg.deterministic.enable_deterministic_inference:
            return "UBTREE does not support deterministic sampling yet."
        if exec_cfg.kernel.sampling_backend != "flashinfer":
            return "UBTREE non-greedy requests require --sampling-backend flashinfer."
    if req.grammar is not None or any(
        constraint is not None
        for constraint in (
            sampling_params.json_schema,
            sampling_params.regex,
            sampling_params.ebnf,
            sampling_params.structural_tag,
        )
    ):
        return "UBTREE does not support grammar decoding yet."
    if (
        sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
        or sampling_params.min_new_tokens > 0
    ):
        return "UBTREE does not support sampling penalties or min_new_tokens."
    if sampling_params.logit_bias is not None:
        return "UBTREE does not support logit_bias."
    if sampling_params.watermark is not None:
        return "UBTREE does not support watermarking."
    return None

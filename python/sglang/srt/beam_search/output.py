# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam search output carrier, along scheduler -> detokenizer -> tokenizer
manager: pack (per-leader BeamSearchOutput) -> decode (sequence texts) ->
build out dict (meta_info.beam_results).

Module-level imports stay off scheduler-only modules so the detokenizer /
tokenizer processes can import this without pulling the scheduler graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from sglang.srt.managers.incremental_detokenize import DetokenizeHeader
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def pack_beam_search_output(req: Req) -> Optional[BeamSearchOutput]:
    """Top num_return sequences, best score first; None for a group that ended
    without results. Finish reasons are JSON: the carrier crosses IPC."""
    # Scheduler-side only; keep schedule_batch out of the module import graph.
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN

    group = req.beam_group
    results = getattr(group, "final_results", None)
    if not results:
        return None
    results = results[: group.num_return]
    reasons = {}  # one dict per distinct reason, which pickle sends once
    finish_reasons = []
    for r in results:
        key = (r.matched_token, None if r.matched_token is not None else len(r.tokens))
        reason = reasons.get(key)
        if reason is None:
            if r.matched_token is not None:
                reason = FINISH_MATCHED_TOKEN(matched=r.matched_token).to_json()
            else:
                reason = FINISH_LENGTH(length=len(r.tokens)).to_json()
            reasons[key] = reason
        finish_reasons.append(reason)
    return BeamSearchOutput(
        tokens=[r.tokens for r in results],
        cum_logprobs=[r.cum_logprob for r in results],
        beam_scores=[r.beam_score for r in results],
        finish_reasons=finish_reasons,
    )


def beam_completion_tokens(beam_output: BeamSearchOutput) -> int:
    """A group's completion_tokens: the total across its returned sequences
    (the leader row's output_ids is a length placeholder, not output)."""
    return sum(map(len, beam_output.tokens))


def is_beam_search_batch(recv_obj: BatchTokenIDOutput) -> bool:
    return (
        recv_obj.beam_search_output is not None and len(recv_obj.beam_search_output) > 0
    )


def decode_beam_search_output(
    recv_obj: BatchTokenIDOutput,
    headers: List[Optional[DetokenizeHeader]],
    *,
    tokenizer,
    disable_batch_decode: bool,
    trim_matched_stop: Callable,
) -> None:
    """Fill each output's sequence `texts` in place."""
    for i, beam_output in enumerate(recv_obj.beam_search_output):
        header = headers[i]
        if beam_output is None or header is None:
            # A non-beam item of a mixed batch, or a row with no detokenize state.
            continue
        # A group's returned beams mix stop-finished and length-finished
        # ones, so the leader's reason would trim the wrong ones.
        trimmed_tokens = [
            trim_matched_stop(tokens, finish_reason, header.no_stop_trim)
            for tokens, finish_reason in zip(
                beam_output.tokens, beam_output.finish_reasons
            )
        ]
        # batch_decode takes scalar skip_special_tokens flags, so decode per
        # request; batching all would apply request 0's flags to everyone.
        flags = dict(
            skip_special_tokens=header.skip_special_tokens,
            spaces_between_special_tokens=header.spaces_between_special_tokens,
        )
        if disable_batch_decode:
            beam_output.texts = [
                tokenizer.decode(tokens, **flags) for tokens in trimmed_tokens
            ]
        else:
            beam_output.texts = tokenizer.batch_decode(trimmed_tokens, **flags)


def build_beam_search_out(out: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten beam_results into a regular out dict: best beam at the top level,
    full list under meta_info, so _wait_one_response reuses its normal path."""
    beam_results = out.get("beam_results", [])
    if not beam_results:
        return out
    first_beam = beam_results[0]
    converted = {
        "text": first_beam.get("text", ""),
        "output_ids": first_beam.get("output_ids", []),
        "meta_info": first_beam.get("meta_info", {}).copy(),
    }
    converted["meta_info"]["beam_results"] = beam_results
    return converted


def try_build_beam_search_out_dict(
    recv_obj: Union[
        BatchStrOutput,
        BatchEmbeddingOutput,
        BatchTokenIDOutput,
    ],
    i: int,
    meta_info: Dict[str, Any],
) -> Optional[dict]:
    """Build the out_dict if item `i` is a finished beam result, else None."""
    if not isinstance(recv_obj, (BatchTokenIDOutput, BatchStrOutput)):
        return None

    beam_search_output = (
        recv_obj.beam_search_output[i]
        if recv_obj.beam_search_output and i < len(recv_obj.beam_search_output)
        else None
    )
    if (
        beam_search_output is None
        or not beam_search_output.tokens
        or recv_obj.finished_reasons[i] is None
    ):
        return None

    return _build_beam_search_out_dict(beam_search_output, meta_info, recv_obj)


def _build_beam_search_out_dict(
    beam_search_output: BeamSearchOutput,
    meta_info: Dict[str, Any],
    recv_obj: Union[BatchStrOutput, BatchTokenIDOutput],
) -> dict:
    columns = zip(
        beam_search_output.tokens,
        beam_search_output.finish_reasons,
        beam_search_output.beam_scores,
    )
    # Reasons are copied: the carrier shares one dict between beams that
    # finished alike, and callers edit a reply's reason in place.
    if isinstance(recv_obj, BatchStrOutput):
        texts = beam_search_output.texts or [None] * len(beam_search_output.tokens)
        beam_results = [
            {
                "output_ids": tokens.copy(),
                "text": text or "",
                "meta_info": {"finish_reason": reason.copy(), "sequence_score": score},
            }
            for (tokens, reason, score), text in zip(columns, texts)
        ]
    else:
        beam_results = [
            {
                "output_ids": tokens.copy(),
                "meta_info": {"finish_reason": reason.copy(), "sequence_score": score},
            }
            for tokens, reason, score in columns
        ]
    # The first beam also carries the request's meta_info, with completion_tokens
    # summed over every beam: recv_obj.completion_tokens[i] counts the first only.
    first = meta_info.copy()
    first["completion_tokens"] = beam_completion_tokens(beam_search_output)
    first.update(beam_results[0]["meta_info"])
    beam_results[0]["meta_info"] = first
    return {"beam_results": beam_results}

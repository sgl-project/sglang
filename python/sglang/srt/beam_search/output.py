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

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from sglang.srt.beam_search.types import BeamSearchSequence
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
    sequences = []
    for r in results:
        if r.matched_token is not None:
            finish_reason = FINISH_MATCHED_TOKEN(matched=r.matched_token)
        else:
            finish_reason = FINISH_LENGTH(length=len(r.tokens))
        sequences.append(
            BeamSearchSequence(
                tokens=r.tokens,
                cum_logprob=r.cum_logprob,
                beam_score=r.beam_score,
                finish_reason=finish_reason.to_json(),
            )
        )
    return BeamSearchOutput(sequences=sequences)


def beam_completion_tokens(beam_output: BeamSearchOutput) -> int:
    """A group's completion_tokens: the total across its returned sequences
    (the leader row's output_ids is a length placeholder, not output)."""
    return sum(len(seq.tokens) for seq in beam_output.sequences)


def is_beam_search_batch(recv_obj: BatchTokenIDOutput) -> bool:
    return (
        recv_obj.beam_search_output is not None and len(recv_obj.beam_search_output) > 0
    )


def decode_beam_search_output(
    recv_obj: BatchTokenIDOutput,
    *,
    tokenizer,
    disable_batch_decode: bool,
    trim_matched_stop: Callable,
) -> None:
    """Fill each candidate sequence's `text` in place."""
    if disable_batch_decode:
        for i, beam_output in enumerate(recv_obj.beam_search_output):
            if beam_output is None:
                # Mixed batch: this item is not a beam request.
                continue
            for beam in beam_output.sequences:
                # A group's returned beams mix stop-finished and length-finished
                # ones, so the leader's reason would trim the wrong ones.
                trimmed_tokens = trim_matched_stop(
                    beam.tokens,
                    beam.finish_reason,
                    recv_obj.no_stop_trim[i],
                )
                beam.text = tokenizer.decode(
                    trimmed_tokens,
                    skip_special_tokens=recv_obj.skip_special_tokens[i],
                    spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[
                        i
                    ],
                )
    else:
        # batch_decode takes scalar skip_special_tokens flags, so decode per
        # request; batching all would apply request 0's flags to everyone.
        for i, beam_output in enumerate(recv_obj.beam_search_output):
            if beam_output is None:
                # Mixed batch: this item is not a beam request.
                continue
            trimmed_tokens = [
                trim_matched_stop(
                    beam.tokens,
                    beam.finish_reason,
                    recv_obj.no_stop_trim[i],
                )
                for beam in beam_output.sequences
            ]
            beam_texts = tokenizer.batch_decode(
                trimmed_tokens,
                skip_special_tokens=recv_obj.skip_special_tokens[i],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[i],
            )
            for beam, text in zip(beam_output.sequences, beam_texts):
                beam.text = text


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
    has_beam_search = (
        beam_search_output is not None
        and hasattr(beam_search_output, "sequences")
        and beam_search_output.sequences
    )
    if not has_beam_search or recv_obj.finished_reasons[i] is None:
        return None

    return _build_beam_search_out_dict(beam_search_output, meta_info, recv_obj)


def _build_beam_search_out_dict(
    beam_search_output: Any,
    meta_info: Dict[str, Any],
    recv_obj: Union[BatchStrOutput, BatchTokenIDOutput],
) -> dict:
    include_text = isinstance(recv_obj, BatchStrOutput)
    beam_results = []
    total_completion_tokens = beam_completion_tokens(beam_search_output)
    for idx, beam_seq in enumerate(beam_search_output.sequences):
        beam_out_dict = {"output_ids": beam_seq.tokens.copy()}
        if include_text:
            beam_out_dict["text"] = beam_seq.text if beam_seq.text else ""
        if idx == 0:
            beam_meta_info = meta_info.copy()
            # Override completion_tokens with the sum of all beam sequences,
            # since recv_obj.completion_tokens[i] only counts the first beam.
            beam_meta_info["completion_tokens"] = total_completion_tokens
        else:
            beam_meta_info = {}
        beam_meta_info["finish_reason"] = beam_seq.finish_reason
        beam_meta_info["sequence_score"] = beam_seq.beam_score
        beam_out_dict["meta_info"] = beam_meta_info

        beam_results.append(beam_out_dict)

    return {"beam_results": beam_results}

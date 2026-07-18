"""Beam search output carrier: packing and per-stage transforms.

The carrier path is scheduler -> detokenizer -> tokenizer manager:
pack_beam_search_output builds the per-leader BeamSearchOutput,
decode_beam_search_output fills sequence texts in the detokenizer, and
build_beam_search_out / try_build_beam_search_out_dict shape the final
meta_info.beam_results response.

Plain functions with narrow parameters; module-level imports stay off
scheduler-only modules so the detokenizer/tokenizer processes can import
this without pulling the scheduler graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from sglang.srt.managers.beam_search_type import BeamSearchSequence
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def pack_beam_search_output(req: Req) -> Optional[BeamSearchOutput]:
    """Build the beam_results carrier for a finished leader.

    Returns the group's top num_return sequences, best score first, with
    JSON-serializable finish reasons (the carrier crosses IPC boundaries).
    Returns None for a group that ended without results (aborted).
    """
    # Scheduler-side only; keep schedule_batch out of the module import graph.
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN

    group = req.group
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


def is_beam_search_batch(recv_obj: BatchTokenIDOutput) -> bool:
    """Check if the batch contains beam search requests."""
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
    """Decode beam search candidate sequences to text."""
    if disable_batch_decode:
        for i, beam_output in enumerate(recv_obj.beam_search_output):
            if beam_output is None:
                # Mixed batch: this item is not a beam request.
                continue
            for beam in beam_output.sequences:
                trimmed_tokens = trim_matched_stop(
                    beam.tokens,
                    recv_obj.finished_reasons[i],
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
        # batch_decode only accepts scalar skip_special_tokens /
        # spaces_between_special_tokens flags, so decode per request (each
        # request's beams share that request's flags). Batching across the whole
        # batch would apply request 0's flags to every request's beams.
        for i, beam_output in enumerate(recv_obj.beam_search_output):
            if beam_output is None:
                # Mixed batch: this item is not a beam request.
                continue
            trimmed_tokens = [
                trim_matched_stop(
                    beam.tokens,
                    recv_obj.finished_reasons[i],
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
    """Convert a beam search out dict (containing beam_results) to a regular out dict.

    Takes the first beam's meta_info as the top-level meta_info, and stores
    the full beam_results list inside meta_info so callers can access all beams.
    All post-processing (logging, metrics, abort handling, timing, etc.) is handled
    by the shared _wait_one_response logic after this conversion.
    """
    beam_results = out.get("beam_results", [])
    if not beam_results:
        return out
    first_beam = beam_results[0]
    # Use the first beam's fields as the top-level out, put all beams in meta_info.
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
    """If this item is a beam search result, build and return the out_dict.
    Returns None if not a beam search item.
    """
    # Only support BatchTokenIDOutput or BatchStrOutput for beam search
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
    """Build the out_dict for a beam search result."""
    # recv_obj is guaranteed to be BatchStrOutput or BatchTokenIDOutput by the
    # only caller (try_build_beam_search_out_dict); the check is loop-invariant.
    # Only BatchStrOutput carries detokenized text.
    include_text = isinstance(recv_obj, BatchStrOutput)
    beam_results = []
    total_completion_tokens = sum(
        len(beam_seq.tokens) for beam_seq in beam_search_output.sequences
    )
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

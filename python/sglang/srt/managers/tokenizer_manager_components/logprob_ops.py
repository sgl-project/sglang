from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from sglang.srt.managers.io_struct import BatchStrOutput
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState

INCREMENTAL_STREAMING_META_INFO_KEYS = (
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
)


def slice_streaming_output_meta_info(
    meta_info: Dict[Any, Any],
    last_output_offset: int,
    customized_info_keys: Optional[Iterable[str]] = None,
) -> None:
    """Align output-side metadata with the current incremental streaming chunk."""
    streaming_meta_info_keys = set(INCREMENTAL_STREAMING_META_INFO_KEYS)
    if customized_info_keys is not None:
        streaming_meta_info_keys.update(customized_info_keys)
    for key in meta_info.keys() & streaming_meta_info_keys:
        meta_info[key] = meta_info[key][last_output_offset:]


def fill_meta_info(
    meta_info: dict,
    state: ReqState,
    *,
    top_logprobs_num: int,
    token_ids_logprob: Optional[List[int]],
    return_text_in_logprobs: bool,
    tokenizer: Optional[Any],
) -> None:
    # 1. Handle regular logprobs
    if len(state.input_token_logprobs_val) > len(state.input_token_logprobs):
        state.input_token_logprobs.extend(
            _detokenize_logprob_tokens(
                state.input_token_logprobs_val[len(state.input_token_logprobs) :],
                state.input_token_logprobs_idx[len(state.input_token_logprobs) :],
                decode_to_text=return_text_in_logprobs,
                tokenizer=tokenizer,
            )
        )

    if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
        state.output_token_logprobs.extend(
            _detokenize_logprob_tokens(
                state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                decode_to_text=return_text_in_logprobs,
                tokenizer=tokenizer,
            )
        )

    meta_info["input_token_logprobs"] = state.input_token_logprobs
    meta_info["output_token_logprobs"] = state.output_token_logprobs
    meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)

    # 2. Handle top logprobs
    if top_logprobs_num > 0:
        if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
            state.input_top_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.input_top_logprobs_val[len(state.input_top_logprobs) :],
                    state.input_top_logprobs_idx[len(state.input_top_logprobs) :],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )
        if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
            state.output_top_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                    state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )

        meta_info["input_top_logprobs"] = state.input_top_logprobs
        meta_info["output_top_logprobs"] = state.output_top_logprobs

    # 3. Handle token_ids_logprob
    if token_ids_logprob is not None:
        if len(state.input_token_ids_logprobs_val) > len(
            state.input_token_ids_logprobs
        ):
            state.input_token_ids_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.input_token_ids_logprobs_val[
                        len(state.input_token_ids_logprobs) :
                    ],
                    state.input_token_ids_logprobs_idx[
                        len(state.input_token_ids_logprobs) :
                    ],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )
        if len(state.output_token_ids_logprobs_val) > len(
            state.output_token_ids_logprobs
        ):
            state.output_token_ids_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.output_token_ids_logprobs_val[
                        len(state.output_token_ids_logprobs) :
                    ],
                    state.output_token_ids_logprobs_idx[
                        len(state.output_token_ids_logprobs) :
                    ],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )

        meta_info["input_token_ids_logprobs"] = state.input_token_ids_logprobs
        meta_info["output_token_ids_logprobs"] = state.output_token_ids_logprobs


def absorb_recv(
    meta_info: dict,
    state: ReqState,
    *,
    top_logprobs_num: int,
    token_ids_logprob: Optional[List[int]],
    return_text_in_logprobs: bool,
    recv_obj: BatchStrOutput,
    recv_obj_index: int,
    tokenizer: Optional[Any],
) -> None:
    if recv_obj.input_token_logprobs_val is None:
        return

    if (
        len(recv_obj.input_token_logprobs_val) > 0
        and recv_obj.input_token_logprobs_val[recv_obj_index] is not None
    ):
        state.input_token_logprobs_val.extend(
            recv_obj.input_token_logprobs_val[recv_obj_index]
        )
        state.input_token_logprobs_idx.extend(
            recv_obj.input_token_logprobs_idx[recv_obj_index]
        )
    state.output_token_logprobs_val.extend(
        recv_obj.output_token_logprobs_val[recv_obj_index]
    )
    state.output_token_logprobs_idx.extend(
        recv_obj.output_token_logprobs_idx[recv_obj_index]
    )

    if top_logprobs_num > 0:
        if len(recv_obj.input_top_logprobs_val) > 0:
            state.input_top_logprobs_val.extend(
                recv_obj.input_top_logprobs_val[recv_obj_index]
            )
            state.input_top_logprobs_idx.extend(
                recv_obj.input_top_logprobs_idx[recv_obj_index]
            )
        state.output_top_logprobs_val.extend(
            recv_obj.output_top_logprobs_val[recv_obj_index]
        )
        state.output_top_logprobs_idx.extend(
            recv_obj.output_top_logprobs_idx[recv_obj_index]
        )

    if token_ids_logprob is not None:
        if len(recv_obj.input_token_ids_logprobs_val) > 0:
            state.input_token_ids_logprobs_val.extend(
                recv_obj.input_token_ids_logprobs_val[recv_obj_index]
            )
            state.input_token_ids_logprobs_idx.extend(
                recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
            )
        state.output_token_ids_logprobs_val.extend(
            recv_obj.output_token_ids_logprobs_val[recv_obj_index]
        )
        state.output_token_ids_logprobs_idx.extend(
            recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
        )

    fill_meta_info(
        meta_info,
        state,
        top_logprobs_num=state.obj.top_logprobs_num,
        token_ids_logprob=state.obj.token_ids_logprob,
        return_text_in_logprobs=return_text_in_logprobs,
        tokenizer=tokenizer,
    )


def _detokenize_logprob_tokens(
    token_logprobs_val: List[float],
    token_logprobs_idx: List[int],
    *,
    decode_to_text: bool,
    tokenizer: Optional[Any],
) -> List[Tuple[float, int, Optional[str]]]:
    if not decode_to_text:
        return [
            (logprob, token_id, None)
            for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
        ]
    else:
        assert tokenizer is not None
        # In transformers v5, batch_decode([1, 2, 3]) concatenates all tokens
        # into one string. Wrap each ID in its own list so they decode separately.
        token_texts = tokenizer.batch_decode([[idx] for idx in token_logprobs_idx])
        return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))


def _detokenize_top_logprobs_tokens(
    token_logprobs_val: List[List[float]],
    token_logprobs_idx: List[List[int]],
    *,
    decode_to_text: bool,
    tokenizer: Optional[Any],
) -> List[Optional[List[Tuple[float, int, Optional[str]]]]]:
    # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
    # We should batch all top-k tokens in all positions.
    ret = []
    for i in range(len(token_logprobs_val)):
        if token_logprobs_val[i]:
            ret.append(
                _detokenize_logprob_tokens(
                    token_logprobs_val[i],
                    token_logprobs_idx[i],
                    decode_to_text=decode_to_text,
                    tokenizer=tokenizer,
                )
            )
        else:
            ret.append(None)
    return ret

# SPDX-License-Identifier: Apache-2.0

from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session


def test_streaming_session_drop_previous_output_commits_without_duplicate():
    session = Session(capacity_of_str_len=1024, session_id="s", streaming=True)
    first = session.create_req(
        _req("r1", [1, 2], max_new_tokens=1),
        tokenizer=None,
        vocab_size=32000,
    )
    first.output_ids = [99]
    session.finish_req(first)

    commit = session.create_req(
        _req(
            "r2",
            [99],
            session_rid="r1",
            max_new_tokens=0,
            drop_previous_output=True,
        ),
        tokenizer=None,
        vocab_size=32000,
    )

    assert commit.to_finish is None
    assert commit.origin_input_ids == [1, 2, 99]


def _req(
    rid,
    input_ids,
    *,
    session_rid=None,
    max_new_tokens,
    drop_previous_output=False,
):
    sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="",
        input_ids=list(input_ids),
        mm_inputs=None,
        sampling_params=sampling_params,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(
            id="s",
            rid=session_rid,
            replace=False,
            drop_previous_output=drop_previous_output,
        ),
    )

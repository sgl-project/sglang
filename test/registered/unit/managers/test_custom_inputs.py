import pytest

from sglang.srt.managers.io_struct import GenerateReqInput


def test_custom_inputs_single_request():
    metadata = {"acoustic_embedding": [[1.0, 2.0]], "input_function_ids": [7]}
    req = GenerateReqInput(input_ids=[1], custom_inputs=metadata)

    req.normalize_batch_and_arguments()

    assert req.is_single
    assert req.custom_inputs is metadata


def test_custom_inputs_batch_normalization():
    metadata = [{"value": 1}, None]
    req = GenerateReqInput(input_ids=[[1], [2]], custom_inputs=metadata)

    req.normalize_batch_and_arguments()

    assert req[0].custom_inputs == {"value": 1}
    assert req[1].custom_inputs is None


def test_custom_inputs_batch_size_must_match():
    req = GenerateReqInput(input_ids=[[1], [2]], custom_inputs=[{"value": 1}])

    with pytest.raises(ValueError, match="length of custom_inputs"):
        req.normalize_batch_and_arguments()


def test_custom_inputs_values_must_be_dicts():
    with pytest.raises(ValueError, match="single request"):
        GenerateReqInput(
            input_ids=[1], custom_inputs=[1]
        ).normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="dict or None"):
        GenerateReqInput(
            input_ids=[[1], [2]], custom_inputs=[{"value": 1}, 2]
        ).normalize_batch_and_arguments()


def test_empty_input_ids_only_allowed_for_session_continuation():
    with pytest.raises(ValueError, match="outside a session continuation"):
        GenerateReqInput(input_ids=[]).normalize_batch_and_arguments()

    req = GenerateReqInput(
        input_ids=[],
        session_params={"id": "voicechat", "rid": None},
        custom_inputs={"input_function_ids": [0]},
    )
    req.normalize_batch_and_arguments()
    assert req.is_single
    assert req.input_ids == []

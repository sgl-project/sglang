from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.utils import (
    resolve_disagg_max_req_input_len,
    validate_input_length,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_resolve_disagg_input_limit_defaults_to_local_limit():
    assert resolve_disagg_max_req_input_len(200, None) == (200, None, None)


def test_resolve_disagg_input_limit_accepts_smaller_pair_limit():
    assert resolve_disagg_max_req_input_len(200, 100) == (100, None, None)


def test_resolve_disagg_input_limit_rejects_invalid_value():
    limit, error, status = resolve_disagg_max_req_input_len(200, 0)
    assert limit is None
    assert "must be positive" in error
    assert status == 400


def test_resolve_disagg_input_limit_rejects_value_above_local_capacity():
    limit, error, status = resolve_disagg_max_req_input_len(100, 200)
    assert limit is None
    assert "exceeds this worker's local capacity" in error
    assert status == 503


def test_validate_input_length_truncates_to_shared_limit_when_enabled():
    req = SimpleNamespace(origin_input_ids=list(range(12)))
    assert validate_input_length(req, 8, allow_auto_truncate=True) is None
    assert req.origin_input_ids == list(range(8))


def test_validate_input_length_rejects_shared_limit_when_disabled():
    req = SimpleNamespace(origin_input_ids=list(range(12)))
    error = validate_input_length(req, 8, allow_auto_truncate=False)
    assert "exceeds the maximum allowed length (8 tokens)" in error
    assert len(req.origin_input_ids) == 12


def _make_tokenizer_manager(
    *,
    allow_auto_truncate: bool,
    context_len: int = 64,
    disaggregation_mode: DisaggregationMode = DisaggregationMode.NULL,
):
    manager = object.__new__(TokenizerManager)
    manager.context_len = context_len
    manager.num_reserved_tokens = 0
    manager.allow_auto_truncate = allow_auto_truncate
    manager.validate_total_tokens = True
    manager.disaggregation_mode = disaggregation_mode
    return manager


def test_tokenizer_truncates_to_shared_limit_before_total_token_validation():
    manager = _make_tokenizer_manager(allow_auto_truncate=True, context_len=20)
    input_ids = list(range(12))
    obj = SimpleNamespace(sampling_params={"max_new_tokens": 10})

    manager._validate_one_request(obj, input_ids, max_req_input_len=8)

    assert input_ids == list(range(8))
    assert obj.sampling_params["max_new_tokens"] == 10


def test_tokenizer_rejects_shared_limit_when_auto_truncate_is_disabled():
    manager = _make_tokenizer_manager(allow_auto_truncate=False)
    input_ids = list(range(12))
    obj = SimpleNamespace(sampling_params={"max_new_tokens": 1})

    try:
        manager._validate_one_request(obj, input_ids, max_req_input_len=8)
    except ValueError as error:
        assert "PD shared input limit (8 tokens)" in str(error)
    else:
        raise AssertionError("expected the shared input limit to reject the request")

    assert input_ids == list(range(12))


def test_prefill_does_not_validate_decode_output_budget_against_local_context():
    manager = _make_tokenizer_manager(
        allow_auto_truncate=False,
        context_len=20,
        disaggregation_mode=DisaggregationMode.PREFILL,
    )
    input_ids = list(range(8))
    obj = SimpleNamespace(sampling_params={"max_new_tokens": 32})

    manager._validate_one_request(obj, input_ids, max_req_input_len=12)

    assert input_ids == list(range(8))
    assert obj.sampling_params["max_new_tokens"] == 32


def test_decode_still_validates_output_budget_against_local_context():
    manager = _make_tokenizer_manager(
        allow_auto_truncate=False,
        context_len=20,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    input_ids = list(range(8))
    obj = SimpleNamespace(sampling_params={"max_new_tokens": 32})

    try:
        manager._validate_one_request(obj, input_ids, max_req_input_len=12)
    except ValueError as error:
        assert "maximum context length" in str(error)
    else:
        raise AssertionError("expected decode to validate the full output budget")


def test_openai_pd_requests_accept_shared_input_limit():
    completion = CompletionRequest(
        model="test-model",
        prompt="hello",
        disagg_max_req_input_len=128,
    )
    chat = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        disagg_max_req_input_len=128,
    )

    assert completion.disagg_max_req_input_len == 128
    assert chat.disagg_max_req_input_len == 128


def test_generate_batch_preserves_scalar_shared_input_limit():
    req = GenerateReqInput(
        input_ids=[[1, 2, 3], [4, 5, 6]],
        sampling_params=[{}, {}],
        disagg_max_req_input_len=128,
    )
    req.normalize_batch_and_arguments()

    assert req[0].disagg_max_req_input_len == 128
    assert req[1].disagg_max_req_input_len == 128


def test_chat_prefill_skips_decode_output_budget_precheck():
    serving = object.__new__(OpenAIServingChat)
    serving.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(
            context_length=20,
            allow_auto_truncate=False,
        ),
        disaggregation_mode=DisaggregationMode.PREFILL,
    )
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=32,
        disagg_max_req_input_len=12,
    )

    assert serving._validate_request(request) is None

    serving.tokenizer_manager.disaggregation_mode = DisaggregationMode.DECODE
    assert "too large" in serving._validate_request(request)

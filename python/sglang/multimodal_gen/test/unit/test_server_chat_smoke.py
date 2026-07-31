from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.runtime.server_warmup import (
    MINIMUM_PICTURE_BASE64_FOR_WARMUP,
)
from sglang.multimodal_gen.test.server.test_server_common import DiffusionServerBase
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)


@pytest.fixture
def understanding_case() -> DiffusionTestCase:
    return DiffusionTestCase(
        "bagel_understanding_i2t",
        DiffusionServerArgs(
            model_path="ByteDance-Seed/BAGEL-7B-MoT",
            modality="text",
            extras=["--pipeline-class-name", "BagelUnderstandingPipeline"],
        ),
        DiffusionSamplingParams(
            prompt="Describe this image.",
            image_path=MINIMUM_PICTURE_BASE64_FOR_WARMUP,
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    )


def _chat_response(
    *,
    choice_count: int = 1,
    content: str = "A solid white square.",
    finish_reason: str = "stop",
    include_usage: bool = True,
    prompt_tokens: int = 12,
    completion_tokens: int = 5,
    total_tokens: int = 17,
) -> SimpleNamespace:
    """Build a minimal OpenAI-compatible Chat response for harness tests."""
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )
        for _ in range(choice_count)
    ]
    usage = (
        SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        if include_usage
        else None
    )
    return SimpleNamespace(id="chatcmpl-test", choices=choices, usage=usage)


def _configure_synchronous_chat(
    server: DiffusionServerBase,
    response: SimpleNamespace,
) -> Mock:
    """Replace external Chat and watchdog boundaries with synchronous fakes."""
    create = Mock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    server._client = Mock(return_value=client)
    server._run_generation_with_server_watchdog = Mock(
        side_effect=lambda ctx, case_id, generate_fn, request_client: generate_fn(
            case_id, request_client
        )
    )
    return create


def test_chat_completion_smoke_validates_request_and_response(
    understanding_case: DiffusionTestCase,
) -> None:
    server = DiffusionServerBase()
    create = _configure_synchronous_chat(server, _chat_response())
    ctx = SimpleNamespace(port=30000)

    server._test_chat_completion_smoke(ctx, understanding_case)

    create.assert_called_once_with(
        model="ByteDance-Seed/BAGEL-7B-MoT",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": MINIMUM_PICTURE_BASE64_FOR_WARMUP},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        max_completion_tokens=64,
        temperature=0,
    )


@pytest.mark.parametrize(
    ("response_kwargs", "error_match"),
    [
        ({"choice_count": 0}, "exactly one chat completion choice"),
        ({"content": "  "}, "assistant content must be non-empty text"),
        ({"finish_reason": "content_filter"}, "unsupported finish reason"),
        ({"include_usage": False}, "token usage must be present"),
        ({"prompt_tokens": 0}, "prompt token count must be positive"),
        ({"completion_tokens": 0}, "completion token count must be positive"),
        ({"total_tokens": 99}, "total token count must equal"),
    ],
)
def test_chat_completion_smoke_rejects_invalid_response_contract(
    understanding_case: DiffusionTestCase,
    response_kwargs: dict[str, Any],
    error_match: str,
) -> None:
    server = DiffusionServerBase()
    _configure_synchronous_chat(server, _chat_response(**response_kwargs))

    with pytest.raises(AssertionError, match=error_match):
        server._test_chat_completion_smoke(
            SimpleNamespace(port=30000),
            understanding_case,
        )


def test_text_task_routes_around_diffusion_generation(
    understanding_case: DiffusionTestCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_GEN_GT", "1")
    server = DiffusionServerBase()
    server._test_chat_completion_smoke = Mock()
    server.run_and_collect = Mock(
        side_effect=AssertionError("text task must not collect diffusion perf")
    )
    server._save_gt_output = Mock()
    server._validate_and_record = Mock()
    server._validate_consistency = Mock()
    ctx = SimpleNamespace(port=30000)

    with patch(
        "sglang.multimodal_gen.test.server.test_server_common.get_generate_fn",
        side_effect=AssertionError("text task must not use media generation"),
    ) as get_generate_fn:
        server._test_diffusion_generation_impl(understanding_case, ctx)

    server._test_chat_completion_smoke.assert_called_once_with(ctx, understanding_case)
    get_generate_fn.assert_not_called()
    server.run_and_collect.assert_not_called()
    server._save_gt_output.assert_not_called()
    server._validate_and_record.assert_not_called()
    server._validate_consistency.assert_not_called()

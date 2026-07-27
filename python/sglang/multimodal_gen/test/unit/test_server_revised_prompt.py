from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.runtime.server_warmup import (
    MINIMUM_PICTURE_BASE64_FOR_WARMUP,
)
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
)


def _image_client(revised_prompt: str | None) -> tuple[SimpleNamespace, Mock]:
    """Build a minimal OpenAI image client with one embedded PNG response."""
    image_b64 = MINIMUM_PICTURE_BASE64_FOR_WARMUP.split(",", maxsplit=1)[1]
    parsed = SimpleNamespace(
        id="image-test",
        created=1700000000,
        data=[
            SimpleNamespace(
                b64_json=image_b64,
                revised_prompt=revised_prompt,
            )
        ],
    )
    generate = Mock(return_value=SimpleNamespace(parse=Mock(return_value=parsed)))
    client = SimpleNamespace(
        images=SimpleNamespace(
            with_raw_response=SimpleNamespace(generate=generate),
        )
    )
    return client, generate


def _thinking_sampling_params() -> DiffusionSamplingParams:
    """Return the small deterministic request used by the Thinking smoke."""
    return DiffusionSamplingParams(
        prompt="A small blue robot holding a red flower.",
        output_size="64x64",
        output_format="png",
        extras={
            "num_inference_steps": 2,
            "max_think_tokens": 16,
            "think_do_sample": False,
        },
    )


def test_generation_accepts_distinct_revised_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_params = _thinking_sampling_params()
    revised_prompt = f"{sampling_params.prompt}\n<think>Use a centered subject.</think>"
    client, generate = _image_client(revised_prompt)
    generate_fn = get_generate_fn(
        model_path="ByteDance-Seed/BAGEL-7B-MoT",
        modality="image",
        sampling_params=sampling_params,
        run_revised_prompt_check=True,
    )
    monkeypatch.chdir(tmp_path)

    with patch(
        "sglang.multimodal_gen.test.server.test_server_utils.upload_file_to_slack"
    ):
        request_id, image_bytes = generate_fn("bagel_thinking_t2i", client)

    assert request_id == "image-test"
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    generate.assert_called_once_with(
        model="ByteDance-Seed/BAGEL-7B-MoT",
        prompt=sampling_params.prompt,
        n=1,
        size="64x64",
        response_format="b64_json",
        output_format="png",
        extra_body={
            "num_inference_steps": 2,
            "max_think_tokens": 16,
            "think_do_sample": False,
        },
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("revised_prompt", "error_match"),
    [
        (None, "revised_prompt must be non-empty"),
        ("  ", "revised_prompt must be non-empty"),
        (
            "A small blue robot holding a red flower.",
            "revised_prompt must differ from the input prompt",
        ),
        (
            "An unrelated rewritten prompt.",
            "revised_prompt must start with the input prompt followed by a newline",
        ),
        (
            "A small blue robot holding a red flower.\n   ",
            "revised_prompt must include a non-empty planning suffix",
        ),
    ],
)
def test_generation_rejects_missing_or_fallback_revised_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revised_prompt: str | None,
    error_match: str,
) -> None:
    sampling_params = _thinking_sampling_params()
    client, _ = _image_client(revised_prompt)
    generate_fn = get_generate_fn(
        model_path="ByteDance-Seed/BAGEL-7B-MoT",
        modality="image",
        sampling_params=sampling_params,
        run_revised_prompt_check=True,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError, match=error_match):
        generate_fn("bagel_thinking_t2i", client)

    assert list(tmp_path.iterdir()) == []

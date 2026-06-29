# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _resolve_output_path_for_request,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)


def test_image_request_defaults_to_server_output_path():
    request = ImageGenerationsRequest(prompt="a cat")

    output_path, is_persistent = _resolve_output_path_for_request(request, "outputs/")

    assert output_path == "outputs/"
    assert is_persistent


def test_image_request_save_output_false_uses_temporary_output_path():
    request = ImageGenerationsRequest(
        prompt="a cat",
        response_format="b64_json",
        save_output=False,
    )

    output_path, is_persistent = _resolve_output_path_for_request(request, "outputs/")

    assert output_path is None
    assert not is_persistent


def test_image_request_save_output_false_from_extra_body():
    request = ImageGenerationsRequest(
        prompt="a cat",
        response_format="b64_json",
        extra_body={"save_output": False},
    )

    output_path, is_persistent = _resolve_output_path_for_request(request, "outputs/")

    assert output_path is None
    assert not is_persistent


def test_image_request_output_path_override():
    request = ImageGenerationsRequest(prompt="a cat", output_path="custom-output")

    output_path, is_persistent = _resolve_output_path_for_request(request, "outputs/")

    assert output_path == "custom-output"
    assert is_persistent

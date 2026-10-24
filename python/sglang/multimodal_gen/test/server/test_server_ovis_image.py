# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP lifecycle checks using a public HF checkpoint or a local directory.

Run with MODEL_PATH=ATH-MaaS/Ovis-Image-7B pytest -s <this file>.
The model path belongs to the external test harness; ordinary CI collection
does not download a full checkpoint.
"""

import base64
import io
import os
import sys

import numpy as np
import pytest
from openai import OpenAI
from PIL import Image

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port
from sglang.test.test_utils import terminate_and_kill_process_tree

pytestmark = pytest.mark.skipif(
    not os.environ.get("MODEL_PATH"),
    reason="set MODEL_PATH to the Ovis-Image checkpoint for full-model HTTP tests",
)


def create_ovis_image_server(model):
    manager = ServerManager(
        model=model,
        port=get_dynamic_server_port(),
        extra_args=(
            "--num-gpus 1 --tp-size 1 --sp-degree 1 --performance-mode manual "
            "--warmup-mode off --attention-backend torch_sdpa "
            "--enable-torch-compile false --enable-breakable-cuda-graph false"
        ),
    )
    return manager.start()


@pytest.fixture(scope="module")
def server():
    context = create_ovis_image_server(os.environ["MODEL_PATH"])
    try:
        yield context
    finally:
        try:
            terminate_and_kill_process_tree(context.process)
        finally:
            context.cleanup()


def _generate(client, model, *, prompt, size="512x512", outputs=1):
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=outputs,
        response_format="b64_json",
        extra_body={
            "seed": 42,
            "num_inference_steps": 4,
            "guidance_scale": 5.0,
            "negative_prompt": "",
            "generator_device": "cpu",
            "output_format": "png",
        },
    )
    assert len(result.data) == outputs
    width, height = map(int, size.split("x"))
    images = []
    for item in result.data:
        with Image.open(io.BytesIO(base64.b64decode(item.b64_json))) as image:
            assert image.size == (width, height)
            pixels = np.asarray(image.convert("RGB"))
        assert np.isfinite(pixels).all()
        assert pixels.std() > 5, "degenerate output"
        images.append(pixels)
    return images


def test_repeated_generation_and_request_state(server):
    """Changed conditioning/shape must not contaminate a repeated seed or output batch."""
    prompt = "A red ceramic teapot on a white table, watercolor painting."
    with OpenAI(
        api_key="EMPTY",
        base_url=f"http://127.0.0.1:{server.port}/v1",
        timeout=600,
        max_retries=0,
    ) as client:
        baseline = _generate(client, server.model, prompt=prompt)
        repeated = _generate(client, server.model, prompt=prompt)
        np.testing.assert_array_equal(baseline[0], repeated[0])

        changed = _generate(
            client,
            server.model,
            prompt="A blue sailboat on a calm sea, watercolor painting.",
        )
        assert not np.array_equal(baseline[0], changed[0])
        _generate(client, server.model, prompt=prompt, size="640x384")
        restored = _generate(client, server.model, prompt=prompt)
        np.testing.assert_array_equal(baseline[0], restored[0])

        outputs = _generate(client, server.model, prompt=prompt, outputs=2)
        repeated_outputs = _generate(client, server.model, prompt=prompt, outputs=2)
        for actual, expected in zip(outputs, repeated_outputs, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert not np.array_equal(outputs[0], outputs[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

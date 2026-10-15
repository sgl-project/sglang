# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP validation: SGLANG_ANIMA_TEST_MODEL=<HF id or local checkpoint>."""

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

pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_ANIMA_TEST_MODEL"),
    reason="set SGLANG_ANIMA_TEST_MODEL for full-checkpoint HTTP tests",
)


@pytest.fixture(scope="module")
def server():
    manager = ServerManager(
        model=os.environ["SGLANG_ANIMA_TEST_MODEL"],
        port=get_dynamic_server_port(),
        extra_args="--num-gpus 1 --performance-mode speed --attention-backend fa",
    )
    context = manager.start()
    try:
        yield context
    finally:
        context.cleanup()


@pytest.mark.parametrize("size,steps,outputs", [(512, 4, 2), (1024, 30, 1)])
def test_repeated_generation(server, size, steps, outputs):
    with OpenAI(
        api_key="EMPTY", base_url=f"http://127.0.0.1:{server.port}/v1"
    ) as client:
        requests = []
        for _ in range(2):
            result = client.images.generate(
                model=server.model,
                prompt="masterpiece, best quality, safe, watercolor landscape, a quiet seaside village at sunset",
                size=f"{size}x{size}",
                n=outputs,
                response_format="b64_json",
                extra_body={
                    "seed": 42,
                    "num_inference_steps": steps,
                    "guidance_scale": 4.0,
                    "negative_prompt": "",
                    "generator_device": "cpu",
                    "output_format": "png",
                },
            )
            assert len(result.data) == outputs
            images = []
            for item in result.data:
                with Image.open(io.BytesIO(base64.b64decode(item.b64_json))) as image:
                    assert image.size == (size, size)
                    pixels = np.asarray(image.convert("RGB"))
                    assert pixels.std() > 5, "degenerate image"
                    images.append(pixels)
            requests.append(images)
        for first, second in zip(*requests):
            np.testing.assert_array_equal(first, second)
        if outputs > 1:
            assert not np.array_equal(requests[0][0], requests[0][1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

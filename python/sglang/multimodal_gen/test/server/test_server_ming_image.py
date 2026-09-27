# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP regression tests for both public Ming checkpoints.

Set SGLANG_TEST_MING_IMAGE=1 on a GPU host with sufficient device/host memory.
"""

import base64
import io
import os

import numpy as np
import pytest
import requests
from PIL import Image, ImageDraw

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_MING_IMAGE") != "1",
    reason="opt-in full-checkpoint Ming-Image test",
)


@pytest.fixture(
    params=[
        ("Design", False, "off", False),
        ("Design-Layer", False, "off", False),
        ("Design-Layer", True, "off", False),
        ("Design", False, "request", False),
        ("Design-Layer", False, "request", False),
        ("Design-Layer", False, "off", True),
    ],
    ids=[
        "design-cold",
        "layer-cold",
        "layer-tiled",
        "design-warm",
        "layer-warm",
        "layer-offload",
    ],
)
def case(request):
    checkpoint, tiled, warmup, offload = request.param
    extras = [
        "--performance-mode speed",
        f"--warmup-mode {warmup}",
        f"--vae-tiling {str(tiled).lower()}",
    ]
    if offload:
        extras.append(
            "--component-residency text_encoder=layerwise-offload transformer=layerwise-offload"
        )
    return DiffusionTestCase(
        f"ming_image_{checkpoint.lower()}_{'offload' if offload else 'tiled' if tiled else 'full'}_{warmup}",
        DiffusionServerArgs(
            model_path=f"inclusionAI/Ming-Image-0.1-{checkpoint}",
            modality="image",
            extras=extras,
        ),
        DiffusionSamplingParams(output_size="512x512"),
    )


def test_repeated_generation_and_editing(diffusion_server, case):
    url = f"http://localhost:{diffusion_server.port}/v1/images"
    layered = case.server_args.model_path.endswith("-Layer")
    reference = Image.new("RGBA", (512, 512), "white")
    ImageDraw.Draw(reference).rectangle((120, 120, 390, 390), fill="red")
    buffer = io.BytesIO()
    reference.save(buffer, format="PNG")
    modes = ["edit"] if layered else ["generation", "edit"]
    for mode in modes:
        payload = {
            "prompt": (
                "Decompose this image into 2 layers."
                if layered
                else "Change the red square to blue."
                if mode == "edit"
                else "A red ceramic teapot on a white background."
            ),
            "size": "512x512",
            "num_inference_steps": 12,
            "seed": 42,
            "output_format": "png",
            "response_format": "b64_json",
        }
        if layered:
            payload["num_layers"] = 2
        previous = {}
        for count in (1, 1, 2, 2):
            payload["n"] = count
            if mode == "edit":
                response = requests.post(
                    f"{url}/edits",
                    data=payload,
                    files={"image": ("input.png", buffer.getvalue(), "image/png")},
                    timeout=300,
                )
            else:
                response = requests.post(
                    f"{url}/generations", json=payload, timeout=300
                )
            assert response.ok, response.text
            data = response.json()["data"]
            assert len(data) == count * (2 if layered else 1)
            images = []
            for item in data:
                with Image.open(
                    io.BytesIO(base64.b64decode(item["b64_json"]))
                ) as image:
                    assert image.mode == "RGBA"
                    assert image.size == (512, 512)
                    pixels = np.array(image)
                    images.append(pixels)
            # a decomposed background layer can legitimately be a solid color
            assert max(pixels[..., :3].std() for pixels in images) > 5
            if count in previous:
                for actual, expected in zip(images, previous[count], strict=True):
                    np.testing.assert_array_equal(actual, expected)
            if count == 2:
                for actual, expected in zip(images, previous[1]):
                    np.testing.assert_array_equal(actual, expected)
            previous[count] = images

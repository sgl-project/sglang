# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-checkpoint test: SGLANG_TEST_FLUX3_IMAGE=1 pytest <this file>."""

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import requests
from PIL import Image
from skimage.metrics import structural_similarity

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_FLUX3_IMAGE") != "1",
    reason="requires access to the full FLUX.3 image checkpoint",
)


@pytest.fixture
def case():
    return DiffusionTestCase(
        "flux3_image_batching",
        DiffusionServerArgs(
            model_path=os.environ.get(
                "SGLANG_TEST_FLUX3_IMAGE_MODEL",
                "black-forest-labs/flux-3-image-klein-base",
            ),
            extras=[
                "--performance-mode manual --component-residency all=resident",
                "--attention-backend torch_sdpa --warmup-mode off",
                "--batching-max-size 2 --batching-delay-ms 50",
            ],
        ),
        DiffusionSamplingParams(output_size="512x512"),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    )


def test_grouped_requests_multi_output_and_repeat(diffusion_server):
    url = f"http://localhost:{diffusion_server.port}/v1/images"
    reference = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(reference, format="PNG")

    def generate(index=0, editing=False, n=1, barrier=None):
        payload = dict(
            prompt=["A red fox in snow", "A vase of yellow tulips"][index],
            seed=42 + index,
            size="512x512",
            num_inference_steps=8,
            n=n,
            quality="exact",
            output_format="png",
            response_format="b64_json",
        )
        if barrier is not None:
            barrier.wait()
        if editing:
            response = requests.post(
                f"{url}/edits",
                data=payload,
                files={"image": ("reference.png", reference.getvalue(), "image/png")},
                timeout=120,
            )
        else:
            response = requests.post(f"{url}/generations", json=payload, timeout=120)
        assert response.ok, response.text
        data = response.json()["data"]
        assert len(data) == n
        return [
            np.asarray(Image.open(io.BytesIO(base64.b64decode(row["b64_json"]))))
            for row in data
        ]

    for editing in (False, True):
        serial = [generate(index, editing)[0] for index in range(2)]
        np.testing.assert_array_equal(generate(editing=editing)[0], serial[0])
        barrier = Barrier(2)
        with ThreadPoolExecutor(2) as pool:
            grouped = list(
                pool.map(
                    lambda index: generate(index, editing, barrier=barrier)[0], range(2)
                )
            )
        for expected, actual in zip(serial, grouped):
            assert actual.shape == expected.shape == (512, 512, 3)
            # Compare lossless PNGs, not JPEG's amplified rounding differences.
            score = structural_similarity(
                expected, actual, channel_axis=2, data_range=255
            )
            assert score > 0.98, score
            assert np.abs(expected.astype(float) - actual).mean() < 3.0
        multi = generate(editing=editing, n=2)
        assert not np.array_equal(multi[0], multi[1])
        np.testing.assert_array_equal(generate(editing=editing)[0], serial[0])
    assert "Processed native grouped batch" in diffusion_server.stdout_file.read_text()

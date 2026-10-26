# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import av
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen import DiffGenerator
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="diffusion-1-gpu-h100")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        os.environ.get("DIFFUSION_PARTITION_ID", "0") != "0",
        reason="model smoke runs on diffusion partition 0",
    ),
]


@pytest.fixture(scope="module")
def generator():
    generator = DiffGenerator.from_pretrained(
        model_path="Efficient-Large-Model/SANA-Video_2.0_5B_720p",
        warmup_mode="off",
    )
    try:
        yield generator
    finally:
        generator.shutdown()


@pytest.mark.parametrize("mode", ["t2v", "ti2v"])
def test_sana_video2_smoke(generator, tmp_path, mode):
    params = dict(
        prompt="A red cube on a white table.",
        width=832,
        height=480,
        num_frames=17,
        fps=24,
        num_inference_steps=8,
        guidance_scale=8.0,
        flow_shift=3.0,
        seed=42,
        save_output=True,
        output_path=str(tmp_path),
        output_file_name=f"{mode}.mp4",
    )
    if mode == "ti2v":
        image_path = tmp_path / "first_frame.png"
        Image.new("RGB", (832, 480), "red").save(image_path)
        params["image_path"] = str(image_path)
    result = generator.generate(params)
    assert Path(result.output_file_path).is_file()
    with av.open(result.output_file_path) as video:
        stream = video.streams.video[0]
        assert (stream.width, stream.height) == (832, 480)
        assert stream.average_rate == 24
        assert sum(1 for _ in video.decode(video=0)) == 17


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

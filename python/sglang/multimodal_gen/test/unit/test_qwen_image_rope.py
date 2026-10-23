# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenEmbedLayer3DRope,
    QwenEmbedRope,
)


def _reference_freqs(coordinates, axes, device):
    frequencies = []
    for index, dim in zip(coordinates, axes):
        index = index.to(device)
        scale = 1.0 / torch.pow(
            10000, torch.arange(0, dim, 2, device=device).float().div(dim)
        )
        phase = index.unsqueeze(-1) * scale
        frequencies.append(torch.polar(torch.ones_like(phase), phase))
    return torch.cat(frequencies, dim=-1)


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("init_device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_qwen_image_rope_coordinates_and_device_lifecycle(
    layered, scale, init_device, dtype, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device)
    axes = [4, 6, 6]
    with torch.device(init_device):
        rope = (QwenEmbedLayer3DRope if layered else QwenEmbedRope)(10000, axes, scale)
    rope.to(dtype=dtype)
    reference_device = device if init_device == "meta" else torch.device("cpu")
    assert not rope.state_dict() and not list(rope.buffers())
    try:
        for shapes in [[(2, 3, 4), (1, 4, 3)], [(1, 1, 1)] * 5]:
            expected_images = []
            text_start = len(shapes) - 1 if layered else 0
            for index, (frames, height, width) in enumerate(shapes):
                frame_index = (
                    torch.tensor([-1])
                    if layered and index == len(shapes) - 1
                    else torch.arange(frames) + index
                )
                coordinates = torch.meshgrid(
                    frame_index,
                    torch.arange(height) - ((height + 1) // 2 if scale else 0),
                    torch.arange(width) - ((width + 1) // 2 if scale else 0),
                    indexing="ij",
                )
                expected_images.append(
                    _reference_freqs(coordinates, axes, reference_device).flatten(0, 2)
                )
                text_start = max(
                    text_start,
                    height // 2 if scale else height,
                    width // 2 if scale else width,
                )
            text_index = torch.arange(text_start, text_start + 7)
            expected = (
                torch.cat(expected_images).to(device),
                _reference_freqs([text_index] * 3, axes, reference_device).to(device),
            )
            for _ in range(2):
                actual = rope([shapes], [3, 7], device)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert rope.pos_freqs.dtype == rope.neg_freqs.dtype == torch.complex64
            assert rope.pos_freqs.device.type == device.type
        assert rope._compute_video_freqs.cache_info().maxsize == (
            None if layered else 128
        )
    finally:
        rope._compute_video_freqs.cache_clear()
        if layered:
            rope._compute_condition_freqs.cache_clear()

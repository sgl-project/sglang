from unittest.mock import patch

import torch

from sglang.srt.utils.video_decoder import _pin_memory_if_available
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_decoded_frames_remain_pageable_without_cuda():
    frames = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)

    with patch("torch.cuda.is_available", return_value=False), patch.object(
        torch.Tensor, "pin_memory", side_effect=AssertionError("must not pin")
    ):
        result = _pin_memory_if_available(frames)

    assert result is frames

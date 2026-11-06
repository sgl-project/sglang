"""Qwen3-VL video chat completion on Intel XPU.

Video frames decode on the XPU media engine (torchcodec-xpu) and flow into
the on-device resize and HF fast processor. The shared video mixin supplies
the clip and the content assertions used by the CUDA VLM suite.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.vlm_utils import VideoOpenAITestMixin

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")

XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--tp=1",
    "--disable-overlap-schedule",
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.70",
    "--attention-backend",
    "intel_xpu",
]


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestQwen3VLVideoXPU(VideoOpenAITestMixin):
    model = "Qwen/Qwen3-VL-2B-Instruct"
    extra_args = XPU_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        # The server process inherits the override, so it decodes on the media engine.
        with envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.override(True):
            super().setUpClass()


# Delete the mixin class so that unittest/pytest do not collect it on its own.
del VideoOpenAITestMixin

if __name__ == "__main__":
    unittest.main()

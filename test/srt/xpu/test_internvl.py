"""
XPU tests for InternVL models (InternVL2.5-2B, InternVL3.5-2B).

Uses the same structure as test_vision_openai_server_a.py: OpenAI /v1 chat API
and ImageOpenAITestMixin. An XPU-specific base injects --device xpu and
--attention-backend intel_xpu.

Usage (pick module path to match your cwd):

  From test/srt/xpu:
    python3 -m unittest test_internvl.TestInternVL25Server.test_single_image_chat_completion
    python3 -m unittest test_internvl

  From test/srt:
    python3 -m unittest xpu.test_internvl.TestInternVL25Server.test_single_image_chat_completion
    python3 -m unittest xpu.test_internvl

  From repo root:
    python3 -m unittest test.srt.xpu.test_internvl.TestInternVL25Server.test_single_image_chat_completion
    python3 -m unittest test.srt.xpu.test_internvl
"""

import os
import unittest

from sglang.test.vlm_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ImageOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    kill_process_tree,
    popen_launch_server,
)

# XPU args injected into server launch for all InternVL XPU tests
XPU_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "intel_xpu",
]


# Longer launch timeout for InternVL3.5 (can be slow to start on XPU)
INTERNVL35_LAUNCH_TIMEOUT = 900


class InternVLXPUServerBase(TestOpenAIMLLMServerBase):
    """Base for InternVL tests on XPU. Injects XPU args and sets SGLANG_USE_SGL_XPU."""

    use_sgl_xpu = True  # subclasses override for Triton backend
    launch_timeout = None  # subclasses can set to override (seconds)

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        os.environ["SGLANG_USE_SGL_XPU"] = "1" if cls.use_sgl_xpu else "0"

        other_args = list(XPU_ARGS) + list(cls.extra_args)
        if cls.trust_remote_code:
            other_args.extend(cls.fixed_args)
        else:
            other_args.extend(
                arg for arg in cls.fixed_args if arg != "--trust-remote-code"
            )

        timeout = (
            cls.launch_timeout
            if cls.launch_timeout is not None
            else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=timeout,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestInternVL25Server(ImageOpenAITestMixin, InternVLXPUServerBase):
    """InternVL2.5-2B on XPU with SGL XPU backend."""

    model = "OpenGVLab/InternVL2_5-2B"
    use_sgl_xpu = True
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]

    def test_video_images_chat_completion(self):
        # Video test uses 10 frames and exceeds max_prefill_tokens (23124 > 16384).
        pass


class TestInternVL25TritonServer(ImageOpenAITestMixin, InternVLXPUServerBase):
    """InternVL2.5-2B on XPU with Triton (non-SGL) backend."""

    model = "OpenGVLab/InternVL2_5-2B"
    use_sgl_xpu = False
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]

    def test_video_images_chat_completion(self):
        # Video test exceeds max_prefill_tokens on XPU with default limits.
        pass


class TestInternVL35_2BServer(ImageOpenAITestMixin, InternVLXPUServerBase):
    """InternVL3.5-2B on XPU with SGL XPU backend."""

    model = "OpenGVLab/InternVL3_5-2B"
    use_sgl_xpu = True
    launch_timeout = INTERNVL35_LAUNCH_TIMEOUT
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]

    def test_video_images_chat_completion(self):
        # Video test exceeds max_prefill_tokens (23202 > 14588) on InternVL3.5.
        pass


class TestInternVL35_2BTritonServer(ImageOpenAITestMixin, InternVLXPUServerBase):
    """InternVL3.5-2B on XPU with Triton (non-SGL) backend."""

    model = "OpenGVLab/InternVL3_5-2B"
    use_sgl_xpu = False
    launch_timeout = INTERNVL35_LAUNCH_TIMEOUT
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]

    def test_video_images_chat_completion(self):
        # Video test exceeds max_prefill_tokens on InternVL3.5.
        pass


if __name__ == "__main__":
    unittest.main()

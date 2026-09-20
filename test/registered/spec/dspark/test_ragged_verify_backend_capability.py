"""Backend opt-in flags for the ragged-verify graphs.

Runs in the GPU suite because importing the backend modules pulls GPU-only
wheels (sgl_kernel) at module scope, which fail to import on CPU runners.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

    def test_ragged_implementing_backends_declare_the_flag(self):
        """Every backend with a ragged-verify metadata path must opt in; a
        dropped flag silently disables ragged graphs for that backend (the
        runner falls back to eager with no other test going red)."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        for backend in (
            TRTLLMHAAttnBackend,
            DeepseekV4AttnBackend,
            FlashAttentionBackend,
        ):
            with self.subTest(backend=backend.__name__):
                self.assertTrue(backend.supports_ragged_verify_graph)


class TestRaggedVerifyCaptureGeometry(CustomTestCase):
    def test_capture_iterates_token_keys_without_width_multiplication(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.capture_num_tokens = [8, 16, 40, 224]
        runner.capture_bs = [8, 16, 24, 32]
        runner.max_bs = 32
        runner.captured_req_width = 7

        self.assertEqual(runner._capture_shape_keys(), [8, 16, 40, 224])
        self.assertEqual(runner._capture_shape_geometry(8), (8, 8))
        self.assertEqual(runner._capture_shape_geometry(40), (32, 40))
        self.assertEqual(runner._capture_shape_geometry(224), (32, 224))

    def test_static_capture_keeps_request_key_semantics(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = False
        runner.capture_num_tokens = None
        runner.capture_bs = [8, 16, 24, 32]
        runner.max_bs = 32
        runner.captured_req_width = 7

        self.assertEqual(runner._capture_shape_keys(), [8, 16, 24, 32])
        self.assertEqual(runner._capture_shape_geometry(8), (8, 56))


if __name__ == "__main__":
    unittest.main()

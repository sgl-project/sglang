"""Backend opt-in flags for the ragged-verify graphs.

Runs in the GPU suite because importing the backend modules pulls GPU-only
wheels (sgl_kernel) at module scope, which fail to import on CPU runners.
"""

import unittest
from types import SimpleNamespace

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

    def test_admission_requires_an_exact_captured_backend_key(self):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        class _Backend:
            def __init__(self, keys):
                self.keys = keys

            def can_run(self, _forward_batch, shape_key):
                return shape_key.size in self.keys

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.attn_backend = SimpleNamespace(supports_ragged_verify_graph=True)
        runner.capture_num_tokens = [8, 16]
        runner.max_bs = 8
        runner.enable_pdmux = False
        runner.record_nolora_graph = False
        runner.attention_graph_variants = None
        runner.require_mlp_sync = False
        runner.is_encoder_decoder = False
        runner.capture_hidden_mode = CaptureHiddenMode.FULL
        runner.backend = _Backend({8})
        forward_batch = SimpleNamespace(
            batch_size=1,
            can_run_decode_cuda_graph=True,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        self.assertTrue(
            runner._can_run_ragged_verify_graph(
                forward_batch, SimpleNamespace(graph_num_tokens=8)
            )
        )
        self.assertFalse(
            runner._can_run_ragged_verify_graph(
                forward_batch, SimpleNamespace(graph_num_tokens=12)
            )
        )
        runner.backend = _Backend(set())
        self.assertFalse(
            runner._can_run_ragged_verify_graph(
                forward_batch, SimpleNamespace(graph_num_tokens=8)
            )
        )

    def test_staging_rejects_an_uncaptured_layout_key(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner._captured_ragged_layouts = {}

        with self.assertRaisesRegex(RuntimeError, "uncaptured token key 8"):
            runner._stage_ragged_verify_layout(SimpleNamespace(), graph_size_key=8)

    def _uniform_width_runner(self, max_bs=32, width=7):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.captured_req_width = width
        runner.max_bs = max_bs
        runner.ragged_verify_uniform_width = True
        return runner

    def test_verify_all_sizes_slots_by_requests_not_tokens(self):
        runner = self._uniform_width_runner()

        # Token-count sizing gave min(num_tokens, max_bs) slots, i.e. up to
        # `width` times more page-table / seq_lens / attention-metadata rows
        # than a verify-all tier can ever hold.
        self.assertEqual(runner._ragged_capture_slots(8), 2)
        self.assertEqual(runner._ragged_capture_slots(40), 6)
        self.assertEqual(runner._ragged_capture_slots(56), 8)
        self.assertEqual(runner._ragged_capture_slots(224), 32)

    def test_verify_all_slot_sizing_keeps_every_batch_admissible(self):
        from sglang.srt.speculative.ragged_verify import (
            build_capture_verify_lens,
            build_ragged_capture_token_buckets,
            round_up_grid,
        )

        width, max_bs = 7, 32
        runner = self._uniform_width_runner(max_bs=max_bs, width=width)
        buckets = build_ragged_capture_token_buckets(
            request_buckets=list(range(1, 9)) + list(range(10, 33, 2)),
            max_num_requests=max_bs,
            num_tokens_per_req=width,
            token_alignment=8,
        )

        for bs in range(1, max_bs + 1):
            key = round_up_grid(bs * width, buckets)
            slots = runner._ragged_capture_slots(key)
            with self.subTest(bs=bs, key=key, slots=slots):
                # Admission needs slots >= bs, and the capture layout has to
                # pack the whole alignment-padded tier into those slots.
                self.assertGreaterEqual(slots, bs)
                self.assertLessEqual(key, slots * width)
                build_capture_verify_lens(
                    num_tokens=key, num_slots=slots, num_draft_tokens=width
                )

    def test_non_uniform_schedule_keeps_token_count_slot_sizing(self):
        runner = self._uniform_width_runner()
        runner.ragged_verify_uniform_width = False

        # A profiled SPS table trims requests, so a tier really can hold up to
        # `num_tokens` single-token requests.
        self.assertEqual(runner._ragged_capture_slots(8), 8)
        self.assertEqual(runner._ragged_capture_slots(40), 32)


if __name__ == "__main__":
    unittest.main()

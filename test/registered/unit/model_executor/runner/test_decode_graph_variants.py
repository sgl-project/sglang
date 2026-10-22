"""Speculative decode-graph variants (``DecodeGraphVariants``) - CPU only.

An algorithm whose verify width changes per step hands the decode runner a
``DecodeGraphVariants`` provider. The runner then captures one graph per
(batch size, variant), each at ``bs * variant.num_tokens_per_req`` tokens, and
replays only the graph captured for the step's variant. Without a provider
(every builtin) nothing changes.

The runner is built with ``__new__`` and only the attributes each method
reads, so no model, GPU or server is needed.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.decode_graph_variants import (
    DecodeGraphRunnerInfo,
    DecodeGraphVariant,
    DecodeGraphVariants,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

W4 = DecodeGraphVariant("w4", 4)
W2 = DecodeGraphVariant("w2", 2)


class _Widths(DecodeGraphVariants):
    """Captures widths 4 and 2 for every bs; a step replays the width its
    spec input carries. Records every call in order."""

    def __init__(self, variant_backends=None):
        self.calls = []
        self.variant_backends = variant_backends or {}
        self.num_replay_variant_calls = 0

    def capture_variants(self, bs):
        return [W4, W2]

    def replay_variant(self, forward_batch):
        self.num_replay_variant_calls += 1
        width = forward_batch.spec_info.num_tokens_per_req
        return {4: W4, 2: W2}.get(width)

    def begin_capture(self, variant, forward_batch, attn_backend, stream_idx):
        self.calls.append(("begin_capture", variant.label))
        forward_batch.spec_info.num_tokens_per_req = variant.num_tokens_per_req
        return self.variant_backends.get(variant.label, attn_backend)

    def begin_replay(self, variant, forward_batch, attn_backend, stream_idx):
        self.calls.append(("begin_replay", variant.label))
        return self.variant_backends.get(variant.label, attn_backend)

    def end_metadata(self, variant, forward_batch):
        self.calls.append(("end_metadata", variant.label))


def _bare_runner(graph_variants=None):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.graph_variants = graph_variants
    runner._captured_graph_variants = set()
    runner.captured_req_width = 4
    runner.capture_bs = [1, 2, 4]
    runner.compile_bs = []
    runner.max_bs = 4
    runner.ragged_verify_mode = False
    runner.record_nolora_graph = False
    runner.attention_graph_variants = None
    runner.enable_pdmux = False
    runner.disable_padding = False
    runner.require_mlp_tp_gather = False
    runner.require_attn_tp_gather = False
    runner.require_mlp_sync = False
    runner.is_encoder_decoder = False
    runner.enable_two_batch_overlap = False
    runner._metadata_glue = None
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.seq_len_fill_value = 1
    runner.dllm_uses_input_embeds = False
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.NONE,
        is_draft_worker=False,
        device="cpu",
        gpu_id=0,
        model=None,
        lora_manager=None,
        canary_manager=None,
        capture_tail_hooks=[],
        hisparse_coordinator=None,
    )
    runner.attn_backend = mock.MagicMock(name="runner_backend")
    runner.backend = mock.MagicMock(name="graph_backend")
    runner.deepep_adapter = mock.MagicMock()
    runner.tbo_plugin = mock.MagicMock()
    return runner


def _verify_batch(bs, width):
    return SimpleNamespace(
        batch_size=bs,
        input_ids=torch.zeros(bs * width, dtype=torch.int64),
        positions=torch.zeros(bs * width, dtype=torch.int64),
        spec_info=SimpleNamespace(num_tokens_per_req=width),
        replace_embeds=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        lora_ids=None,
        input_embeds=None,
        needs_forward_metadata_init=lambda: True,
    )


class TestDefaultIsNoVariants(CustomTestCase):
    def test_builtin_and_plugin_defaults_return_none(self):
        info = DecodeGraphRunnerInfo(False, ForwardMode.TARGET_VERIFY, 4, "cpu")
        self.assertIsNone(SpeculativeAlgorithm.EAGLE.create_decode_graph_variants(info))
        plugin = CustomSpecAlgo(name="PLUGIN", factory=lambda server_args: None)
        self.assertIsNone(plugin.create_decode_graph_variants(info))

    def test_init_asks_the_algorithm_and_disables_metadata_glue(self):
        provider = _Widths()
        asked = []
        runner = _bare_runner()
        runner.model_runner.spec_algorithm = SimpleNamespace(
            create_decode_graph_variants=lambda info: asked.append(info) or provider
        )
        runner._metadata_glue = object()
        runner._init_graph_variants()
        self.assertIs(runner.graph_variants, provider)
        self.assertIsNone(runner._metadata_glue)
        # The provider learns the runner's role, mode, width and device only.
        self.assertEqual(
            asked,
            [DecodeGraphRunnerInfo(False, ForwardMode.TARGET_VERIFY, 4, "cpu")],
        )

        plain = _bare_runner()
        glue = object()
        plain._metadata_glue = glue
        plain._init_graph_variants()
        self.assertIsNone(plain.graph_variants)
        self.assertIs(plain._metadata_glue, glue)

    def test_capture_without_provider_is_one_default_graph_per_bs(self):
        runner = _bare_runner()
        self.assertEqual(runner._capture_graph_variants(2), [None])


class TestCaptureLoop(CustomTestCase):
    def _capture(self, runner):
        recorded = []

        @contextlib.contextmanager
        def patch_model(model, compile_enabled, num_tokens, tp_group):
            yield ("forward", num_tokens)

        runner.capture_one_shape = lambda bs, forward, *args, **kwargs: recorded.append(
            (bs, forward[1], kwargs.get("spec_variant"))
        )
        with (
            mock.patch.object(mod, "get_available_gpu_memory", return_value=1.0),
            mock.patch.object(
                mod,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=1, tp_group=None),
            ),
            mock.patch.object(mod.torch_compile_decoration, "patch_model", patch_model),
        ):
            runner._capture_one_stream()
        return recorded

    def test_one_graph_per_bs_and_variant_at_the_variant_width(self):
        recorded = self._capture(_bare_runner(_Widths()))
        self.assertEqual(
            recorded,
            [
                (4, 16, W4),
                (4, 8, W2),
                (2, 8, W4),
                (2, 4, W2),
                (1, 4, W4),
                (1, 2, W2),
            ],
        )

    def test_without_provider_the_loop_is_unchanged(self):
        recorded = self._capture(_bare_runner())
        self.assertEqual(recorded, [(4, 16, None), (2, 8, None), (1, 4, None)])

    def test_a_variant_wider_than_the_buffers_is_rejected(self):
        runner = _bare_runner(_Widths())
        runner.captured_req_width = 3
        with self.assertRaises(ValueError):
            runner._capture_graph_variants(1)


class TestCaptureOneShape(CustomTestCase):
    def test_variant_capture_uses_its_width_backend_and_key(self):
        variant_backend = mock.MagicMock(name="w2_backend")
        provider = _Widths({"w2": variant_backend})
        runner = _bare_runner(provider)
        fb = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=4), lora_ids=None
        )
        variant_backend.init_forward_metadata_out_graph.side_effect = lambda *a, **k: (
            provider.calls.append(("metadata", "w2"))
        )
        runner.capture_prepare = mock.MagicMock(
            return_value=(fb, runner.attn_backend, None)
        )
        with (
            mock.patch.object(
                mod,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(debug_cuda_graph=False)
                ),
            ),
            mock.patch.object(mod, "maybe_flashinfer_autotune_speculative_draft"),
        ):
            runner.capture_one_shape(2, forward=None, spec_variant=W2)

        runner.capture_prepare.assert_called_once_with(2, stream_idx=None, num_tokens=4)
        self.assertEqual(fb.spec_info.num_tokens_per_req, 2)
        variant_backend.init_forward_metadata_out_graph.assert_called_once_with(
            fb, in_capture=True
        )
        runner.attn_backend.init_forward_metadata_out_graph.assert_not_called()
        self.assertEqual(
            provider.calls,
            [("begin_capture", "w2"), ("metadata", "w2"), ("end_metadata", "w2")],
        )
        shape_key = runner.backend.capture_one.call_args.args[0]
        self.assertEqual(shape_key, ShapeKey(size=2, spec_variant="w2"))
        self.assertEqual(runner._captured_graph_variants, {(2, "w2")})


class TestReplay(CustomTestCase):
    def test_can_run_only_captured_variants(self):
        runner = _bare_runner(_Widths())
        runner._captured_graph_variants = {(2, "w2"), (2, "w4")}
        self.assertTrue(runner.can_run_graph(_verify_batch(2, 2)))
        self.assertTrue(runner.can_run_graph(_verify_batch(2, 4)))
        # bs 1 pads to bucket 1, which has no captured variant.
        self.assertFalse(runner.can_run_graph(_verify_batch(1, 2)))
        # The provider has no variant for width 3: eager.
        self.assertFalse(runner.can_run_graph(_verify_batch(2, 3)))
        # Token count must match the variant width.
        mismatched = _verify_batch(2, 2)
        mismatched.input_ids = torch.zeros(5, dtype=torch.int64)
        self.assertFalse(runner.can_run_graph(mismatched))

    def test_without_provider_a_narrow_width_runs_eager(self):
        runner = _bare_runner()
        self.assertFalse(runner.can_run_graph(_verify_batch(2, 2)))
        self.assertTrue(runner.can_run_graph(_verify_batch(2, 4)))

    def test_load_batch_fills_and_plans_at_the_variant_width(self):
        variant_backend = mock.MagicMock(name="w2_backend")
        provider = _Widths({"w2": variant_backend})
        variant_backend.init_forward_metadata_out_graph.side_effect = lambda *a, **k: (
            provider.calls.append(("metadata", "w2"))
        )
        runner = _bare_runner(provider)
        runner._captured_graph_variants = {(1, "w2")}
        runner.buffers = SimpleNamespace(custom_mask=None, num_token_non_padded=None)
        runner.buffer_registry = mock.MagicMock()
        runner._validate_capture_hidden_mode = lambda fb: None
        fb = _verify_batch(1, 2)
        views = []

        def fake_view(**kwargs):
            views.append(kwargs)
            return SimpleNamespace(**kwargs)

        with mock.patch.object(mod, "build_replay_fb_view", side_effect=fake_view):
            self.assertTrue(runner.can_run_graph(fb))
            runner.load_batch(fb)

        # load_batch replays the variant can_run_graph resolved.
        self.assertEqual(provider.num_replay_variant_calls, 1)

        fill = runner.buffer_registry.fill_from.call_args.kwargs
        self.assertEqual((fill["raw_num_tokens"], fill["padded_num_tokens"]), (2, 2))
        self.assertEqual(views[0]["num_tokens"], 2)
        variant_backend.init_forward_metadata_out_graph.assert_called_once()
        runner.attn_backend.init_forward_metadata_out_graph.assert_not_called()
        self.assertEqual(
            provider.calls,
            [("begin_replay", "w2"), ("metadata", "w2"), ("end_metadata", "w2")],
        )
        self.assertEqual(runner._replay_graph_key, ShapeKey(size=1, spec_variant="w2"))
        self.assertEqual(runner.raw_num_token, 2)


if __name__ == "__main__":
    unittest.main()

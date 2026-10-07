"""Argument-contract tests for the GLM-5-Next vision-tower precompile hook.

These tests verify what ``precompile_kernels_after_loading`` passes to
``context_attention_fwd`` and when it declines to call it: the q/k/v/o shape
``(1, 1, head_dim)`` in the tower's dtype and device, int32 ``b_start_loc`` and
``b_seq_len``, ``max_input_len=1``, ``is_causal=False``; no call for
language-only models, for non-Triton vision backends, or on pipeline ranks
other than the first; a synchronous failure is logged at WARNING and swallowed.

They do not run Triton: the kernel entry point is a recording stub and the
vision tower is a fake, so nothing here shows that the hook's call and a real
vision call share one compiled kernel. That is the GPU test's job
(``test_glm5_next_vision_precompile_gpu.py``).

The vision-MLP tests check the warmup's input ranks and grad mode, and
``TestGlm5NextVisionPrecompileReuse`` runs the real ``torch.compile``d
activation on CPU to show that serving-shaped calls in each serving grad mode
reuse the warmed graphs instead of compiling again.
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.glm5_next as glm5_next
from sglang.srt.utils.common import DynamicGradMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeVisionAttention(nn.Module):
    def __init__(self, backend: str):
        super().__init__()
        self.qkv_backend_name = backend


class _FakeVisionTower(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, backend: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn = _FakeVisionAttention(backend)
        self.proj = nn.Linear(2, 2, bias=False).to(torch.bfloat16)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device


class _RecordingActivationModule(nn.Module):
    """Stands in for the block MLP or the patch merger: records the token
    counts it is run at and exposes the first linear's input width."""

    def __init__(self, first_linear: str, in_features: int, calls: list):
        nn.Module.__init__(self)
        setattr(self, first_linear, SimpleNamespace(input_size=in_features))
        self.calls = calls
        self.grad_modes = []
        self.in_features = in_features

    def forward(self, x):
        self.calls.append((tuple(x.shape), x.dtype, x.device))
        self.grad_modes.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        return x


class _RecordingMLP(_RecordingActivationModule, glm5_next.Glm5NextVisionMLP):
    def __init__(self, in_features, calls):
        _RecordingActivationModule.__init__(self, "gate_up_proj", in_features, calls)


class _RecordingMerger(_RecordingActivationModule, glm5_next.Glm5NextVisionPatchMerger):
    def __init__(self, in_features, calls):
        _RecordingActivationModule.__init__(self, "proj", in_features, calls)


def _make_model(visual, is_first_rank: bool = True):
    model = glm5_next.Glm5NextForConditionalGeneration.__new__(
        glm5_next.Glm5NextForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.encoder_only = True
    model.visual = visual
    model.pp_group = SimpleNamespace(is_first_rank=is_first_rank)
    return model


class TestGlm5NextVisionPrecompile(CustomTestCase):
    def _run_hook(self, model, stub):
        prefill_attention = ModuleType("sglang.kernels.ops.attention.prefill_attention")
        prefill_attention.context_attention_fwd = stub
        with (
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.attention.prefill_attention": prefill_attention},
            ),
            patch.object(glm5_next, "VisionAttention", _FakeVisionAttention),
        ):
            model.precompile_kernels_after_loading()

    def test_triton_backend_precompiles_with_vision_head_dim(self):
        calls = []

        def stub(*args, **kwargs):
            calls.append((args, kwargs))

        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="triton_attn")
        self._run_hook(_make_model(visual), stub)

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        q, k, v, o, start_loc, seq_len, max_input_len = args
        for t in (q, k, v, o):
            self.assertEqual(tuple(t.shape), (1, 1, 128))
            self.assertEqual(t.dtype, torch.bfloat16)
            self.assertEqual(t.device, visual.device)
        self.assertEqual(start_loc.dtype, torch.int32)
        self.assertEqual(start_loc.tolist(), [0])
        self.assertEqual(seq_len.dtype, torch.int32)
        self.assertEqual(seq_len.tolist(), [1])
        self.assertEqual(max_input_len, 1)
        self.assertEqual(kwargs, {"is_causal": False})

    def test_language_only_model_skips(self):
        calls = []
        self._run_hook(_make_model(None), lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])

    def test_non_triton_vision_backend_skips(self):
        calls = []
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        self._run_hook(_make_model(visual), lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])

    def test_non_first_pipeline_rank_skips(self):
        # Only the first PP rank embeds images (general_mm_embed_routine), so a
        # later rank must not compile or hold the vision kernel.
        calls = []
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="triton_attn")
        self._run_hook(
            _make_model(visual, is_first_rank=False), lambda *a, **k: calls.append(a)
        )
        self.assertEqual(calls, [])

    def test_mlp_and_merger_run_at_two_token_counts(self):
        # Two distinct token counts make dynamo compile the static kernel and
        # then settle on the dynamic one before the pools exist.
        mlp_calls, merger_calls = [], []
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        visual.mlp = _RecordingMLP(3072, mlp_calls)
        visual.merger = _RecordingMerger(1536, merger_calls)
        self._run_hook(_make_model(visual), lambda *a, **k: None)
        # The block MLP is served (S, 1, H) by GlmOcrVisionBlock.forward and
        # the merger (S, H); dynamo guards on rank, so the warmup matches both.
        self.assertEqual(
            [shape for shape, _, _ in mlp_calls], [(64, 1, 3072), (4096, 1, 3072)]
        )
        self.assertEqual(
            [shape for shape, _, _ in merger_calls], [(64, 1536), (4096, 1536)]
        )
        for _, dtype, device in mlp_calls + merger_calls:
            self.assertEqual(dtype, torch.bfloat16)
            self.assertEqual(device, visual.device)

    def _warmup_grad_modes(self, encoder_only: bool):
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        visual.mlp = _RecordingMLP(3072, [])
        visual.merger = _RecordingMerger(1536, [])
        model = _make_model(visual)
        model.encoder_only = encoder_only
        self._run_hook(model, lambda *a, **k: None)
        return visual.mlp.grad_modes + visual.merger.grad_modes

    def test_mlp_warmup_uses_the_scheduler_grad_mode(self):
        # (grad_enabled, inference_mode) as DynamicGradMode applies it.
        self.assertEqual(set(self._warmup_grad_modes(False)), {(False, False)})
        DynamicGradMode.set_inference_mode(True)
        try:
            self.assertEqual(set(self._warmup_grad_modes(False)), {(False, True)})
        finally:
            DynamicGradMode.set_inference_mode(False)

    def test_mlp_warmup_uses_inference_mode_in_the_encoder_server(self):
        self.assertEqual(set(self._warmup_grad_modes(True)), {(False, True)})

    def test_mlp_precompile_skips_without_the_modules_or_off_first_rank(self):
        calls = []
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        self._run_hook(_make_model(visual), lambda *a, **k: None)  # no modules
        visual.mlp = _RecordingMLP(3072, calls)
        self._run_hook(_make_model(visual, is_first_rank=False), lambda *a, **k: None)
        self.assertEqual(calls, [])

    def test_mlp_failure_is_logged_not_raised(self):
        class _Failing(_RecordingMLP):
            def forward(self, x):
                raise RuntimeError("mlp compile failed")

        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        visual.mlp = _Failing(3072, [])
        with self.assertLogs(glm5_next.logger, level="WARNING") as logs:
            self._run_hook(_make_model(visual), lambda *a, **k: None)
        self.assertTrue(
            any("MLP precompile failed" in line for line in logs.output), logs.output
        )

    def test_kernel_failure_is_logged_not_raised(self):
        def stub(*args, **kwargs):
            raise RuntimeError("compile failed")

        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="triton_attn")
        with self.assertLogs(glm5_next.logger, level="WARNING") as logs:
            self._run_hook(_make_model(visual), stub)
        self.assertTrue(
            any(
                "precompile failed" in line and "RuntimeError: compile failed" in line
                for line in logs.output
            ),
            logs.output,
        )
        self.assertFalse(
            any(line.startswith("ERROR") for line in logs.output), logs.output
        )


class _CompiledActivation(nn.Module):
    """Runs the real ``torch.compile``d ``swiglu_clamped`` on a gate_up-shaped
    input that keeps the caller's leading dimensions, as the real modules do."""

    def __init__(self, first_linear: str, in_features: int):
        nn.Module.__init__(self)
        setattr(self, first_linear, SimpleNamespace(input_size=in_features))

    def forward(self, x):
        return glm5_next.swiglu_clamped(torch.cat([x, x], dim=-1), 7.0)


class _CompiledMLP(_CompiledActivation, glm5_next.Glm5NextVisionMLP):
    def __init__(self, in_features):
        _CompiledActivation.__init__(self, "gate_up_proj", in_features)


class _CompiledMerger(_CompiledActivation, glm5_next.Glm5NextVisionPatchMerger):
    def __init__(self, in_features):
        _CompiledActivation.__init__(self, "proj", in_features)


class TestGlm5NextVisionPrecompileReuse(CustomTestCase):
    """Runs the real compiled activation: after the hook, serving-shaped calls
    in the serving grad mode at a new token count must not compile again."""

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        DynamicGradMode.set_inference_mode(False)
        torch._dynamo.reset()

    def _assert_serving_calls_reuse(self, encoder_only, serving_mode):
        from torch._dynamo.utils import counters

        visual = _FakeVisionTower(hidden_size=32, num_heads=2, backend="fa3")
        visual.mlp = _CompiledMLP(32)
        visual.merger = _CompiledMerger(16)
        model = _make_model(visual)
        model.encoder_only = encoder_only
        with self.assertNoLogs(glm5_next.logger, level="WARNING"):
            model.precompile_kernels_after_loading()
        graphs = counters["stats"]["unique_graphs"]
        self.assertGreater(graphs, 0)

        with serving_mode():
            for tokens in (1000, 1500):
                # GlmOcrVisionBlock.forward hands its MLP (S, 1, H); the
                # vision model hands the merger (S, H).
                visual.mlp(torch.zeros((tokens, 1, 32), dtype=torch.bfloat16))
                visual.merger(torch.zeros((tokens // 4, 16), dtype=torch.bfloat16))
        self.assertEqual(counters["stats"]["unique_graphs"], graphs)

    def test_scheduler_no_grad_serving_reuses_the_warmup(self):
        self._assert_serving_calls_reuse(False, DynamicGradMode)

    def test_scheduler_inference_mode_serving_reuses_the_warmup(self):
        DynamicGradMode.set_inference_mode(True)
        self._assert_serving_calls_reuse(False, DynamicGradMode)

    def test_encoder_server_serving_reuses_the_warmup(self):
        self._assert_serving_calls_reuse(True, torch.inference_mode)


if __name__ == "__main__":
    unittest.main()

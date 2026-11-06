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

The vision-MLP tests check that the warmup runs only the activation, never
the modules or their tensor-parallel ``down_proj`` all-reduce, at the serving
input ranks and grad mode. ``TestGlm5NextVisionPrecompileReuse`` runs the real
``torch.compile``d activation on CPU to show that serving-shaped calls in each
serving grad mode reuse the warmed graphs instead of compiling again.
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


class _ActivationCaller(nn.Module):
    """Stands in for the block MLP or the patch merger. Exposes what the warmup
    reads (the per-rank gate_up width and ``swiglu_limit``) and records any
    call into the module itself: ``down_proj`` all-reduces across
    tensor-parallel ranks, so the warmup must reach neither it nor ``forward``."""

    def __init__(self, gate_up_width: int, swiglu_limit: float = 7.0):
        nn.Module.__init__(self)
        # Input widths too, so a warmup that ran the whole module would reach
        # forward here and be caught rather than fail on a missing attribute.
        self.gate_up_proj = SimpleNamespace(
            input_size=gate_up_width // 2, output_size_per_partition=gate_up_width
        )
        self.proj = SimpleNamespace(input_size=gate_up_width // 2)
        self.swiglu_limit = swiglu_limit
        self.module_calls = []
        self.down_proj = lambda *a, **k: self.module_calls.append("down_proj")

    def forward(self, x):
        self.module_calls.append("forward")
        return x


class _FakeMLP(_ActivationCaller, glm5_next.Glm5NextVisionMLP):
    pass


class _FakeMerger(_ActivationCaller, glm5_next.Glm5NextVisionPatchMerger):
    pass


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
    def _run_hook(self, model, stub, activation=lambda y, limit: y):
        prefill_attention = ModuleType("sglang.kernels.ops.attention.prefill_attention")
        prefill_attention.context_attention_fwd = stub
        with (
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.attention.prefill_attention": prefill_attention},
            ),
            patch.object(glm5_next, "VisionAttention", _FakeVisionAttention),
            patch.object(glm5_next, "swiglu_clamped", activation),
        ):
            model.precompile_kernels_after_loading()

    def _run_hook_recording_activation(self, model):
        calls = []

        def activation(y, limit):
            calls.append(
                (
                    tuple(y.shape),
                    y.dtype,
                    y.device,
                    limit,
                    torch.is_grad_enabled(),
                    torch.is_inference_mode_enabled(),
                )
            )
            return y

        self._run_hook(model, lambda *a, **k: None, activation)
        return calls

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

    def test_mlp_and_merger_activations_run_at_two_token_counts(self):
        # Two distinct token counts make dynamo compile the static kernel and
        # then settle on the dynamic one before the pools exist.
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        visual.mlp = _FakeMLP(6144, swiglu_limit=7.0)
        visual.merger = _FakeMerger(3072, swiglu_limit=5.0)
        calls = self._run_hook_recording_activation(_make_model(visual))
        # gate_up_proj keeps its input's leading dims: the block MLP is served
        # (S, 1, H) by GlmOcrVisionBlock.forward and the merger (S, H), and
        # dynamo guards on rank, so the warmup matches both.
        self.assertEqual(
            [(shape, limit) for shape, _, _, limit, _, _ in calls],
            [
                ((64, 1, 6144), 7.0),
                ((4096, 1, 6144), 7.0),
                ((64, 3072), 5.0),
                ((4096, 3072), 5.0),
            ],
        )
        for _, dtype, device, _, _, _ in calls:
            self.assertEqual(dtype, torch.bfloat16)
            self.assertEqual(device, visual.device)
        self.assertEqual(visual.mlp.module_calls + visual.merger.module_calls, [])

    def _warmup_grad_modes(self, encoder_only: bool):
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        visual.mlp = _FakeMLP(6144)
        visual.merger = _FakeMerger(3072)
        model = _make_model(visual)
        model.encoder_only = encoder_only
        calls = self._run_hook_recording_activation(model)
        self.assertEqual(len(calls), 4)
        return {(grad, inference) for *_, grad, inference in calls}

    def test_mlp_warmup_uses_the_scheduler_grad_mode(self):
        # (grad_enabled, inference_mode) as DynamicGradMode applies it.
        self.assertEqual(self._warmup_grad_modes(False), {(False, False)})
        DynamicGradMode.set_inference_mode(True)
        try:
            self.assertEqual(self._warmup_grad_modes(False), {(False, True)})
        finally:
            DynamicGradMode.set_inference_mode(False)

    def test_mlp_warmup_uses_inference_mode_in_the_encoder_server(self):
        self.assertEqual(self._warmup_grad_modes(True), {(False, True)})

    def test_mlp_precompile_skips_without_the_modules_or_off_first_rank(self):
        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        calls = self._run_hook_recording_activation(_make_model(visual))
        visual.mlp = _FakeMLP(6144)
        calls += self._run_hook_recording_activation(
            _make_model(visual, is_first_rank=False)
        )
        self.assertEqual(calls, [])
        self.assertEqual(visual.mlp.module_calls, [])

    def test_failure_on_one_tp_rank_enters_no_collective(self):
        # A rank whose warmup fails logs and returns. Its peers must not be
        # left waiting in a collective it never reaches, so the warmup itself
        # must issue none: neither the modules nor the all-reduce run.
        def failing_activation(y, limit):
            raise RuntimeError("activation compile failed")

        visual = _FakeVisionTower(hidden_size=1536, num_heads=12, backend="fa3")
        visual.mlp = _FakeMLP(6144)
        visual.merger = _FakeMerger(3072)
        all_reduces = []
        with (
            patch(
                "sglang.srt.layers.linear.tensor_model_parallel_all_reduce",
                side_effect=lambda t: all_reduces.append(t) or t,
            ),
            self.assertLogs(glm5_next.logger, level="WARNING") as logs,
        ):
            self._run_hook(
                _make_model(visual), lambda *a, **k: None, failing_activation
            )
        self.assertTrue(
            any(
                "MLP precompile failed" in line
                and "RuntimeError: activation compile failed" in line
                for line in logs.output
            ),
            logs.output,
        )
        self.assertEqual(all_reduces, [])
        self.assertEqual(visual.mlp.module_calls + visual.merger.module_calls, [])

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


class TestGlm5NextVisionPrecompileReuse(CustomTestCase):
    """Runs the real compiled activation: after the hook, serving-shaped calls
    in the serving grad mode at a new token count must not compile again."""

    def tearDown(self):
        DynamicGradMode.set_inference_mode(False)
        torch._dynamo.reset()

    def _assert_serving_calls_reuse(self, encoder_only, serving_mode):
        from torch._dynamo.utils import counters

        # CustomTestCase retries the test method in CI without rerunning
        # setUp, so start every attempt from an empty dynamo cache; otherwise
        # a retry would reuse graphs the failed attempt compiled while serving.
        torch._dynamo.reset()
        baseline = counters["stats"]["unique_graphs"]

        visual = _FakeVisionTower(hidden_size=32, num_heads=2, backend="fa3")
        visual.mlp = _FakeMLP(64)
        visual.merger = _FakeMerger(32)
        model = _make_model(visual)
        model.encoder_only = encoder_only
        with self.assertNoLogs(glm5_next.logger, level="WARNING"):
            model.precompile_kernels_after_loading()
        warmed = counters["stats"]["unique_graphs"]
        self.assertGreater(warmed, baseline)

        with serving_mode():
            for tokens in (1000, 1500):
                # gate_up_proj keeps the leading dims of what it is handed:
                # (S, 1, H) in GlmOcrVisionBlock.forward, (S, H) in the merger.
                glm5_next.swiglu_clamped(
                    torch.zeros((tokens, 1, 64), dtype=torch.bfloat16), 7.0
                )
                glm5_next.swiglu_clamped(
                    torch.zeros((tokens // 4, 32), dtype=torch.bfloat16), 7.0
                )
        self.assertEqual(counters["stats"]["unique_graphs"], warmed)

    def test_scheduler_no_grad_serving_reuses_the_warmup(self):
        self._assert_serving_calls_reuse(False, DynamicGradMode)

    def test_scheduler_inference_mode_serving_reuses_the_warmup(self):
        DynamicGradMode.set_inference_mode(True)
        self._assert_serving_calls_reuse(False, DynamicGradMode)

    def test_encoder_server_serving_reuses_the_warmup(self):
        self._assert_serving_calls_reuse(True, torch.inference_mode)


if __name__ == "__main__":
    unittest.main()

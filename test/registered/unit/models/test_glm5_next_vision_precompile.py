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
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.glm5_next as glm5_next
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
        self.in_features = in_features

    def forward(self, x):
        self.calls.append((tuple(x.shape), x.dtype, x.device))
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
    model.visual = visual
    model.pp_group = SimpleNamespace(is_first_rank=is_first_rank)
    return model


class TestGlm5NextVisionPrecompile(CustomTestCase):
    def _run_hook(self, model, stub):
        prefill_attention = ModuleType("sglang.kernels.ops.attention.prefill_attention")
        prefill_attention.context_attention_fwd = stub
        with patch.dict(
            sys.modules,
            {"sglang.kernels.ops.attention.prefill_attention": prefill_attention},
        ), patch.object(glm5_next, "VisionAttention", _FakeVisionAttention):
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
        self.assertEqual(
            [shape for shape, _, _ in mlp_calls], [(64, 3072), (4096, 3072)]
        )
        self.assertEqual(
            [shape for shape, _, _ in merger_calls], [(64, 1536), (4096, 1536)]
        )
        for _, dtype, device in mlp_calls + merger_calls:
            self.assertEqual(dtype, torch.bfloat16)
            self.assertEqual(device, visual.device)

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


if __name__ == "__main__":
    unittest.main()

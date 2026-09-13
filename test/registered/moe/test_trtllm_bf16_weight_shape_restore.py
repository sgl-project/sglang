"""Regression tests for issue #27787.

The flashinfer TRT-LLM BF16 postprocess in
``UnquantizedFusedMoEMethod.process_weights_after_loading`` reshapes expert
weights into block layout whenever ``use_flashinfer_trtllm_moe`` is set, which
covers both the routed and the non-routed TRT-LLM backends. The restore hook
``maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load`` used to gate on
``is_flashinfer_trtllm_routed()`` only, so with the non-routed
``--moe-runner-backend flashinfer_trtllm`` the blocked layout was never
restored on the ``update_weights_from_tensor`` path and ``_load_w13`` crashed
with a 64-vs-hidden_size shape mismatch.

These tests exercise the restore hook directly (pure tensor logic; no GPU
kernels involved).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

NUM_EXPERTS = 2
INTERMEDIATE = 128
HIDDEN = 64
BLOCK = 64


def _make_layer():
    return SimpleNamespace(
        num_local_experts=NUM_EXPERTS,
        intermediate_size_per_partition=INTERMEDIATE,
        hidden_size=HIDDEN,
        moe_runner_config=SimpleNamespace(is_gated=True),
    )


def _blocked_w13_param():
    """w13 weight in the trtllm block layout [E, 2I, H // B, B]."""
    canonical = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    blocked = canonical.reshape(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // BLOCK, BLOCK)
    return torch.nn.Parameter(blocked, requires_grad=False)


def _blocked_w2_param():
    """w2 weight in the trtllm block layout [E, H, I // B, B]."""
    canonical = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    blocked = canonical.reshape(NUM_EXPERTS, HIDDEN, INTERMEDIATE // BLOCK, BLOCK)
    return torch.nn.Parameter(blocked, requires_grad=False)


def _restore(method, param, weight_name):
    method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        layer=_make_layer(),
        param=param,
        weight_name=weight_name,
    )


class TestTrtllmBf16WeightShapeRestore(CustomTestCase):
    def test_non_routed_backend_restores_w13_and_w2(self):
        """The non-routed trtllm backend reshapes at load, so the restore hook
        must fire for it too (the #27787 crash path)."""
        method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        with patch(
            "sglang.srt.layers.moe.utils.MOE_RUNNER_BACKEND",
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            w13 = _blocked_w13_param()
            _restore(method, w13, "model.layers.0.mlp.experts.w13_weight")
            self.assertEqual(
                tuple(w13.data.shape), (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
            )

            w2 = _blocked_w2_param()
            _restore(method, w2, "model.layers.0.mlp.experts.w2_weight")
            self.assertEqual(tuple(w2.data.shape), (NUM_EXPERTS, HIDDEN, INTERMEDIATE))

    def test_routed_backend_still_restores(self):
        method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        with patch(
            "sglang.srt.layers.moe.utils.MOE_RUNNER_BACKEND",
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        ):
            w13 = _blocked_w13_param()
            _restore(method, w13, "model.layers.0.mlp.experts.w13_weight")
            self.assertEqual(
                tuple(w13.data.shape), (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
            )

    def test_without_trtllm_flag_is_a_noop(self):
        method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=False)
        with patch(
            "sglang.srt.layers.moe.utils.MOE_RUNNER_BACKEND",
            MoeRunnerBackend.TRITON,
        ):
            w13 = _blocked_w13_param()
            original_shape = tuple(w13.data.shape)
            _restore(method, w13, "model.layers.0.mlp.experts.w13_weight")
            self.assertEqual(tuple(w13.data.shape), original_shape)

    def test_canonical_shape_is_a_noop(self):
        method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        with patch(
            "sglang.srt.layers.moe.utils.MOE_RUNNER_BACKEND",
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            canonical = torch.nn.Parameter(
                torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                requires_grad=False,
            )
            _restore(method, canonical, "model.layers.0.mlp.experts.w13_weight")
            self.assertEqual(
                tuple(canonical.data.shape), (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
            )

    def test_numel_mismatch_raises(self):
        method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        with patch(
            "sglang.srt.layers.moe.utils.MOE_RUNNER_BACKEND",
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            wrong = torch.nn.Parameter(
                torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN),
                requires_grad=False,
            )
            with self.assertRaises(RuntimeError):
                _restore(method, wrong, "model.layers.0.mlp.experts.w13_weight")


if __name__ == "__main__":
    unittest.main()

"""Block-FP8 MoE finalization shuffles the experts into AITER's layout only for
the AITER runner; every other runner reads the weights as they are."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.quantization.fp8"
WEIGHTS = ("w13_weight", "w2_weight")


def _shuffle(weight, layout):
    # Any byte permutation shows whether a shuffle reached the weights.
    assert layout == (16, 16)
    return weight.view(torch.uint8).flip(-1).contiguous().view(weight.dtype)


def _moe(runner_backend):
    method = object.__new__(Fp8MoEMethod)
    method.quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    method.block_quant = True
    method.use_mxfp8 = False
    method.convert_mxfp8_to_block = False
    method.is_fp4_expert = False
    method.is_checkpoint_fp8_serialized = True
    if runner_backend is not None:
        method.runner = SimpleNamespace(runner_backend=runner_backend)
    layer = torch.nn.Module()
    generator = torch.Generator().manual_seed(0)
    for name, shape in zip(WEIGHTS, ((2, 256, 256), (2, 256, 128))):
        weight = (
            (torch.randn(shape, generator=generator) * 100)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        scale = torch.rand((shape[0], 2, shape[2] // 128), generator=generator) + 0.1
        layer.register_parameter(name, torch.nn.Parameter(weight, requires_grad=False))
        layer.register_parameter(
            name + "_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
        )
    return method, layer


def _bytes(layer):
    return {name: p.view(torch.uint8).clone() for name, p in layer.named_parameters()}


class TestFp8MoeAiterShuffleFollowsRunner(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        # CPU tensors throughout, whatever default device an earlier test left behind.
        stack.enter_context(torch.device("cpu"))
        stack.enter_context(patch(f"{MODULE}._use_hip_int4", False))
        self.shuffle = stack.enter_context(
            patch(f"{MODULE}.shuffle_weight", side_effect=_shuffle, create=True)
        )

    def _finalize(self, runner_backend, *, use_aiter, fnuz):
        method, layer = _moe(runner_backend)
        with (
            patch(f"{MODULE}._use_aiter", use_aiter),
            patch(f"{MODULE}._is_fp8_fnuz", fnuz),
        ):
            method.process_weights_after_loading_block_quant(layer)
        return layer

    def _assert_same_bytes(self, actual, expected):
        for name, value in _bytes(actual).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_fnuz_experts_for_the_triton_runner_are_not_shuffled(self):
        without_aiter = _bytes(
            self._finalize(MoeRunnerBackend.TRITON, use_aiter=False, fnuz=True)
        )
        # No runner is read the way the MXFP8 branch already reads it: not AITER.
        for runner_backend in (MoeRunnerBackend.TRITON, None):
            with self.subTest(runner_backend=runner_backend):
                self.shuffle.reset_mock()
                layer = self._finalize(runner_backend, use_aiter=True, fnuz=True)
                self.shuffle.assert_not_called()
                for name in WEIGHTS:
                    weight = getattr(layer, name)
                    self.assertEqual(weight.dtype, torch.float8_e4m3fnuz)
                    self.assertFalse(getattr(weight, "is_shuffled", False))
                self._assert_same_bytes(layer, without_aiter)

    def test_fnuz_experts_for_the_aiter_runner_are_shuffled(self):
        layer = self._finalize(MoeRunnerBackend.AITER, use_aiter=True, fnuz=True)
        self.assertEqual(self.shuffle.call_count, 2)
        for name in WEIGHTS:
            weight = getattr(layer, name)
            self.assertEqual(weight.dtype, torch.float8_e4m3fnuz)
            self.assertTrue(weight.is_shuffled)
        self.assertFalse(layer._aiter_gate_up_interleaved)

    def test_non_fnuz_experts_for_the_triton_runner_are_not_shuffled(self):
        loaded = _bytes(_moe(None)[1])
        for runner_backend in (MoeRunnerBackend.TRITON, None):
            with self.subTest(runner_backend=runner_backend):
                self.shuffle.reset_mock()
                layer = self._finalize(runner_backend, use_aiter=True, fnuz=False)
                self.shuffle.assert_not_called()
                for name in WEIGHTS:
                    weight = getattr(layer, name)
                    self.assertEqual(weight.dtype, torch.float8_e4m3fn)
                    self.assertFalse(getattr(weight, "is_shuffled", False))
                self._assert_same_bytes(layer, loaded)

    def test_non_fnuz_experts_for_the_aiter_runner_are_shuffled(self):
        _, loaded = _moe(None)
        layer = self._finalize(MoeRunnerBackend.AITER, use_aiter=True, fnuz=False)
        self.assertEqual(self.shuffle.call_count, 2)
        for name in WEIGHTS:
            weight = getattr(layer, name)
            self.assertTrue(weight.is_shuffled)
            torch.testing.assert_close(
                weight.view(torch.uint8),
                _shuffle(getattr(loaded, name), (16, 16)).view(torch.uint8),
                rtol=0,
                atol=0,
            )
        self.assertFalse(layer._aiter_gate_up_interleaved)


if __name__ == "__main__":
    unittest.main()

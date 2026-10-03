"""gfx94x block-FP8 weights survive in-place reloads byte for byte."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.quantization.fp8"


def _shuffle_reference(weight, layout):
    # AITER's (16, 16) FP8 layout, written independently of the inverse under test.
    assert layout == (16, 16)
    shape = weight.shape
    return (
        weight.view(torch.uint8)
        .reshape(-1, shape[-2] // 16, 16, shape[-1] // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .reshape(shape)
        .view(weight.dtype)
    )


def _make(moe, seed=1):
    config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    method = object.__new__(Fp8MoEMethod if moe else Fp8LinearMethod)
    method.quant_config = config
    method.block_quant = True
    method.use_mxfp8 = False
    method.convert_mxfp8_to_block = False
    method.is_fp4_expert = False
    method.block_fp8_as_mxfp8 = False
    method.is_checkpoint_fp8_serialized = True
    if moe:
        # The runner AITER-on configurations pick; the experts are shuffled only for it.
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.AITER)
    layer = torch.nn.Module()
    shapes = (
        {"w13_weight": (2, 256, 256), "w2_weight": (2, 256, 128)}
        if moe
        else {"weight": (256, 256)}
    )
    generator = torch.Generator().manual_seed(seed)
    raw = {}
    for name, shape in shapes.items():
        # Values up to 448: a numeric FN -> FNUZ copy turns everything above 240 into NaN.
        value = (
            (torch.randn(shape, generator=generator) * 100)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        value.view(torch.uint8).flatten()[0] = 0x80  # FN negative zero
        scale = (
            torch.rand(
                (*shape[:-2], shape[-2] // 128, shape[-1] // 128), generator=generator
            )
            + 0.1
        )
        for param_name, tensor in ((name, value), (name + "_scale_inv", scale)):
            param = torch.nn.Parameter(tensor, requires_grad=False)
            param.weight_loader = lambda dest, src: dest.data.copy_(src)
            param.format_ue8m0 = False
            layer.register_parameter(param_name, param)
            raw[param_name] = tensor.clone()
    return method, layer, raw


class TestFp8FnuzWeightReload(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch(f"{MODULE}._is_fp8_fnuz", True))
        self.stack.enter_context(patch(f"{MODULE}._use_hip_int4", False))
        self.stack.enter_context(patch(f"{MODULE}._use_aiter_bpreshuffle_gfx95", False))
        self.stack.enter_context(
            patch(
                f"{MODULE}.shuffle_weight", side_effect=_shuffle_reference, create=True
            )
        )

    def _finalize(self, method, layer):
        method.process_weights_after_loading_block_quant(layer)

    def _assert_same(self, actual, expected):
        for name, param in actual.named_parameters():
            reference = getattr(expected, name)
            self.assertEqual(param.dtype, reference.dtype, name)
            # torch.equal is not implemented for CPU float8.
            torch.testing.assert_close(
                param.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
            )

    def test_a_reload_matches_a_fresh_load_and_keeps_every_parameter(self):
        # The plain block-FP8 MoE branch shuffles for AITER whenever AITER is on.
        for moe, aiter in ((False, False), (True, False), (True, True)):
            with (
                self.subTest(moe=moe, aiter=aiter),
                patch(f"{MODULE}._use_aiter", aiter),
            ):
                method, layer, _ = _make(moe)
                identity = {
                    name: (id(p), p.data_ptr(), p.weight_loader)
                    for name, p in layer.named_parameters()
                }
                self._finalize(method, layer)
                if moe:
                    self.assertEqual(
                        bool(getattr(layer.w13_weight, "is_shuffled", False)), aiter
                    )
                for seed in (1, 5, 9):
                    fresh_method, fresh, incoming = _make(moe, seed)
                    self._finalize(fresh_method, fresh)

                    method.restore_weights_before_loading(layer)
                    method.restore_weights_before_loading(layer)  # begin is idempotent
                    if moe:
                        self.assertFalse(
                            getattr(layer.w13_weight, "is_shuffled", False)
                        )
                    for name, param in layer.named_parameters():
                        self.assertEqual(param.dtype, incoming[name].dtype, name)
                        param.weight_loader(param, incoming[name])
                    self._finalize(method, layer)
                    self._finalize(method, layer)  # end is idempotent

                    self._assert_same(layer, fresh)
                    for name, param in layer.named_parameters():
                        self.assertEqual(id(param), identity[name][0])
                        self.assertEqual(param.data_ptr(), identity[name][1])
                        self.assertIs(param.weight_loader, identity[name][2])
                        self.assertFalse(param.format_ue8m0)

    def test_empty_and_partial_sessions_leave_untouched_weights_valid(self):
        for moe in (False, True):
            with self.subTest(moe=moe), patch(f"{MODULE}._use_aiter", moe):
                method, layer, _ = _make(moe)
                reference_method, reference, _ = _make(moe)
                self._finalize(method, layer)
                self._finalize(reference_method, reference)
                for _ in range(3):
                    method.restore_weights_before_loading(layer)
                    self._finalize(method, layer)
                    self._assert_same(layer, reference)
                # Rewrite one block scale and leave every weight byte as it is.
                method.restore_weights_before_loading(layer)
                reference_method.restore_weights_before_loading(reference)
                scale_name = "w2_weight_scale_inv" if moe else "weight_scale_inv"
                getattr(layer, scale_name).data.flatten()[0] = 0.75
                getattr(reference, scale_name).data.flatten()[0] = 0.75
                self._finalize(method, layer)
                self._finalize(reference_method, reference)
                self._assert_same(layer, reference)

    def _assert_restore_is_a_no_op(self, method, layer):
        before = {
            name: (p.dtype, p.view(torch.uint8).clone())
            for name, p in layer.named_parameters()
        }
        method.restore_weights_before_loading(layer)
        for name, param in layer.named_parameters():
            self.assertEqual(param.dtype, before[name][0], name)
            torch.testing.assert_close(
                param.view(torch.uint8), before[name][1], rtol=0, atol=0
            )

    def test_only_weights_the_block_fp8_finalization_normalized_are_restored(self):
        with patch(f"{MODULE}._use_aiter", False):
            with self.subTest("a converted MXFP8 checkpoint reloads as MXFP8"):
                method, layer, _ = _make(moe=False)
                method.quant_config.use_mxfp8 = True
                self._finalize(method, layer)
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fnuz)
                self._assert_restore_is_a_no_op(method, layer)
            for moe in (False, True):
                with self.subTest("never finalized", moe=moe):
                    method, layer, _ = _make(moe)
                    self._assert_restore_is_a_no_op(method, layer)
                with self.subTest("FNUZ weights from another path", moe=moe):
                    method, layer, _ = _make(moe)
                    for param in layer.parameters():
                        if param.dtype == torch.float8_e4m3fn:
                            param.data = param.data.view(torch.float8_e4m3fnuz)
                    self._assert_restore_is_a_no_op(method, layer)


if __name__ == "__main__":
    unittest.main()

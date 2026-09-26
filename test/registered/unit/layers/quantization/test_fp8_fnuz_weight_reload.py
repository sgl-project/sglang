"""gfx94x block-FP8 load/update cycles preserve bytes, scales and parameters."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.quantization.fp8"


def _shuffle_reference(weight, layout):
    # AITER's (16,16) FP8 layout, deliberately independent of the inverse.
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


class TestFp8FnuzWeightReload(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch(f"{MODULE}._is_fp8_fnuz", True))
        self.stack.enter_context(patch(f"{MODULE}._use_hip_int4", False))
        self.stack.enter_context(patch(f"{MODULE}._use_aiter_bpreshuffle_gfx95", False))
        self.stack.enter_context(patch(f"{MODULE}._use_aiter", True))
        self.shuffle = self.stack.enter_context(
            patch(
                f"{MODULE}.shuffle_weight", side_effect=_shuffle_reference, create=True
            )
        )

    def _make(self, moe, shuffled=True, seed=1):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
        cls = Fp8MoEMethod if moe else Fp8LinearMethod
        method = object.__new__(cls)
        method.quant_config = config
        method.block_quant = True
        method.use_mxfp8 = False
        method.convert_mxfp8_to_block = False
        method.is_fp4_expert = False
        method.is_checkpoint_fp8_serialized = True
        method.runner = SimpleNamespace(
            runner_backend=SimpleNamespace(is_aiter=lambda: shuffled)
        )
        layer = torch.nn.Module()
        shapes = (
            {"w13_weight": (2, 256, 256), "w2_weight": (2, 256, 128)}
            if moe
            else {"weight": (256, 256)}
        )
        generator = torch.Generator().manual_seed(seed)
        raw = {}
        for name, shape in shapes.items():
            # Include saturation-edge values: a numeric FN -> FNUZ copy overflows.
            value = (
                (torch.randn(shape, generator=generator) * 100)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            value.view(torch.uint8).flatten()[0] = 128  # FN negative zero
            scale_name = name + "_scale_inv"
            scale = (
                torch.rand(
                    (*shape[:-2], shape[-2] // 128, shape[-1] // 128),
                    generator=generator,
                )
                + 0.1
            )
            for param_name, tensor in ((name, value), (scale_name, scale)):
                param = torch.nn.Parameter(tensor, requires_grad=False)
                param.weight_loader = lambda dest, src: dest.data.copy_(src)
                param.format_ue8m0 = False
                layer.register_parameter(param_name, param)
                raw[param_name] = tensor.clone()
        return method, layer, raw

    def _assert_same(self, actual, expected):
        for name, param in actual.named_parameters():
            reference = getattr(expected, name)
            self.assertEqual(param.dtype, reference.dtype)
            # torch.equal is not implemented for CPU float8.
            torch.testing.assert_close(
                param.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
            )

    def test_reload_matches_fresh_load_and_keeps_storage(self):
        for moe, shuffled in ((False, False), (True, False), (True, True)):
            with self.subTest(moe=moe, shuffled=shuffled):
                method, layer, raw = self._make(moe, shuffled)
                identity = {
                    name: (id(p), p.data_ptr(), p.weight_loader)
                    for name, p in layer.named_parameters()
                }
                method.process_weights_after_loading_block_quant(layer)
                if moe and shuffled:
                    self.assertTrue(layer.w13_weight.is_shuffled)
                    self.assertTrue(layer.w2_weight.is_shuffled)
                for seed in (1, 5, 9):
                    fresh_method, fresh, incoming = self._make(moe, shuffled, seed)
                    fresh_method.process_weights_after_loading_block_quant(fresh)
                    method.restore_weights_before_loading(layer)
                    if moe and shuffled:
                        self.assertFalse(layer.w13_weight.is_shuffled)
                        self.assertFalse(layer.w2_weight.is_shuffled)
                    method.restore_weights_before_loading(layer)  # idempotent begin
                    for name, param in layer.named_parameters():
                        param.weight_loader(param, incoming[name])
                    method.process_weights_after_loading_block_quant(layer)
                    method.process_weights_after_loading_block_quant(
                        layer
                    )  # idempotent end
                    self._assert_same(layer, fresh)
                    for name, param in layer.named_parameters():
                        self.assertEqual(id(param), identity[name][0])
                        self.assertEqual(param.data_ptr(), identity[name][1])
                        self.assertIs(param.weight_loader, identity[name][2])
                        self.assertFalse(param.format_ue8m0)

    def test_empty_and_partial_session_preserve_untouched_weights(self):
        for moe in (False, True):
            with self.subTest(moe=moe):
                method, layer, _ = self._make(moe)
                reference_method, reference, _ = self._make(moe)
                method.process_weights_after_loading_block_quant(layer)
                reference_method.process_weights_after_loading_block_quant(reference)
                for _ in range(3):
                    method.restore_weights_before_loading(layer)
                    method.process_weights_after_loading_block_quant(layer)
                    self._assert_same(layer, reference)
                # Update just one block scale, leaving every weight byte intact.
                method.restore_weights_before_loading(layer)
                reference_method.restore_weights_before_loading(reference)
                scale_name = "w2_weight_scale_inv" if moe else "weight_scale_inv"
                getattr(layer, scale_name).data.flatten()[0] = 0.75
                getattr(reference, scale_name).data.flatten()[0] = 0.75
                method.process_weights_after_loading_block_quant(layer)
                reference_method.process_weights_after_loading_block_quant(reference)
                self._assert_same(layer, reference)

    def test_triton_runner_does_not_shuffle(self):
        method, layer, _ = self._make(moe=True, shuffled=False)
        method.process_weights_after_loading_block_quant(layer)
        self.shuffle.assert_not_called()

    def test_restore_does_not_touch_mxfp8_or_non_block(self):
        for moe in (False, True):
            for mxfp8 in (False, True):
                with self.subTest(moe=moe, mxfp8=mxfp8):
                    method, layer, _ = self._make(moe)
                    method.process_weights_after_loading_block_quant(layer)
                    before = {
                        name: p.view(torch.uint8).clone()
                        for name, p in layer.named_parameters()
                    }
                    method.quant_config.use_mxfp8 = mxfp8
                    method.block_quant = mxfp8
                    method.restore_weights_before_loading(layer)
                    for name, param in layer.named_parameters():
                        torch.testing.assert_close(
                            param.view(torch.uint8), before[name], rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()

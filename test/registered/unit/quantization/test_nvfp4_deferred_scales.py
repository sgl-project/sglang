"""Do not allocate derived CUTLASS scale storage before checkpoint loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.quantization.modelopt_quant as modelopt
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeferredNvfp4Scales(CustomTestCase):
    def setUp(self):
        super().setUp()
        for target, value in [
            (
                "sglang.srt.layers.moe.get_moe_runner_backend",
                MoeRunnerBackend.FLASHINFER_CUTLASS,
            ),
            (
                "sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend",
                MoeRunnerBackend.FLASHINFER_CUTLASS,
            ),
            (
                "sglang.srt.layers.quantization.modelopt_quant.get_moe_a2a_backend",
                MoeA2ABackend.NONE,
            ),
            (
                "sglang.srt.layers.quantization.modelopt_quant._use_nvfp4_dispatch",
                False,
            ),
        ]:
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_layer(self, serialized=True, gated=True, intermediate_size=128):
        method = object.__new__(modelopt.ModelOptNvFp4FusedMoEMethod)
        method.enable_flashinfer_trtllm_moe = False
        method.quant_config = SimpleNamespace(
            is_checkpoint_nvfp4_serialized=serialized,
            group_size=16,
            use_per_token_activation=False,
            get_name=lambda: "modelopt" if serialized else "nvfp4_online",
        )
        # Loader callbacks are not invoked by this allocation/lifecycle test.
        method.prepare_weight_loader = Mock(return_value=None)
        layer = torch.nn.Module()
        layer.num_experts = layer.num_local_experts = 2
        layer.moe_ep_size = 1
        layer.moe_ep_rank = 0
        layer.moe_runner_config = SimpleNamespace(is_gated=gated)
        layer.dispatcher = Mock()
        method.moe_runner_config = layer.moe_runner_config
        with torch.device("cpu"):
            method.create_weights(layer, 2, 128, intermediate_size, torch.bfloat16)
        return method, layer

    def load_scales(self, layer, offset):
        expected = {}
        for prefix in ("w13", "w2"):
            scale = getattr(layer, prefix + "_weight_scale")
            values = (
                torch.arange(scale.numel(), dtype=torch.float32).reshape(scale.shape)
                % 32
                + offset
            ).to(torch.float8_e4m3fn)
            scale.data.copy_(values)
            getattr(layer, prefix + "_weight_scale_2").data.fill_(0.5)
            expected[prefix] = modelopt.swizzle_blockscale(values).view(torch.uint8)
        return expected

    def assert_loaded_scales(self, layer, expected):
        for prefix, value in expected.items():
            source = getattr(layer, prefix + "_weight_scale")
            derived = getattr(layer, prefix + "_blockscale_swizzled")
            self.assertTrue(torch.equal(derived.view(torch.uint8), value))
            self.assertEqual(source.data_ptr(), derived.data_ptr())

    def test_serialized_cutlass_skips_unloaded_scale_work(self):
        for gated in (True, False):
            with (
                self.subTest(gated=gated),
                patch.object(
                    modelopt, "swizzle_blockscale", wraps=modelopt.swizzle_blockscale
                ) as swizzle,
            ):
                _, layer = self.make_layer(gated=gated)
                swizzle.assert_not_called()
                self.assertIsNone(layer.w13_blockscale_swizzled)
                self.assertIsNone(layer.w2_blockscale_swizzled)
                self.assertGreater(layer.w13_weight_scale.numel(), 0)
                self.assertGreater(layer.w2_weight_scale.numel(), 0)

    def test_online_cutlass_retains_initialization(self):
        with patch.object(
            modelopt, "swizzle_blockscale", wraps=modelopt.swizzle_blockscale
        ) as swizzle:
            _, layer = self.make_layer(serialized=False)
        self.assertEqual(swizzle.call_count, 2)
        self.assertIsInstance(layer.w13_blockscale_swizzled, torch.nn.Parameter)
        self.assertIsInstance(layer.w2_blockscale_swizzled, torch.nn.Parameter)

    def test_postload_populates_deferred_buffers(self):
        for gated in (True, False):
            with self.subTest(gated=gated):
                method, layer = self.make_layer(gated=gated)
                expected = self.load_scales(layer, 1)
                method.process_weights_after_loading(layer)
                self.assert_loaded_scales(layer, expected)

    def test_reload_refreshes_the_shared_scale_storage(self):
        method, layer = self.make_layer()
        self.load_scales(layer, 1)
        method.process_weights_after_loading(layer)
        pointer = layer.w13_weight_scale.data_ptr()
        expected = self.load_scales(layer, 3)
        method.process_weights_after_loading(layer)
        self.assert_loaded_scales(layer, expected)
        self.assertEqual(pointer, layer.w13_weight_scale.data_ptr())

    def test_non_gated_padding_binds_a_larger_derived_buffer(self):
        method, layer = self.make_layer(gated=False, intermediate_size=192)
        expected = self.load_scales(layer, 1)
        padded_down_scale = torch.nn.functional.pad(
            layer.w2_weight_scale.detach(), (0, 4)
        )
        expected["w2"] = modelopt.swizzle_blockscale(padded_down_scale).view(
            torch.uint8
        )
        method.process_weights_after_loading(layer)
        for prefix, value in expected.items():
            derived = getattr(layer, prefix + "_blockscale_swizzled")
            self.assertTrue(torch.equal(derived.view(torch.uint8), value))
        self.assertEqual(layer.w13_weight.shape[1], 256)
        self.assertEqual(layer.w2_weight.shape[2], 128)
        self.assertTrue(torch.all(layer.w13_weight[:, 192:] == 0))
        self.assertTrue(torch.all(layer.w2_weight[:, :, 96:] == 0))


if __name__ == "__main__":
    unittest.main()

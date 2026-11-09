"""CPU regression tests for ModelSlim MXFP8 weight-scale layouts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.quantization.modelslim.schemes.modelslim_mxfp8 import (
    ModelSlimMXFP8Scheme,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestModelSlimMXFP8ScaleLayout(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.scheme = ModelSlimMXFP8Scheme()

    def make_layer(self, input_size=4304, tp_size=1, tp_rank=0, presharded=False):
        layer = torch.nn.Module()
        layer.tp_group = SimpleNamespace(rank_in_group=tp_rank, world_size=tp_size)
        layer.use_presharded_weights = presharded
        self.scheme.create_weights(
            layer,
            input_size // tp_size,
            [2],
            input_size,
            2,
            torch.bfloat16,
        )
        return layer

    def test_tp_shards_preserve_checkpoint_blocks(self):
        k = 4304
        weight = (torch.arange(2 * k).reshape(2, k) % 7 - 3).to(torch.float8_e4m3fn)
        scales = (torch.arange(2 * 135).reshape(2, 135) % 7 + 124).to(torch.uint8)
        reference = (
            weight.float()
            * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)[:, :k]
        )
        x = (torch.arange(k) % 5 - 2).float().unsqueeze(0)

        for tp_size in (1, 2, 4, 8):
            result = torch.zeros((1, 2))
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    layer = self.make_layer(tp_size=tp_size, tp_rank=rank)
                    start, end = rank * (k // tp_size), (rank + 1) * (k // tp_size)
                    blocks = torch.arange(start, end) // 32
                    expected_scales = scales[:, blocks.unique_consecutive()]
                    if tp_size == 1:
                        layer.weight_scale.data.copy_(scales)
                    else:
                        layer.weight_scale.weight_loader(layer.weight_scale, scales)
                    torch.testing.assert_close(layer.weight_scale, expected_scales)
                    if tp_size == 4 and rank in (1, 3):
                        self.assertEqual(layer.weight_scale.shape[1], 35)
                    layer.weight.data.copy_(weight[:, start:end])
                    self.scheme.process_weights_after_loading(layer)

                    # Undo the kernel's paired scale layout and compare every
                    # real weight element against the global checkpoint blocks.
                    local_scales = layer.weight_scale_inv.transpose(0, 1).reshape(2, -1)
                    local_weight = (
                        layer.weight.t().float()
                        * torch.exp2(local_scales.float() - 127).repeat_interleave(
                            32, dim=1
                        )[:, : layer.weight.shape[0]]
                    )
                    left, right = getattr(layer, "mxfp8_tp_padding", (0, 0))
                    torch.testing.assert_close(
                        local_weight[:, left : left + end - start],
                        reference[:, start:end],
                    )
                    self.assertEqual(torch.count_nonzero(local_weight[:, :left]), 0)
                    if right:
                        self.assertEqual(
                            torch.count_nonzero(local_weight[:, -right:]), 0
                        )
                    padded_x = torch.nn.functional.pad(x[:, start:end], (left, right))
                    result += padded_x @ local_weight.t()
            torch.testing.assert_close(result, x @ reference.t())

    def test_presharded_scales_are_loaded_without_reslicing(self):
        layer = self.make_layer(tp_size=4, tp_rank=3, presharded=True)
        scales = torch.arange(70, dtype=torch.uint8).reshape(2, 35)
        layer.weight_scale.weight_loader(layer.weight_scale, scales)
        torch.testing.assert_close(layer.weight_scale, scales)
        with self.assertRaises(AssertionError):
            layer.weight_scale.weight_loader(layer.weight_scale, scales[:, :-1])

    def test_unaligned_input_padding_precedes_kernel(self):
        layer = self.make_layer(tp_size=4, tp_rank=3)
        self.scheme.kernel = Mock()
        for shape in ((2, 1076), (2, 3, 1076)):
            with self.subTest(shape=shape):
                x = torch.ones(shape, dtype=torch.bfloat16)
                bias = torch.ones(2)
                result = self.scheme.apply_weights(layer, x, bias)
                self.assertIs(result, self.scheme.kernel.apply.return_value)
                padded = self.scheme.kernel.apply.call_args.args[1]
                torch.testing.assert_close(padded, torch.nn.functional.pad(x, (28, 16)))
                self.assertIs(self.scheme.kernel.apply.call_args.args[2], bias)
        self.scheme.kernel.reset_mock()
        with self.assertRaisesRegex(ValueError, "unquantized inputs"):
            self.scheme.apply_weights(layer, (torch.ones(1), torch.ones(1)))
        self.scheme.kernel.apply.assert_not_called()

    def test_unsharded_column_and_aligned_paths_keep_original_loader(self):
        for local_k, full_k, output_sizes, output_size in (
            (4304, 4304, [2], 2),  # TP1, partial final block.
            (4304, 4304, [1], 2),  # Column-parallel partition.
            (96, 192, [2], 2),  # Row-parallel, odd but aligned scale count.
        ):
            with self.subTest(local_k=local_k, output_sizes=output_sizes):
                layer = torch.nn.Module()
                loader = Mock()
                self.scheme.create_weights(
                    layer,
                    local_k,
                    output_sizes,
                    full_k,
                    output_size,
                    torch.bfloat16,
                    weight_loader=loader,
                )
                self.assertIs(layer.weight_scale.weight_loader, loader)
                self.assertIs(layer.weight.weight_loader, loader)
                self.assertFalse(hasattr(layer, "mxfp8_tp_padding"))
                self.scheme.kernel = Mock()
                for x in (torch.ones(2, local_k), (torch.ones(1), torch.ones(1))):
                    self.scheme.apply_weights(layer, x)
                    self.assertIs(self.scheme.kernel.apply.call_args.args[1], x)

    def test_scale_placeholder_rounds_up_partial_block(self):
        layer = torch.nn.Module()

        self.scheme.create_weights(
            layer=layer,
            input_size_per_partition=4304,
            output_partition_sizes=[2],
            input_size=4304,
            output_size=2,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(layer.weight_scale.shape, (2, 135))

    def test_post_load_pads_odd_scale_count_for_pair_layout(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((2, 4304), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones((2, 135), dtype=torch.uint8), requires_grad=False
            ),
        )
        layer.register_parameter("bias", None)

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (4304, 2))
        self.assertEqual(layer.weight_scale_inv.shape, (68, 2, 2))
        self.assertTrue(torch.all(layer.weight_scale_inv[-1, :, 1] == 0))
        self.assertFalse(hasattr(layer, "weight_scale"))


if __name__ == "__main__":
    unittest.main()

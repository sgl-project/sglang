"""CPU regression tests for the Ascend MXFP4 W4A4 online config (Linear gate)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Initialize quantization before its NPU helpers to avoid a cold-import cycle.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUDualLevelMXFP4LinearMethod,
    NPUMXFP4W4A8LinearMethod,
    _prepare_dual_level_weight_l0_scale,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.npu_mxfp4 import Mxfp4W4A8Config
from sglang.srt.layers.quantization.npu_mxfp4_utils import MXFP4_BLOCK_SIZE
from sglang.srt.layers.quantization.npu_mxfp4_w4a4 import Mxfp4W4A4Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNPUDualLevelMXFP4Linear(CustomTestCase):
    def test_preserves_l0_scale_blocks_and_singleton_dimensions(self):
        for shape in ((4, 1), (4, 1, 1), (1, 2), (4, 2, 1)):
            with self.subTest(shape=shape):
                scale = torch.arange(torch.tensor(shape).prod()).reshape(shape)
                prepared = _prepare_dual_level_weight_l0_scale(scale)
                expected = scale.reshape(shape[0], -1).transpose(0, 1)
                self.assertEqual(prepared.shape, expected.shape)
                self.assertTrue(torch.equal(prepared, expected))
                self.assertTrue(prepared.is_contiguous())


class TestMxfp4W4A4ConfigGroupAlignment(CustomTestCase):
    """Guards the K (input dim) 32-alignment gate in ``get_quant_method``.

    MXFP4 block scales use group_size=32, so a reduction dim K that is not a
    multiple of 32 does not tile evenly. Qwen3.5's vision MLP ``linear_fc2`` has
    K=4304 (4304/32=134.5); serving Qwen3.5-VL online with ``--quantization
    mxfp4`` must fall back to BF16 for it rather than route it to the dual-level
    FP4 method, mirroring how the offline msmodelslim yaml excludes linear_fc2.
    """

    @staticmethod
    def _make_linear_layer(input_size: int) -> LinearBase:
        # Build a bare LinearBase without the full (distributed) __init__; only
        # input_size is read by the alignment gate. Mirrors test_modelslim_config.
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.input_size = input_size
        return layer

    def test_online_config_is_not_static(self):
        self.assertFalse(Mxfp4W4A4Config().is_static_cfg())

    def test_skips_group_unaligned_linear(self):
        # K=4304 is not a multiple of 32 -> must fall back to BF16, not the
        # dual-level FP4 method (whose block scales cannot tile K=4304).
        self.assertNotEqual(4304 % MXFP4_BLOCK_SIZE, 0)
        layer = self._make_linear_layer(4304)

        method = Mxfp4W4A4Config().get_quant_method(
            layer, "model.visual.blocks.0.mlp.linear_fc2"
        )

        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_aligned_npu_linear_uses_dual_level_and_honors_ignored_layers(self):
        layer = self._make_linear_layer(4096)
        prefix = "model.layers.0.self_attn.o_proj"
        with patch(
            "sglang.srt.layers.quantization.npu_mxfp4_w4a4.is_npu", return_value=True
        ):
            self.assertIsInstance(
                Mxfp4W4A4Config().get_quant_method(layer, prefix),
                NPUDualLevelMXFP4LinearMethod,
            )
            self.assertIsInstance(
                Mxfp4W4A4Config(ignored_layers=[prefix]).get_quant_method(
                    layer, prefix
                ),
                UnquantizedLinearMethod,
            )

    def test_both_configs_use_the_bound_group_local_k(self):
        prefix = "model.layers.0.self_attn.o_proj"
        for config_cls, method_cls in (
            (Mxfp4W4A4Config, NPUDualLevelMXFP4LinearMethod),
            (Mxfp4W4A8Config, NPUMXFP4W4A8LinearMethod),
        ):
            for linear_cls, size, tp, aligned in (
                (RowParallelLinear, 4352, 16, False),
                (RowParallelLinear, 4096, 16, True),
                (RowParallelLinear, 4352, 1, True),
                (RowParallelLinear, 4304, 1, False),
                (ColumnParallelLinear, 4352, 16, True),
                (ReplicatedLinear, 4352, 16, True),
            ):
                layer = linear_cls.__new__(linear_cls)
                torch.nn.Module.__init__(layer)
                layer.input_size = size
                layer.tp_group = SimpleNamespace(rank=0, size=tp)
                self.assertFalse(hasattr(layer, "input_size_per_partition"))
                with (
                    self.subTest(config=config_cls, layer=linear_cls, size=size, tp=tp),
                    patch(f"{config_cls.__module__}.is_npu", return_value=True),
                    patch(
                        "sglang.srt.distributed.utils.get_group_rank_size",
                        return_value=(0, tp),
                    ) as group,
                ):
                    method = config_cls().get_quant_method(layer, prefix)
                    self.assertIsInstance(
                        method, method_cls if aligned else UnquantizedLinearMethod
                    )
                    if linear_cls is RowParallelLinear:
                        group.assert_called_once_with(layer.tp_group)
                    else:
                        group.assert_not_called()
                    ignored = config_cls(ignored_layers=[prefix]).get_quant_method(
                        layer, prefix
                    )
                    self.assertIsInstance(ignored, UnquantizedLinearMethod)

    def test_aligned_linear_not_skipped(self):
        # A 32-aligned K must NOT hit the fallback: on CPU (is_npu() False) the
        # method resolution reaches the NPU-only branch and raises, proving the
        # alignment gate did not short-circuit it.
        self.assertEqual(4096 % MXFP4_BLOCK_SIZE, 0)
        layer = self._make_linear_layer(4096)
        config = Mxfp4W4A4Config()

        if is_npu():
            self.skipTest("aligned-path fallback assertion is CPU-only")
        with self.assertRaises(NotImplementedError):
            config.get_quant_method(layer, "model.layers.0.self_attn.o_proj")


if __name__ == "__main__":
    unittest.main()

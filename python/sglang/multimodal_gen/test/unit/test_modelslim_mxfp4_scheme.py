# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ModelSlimMXFP4Scheme's dual-scale/mul_scale bugs.

Both bugs only trigger for ``W4A4_MXFP4_DUALSCALE``, which does not import
the NPU-only kernel path in ``__init__``, so these are safe to construct on
CPU.
"""

import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp4_scheme import (
    ModelSlimMXFP4Scheme,
)


class TestDualScaleKeyCheck(unittest.TestCase):
    def test_construction_does_not_raise_when_dual_scale_present(self):
        # Previously read the never-set attribute `self.is_dual_scale_key`,
        # raising AttributeError on every dual-scale layer regardless of
        # whether the checkpoint actually had the dual scale.
        scheme = ModelSlimMXFP4Scheme(
            quant_config={"layer.weight_dual_scale": "..."},
            prefix="layer",
            quant_type="W4A4_MXFP4_DUALSCALE",
        )
        self.assertEqual(scheme.dual_scale_key, "layer.weight_dual_scale")


class TestHasMulScaleAcceptsEitherKey(unittest.TestCase):
    def test_repacked_checkpoint_with_only_the_new_key_is_detected(self):
        # tools/wan_repack.py renames `.div.mul_scale` to `.mul_scale`, so a
        # repacked checkpoint only ever has the new key, never both.
        scheme = ModelSlimMXFP4Scheme(
            quant_config={
                "layer.weight_dual_scale": "...",
                "layer.mul_scale": "...",
            },
            prefix="layer",
            quant_type="W4A4_MXFP4_DUALSCALE",
        )
        self.assertTrue(scheme.has_mul_scale)

    def test_unrepacked_checkpoint_with_only_the_legacy_key_is_detected(self):
        scheme = ModelSlimMXFP4Scheme(
            quant_config={
                "layer.weight_dual_scale": "...",
                "layer.div.mul_scale": "...",
            },
            prefix="layer",
            quant_type="W4A4_MXFP4_DUALSCALE",
        )
        self.assertTrue(scheme.has_mul_scale)

    def test_neither_key_present_is_not_detected(self):
        scheme = ModelSlimMXFP4Scheme(
            quant_config={"layer.weight_dual_scale": "..."},
            prefix="layer",
            quant_type="W4A4_MXFP4_DUALSCALE",
        )
        self.assertFalse(scheme.has_mul_scale)


class TestMulScaleTPSharding(unittest.TestCase):
    def _param(self, size):
        from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp4_scheme import (
            _MulScaleParameter,
        )

        return _MulScaleParameter(
            data=torch.empty(size, dtype=torch.float32),
            input_dim=0,
            weight_loader=lambda *a, **k: None,
        )

    def test_row_parallel_shards_input_dim_on_both_ranks(self):
        full = torch.arange(8, dtype=torch.float32)
        for rank in (0, 1):
            p = self._param(4)
            with mock.patch(
                "sglang.multimodal_gen.runtime.models.parameter.get_tp_rank",
                return_value=rank,
            ):
                p.load_row_parallel_weight(full)
            self.assertTrue(torch.equal(p.data, full[rank * 4 : rank * 4 + 4]))

    def test_column_parallel_loads_full_vector(self):
        full = torch.arange(8, dtype=torch.float32)
        p = self._param(8)
        p.load_column_parallel_weight(full)
        self.assertTrue(torch.equal(p.data, full))


if __name__ == "__main__":
    unittest.main()

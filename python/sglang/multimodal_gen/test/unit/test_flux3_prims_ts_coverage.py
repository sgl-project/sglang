# SPDX-License-Identifier: Apache-2.0
"""Regression test: verify_prims_ts_coverage must not double-report a
missing QKV projection.

When a ``to_qkv`` projection had no planned PrimsTS role, the loop appended
it to the ``missing`` list twice: once as the plain prefix, once with the
"(QKV/QKNorm/RoPE epilogue)" suffix. Call the method unbound against a
minimal fake config object, so the test does not need a real
``Flux3ModelOptFp4Config`` or any FlashInfer/CUDA hardware.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts as mod


class TestVerifyPrimsTsCoverageQkvReporting(unittest.TestCase):
    def _make_layer(self, prefix, quant_method):
        layer = mock.Mock(spec=LinearBase)
        layer.prefix = prefix
        layer.quant_method = quant_method
        return layer

    def test_missing_qkv_is_reported_once(self):
        qkv_method = mock.Mock(spec=mod.Flux3Nvfp4LinearMethod)
        qkv_method.prims_ts_role = None  # no plan: the bug case

        qkv_layer = self._make_layer("blk.0.to_qkv", qkv_method)
        attn_out_layer = self._make_layer(
            "blk.0.attn_out",
            SimpleNamespace(prims_ts_role=mod.Flux3PrimsTsRole.LINEAR),
        )
        mlp_in_layer = self._make_layer(
            "blk.0.mlp_in",
            SimpleNamespace(prims_ts_role=mod.Flux3PrimsTsRole.SWIGLU),
        )
        mlp_out_layer = self._make_layer(
            "blk.0.mlp_out",
            SimpleNamespace(prims_ts_role=mod.Flux3PrimsTsRole.LINEAR),
        )

        model = mock.Mock()
        model.modules.return_value = [
            qkv_layer,
            attn_out_layer,
            mlp_in_layer,
            mlp_out_layer,
        ]

        fake_self = SimpleNamespace(
            enable_prims_ts=True,
            dynamic_input_scale=False,
            prims_ts_targets=("blk.",),
        )

        with mock.patch.object(mod, "flux3_prims_ts_available", return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                mod.Flux3ModelOptFp4Config.verify_prims_ts_coverage(fake_self, model)

        self.assertEqual(str(ctx.exception).count("to_qkv"), 1)


if __name__ == "__main__":
    unittest.main()

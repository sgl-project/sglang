# SPDX-License-Identifier: Apache-2.0
"""Reject W4AFP8 MoE group_size != 128 (cutlass chunk_size silent-wrong path)."""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8_moe import (  # noqa: E402
    CompressedTensorsW4AFP8MoE,
)
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_scheme(group_size: int) -> CompressedTensorsW4AFP8MoE:
    weights = SimpleNamespace(num_bits=4, group_size=group_size, symmetric=True)
    quant_config = SimpleNamespace(
        target_scheme_map={"Linear": {"weights": weights}},
        quant_format="pack-quantized",
    )
    return CompressedTensorsW4AFP8MoE(quant_config, weight_quant=None, input_quant=None)


class TestW4AFP8MoEGroupSizeGuard(CustomTestCase):
    def test_compressed_tensors_scheme_rejects_non_128(self):
        with self.assertRaisesRegex(ValueError, "group_size=128"):
            _make_scheme(64)

    def test_compressed_tensors_scheme_accepts_128(self):
        scheme = _make_scheme(128)
        self.assertEqual(scheme.group_size, 128)

    def test_w4afp8_method_rejects_non_128(self):
        cfg = W4AFp8Config(group_size=64)
        with self.assertRaisesRegex(ValueError, "group_size=128"):
            W4AFp8MoEMethod(cfg)

    def test_w4afp8_method_accepts_128(self):
        method = W4AFp8MoEMethod(W4AFp8Config(group_size=128))
        self.assertEqual(method.quant_config.group_size, 128)

    def test_cutlass_w4a8_moe_exposes_group_size(self):
        import inspect

        from sglang.srt.layers.moe import cutlass_w4a8_moe as mod

        sig = inspect.signature(mod.cutlass_w4a8_moe)
        self.assertIn("group_size", sig.parameters)
        self.assertEqual(sig.parameters["group_size"].default, 128)

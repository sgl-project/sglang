import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.cpu.quantization.gptq_kernels import (
    GPTQIntelAMXLinearKernel,
    GPTQIntelAMXMoEKernel,
    _normalize_symmetric_qzeros,
)
from sglang.srt.layers.quantization.gptq import CPUGPTQConfig
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cpu_ci(est_time=2, suite="extra-test-cpu-intel")

_PACKER = "sglang.srt.hardware_backend.cpu.quantization.gptq_kernels._amx_process_weight_after_loading"
_PACKED_EIGHTS = -2004318072  # 0x88888888 as int32
_PACKED_SEVENS = 0x77777777


def _config(sym=None, dynamic=None):
    data = {"bits": 4, "group_size": 128, "desc_act": False, "dynamic": dynamic or {}}
    if sym is not None:
        data["sym"] = sym
    return CPUGPTQConfig.from_config(data)


def _layer(names, shape, value):
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.proj"
    for name in names:
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.full(shape, value, dtype=torch.int32), requires_grad=False
            ),
        )
    return layer


class TestCPUGPTQZeroPoints(unittest.TestCase):
    def test_kernel_and_quantization_import_orders(self):
        statements = [
            "from sglang.srt.hardware_backend.cpu.quantization.gptq_kernels import GPTQIntelAMXLinearKernel, GPTQIntelAMXMoEKernel",
            "from sglang.srt.layers.quantization.gptq import CPUGPTQConfig",
        ]
        for imports in (statements, statements[::-1]):
            with self.subTest(first=imports[0]):
                result = subprocess.run(
                    [sys.executable, "-c", "; ".join(imports)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_config_preserves_explicit_or_missing_sym(self):
        for sym in (True, False, None):
            with self.subTest(sym=sym):
                self.assertIs(_config(sym).sym, sym)

    def test_symmetric_legacy_and_explicit_zeros_have_same_math(self):
        codes = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        scales = torch.linspace(0.01, 0.08, 8)
        x = torch.tensor([[0.5, 1.0]])
        expected = x @ ((codes - 8) * scales)
        for shape in ((1, 1), (2, 1, 1)):
            for stored in (_PACKED_SEVENS, _PACKED_EIGHTS):
                with self.subTest(shape=shape, stored=stored):
                    layer = _layer(["qzeros"], shape, stored)
                    original_storage = layer.qzeros.detach()
                    layer.qzeros.weight_loader = "preserve loader metadata"
                    _normalize_symmetric_qzeros(layer, "qzeros", _config(True))
                    self.assertTrue(torch.all(layer.qzeros == _PACKED_SEVENS))
                    # Simulate only the legacy C++ zero decoding, then check
                    # independent nonzero-scale linear algebra against zp=8.
                    zero = (int(layer.qzeros.flatten()[0]) & 15) + 1
                    torch.testing.assert_close(x @ ((codes - zero) * scales), expected)
                    self.assertTrue(torch.all(original_storage == stored))
                    self.assertEqual(
                        layer.qzeros.weight_loader, "preserve loader metadata"
                    )

    def test_asymmetric_unknown_and_mixed_zeros_are_unchanged(self):
        for sym in (False, None):
            layer = _layer(["qzeros"], (2, 2), _PACKED_EIGHTS)
            original = layer.qzeros.data_ptr()
            _normalize_symmetric_qzeros(layer, "qzeros", _config(sym))
            self.assertEqual(layer.qzeros.data_ptr(), original)
            self.assertTrue(torch.all(layer.qzeros == _PACKED_EIGHTS))
        layer = _layer(["qzeros"], (2, 2), _PACKED_EIGHTS)
        layer.qzeros.data[0, 0] = _PACKED_SEVENS
        original = layer.qzeros.clone()
        _normalize_symmetric_qzeros(layer, "qzeros", _config(True))
        torch.testing.assert_close(layer.qzeros, original)

    def test_dynamic_symmetry_overrides_global_config(self):
        for global_sym, layer_sym in ((True, False), (False, True)):
            for attr in ("prefix", "layer_name"):
                with self.subTest(global_sym=global_sym, attr=attr):
                    layer = _layer(["qzeros"], (1, 1), _PACKED_EIGHTS)
                    if attr == "layer_name":
                        del layer.prefix
                        layer.layer_name = "model.layers.0.proj"
                    config = _config(global_sym, {"+:.*proj$": {"sym": layer_sym}})
                    _normalize_symmetric_qzeros(layer, "qzeros", config)
                    expected = _PACKED_SEVENS if layer_sym else _PACKED_EIGHTS
                    self.assertTrue(torch.all(layer.qzeros == expected))

    def test_linear_and_moe_normalize_before_native_packing(self):
        for stored in (_PACKED_SEVENS, _PACKED_EIGHTS):
            for kernel_class, names, shape in (
                (GPTQIntelAMXLinearKernel, ["qweight", "qzeros", "scales"], (1, 1)),
                (
                    GPTQIntelAMXMoEKernel,
                    [
                        "w13_qweight",
                        "w13_qzeros",
                        "w13_scales",
                        "w2_qweight",
                        "w2_qzeros",
                        "w2_scales",
                    ],
                    (2, 1, 1),
                ),
            ):
                with self.subTest(kernel=kernel_class.__name__, stored=stored):
                    layer = _layer(names, shape, stored)
                    seen = []

                    def pack(module, weight_names, transpose_dims, method):
                        self.assertEqual(method, "gptq")
                        self.assertTrue(
                            torch.all(
                                getattr(module, weight_names[1]) == _PACKED_SEVENS
                            )
                        )
                        seen.append(weight_names[1])

                    with patch(_PACKER, side_effect=pack):
                        kernel_class(_config(True)).process_weights_after_loading(layer)
                    self.assertEqual(seen, [name for name in names if "qzeros" in name])

    @unittest.skipUnless(
        cpu_has_amx_support(),
        "Requires the Intel AMX GPTQ operator",
    )
    def test_nonzero_native_gemm_for_legacy_and_explicit_symmetric_zeros(self):
        import sgl_kernel  # noqa: F401

        if not hasattr(torch.ops.sgl_kernel, "int4_scaled_mm_cpu"):
            self.skipTest("CPU GPTQ operator is not installed")
        k, n = 128, 64
        codes = torch.arange(k * n, dtype=torch.int64).reshape(k, n) % 16
        shifts = (torch.arange(8, dtype=torch.int64) * 4).reshape(1, 8, 1)
        packed_weight = (
            (codes.reshape(k // 8, 8, n) << shifts).sum(dim=1).to(torch.int32)
        )
        scales = torch.linspace(0.01, 0.1, n).reshape(1, n).to(torch.bfloat16)
        x = torch.linspace(0.25, 1.0, k).repeat(4, 1).to(torch.bfloat16)
        expected = x.float() @ ((codes.float() - 8) * scales.float())
        for stored in (_PACKED_SEVENS, _PACKED_EIGHTS):
            with self.subTest(stored=stored):
                layer = torch.nn.Module()
                layer.register_parameter(
                    "qweight",
                    torch.nn.Parameter(packed_weight.clone(), requires_grad=False),
                )
                layer.register_parameter(
                    "qzeros",
                    torch.nn.Parameter(
                        torch.full((1, n // 8), stored, dtype=torch.int32),
                        requires_grad=False,
                    ),
                )
                layer.register_parameter(
                    "scales", torch.nn.Parameter(scales.clone(), requires_grad=False)
                )
                kernel = GPTQIntelAMXLinearKernel(_config(True))
                kernel.process_weights_after_loading(layer)
                actual = kernel.apply(layer, x)
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.02, atol=0.04
                )

    @unittest.skipUnless(cpu_has_amx_support(), "Requires the Intel AMX MoE operator")
    def test_nonzero_native_moe_preserves_legacy_results(self):
        import sgl_kernel  # noqa: F401

        if not hasattr(torch.ops.sgl_kernel, "fused_experts_cpu"):
            self.skipTest("CPU MoE operator is not installed")
        torch.manual_seed(20261123)
        experts, hidden, intermediate = 2, 128, 128
        weights = {
            "w13_qweight": torch.randint(
                -(2**31),
                2**31 - 1,
                (experts, hidden // 8, 2 * intermediate),
                dtype=torch.int32,
            ),
            "w2_qweight": torch.randint(
                -(2**31),
                2**31 - 1,
                (experts, intermediate // 8, hidden),
                dtype=torch.int32,
            ),
            "w13_scales": (torch.rand(experts, 1, 2 * intermediate) * 0.02 + 0.01).to(
                torch.bfloat16
            ),
            "w2_scales": (torch.rand(experts, 1, hidden) * 0.02 + 0.01).to(
                torch.bfloat16
            ),
        }
        dispatch = SimpleNamespace(
            hidden_states=(torch.rand(4, hidden) * 0.75 + 0.25).to(torch.bfloat16),
            topk_output=(
                torch.full((4, 2), 0.5),
                torch.tensor([[0, 1]] * 4, dtype=torch.int32),
                None,
            ),
        )
        outputs = []
        for stored in (_PACKED_SEVENS, _PACKED_EIGHTS):
            layer = torch.nn.Module()
            layer.layer_name = "model.layers.0.mlp.experts"
            for name, value in weights.items():
                layer.register_parameter(
                    name, torch.nn.Parameter(value.clone(), requires_grad=False)
                )
            for name, shape in (
                ("w13_qzeros", (experts, 1, 2 * intermediate // 8)),
                ("w2_qzeros", (experts, 1, hidden // 8)),
            ):
                layer.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.full(shape, stored, dtype=torch.int32),
                        requires_grad=False,
                    ),
                )
            kernel = GPTQIntelAMXMoEKernel(_config(True))
            kernel.moe_runner_config = SimpleNamespace(activation="silu")
            kernel.process_weights_after_loading(layer)
            outputs.append(kernel.apply(layer, dispatch).hidden_states)
        self.assertGreater(torch.count_nonzero(outputs[0]).item(), 0)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

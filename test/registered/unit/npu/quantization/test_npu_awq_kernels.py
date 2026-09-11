"""
Unit tests for sglang.srt.hardware_backend.npu.quantization.awq_kernels.

The awq_kernels module depends on torch_npu (only available on Ascend NPU
hardware) and on a heavy sglang import chain (triton/transformers/...).
To keep these unit tests runnable on any machine, the heavy infrastructure is
stubbed in ``sys.modules`` and the real ``awq_kernels.py`` source file is loaded
directly by path with importlib. The CI marker ``register_npu_ci`` is still
emitted (as a no-op) so the AST-based CI register picks this suite up.
"""

import importlib.util
import os
import sys
import types
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stub heavy infrastructure in sys.modules BEFORE loading the source module.
# This avoids triggering sglang/__init__.py (which pulls in triton/IPython/...).
# ---------------------------------------------------------------------------
_NP = types.ModuleType("torch_npu")
_NP.npu_weight_quant_batchmatmul = MagicMock()
sys.modules.setdefault("torch_npu", _NP)
sys.modules.setdefault("torch_npu.contrib", MagicMock())
sys.modules.setdefault("sgl_kernel_npu", MagicMock())


def _ensure_pkg(dotted: str) -> ModuleType:
    """Ensure a dotted package (and all parents) exist in sys.modules.

    Each missing name is created as a plain ModuleType and registered as an
    attribute of its parent, so ``from a.b.c import x`` resolves without ever
    running a real __init__.py.
    """
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            setattr(parent, parts[i - 1], sys.modules[name])
    return sys.modules[dotted]


def _install_stub(dotted: str, mod: ModuleType) -> None:
    """Install a leaf module stub and link it onto its parent package."""
    _ensure_pkg(".".join(dotted.split(".")[:-1]))
    sys.modules[dotted] = mod
    parent = sys.modules[".".join(dotted.split(".")[:-1])]
    setattr(parent, dotted.split(".")[-1], mod)


# sglang package chain (prevents real __init__.py from running)
for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.quantization",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.quantization",
    "sglang.test",
    "sglang.test.ci",
):
    _ensure_pkg(_pkg)

# register_npu_ci: load the REAL marker from sglang's ci_register.py (by path,
# so sglang/__init__.py —which needs triton —does not run) and register it in
# sys.modules so the literal `from sglang.test.ci.ci_register import ...` used
# by the attention-test files resolves to the real no-op marker. CI
# registration is AST-based; the call is a runtime no-op.
_ci_src = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "test",
        "ci",
        "ci_register.py",
    )
)
_ci_spec = importlib.util.spec_from_file_location("sglang.test.ci.ci_register", _ci_src)
_ci_mod = importlib.util.module_from_spec(_ci_spec)
sys.modules["sglang.test.ci.ci_register"] = _ci_mod
_ci_spec.loader.exec_module(_ci_mod)
from sglang.test.ci.ci_register import register_npu_ci

# moe_methods: opaque stub (awq_kernels only stores the instance)
_moe_mod = ModuleType("sglang.srt.hardware_backend.npu.quantization.moe_methods")


class _StubNPUWNA16Int4MoEMethod:
    def __init__(self, *args, **kwargs):
        pass


_moe_mod.NPUWNA16Int4MoEMethod = _StubNPUWNA16Int4MoEMethod
_install_stub("sglang.srt.hardware_backend.npu.quantization.moe_methods", _moe_mod)

# replace_parameter: faithful reimplementation (the real utils.py imports
# triton via fp8_kernel, which is unavailable here; the function itself is
# pure-torch).
_utils_mod = ModuleType("sglang.srt.layers.quantization.utils")


def replace_parameter(mod, name, new):
    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        old.copy_(new)
    else:
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))


_utils_mod.replace_parameter = replace_parameter
_install_stub("sglang.srt.layers.quantization.utils", _utils_mod)

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Load the real awq_kernels.py source file by path.
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _TEST_DIR
for _ in range(5):  # test/registered/unit/npu/quantization -> repo root
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
_SRC_PATH = os.path.join(
    _REPO_ROOT,
    "python",
    "sglang",
    "srt",
    "hardware_backend",
    "npu",
    "quantization",
    "awq_kernels.py",
)

_spec = importlib.util.spec_from_file_location("awq_kernels_under_test", _SRC_PATH)
awq_kernels_mod = importlib.util.module_from_spec(_spec)
sys.modules["awq_kernels_under_test"] = awq_kernels_mod
_spec.loader.exec_module(awq_kernels_mod)

AWQAscendLinearKernel = awq_kernels_mod.AWQAscendLinearKernel
AWQAscendMoEKernel = awq_kernels_mod.AWQAscendMoEKernel

PACK_FACTOR = 8  # 4-bit weights packed into int32 (32 // 4)


def _make_quant_config(pack_factor=PACK_FACTOR):
    return SimpleNamespace(pack_factor=pack_factor)


def _make_linear_layer(scales, qweight, qzeros):
    """Build a real nn.Module layer carrying scales/qweight/qzeros parameters."""
    layer = torch.nn.Module()
    layer.scales = torch.nn.Parameter(scales.clone(), requires_grad=False)
    layer.qweight = torch.nn.Parameter(qweight.clone(), requires_grad=False)
    layer.qzeros = torch.nn.Parameter(qzeros.clone(), requires_grad=False)
    return layer


def _make_moe_layer(
    w13_qweight, w13_qzeros, w13_scales, w2_qweight, w2_qzeros, w2_scales
):
    layer = torch.nn.Module()
    layer.w13_qweight = torch.nn.Parameter(w13_qweight.clone(), requires_grad=False)
    layer.w13_qzeros = torch.nn.Parameter(w13_qzeros.clone(), requires_grad=False)
    layer.w13_scales = torch.nn.Parameter(w13_scales.clone(), requires_grad=False)
    layer.w2_qweight = torch.nn.Parameter(w2_qweight.clone(), requires_grad=False)
    layer.w2_qzeros = torch.nn.Parameter(w2_qzeros.clone(), requires_grad=False)
    layer.w2_scales = torch.nn.Parameter(w2_scales.clone(), requires_grad=False)
    return layer


# =============================================================================
# AWQAscendLinearKernel —__init__
# =============================================================================
class TestLinearKernelInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = _make_quant_config()
        kernel = AWQAscendLinearKernel(cfg)
        self.assertIs(kernel.quant_config, cfg)

    def test_default_quant_config_none(self):
        kernel = AWQAscendLinearKernel()
        self.assertIsNone(kernel.quant_config)


# =============================================================================
# AWQAscendLinearKernel —process_weights_after_loading (NPU fast path)
# =============================================================================
class TestLinearProcessNpuFastPath(unittest.TestCase):
    """group_size is a multiple of 32 and < K -> NPU fast path."""

    def setUp(self):
        # K=64, N=8, num_groups=2 -> group_size=32 (mult of 32, < 64)
        self.K, self.N, self.num_groups = 64, 8, 2
        self.scales = torch.ones(self.num_groups, self.N, dtype=torch.float32)
        # qweight all-zero so each packed int32 nibble is 0.
        self.qweight = torch.zeros(self.K, self.N // PACK_FACTOR, dtype=torch.int32)
        # qzeros: group 0 = 0x76543210, group 1 = 0x00000000
        self.qzeros = torch.tensor([[0x76543210], [0x00000000]], dtype=torch.int32)
        self.kernel = AWQAscendLinearKernel(_make_quant_config())
        self.layer = _make_linear_layer(self.scales, self.qweight, self.qzeros)
        self.kernel.process_weights_after_loading(self.layer)

    def test_use_npu_matmul_flag(self):
        self.assertTrue(self.layer.use_npu_matmul)

    def test_npu_group_size(self):
        self.assertEqual(self.layer.npu_group_size, 32)

    def test_weight_is_parameter(self):
        self.assertIsInstance(self.layer.weight, torch.nn.Parameter)

    def test_weight_dtype_int32(self):
        self.assertEqual(self.layer.weight.dtype, torch.int32)

    def test_weight_shape(self):
        self.assertEqual(self.layer.weight.shape, (self.K, self.N // PACK_FACTOR))

    def test_zeros_is_parameter(self):
        self.assertIsInstance(self.layer.zeros, torch.nn.Parameter)

    def test_zeros_shape(self):
        self.assertEqual(self.layer.zeros.shape, (self.num_groups, self.N))

    def test_zeros_dtype_matches_scales(self):
        self.assertEqual(self.layer.zeros.dtype, torch.float32)

    def test_scales_preserved_as_parameter(self):
        self.assertIsInstance(self.layer.scales, torch.nn.Parameter)
        self.assertEqual(self.layer.scales.shape, (self.num_groups, self.N))

    def test_qweight_removed(self):
        self.assertFalse(hasattr(self.layer, "qweight"))

    def test_qzeros_removed(self):
        self.assertFalse(hasattr(self.layer, "qzeros"))

    def test_weight_xor_applied(self):
        """All-zero input XOR 0x88888888 -> every int32 == 0x88888888."""
        expected = torch.tensor(0x88888888, dtype=torch.uint32)
        weight_u32 = self.layer.weight.data.view(torch.uint32)
        self.assertTrue(torch.equal(weight_u32, expected.expand_as(weight_u32)))

    def test_zero_point_conversion(self):
        """qzeros nibbles converted via -(nibble - 8) = 8 - nibble."""
        # group 0 = 0x76543210 -> nibbles [0,4,1,5,2,6,3,7] (in extract order)
        # -> -(z-8) -> [8,4,7,3,6,2,5,1]
        expected_g0 = torch.tensor([8, 4, 7, 3, 6, 2, 5, 1], dtype=torch.float32)
        self.assertTrue(torch.equal(self.layer.zeros.data[0], expected_g0))
        # group 1 = 0 -> all nibbles 0 -> 8
        expected_g1 = torch.full((self.N,), 8.0, dtype=torch.float32)
        self.assertTrue(torch.equal(self.layer.zeros.data[1], expected_g1))


# =============================================================================
# AWQAscendLinearKernel —process_weights_after_loading (fallback path)
# =============================================================================
class TestLinearProcessFallback(unittest.TestCase):
    """group_size not a multiple of 32 -> dequantize + FP16 linear fallback."""

    def setUp(self):
        # K=64, N=8, num_groups=4 -> group_size=16 (not mult of 32) -> fallback
        self.K, self.N, self.num_groups = 64, 8, 4
        self.scales = torch.full((self.num_groups, self.N), 2.0, dtype=torch.float32)
        self.qweight = torch.zeros(self.K, self.N // PACK_FACTOR, dtype=torch.int32)
        # qzeros all nibbles == 1 (0x11111111) -> zeros_u8 == 1
        self.qzeros = torch.full(
            (self.num_groups, self.N // PACK_FACTOR), 0x11111111, dtype=torch.int32
        )
        self.kernel = AWQAscendLinearKernel(_make_quant_config())
        self.layer = _make_linear_layer(self.scales, self.qweight, self.qzeros)
        self.kernel.process_weights_after_loading(self.layer)

    def test_use_npu_matmul_flag_false(self):
        self.assertFalse(self.layer.use_npu_matmul)

    def test_weight_is_parameter(self):
        self.assertIsInstance(self.layer.weight, torch.nn.Parameter)

    def test_weight_dtype_bfloat16(self):
        self.assertEqual(self.layer.weight.dtype, torch.bfloat16)

    def test_weight_shape_transposed(self):
        # dequantized (K, N) then .t() -> (N, K)
        self.assertEqual(self.layer.weight.shape, (self.N, self.K))

    def test_weight_values_dequantized(self):
        # weight_u8=0, zeros=1, scales=2 -> (0 - 1) * 2 = -2
        self.assertTrue(
            torch.allclose(
                self.layer.weight.data.float(),
                torch.full((self.N, self.K), -2.0),
                atol=1e-2,
            )
        )

    def test_scales_removed(self):
        self.assertFalse(hasattr(self.layer, "scales"))

    def test_qweight_removed(self):
        self.assertFalse(hasattr(self.layer, "qweight"))

    def test_qzeros_removed(self):
        self.assertFalse(hasattr(self.layer, "qzeros"))


# =============================================================================
# AWQAscendLinearKernel —process_weights_after_loading (per-tensor)
# =============================================================================
class TestLinearProcessPerTensorFallback(unittest.TestCase):
    """num_groups=1 -> group_size==K -> falls back (group_size < K is False)."""

    def test_per_tensor_uses_fallback(self):
        K, N = 64, 8
        scales = torch.ones(1, N, dtype=torch.float32)
        qweight = torch.zeros(K, N // PACK_FACTOR, dtype=torch.int32)
        qzeros = torch.zeros(1, N // PACK_FACTOR, dtype=torch.int32)
        kernel = AWQAscendLinearKernel(_make_quant_config())
        layer = _make_linear_layer(scales, qweight, qzeros)
        kernel.process_weights_after_loading(layer)
        self.assertFalse(layer.use_npu_matmul)
        # dequantized weight is (N, K)
        self.assertEqual(layer.weight.shape, (N, K))


# =============================================================================
# AWQAscendLinearKernel —process_weights_after_loading (errors)
# =============================================================================
class TestLinearProcessErrors(unittest.TestCase):
    def test_k_not_divisible_by_groups_raises(self):
        # K=64, num_groups=3 -> 64 % 3 != 0
        scales = torch.ones(3, 8, dtype=torch.float32)
        qweight = torch.zeros(64, 1, dtype=torch.int32)
        qzeros = torch.zeros(3, 1, dtype=torch.int32)
        kernel = AWQAscendLinearKernel(_make_quant_config())
        layer = _make_linear_layer(scales, qweight, qzeros)
        with self.assertRaises(RuntimeError):
            kernel.process_weights_after_loading(layer)


# =============================================================================
# AWQAscendLinearKernel —apply (NPU matmul path)
# =============================================================================
class TestLinearApplyNpuMatmul(unittest.TestCase):
    def setUp(self):
        self.K, self.N = 64, 8
        self.group_size = 32
        self.kernel = AWQAscendLinearKernel(_make_quant_config())
        self.layer = torch.nn.Module()
        self.layer.use_npu_matmul = True
        self.layer.npu_group_size = self.group_size
        self.layer.weight = torch.nn.Parameter(
            torch.zeros(self.K, self.N // PACK_FACTOR, dtype=torch.int32),
            requires_grad=False,
        )
        self.layer.scales = torch.nn.Parameter(
            torch.ones(2, self.N, dtype=torch.float32), requires_grad=False
        )
        self.layer.zeros = torch.nn.Parameter(
            torch.full((2, self.N), 8.0, dtype=torch.float32), requires_grad=False
        )
        self.mock_mm = awq_kernels_mod.torch_npu.npu_weight_quant_batchmatmul
        self.mock_mm.reset_mock()
        self.mock_mm.side_effect = None
        self.mock_mm.return_value = None

    def _set_return_tensor(self, rows):
        N = self.N

        def _fake(x, qweight, **kwargs):
            return torch.randn(rows, N)

        self.mock_mm.side_effect = _fake

    def test_calls_npu_weight_quant_batchmatmul(self):
        x = torch.randn(4, self.K)
        self._set_return_tensor(4)
        self.kernel.apply(self.layer, x)
        self.mock_mm.assert_called_once()

    def test_input_is_reshaped_to_2d(self):
        # 3-D input (B, S, K) -> reshaped_x (B*S, K)
        x = torch.randn(2, 3, self.K)
        self._set_return_tensor(6)
        self.kernel.apply(self.layer, x)
        reshaped = self.mock_mm.call_args.args[0]
        self.assertEqual(reshaped.shape, (6, self.K))

    def test_output_shape_2d(self):
        x = torch.randn(4, self.K)
        self._set_return_tensor(4)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (4, self.N))

    def test_output_shape_3d(self):
        x = torch.randn(2, 3, self.K)
        self._set_return_tensor(6)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (2, 3, self.N))

    def test_passes_scales_offset_and_group_size(self):
        x = torch.randn(2, self.K)
        self._set_return_tensor(2)
        self.kernel.apply(self.layer, x)
        kwargs = self.mock_mm.call_args.kwargs
        self.assertIs(kwargs["antiquant_scale"], self.layer.scales)
        self.assertIs(kwargs["antiquant_offset"], self.layer.zeros)
        self.assertEqual(kwargs["antiquant_group_size"], self.group_size)

    def test_passes_qweight_as_second_arg(self):
        x = torch.randn(2, self.K)
        self._set_return_tensor(2)
        self.kernel.apply(self.layer, x)
        self.assertIs(self.mock_mm.call_args.args[1], self.layer.weight)

    def test_bias_none(self):
        x = torch.randn(2, self.K)
        self._set_return_tensor(2)
        self.kernel.apply(self.layer, x, bias=None)
        self.assertIsNone(self.mock_mm.call_args.kwargs["bias"])

    def test_bias_bfloat16_converted_to_float(self):
        x = torch.randn(2, self.K)
        self._set_return_tensor(2)
        bias = torch.randn(self.N, dtype=torch.bfloat16)
        self.kernel.apply(self.layer, x, bias=bias)
        passed_bias = self.mock_mm.call_args.kwargs["bias"]
        self.assertEqual(passed_bias.dtype, torch.float32)

    def test_bias_float32_not_converted(self):
        x = torch.randn(2, self.K)
        self._set_return_tensor(2)
        bias = torch.randn(self.N, dtype=torch.float32)
        self.kernel.apply(self.layer, x, bias=bias)
        passed_bias = self.mock_mm.call_args.kwargs["bias"]
        self.assertEqual(passed_bias.dtype, torch.float32)

    def test_returns_reshaped_tensor(self):
        x = torch.randn(2, self.K)
        self._set_return_tensor(2)
        out = self.kernel.apply(self.layer, x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2, self.N))


# =============================================================================
# AWQAscendLinearKernel —apply (fallback linear path)
# =============================================================================
class TestLinearApplyFallback(unittest.TestCase):
    def setUp(self):
        self.K, self.N = 16, 8
        self.kernel = AWQAscendLinearKernel(_make_quant_config())
        self.layer = torch.nn.Module()
        self.layer.use_npu_matmul = False
        # F.linear expects weight of shape (out, in) = (N, K)
        self.layer.weight = torch.nn.Parameter(torch.randn(self.N, self.K))

    def test_uses_linear(self):
        x = torch.randn(3, self.K)
        out = self.kernel.apply(self.layer, x)
        expected = F.linear(x, self.layer.weight)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_with_bias(self):
        x = torch.randn(3, self.K)
        bias = torch.randn(self.N)
        out = self.kernel.apply(self.layer, x, bias=bias)
        expected = F.linear(x, self.layer.weight, bias)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_output_shape(self):
        x = torch.randn(3, self.K)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (3, self.N))


# =============================================================================
# AWQAscendMoEKernel —__init__
# =============================================================================
class TestMoeKernelInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = _make_quant_config()
        kernel = AWQAscendMoEKernel(cfg)
        self.assertIs(kernel.quant_config, cfg)

    def test_default_quant_config_none(self):
        kernel = AWQAscendMoEKernel()
        self.assertIsNone(kernel.quant_config)

    def test_creates_w13_kernel(self):
        kernel = AWQAscendMoEKernel(_make_quant_config())
        self.assertIsInstance(kernel.w13_kernel, _StubNPUWNA16Int4MoEMethod)

    def test_creates_w2_kernel(self):
        kernel = AWQAscendMoEKernel(_make_quant_config())
        self.assertIsInstance(kernel.w2_kernel, _StubNPUWNA16Int4MoEMethod)

    def test_w13_and_w2_are_distinct(self):
        kernel = AWQAscendMoEKernel(_make_quant_config())
        self.assertIsNot(kernel.w13_kernel, kernel.w2_kernel)


# =============================================================================
# AWQAscendMoEKernel —_register_or_replace_parameter (static)
# =============================================================================
class TestMoeRegisterOrReplaceParameter(unittest.TestCase):
    def test_registers_new_parameter(self):
        layer = torch.nn.Module()
        tensor = torch.randn(4, 5)
        AWQAscendMoEKernel._register_or_replace_parameter(layer, "p", tensor)
        self.assertIsInstance(layer.p, torch.nn.Parameter)
        self.assertTrue(torch.equal(layer.p.data, tensor))

    def test_replaces_existing_parameter(self):
        layer = torch.nn.Module()
        old = torch.nn.Parameter(torch.ones(3, 2), requires_grad=False)
        layer.p = old
        new = torch.zeros(3, 2)
        AWQAscendMoEKernel._register_or_replace_parameter(layer, "p", new)
        self.assertIsInstance(layer.p, torch.nn.Parameter)
        self.assertTrue(torch.equal(layer.p.data, new))

    def test_replaces_with_parameter_value(self):
        layer = torch.nn.Module()
        layer.p = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        new = torch.nn.Parameter(torch.zeros(3))
        AWQAscendMoEKernel._register_or_replace_parameter(layer, "p", new)
        self.assertTrue(torch.equal(layer.p.data, new.data))


# =============================================================================
# AWQAscendMoEKernel —process_weights_after_loading
# =============================================================================
class TestMoeProcessWeights(unittest.TestCase):
    def setUp(self):
        # E=2, K=64, N=8, G=2 for w13; E=2, K2=32, N2=8, G2=1 for w2
        self.E, self.K, self.N, self.G = 2, 64, 8, 2
        self.K2, self.G2 = 32, 1
        self.pack = PACK_FACTOR

        self.w13_scales = torch.ones(self.E, self.G, self.N, dtype=torch.float32)
        self.w2_scales = torch.ones(self.E, self.G2, self.N, dtype=torch.float32)

        # qweight all-zero -> after XOR, 0x88888888 everywhere
        self.w13_qweight = torch.zeros(
            self.E, self.K, self.N // self.pack, dtype=torch.int32
        )
        self.w2_qweight = torch.zeros(
            self.E, self.K2, self.N // self.pack, dtype=torch.int32
        )

        # qzeros: w13 = 0x76543210, w2 = 0x00000000
        self.w13_qzeros = torch.full(
            (self.E, self.G, self.N // self.pack), 0x76543210, dtype=torch.int32
        )
        self.w2_qzeros = torch.full(
            (self.E, self.G2, self.N // self.pack), 0x00000000, dtype=torch.int32
        )

        self.kernel = AWQAscendMoEKernel(_make_quant_config())
        self.layer = _make_moe_layer(
            self.w13_qweight,
            self.w13_qzeros,
            self.w13_scales,
            self.w2_qweight,
            self.w2_qzeros,
            self.w2_scales,
        )
        self.kernel.process_weights_after_loading(self.layer)

    def test_w13_qweight_is_parameter(self):
        self.assertIsInstance(self.layer.w13_qweight, torch.nn.Parameter)

    def test_w2_qweight_is_parameter(self):
        self.assertIsInstance(self.layer.w2_qweight, torch.nn.Parameter)

    def test_w13_qzeros_is_parameter(self):
        self.assertIsInstance(self.layer.w13_qzeros, torch.nn.Parameter)

    def test_w2_qzeros_is_parameter(self):
        self.assertIsInstance(self.layer.w2_qzeros, torch.nn.Parameter)

    def test_w13_qweight_shape(self):
        self.assertEqual(
            self.layer.w13_qweight.shape, (self.E, self.K, self.N // self.pack)
        )

    def test_w2_qweight_shape(self):
        self.assertEqual(
            self.layer.w2_qweight.shape, (self.E, self.K2, self.N // self.pack)
        )

    def test_w13_qzeros_shape(self):
        self.assertEqual(self.layer.w13_qzeros.shape, (self.E, self.G, self.N))

    def test_w2_qzeros_shape(self):
        self.assertEqual(self.layer.w2_qzeros.shape, (self.E, self.G2, self.N))

    def test_w13_qweight_xor_applied(self):
        expected = torch.tensor(0x88888888, dtype=torch.uint32)
        weight_u32 = self.layer.w13_qweight.data.view(torch.uint32)
        self.assertTrue(torch.equal(weight_u32, expected.expand_as(weight_u32)))

    def test_w2_qweight_xor_applied(self):
        expected = torch.tensor(0x88888888, dtype=torch.uint32)
        weight_u32 = self.layer.w2_qweight.data.view(torch.uint32)
        self.assertTrue(torch.equal(weight_u32, expected.expand_as(weight_u32)))

    def test_w13_qzeros_conversion(self):
        # 0x76543210 -> nibbles [0,4,1,5,2,6,3,7] -> -(z-8) -> [8,4,7,3,6,2,5,1]
        expected = torch.tensor([8, 4, 7, 3, 6, 2, 5, 1], dtype=torch.float32)
        for e in range(self.E):
            for g in range(self.G):
                self.assertTrue(
                    torch.equal(self.layer.w13_qzeros.data[e, g], expected),
                    f"mismatch at e={e}, g={g}",
                )

    def test_w2_qzeros_conversion(self):
        # 0x00000000 -> all nibbles 0 -> -(0-8) = 8
        expected = torch.full((self.N,), 8.0, dtype=torch.float32)
        self.assertTrue(
            torch.equal(
                self.layer.w2_qzeros.data, expected.expand_as(self.layer.w2_qzeros.data)
            )
        )

    def test_w13_qzeros_dtype_matches_scales(self):
        self.assertEqual(self.layer.w13_qzeros.dtype, self.w13_scales.dtype)

    def test_w2_qzeros_dtype_matches_scales(self):
        self.assertEqual(self.layer.w2_qzeros.dtype, self.w2_scales.dtype)

    def test_w13_scales_unchanged(self):
        self.assertTrue(torch.equal(self.layer.w13_scales.data, self.w13_scales))

    def test_w2_scales_unchanged(self):
        self.assertTrue(torch.equal(self.layer.w2_scales.data, self.w2_scales))


if __name__ == "__main__":
    unittest.main()

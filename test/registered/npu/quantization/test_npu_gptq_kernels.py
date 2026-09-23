"""
Unit tests for sglang.srt.hardware_backend.npu.quantization.gptq_kernels.

The gptq_kernels module reaches NPU only through ``torch.ops.npu.*`` ops
(``npu_convert_weight_to_int4pack`` / ``npu_weight_quant_batchmatmul``),
called at runtime rather than import time. To keep these unit tests runnable
on any machine, ``torch.ops.npu`` is stubbed and the real ``gptq_kernels.py``
source is loaded directly by path with importlib. The CI marker
``register_npu_ci`` is still emitted (as a no-op) so the AST-based CI register
picks this suite up.
"""

import importlib.util
import os
import sys
import types
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch

# ---------------------------------------------------------------------------
# Stub heavy sglang infrastructure so the test file's own
# ``from sglang.test.ci.ci_register import register_npu_ci`` does not trigger
# sglang/__init__.py (which pulls in triton/IPython/...).
# ---------------------------------------------------------------------------
_NP = types.ModuleType("torch_npu")
_NP.npu_weight_quant_batchmatmul = MagicMock()
sys.modules.setdefault("torch_npu", _NP)


def _ensure_pkg(dotted: str) -> ModuleType:
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
    _ensure_pkg(".".join(dotted.split(".")[:-1]))
    sys.modules[dotted] = mod
    setattr(sys.modules[".".join(dotted.split(".")[:-1])], dotted.split(".")[-1], mod)


for _pkg in ("sglang", "sglang.test", "sglang.test.ci"):
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

# ---------------------------------------------------------------------------
# Stub torch.ops.npu with a shared MagicMock so co-run test files (e.g. the
# linear_method_npu test) don't clobber each other's ops. Each op is a cached
# child mock of the same object; npu_convert_weight_to_int4pack is a real
# passthrough lambda (returns the int32 tensor so downstream .reshape/Parameter
# work —the real packing is the NPU op's job and is not replicated here).
# ---------------------------------------------------------------------------
if "_test_npu_ops_stub" not in sys.modules:
    sys.modules["_test_npu_ops_stub"] = MagicMock()
_NPU = sys.modules["_test_npu_ops_stub"]
torch.ops.npu = _NPU
_NPU.npu_convert_weight_to_int4pack = lambda t, *a, **k: t
_MATMUL_MOCK = _NPU.npu_weight_quant_batchmatmul

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Load the real gptq_kernels.py source file by path.
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
    "gptq_kernels.py",
)

_spec = importlib.util.spec_from_file_location("gptq_kernels_under_test", _SRC_PATH)
gptq_kernels_mod = importlib.util.module_from_spec(_spec)
sys.modules["gptq_kernels_under_test"] = gptq_kernels_mod
_spec.loader.exec_module(gptq_kernels_mod)

unpack_from_int32 = gptq_kernels_mod.unpack_from_int32
GPTQLinearAscendKernel = gptq_kernels_mod.GPTQLinearAscendKernel
GPTQMoEAscendKernel = gptq_kernels_mod.GPTQMoEAscendKernel

# sglang monkeypatches logging.Logger with `warning_once` at runtime; the
# standalone-loaded module's logger lacks it, so emulate it (no-op) so the
# "skip neg-scale correction" code path does not crash.
_gptq_logger = gptq_kernels_mod.logger
if not hasattr(_gptq_logger, "warning_once"):
    _gptq_logger.warning_once = lambda msg, *args, **kwargs: None


def _make_gptq_config(weight_bits=4, checkpoint_format="gptq", group_size=32):
    return SimpleNamespace(
        weight_bits=weight_bits,
        checkpoint_format=checkpoint_format,
        group_size=group_size,
    )


def _make_linear_layer(scales, qweight, qzeros):
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
# unpack_from_int32 —pure function (no NPU), fully tested with real values
# =============================================================================
class TestUnpackFromInt32(unittest.TestCase):
    def test_4bit_dim1_values(self):
        # 0x76543210 -> nibbles [0,1,2,3,4,5,6,7] -> minus offset 8 -> [-8..-1]
        w = torch.tensor([[0x76543210]], dtype=torch.int32)
        out = unpack_from_int32(w, 4, packed_dim=1)
        self.assertEqual(out.dtype, torch.int8)
        self.assertEqual(out.shape, (1, 8))
        self.assertEqual(out[0].tolist(), [-8, -7, -6, -5, -4, -3, -2, -1])

    def test_4bit_dim0_values(self):
        w = torch.tensor([[0x76543210]], dtype=torch.int32)
        out = unpack_from_int32(w, 4, packed_dim=0)
        self.assertEqual(out.shape, (8, 1))
        self.assertEqual(out.reshape(-1).tolist(), [-8, -7, -6, -5, -4, -3, -2, -1])

    def test_8bit_dim1_values(self):
        # 0x04030201 -> bytes [1,2,3,4] -> minus offset 128 -> [-127..-124]
        w = torch.tensor([[0x04030201]], dtype=torch.int32)
        out = unpack_from_int32(w, 8, packed_dim=1)
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(out[0].tolist(), [-127, -126, -125, -124])

    def test_shape_dim1_expands_last(self):
        # (rows, cols) -> (rows, cols * pack_factor)
        w = torch.zeros(3, 2, dtype=torch.int32)
        out = unpack_from_int32(w, 4, packed_dim=1)
        self.assertEqual(out.shape, (3, 16))

    def test_shape_dim0_expands_first(self):
        # (rows, cols) -> (rows * pack_factor, cols)
        w = torch.zeros(3, 2, dtype=torch.int32)
        out = unpack_from_int32(w, 4, packed_dim=0)
        self.assertEqual(out.shape, (24, 2))

    def test_offset_4bit_all_zero(self):
        # all-zero packed -> nibble 0 -> 0 - 8 = -8
        w = torch.zeros(1, 2, dtype=torch.int32)
        out = unpack_from_int32(w, 4, packed_dim=1)
        self.assertTrue(torch.equal(out, torch.full((1, 16), -8, dtype=torch.int8)))

    def test_offset_8bit_all_zero(self):
        # all-zero packed -> byte 0 -> 0 - 128 = -128
        w = torch.zeros(1, 1, dtype=torch.int32)
        out = unpack_from_int32(w, 8, packed_dim=1)
        self.assertTrue(torch.equal(out, torch.full((1, 4), -128, dtype=torch.int8)))

    def test_default_packed_dim_is_1(self):
        w = torch.tensor([[0x76543210]], dtype=torch.int32)
        out_default = unpack_from_int32(w, 4)
        out_dim1 = unpack_from_int32(w, 4, packed_dim=1)
        self.assertTrue(torch.equal(out_default, out_dim1))

    def test_2bit_values(self):
        # num_bits=2 -> pack_factor=16, offset=2; 0b100110 = 0b01 10 01 -> wait
        # build a value: nibble... actually 2-bit groups. value 0b11_10_01_00
        # groups (LSB first): 00,01,10,11 -> 0,1,2,3 -> minus 2 -> -2,-1,0,1
        val = 0b11_10_01_00  # 0xE4
        w = torch.tensor([[val]], dtype=torch.int32)
        out = unpack_from_int32(w, 2, packed_dim=1)
        # groups i=0..15: (val >> 2*i) & 0b11
        self.assertEqual(out.shape, (1, 16))
        expected = torch.tensor(
            [((val >> (2 * i)) & 0x3) - 2 for i in range(16)], dtype=torch.int8
        )
        self.assertTrue(torch.equal(out[0], expected))

    def test_assert_non_int32_raises(self):
        w = torch.zeros(2, 2, dtype=torch.int8)
        with self.assertRaises(AssertionError):
            unpack_from_int32(w, 4)

    def test_assert_num_bits_gt_8_raises(self):
        w = torch.zeros(1, 1, dtype=torch.int32)
        with self.assertRaises(AssertionError):
            unpack_from_int32(w, 16)


# =============================================================================
# GPTQLinearAscendKernel —__init__
# =============================================================================
class TestLinearKernelInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = _make_gptq_config()
        kernel = GPTQLinearAscendKernel(cfg)
        self.assertIs(kernel.quant_config, cfg)

    def test_use_v2_format_false_for_gptq(self):
        kernel = GPTQLinearAscendKernel(_make_gptq_config(checkpoint_format="gptq"))
        self.assertFalse(kernel.use_v2_format)

    def test_use_v2_format_true_for_gptq_v2(self):
        kernel = GPTQLinearAscendKernel(_make_gptq_config(checkpoint_format="gptq_v2"))
        self.assertTrue(kernel.use_v2_format)


# =============================================================================
# GPTQLinearAscendKernel —process_weights_after_loading (4-bit, v2)
# =============================================================================
class TestLinearProcess4BitV2(unittest.TestCase):
    """v2 format -> qzeros NOT incremented by 1."""

    def setUp(self):
        self.K, self.N, self.groups = 64, 8, 2
        self.scales = torch.ones(self.groups, self.N, dtype=torch.float32)
        # all-zero packed -> unpacked = -8
        self.qzeros = torch.zeros(self.groups, self.N // 8, dtype=torch.int32)
        self.qweight = torch.zeros(self.K // 8, self.N, dtype=torch.int32)
        self.kernel = GPTQLinearAscendKernel(
            _make_gptq_config(weight_bits=4, checkpoint_format="gptq_v2")
        )
        self.layer = _make_linear_layer(self.scales, self.qweight, self.qzeros)
        self.kernel.process_weights_after_loading(self.layer)

    def test_qzeros_is_parameter(self):
        self.assertIsInstance(self.layer.qzeros, torch.nn.Parameter)

    def test_qzeros_shape(self):
        # packed_dim=1, pack_factor=8 -> (groups, N//8) -> (groups, N)
        self.assertEqual(self.layer.qzeros.shape, (self.groups, self.N))

    def test_qzeros_dtype_matches_scales(self):
        self.assertEqual(self.layer.qzeros.dtype, torch.float32)

    def test_qzeros_no_increment(self):
        # v2: unpacked = -8, no +1 -> stays -8
        self.assertTrue(
            torch.equal(self.layer.qzeros.data, torch.full((self.groups, self.N), -8.0))
        )

    def test_qweight_is_parameter(self):
        self.assertIsInstance(self.layer.qweight, torch.nn.Parameter)

    def test_qweight_uses_int4pack_op(self):
        # qweight for 4-bit goes through npu_convert_weight_to_int4pack (passthrough)
        # unpacked qweight (K, N) int8 -> to(int32) -> passthrough
        self.assertEqual(self.layer.qweight.dtype, torch.int32)


# =============================================================================
# GPTQLinearAscendKernel —process_weights_after_loading (4-bit, non-v2)
# =============================================================================
class TestLinearProcess4BitNonV2(unittest.TestCase):
    """non-v2 format -> qzeros += 1 after unpacking."""

    def setUp(self):
        self.K, self.N, self.groups = 64, 8, 2
        self.scales = torch.ones(self.groups, self.N, dtype=torch.float32)
        self.qzeros = torch.zeros(self.groups, self.N // 8, dtype=torch.int32)
        self.qweight = torch.zeros(self.K // 8, self.N, dtype=torch.int32)
        self.kernel = GPTQLinearAscendKernel(
            _make_gptq_config(weight_bits=4, checkpoint_format="gptq")
        )
        self.layer = _make_linear_layer(self.scales, self.qweight, self.qzeros)
        self.kernel.process_weights_after_loading(self.layer)

    def test_qzeros_incremented(self):
        # unpacked = -8, +1 -> -7
        self.assertTrue(
            torch.equal(self.layer.qzeros.data, torch.full((self.groups, self.N), -7.0))
        )

    def test_qzeros_shape(self):
        self.assertEqual(self.layer.qzeros.shape, (self.groups, self.N))


# =============================================================================
# GPTQLinearAscendKernel —process_weights_after_loading (8-bit)
# =============================================================================
class TestLinearProcess8Bit(unittest.TestCase):
    """8-bit -> no npu_convert_weight_to_int4pack, qweight stored as int8."""

    def setUp(self):
        self.K, self.N, self.groups = 32, 8, 2
        self.scales = torch.ones(self.groups, self.N, dtype=torch.float32)
        # all-zero packed -> 8-bit unpacked = -128
        self.qzeros = torch.zeros(self.groups, self.N // 4, dtype=torch.int32)
        self.qweight = torch.zeros(self.K // 4, self.N, dtype=torch.int32)
        self.kernel = GPTQLinearAscendKernel(
            _make_gptq_config(weight_bits=8, checkpoint_format="gptq")
        )
        self.layer = _make_linear_layer(self.scales, self.qweight, self.qzeros)
        self.kernel.process_weights_after_loading(self.layer)

    def test_qzeros_shape(self):
        # 8-bit pack_factor=4 -> (groups, N//4) -> (groups, N)
        self.assertEqual(self.layer.qzeros.shape, (self.groups, self.N))

    def test_qzeros_incremented(self):
        # unpacked = -128, +1 -> -127
        self.assertTrue(
            torch.equal(
                self.layer.qzeros.data, torch.full((self.groups, self.N), -127.0)
            )
        )

    def test_qweight_is_int8(self):
        # 8-bit stores qweight_tmp directly as int8 (no npu op)
        self.assertEqual(self.layer.qweight.dtype, torch.int8)

    def test_qweight_shape(self):
        # packed_dim=0, pack_factor=4 -> (K//4, N) -> (K, N)
        self.assertEqual(self.layer.qweight.shape, (self.K, self.N))

    def test_qweight_values(self):
        # all-zero packed -> all -128
        self.assertTrue(
            torch.equal(
                self.layer.qweight.data,
                torch.full((self.K, self.N), -128, dtype=torch.int8),
            )
        )


# =============================================================================
# GPTQLinearAscendKernel —apply (4-bit, mocked matmul)
# =============================================================================
class TestLinearApply4Bit(unittest.TestCase):
    def setUp(self):
        self.K, self.N = 64, 8
        self.kernel = GPTQLinearAscendKernel(
            _make_gptq_config(weight_bits=4, group_size=32)
        )
        self.layer = torch.nn.Module()
        # 4-bit apply: out last dim = qweight.shape[-1] * 8, so qweight packed
        # shape (K, N//8) -> out last = N
        self.layer.qweight = torch.nn.Parameter(
            torch.zeros(self.K, self.N // 8, dtype=torch.int32), requires_grad=False
        )
        self.layer.scales = torch.nn.Parameter(
            torch.ones(2, self.N, dtype=torch.float32), requires_grad=False
        )
        self.layer.qzeros = torch.nn.Parameter(
            torch.zeros(2, self.N, dtype=torch.float32), requires_grad=False
        )
        _MATMUL_MOCK.reset_mock()
        _MATMUL_MOCK.side_effect = None
        _MATMUL_MOCK.return_value = None

    def _set_return(self, rows):
        N = self.N

        def _fake(x, qweight, **kwargs):
            return torch.randn(rows, N)

        _MATMUL_MOCK.side_effect = _fake

    def test_calls_matmul(self):
        x = torch.randn(4, self.K)
        self._set_return(4)
        self.kernel.apply(self.layer, x)
        _MATMUL_MOCK.assert_called_once()

    def test_input_reshaped_to_2d(self):
        x = torch.randn(2, 3, self.K)
        self._set_return(6)
        self.kernel.apply(self.layer, x)
        self.assertEqual(_MATMUL_MOCK.call_args.args[0].shape, (6, self.K))

    def test_output_shape_2d(self):
        x = torch.randn(4, self.K)
        self._set_return(4)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (4, self.N))

    def test_output_shape_3d(self):
        x = torch.randn(2, 3, self.K)
        self._set_return(6)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (2, 3, self.N))

    def test_passes_antiquant_params(self):
        x = torch.randn(2, self.K)
        self._set_return(2)
        self.kernel.apply(self.layer, x)
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertIs(kw["antiquant_scale"], self.layer.scales)
        self.assertIs(kw["antiquant_offset"], self.layer.qzeros)
        self.assertEqual(kw["antiquant_group_size"], 32)

    def test_passes_qweight_as_second_arg(self):
        x = torch.randn(2, self.K)
        self._set_return(2)
        self.kernel.apply(self.layer, x)
        self.assertIs(_MATMUL_MOCK.call_args.args[1], self.layer.qweight)

    def test_bias_none(self):
        x = torch.randn(2, self.K)
        self._set_return(2)
        self.kernel.apply(self.layer, x, bias=None)
        self.assertIsNone(_MATMUL_MOCK.call_args.kwargs["bias"])

    def test_bias_bfloat16_converted_to_float(self):
        x = torch.randn(2, self.K)
        self._set_return(2)
        bias = torch.randn(self.N, dtype=torch.bfloat16)
        self.kernel.apply(self.layer, x, bias=bias)
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32)

    def test_bias_float32_not_converted(self):
        x = torch.randn(2, self.K)
        self._set_return(2)
        bias = torch.randn(self.N, dtype=torch.float32)
        self.kernel.apply(self.layer, x, bias=bias)
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32)


# =============================================================================
# GPTQLinearAscendKernel —apply (8-bit)
# =============================================================================
class TestLinearApply8Bit(unittest.TestCase):
    def setUp(self):
        self.K, self.N = 32, 8
        self.kernel = GPTQLinearAscendKernel(
            _make_gptq_config(weight_bits=8, group_size=16)
        )
        self.layer = torch.nn.Module()
        # 8-bit: out last dim = qweight.shape[-1] (no *8)
        self.layer.qweight = torch.nn.Parameter(
            torch.zeros(self.K, self.N, dtype=torch.int8), requires_grad=False
        )
        self.layer.scales = torch.nn.Parameter(
            torch.ones(2, self.N, dtype=torch.float32), requires_grad=False
        )
        self.layer.qzeros = torch.nn.Parameter(
            torch.zeros(2, self.N, dtype=torch.float32), requires_grad=False
        )
        _MATMUL_MOCK.reset_mock()
        _MATMUL_MOCK.side_effect = None

    def test_output_shape_2d(self):
        N = self.N

        def _fake(x, qweight, **kwargs):
            return torch.randn(x.shape[0], N)

        _MATMUL_MOCK.side_effect = _fake
        x = torch.randn(4, self.K)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (4, self.N))

    def test_group_size_passed(self):
        N = self.N

        def _fake(x, qweight, **kwargs):
            return torch.randn(x.shape[0], N)

        _MATMUL_MOCK.side_effect = _fake
        self.kernel.apply(self.layer, torch.randn(2, self.K))
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["antiquant_group_size"], 16)


# =============================================================================
# GPTQMoEAscendKernel —__init__
# =============================================================================
class TestMoeKernelInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = _make_gptq_config()
        kernel = GPTQMoEAscendKernel(cfg)
        self.assertIs(kernel.quant_config, cfg)

    def test_use_v2_format_false_for_gptq(self):
        kernel = GPTQMoEAscendKernel(_make_gptq_config(checkpoint_format="gptq"))
        self.assertFalse(kernel.use_v2_format)

    def test_use_v2_format_true_for_gptq_v2(self):
        kernel = GPTQMoEAscendKernel(_make_gptq_config(checkpoint_format="gptq_v2"))
        self.assertTrue(kernel.use_v2_format)


# =============================================================================
# GPTQMoEAscendKernel —process_weights_after_loading (4-bit, v2, no neg scale)
# =============================================================================
class TestMoeProcess4BitV2(unittest.TestCase):
    def setUp(self):
        # E=2, K=8, N=8, G=2; w2 K2=8, G2=2; group_size=4 -> G*gs=8=K (compatible)
        self.E, self.K, self.N, self.G = 2, 8, 8, 2
        self.K2, self.G2 = 8, 2
        self.gs = 4

        self.w13_scales = torch.ones(self.E, self.G, self.N, dtype=torch.float32)
        self.w2_scales = torch.ones(self.E, self.G2, self.N, dtype=torch.float32)
        self.w13_qweight = torch.zeros(self.E, self.K // 8, self.N, dtype=torch.int32)
        self.w2_qweight = torch.zeros(self.E, self.K2 // 8, self.N, dtype=torch.int32)
        self.w13_qzeros = torch.zeros(self.E, self.G, self.N // 8, dtype=torch.int32)
        self.w2_qzeros = torch.zeros(self.E, self.G2, self.N // 8, dtype=torch.int32)

        self.kernel = GPTQMoEAscendKernel(
            _make_gptq_config(
                weight_bits=4, checkpoint_format="gptq_v2", group_size=self.gs
            )
        )
        self.layer = _make_moe_layer(
            self.w13_qweight,
            self.w13_qzeros,
            self.w13_scales,
            self.w2_qweight,
            self.w2_qzeros,
            self.w2_scales,
        )
        self.kernel.process_weights_after_loading(self.layer)

    def test_w13_qzeros_is_parameter(self):
        self.assertIsInstance(self.layer.w13_qzeros, torch.nn.Parameter)

    def test_w2_qzeros_is_parameter(self):
        self.assertIsInstance(self.layer.w2_qzeros, torch.nn.Parameter)

    def test_w13_qzeros_shape(self):
        # (E, G, N//8) unpacked packed_dim=1 -> reshape(E, G, N)
        self.assertEqual(self.layer.w13_qzeros.shape, (self.E, self.G, self.N))

    def test_w2_qzeros_shape(self):
        self.assertEqual(self.layer.w2_qzeros.shape, (self.E, self.G2, self.N))

    def test_w13_qzeros_no_increment(self):
        # v2: unpacked = -8, no +1
        self.assertTrue(
            torch.equal(
                self.layer.w13_qzeros.data, torch.full((self.E, self.G, self.N), -8.0)
            )
        )

    def test_w2_qzeros_no_increment(self):
        self.assertTrue(
            torch.equal(
                self.layer.w2_qzeros.data, torch.full((self.E, self.G2, self.N), -8.0)
            )
        )

    def test_w13_qweight_is_parameter(self):
        self.assertIsInstance(self.layer.w13_qweight, torch.nn.Parameter)

    def test_w2_qweight_is_parameter(self):
        self.assertIsInstance(self.layer.w2_qweight, torch.nn.Parameter)

    def test_w13_qweight_shape(self):
        # (E, K//8*8=K, N) = (E, K, N) after op passthrough + reshape
        self.assertEqual(self.layer.w13_qweight.shape, (self.E, self.K, self.N))

    def test_w2_qweight_shape(self):
        self.assertEqual(self.layer.w2_qweight.shape, (self.E, self.K2, self.N))

    def test_w13_scales_unchanged(self):
        self.assertTrue(torch.equal(self.layer.w13_scales.data, self.w13_scales))


# =============================================================================
# GPTQMoEAscendKernel —process_weights_after_loading (4-bit, non-v2)
# =============================================================================
class TestMoeProcess4BitNonV2(unittest.TestCase):
    def setUp(self):
        self.E, self.K, self.N, self.G = 2, 8, 8, 2
        self.K2, self.G2 = 8, 2
        self.gs = 4

        self.w13_scales = torch.ones(self.E, self.G, self.N, dtype=torch.float32)
        self.w2_scales = torch.ones(self.E, self.G2, self.N, dtype=torch.float32)
        self.w13_qweight = torch.zeros(self.E, self.K // 8, self.N, dtype=torch.int32)
        self.w2_qweight = torch.zeros(self.E, self.K2 // 8, self.N, dtype=torch.int32)
        self.w13_qzeros = torch.zeros(self.E, self.G, self.N // 8, dtype=torch.int32)
        self.w2_qzeros = torch.zeros(self.E, self.G2, self.N // 8, dtype=torch.int32)

        self.kernel = GPTQMoEAscendKernel(
            _make_gptq_config(
                weight_bits=4, checkpoint_format="gptq", group_size=self.gs
            )
        )
        self.layer = _make_moe_layer(
            self.w13_qweight,
            self.w13_qzeros,
            self.w13_scales,
            self.w2_qweight,
            self.w2_qzeros,
            self.w2_scales,
        )
        self.kernel.process_weights_after_loading(self.layer)

    def test_w13_qzeros_incremented(self):
        # non-v2: unpacked -8 + 1 = -7
        self.assertTrue(
            torch.equal(
                self.layer.w13_qzeros.data, torch.full((self.E, self.G, self.N), -7.0)
            )
        )

    def test_w2_qzeros_incremented(self):
        self.assertTrue(
            torch.equal(
                self.layer.w2_qzeros.data, torch.full((self.E, self.G2, self.N), -7.0)
            )
        )


# =============================================================================
# GPTQMoEAscendKernel —process_weights_after_loading (negative-scale correction)
# =============================================================================
class TestMoeProcessNegScale(unittest.TestCase):
    """All-negative scales -> weights negated+clamped to 7, scales -> abs."""

    def setUp(self):
        self.E, self.K, self.N, self.G = 2, 8, 8, 2
        self.K2, self.G2 = 8, 2
        self.gs = 4  # G*gs = 2*4 = 8 = K (compatible -> correction branch)

        # all-negative scales
        self.w13_scales = torch.full(
            (self.E, self.G, self.N), -1.0, dtype=torch.float32
        )
        self.w2_scales = torch.full(
            (self.E, self.G2, self.N), -1.0, dtype=torch.float32
        )
        # all-zero packed qweight -> unpacked = -8 -> negated = 8 -> clamped 7
        self.w13_qweight = torch.zeros(self.E, self.K // 8, self.N, dtype=torch.int32)
        self.w2_qweight = torch.zeros(self.E, self.K2 // 8, self.N, dtype=torch.int32)
        self.w13_qzeros = torch.zeros(self.E, self.G, self.N // 8, dtype=torch.int32)
        self.w2_qzeros = torch.zeros(self.E, self.G2, self.N // 8, dtype=torch.int32)

        self.kernel = GPTQMoEAscendKernel(
            _make_gptq_config(
                weight_bits=4, checkpoint_format="gptq_v2", group_size=self.gs
            )
        )
        self.layer = _make_moe_layer(
            self.w13_qweight,
            self.w13_qzeros,
            self.w13_scales,
            self.w2_qweight,
            self.w2_qzeros,
            self.w2_scales,
        )
        self.kernel.process_weights_after_loading(self.layer)

    def test_w13_scales_made_absolute(self):
        self.assertTrue((self.layer.w13_scales.data >= 0).all())
        self.assertTrue(
            torch.equal(
                self.layer.w13_scales.data, torch.full((self.E, self.G, self.N), 1.0)
            )
        )

    def test_w2_scales_made_absolute(self):
        self.assertTrue(
            torch.equal(
                self.layer.w2_scales.data, torch.full((self.E, self.G2, self.N), 1.0)
            )
        )

    def test_w13_qweight_negated_and_clamped(self):
        # all-zero -> -8 -> negated 8 -> clamped 7; stored as int32 via passthrough op
        self.assertTrue(
            torch.equal(
                self.layer.w13_qweight.data,
                torch.full((self.E, self.K, self.N), 7, dtype=torch.int32),
            )
        )

    def test_w2_qweight_negated_and_clamped(self):
        self.assertTrue(
            torch.equal(
                self.layer.w2_qweight.data,
                torch.full((self.E, self.K2, self.N), 7, dtype=torch.int32),
            )
        )


# =============================================================================
# GPTQMoEAscendKernel —process_weights_after_loading (skip neg-scale, incompatible)
# =============================================================================
class TestMoeProcessSkipNegScale(unittest.TestCase):
    """Incompatible scales (G*gs != K) -> skip correction, pack directly.

    Negative scales are LEFT negative (no abs), proving the correction branch
    was skipped.
    """

    def test_skip_branch_leaves_scales_negative(self):
        E, K, N, G = 2, 8, 8, 2
        # group_size=3 -> G=K//3=2, G*gs=6 != 8=K -> skip branch
        w13_scales = torch.full((E, G, N), -1.0, dtype=torch.float32)
        w2_scales = torch.full((E, G, N), -1.0, dtype=torch.float32)
        w13_qweight = torch.zeros(E, K // 8, N, dtype=torch.int32)
        w2_qweight = torch.zeros(E, K // 8, N, dtype=torch.int32)
        w13_qzeros = torch.zeros(E, G, N // 8, dtype=torch.int32)
        w2_qzeros = torch.zeros(E, G, N // 8, dtype=torch.int32)

        kernel = GPTQMoEAscendKernel(
            _make_gptq_config(weight_bits=4, checkpoint_format="gptq_v2", group_size=3)
        )
        layer = _make_moe_layer(
            w13_qweight,
            w13_qzeros,
            w13_scales,
            w2_qweight,
            w2_qzeros,
            w2_scales,
        )
        kernel.process_weights_after_loading(layer)
        # scales NOT abs'd (still negative)
        self.assertTrue((layer.w13_scales.data < 0).all())
        self.assertTrue((layer.w2_scales.data < 0).all())
        # qweight still a Parameter
        self.assertIsInstance(layer.w13_qweight, torch.nn.Parameter)
        self.assertIsInstance(layer.w2_qweight, torch.nn.Parameter)


# =============================================================================
# GPTQMoEAscendKernel —process_weights_after_loading (8-bit)
# =============================================================================
class TestMoeProcess8Bit(unittest.TestCase):
    def setUp(self):
        # 8-bit: no npu_convert_weight_to_int4pack, qweight stored as int8 (transposed)
        self.E, self.K, self.N, self.G = 2, 8, 8, 2
        self.K2, self.G2 = 8, 2

        self.w13_scales = torch.ones(self.E, self.G, self.N, dtype=torch.float32)
        self.w2_scales = torch.ones(self.E, self.G2, self.N, dtype=torch.float32)
        # 8-bit pack_factor=4 -> qweight (E, K//4, N), qzeros (E, G, N//4)
        self.w13_qweight = torch.zeros(self.E, self.K // 4, self.N, dtype=torch.int32)
        self.w2_qweight = torch.zeros(self.E, self.K2 // 4, self.N, dtype=torch.int32)
        self.w13_qzeros = torch.zeros(self.E, self.G, self.N // 4, dtype=torch.int32)
        self.w2_qzeros = torch.zeros(self.E, self.G2, self.N // 4, dtype=torch.int32)

        self.kernel = GPTQMoEAscendKernel(
            _make_gptq_config(weight_bits=8, checkpoint_format="gptq", group_size=4)
        )
        self.layer = _make_moe_layer(
            self.w13_qweight,
            self.w13_qzeros,
            self.w13_scales,
            self.w2_qweight,
            self.w2_qzeros,
            self.w2_scales,
        )
        self.kernel.process_weights_after_loading(self.layer)

    def test_w13_qzeros_shape(self):
        # (E, G, N//4) -> unpack packed_dim=1 -> reshape(E, G, N)
        self.assertEqual(self.layer.w13_qzeros.shape, (self.E, self.G, self.N))

    def test_w2_qzeros_shape(self):
        self.assertEqual(self.layer.w2_qzeros.shape, (self.E, self.G2, self.N))

    def test_w13_qzeros_incremented(self):
        # 8-bit offset=128 -> unpacked -128, +1 -> -127
        self.assertTrue(
            torch.equal(
                self.layer.w13_qzeros.data, torch.full((self.E, self.G, self.N), -127.0)
            )
        )

    def test_w2_qzeros_incremented(self):
        # w2 same path: unpacked -128, +1 -> -127
        self.assertTrue(
            torch.equal(
                self.layer.w2_qzeros.data, torch.full((self.E, self.G2, self.N), -127.0)
            )
        )

    def test_w13_qweight_is_int8(self):
        self.assertEqual(self.layer.w13_qweight.dtype, torch.int8)

    def test_w2_qweight_is_int8(self):
        # w2 also takes the 8-bit else branch (no npu op, stays int8)
        self.assertEqual(self.layer.w2_qweight.dtype, torch.int8)

    def test_w13_qweight_shape(self):
        # (E, K//4, N) -> unpack packed_dim=1 -> (E*N, K) -> reshape(E, N, K) -> transpose -> (E, K, N)
        self.assertEqual(self.layer.w13_qweight.shape, (self.E, self.K, self.N))

    def test_w2_qweight_shape(self):
        self.assertEqual(self.layer.w2_qweight.shape, (self.E, self.K2, self.N))


if __name__ == "__main__":
    unittest.main()

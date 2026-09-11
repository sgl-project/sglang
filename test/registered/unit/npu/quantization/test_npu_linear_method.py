"""
Unit tests for sglang.srt.hardware_backend.npu.quantization.linear_method_npu.

The module reaches NPU only through ``torch.ops.npu.*`` ops and the
``npu_format_cast`` helper (called at runtime, not import time). To keep these
unit tests runnable on any machine, the heavy sglang infrastructure
(LinearMethodBase / npu utils / envs / is_npu / lazy layer imports) is stubbed
in ``sys.modules`` and the real ``linear_method_npu.py`` source is loaded
directly by path with importlib. The CI marker ``register_npu_ci`` is still
emitted (as a no-op) so the AST-based CI register picks this suite up.

Scope: the CPU-runnable subset —module-level dtype helpers, the base class
``_NPULinearMethodBase``, and the four linear methods whose
``process_weights_after_loading`` does not call ``.to("npu")`` /
``weight.is_npu`` (those code paths require real NPU hardware and are skipped
here, except their ``apply`` which is exercised via mocked NPU ops).
"""

import importlib.util
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.nn.parameter import Parameter

# ---------------------------------------------------------------------------
# Stub heavy sglang infrastructure in sys.modules BEFORE loading the source.
# ---------------------------------------------------------------------------
sys.modules.setdefault("torch_npu", MagicMock())


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

# LinearMethodBase: plain base class (no abstractmethods, so subclasses
# instantiate cleanly). QuantizationConfig is only a type hint at runtime.
_base_mod = ModuleType("sglang.srt.layers.quantization.base_config")


class LinearMethodBase:
    def __init__(self, *args, **kwargs):
        pass


_base_mod.LinearMethodBase = LinearMethodBase
_base_mod.QuantizationConfig = type("QuantizationConfig", (), {})
_install_stub("sglang.srt.layers.quantization.base_config", _base_mod)

# RowParallelLinear (lazy-imported in W8A8Int8.apply) —plain marker class so
# isinstance(layer, RowParallelLinear) is False for the test layer.
_linear_mod = ModuleType("sglang.srt.layers.linear")


class RowParallelLinear:
    pass


_linear_mod.RowParallelLinear = RowParallelLinear
_install_stub("sglang.srt.layers.linear", _linear_mod)

# ModelWeightParameter (lazy-imported in create_weights) —callable returning a
# Parameter so register_parameter works.
_param_mod = ModuleType("sglang.srt.layers.parameter")


def ModelWeightParameter(data=None, **kwargs):
    if data is None:
        data = torch.empty(0)
    return Parameter(data, requires_grad=False)


_param_mod.ModelWeightParameter = ModelWeightParameter
_install_stub("sglang.srt.layers.parameter", _param_mod)

# npu utils: NPUACLFormat marker + npu_format_cast passthrough.
_npu_utils = ModuleType("sglang.srt.hardware_backend.npu.utils")


class NPUACLFormat:
    ACL_FORMAT_FRACTAL_NZ = 29  # value the real enum uses; passthrough ignores it


def npu_format_cast(t, *args, **kwargs):
    return t  # passthrough —keeps the real tensor so downstream ops work


_npu_utils.NPUACLFormat = NPUACLFormat
_npu_utils.npu_format_cast = npu_format_cast
_install_stub("sglang.srt.hardware_backend.npu.utils", _npu_utils)

# envs: only SGLANG_NPU_W4A4_NEW_PACKING is read; make it toggleable.
_environ_mod = ModuleType("sglang.srt.environ")


class _EnvFlag:
    def __init__(self, v=False):
        self._v = v

    def get(self):
        return self._v


_envs = SimpleNamespace()
_envs.SGLANG_NPU_W4A4_NEW_PACKING = _EnvFlag(False)
_environ_mod.envs = _envs
_install_stub("sglang.srt.environ", _environ_mod)

# is_npu (lazy-imported in _get_float4_e2m1fn_x2_dtype) —always False on CPU.
_utils_mod = ModuleType("sglang.srt.utils")


def is_npu():
    return False


_utils_mod.is_npu = is_npu
_install_stub("sglang.srt.utils", _utils_mod)

# ---------------------------------------------------------------------------
# Stub torch.ops.npu with a shared MagicMock so co-run test files don't clobber
# each other's ops. npu_convert_weight_to_int4pack is a real passthrough lambda;
# the quantize/matmul ops are cached child mocks configured per-test.
# ---------------------------------------------------------------------------
if "_test_npu_ops_stub" not in sys.modules:
    sys.modules["_test_npu_ops_stub"] = MagicMock()
_NPU = sys.modules["_test_npu_ops_stub"]
torch.ops.npu = _NPU
_NPU.npu_convert_weight_to_int4pack = lambda t, *a, **k: t
_QUANTIZE_MOCK = _NPU.npu_quantize
_MATMUL_MOCK = _NPU.npu_quant_matmul
_DYN_QUANT_MOCK = _NPU.npu_dynamic_quant
_MX_QUANT_MOCK = _NPU.npu_dynamic_mx_quant
_DUAL_MX_QUANT_MOCK = _NPU.npu_dynamic_dual_level_mx_quant
_DUAL_MATMUL_MOCK = _NPU.npu_dual_level_quant_matmul

register_npu_ci(est_time=6, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Load the real linear_method_npu.py source file by path.
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
    "linear_method_npu.py",
)

_spec = importlib.util.spec_from_file_location(
    "linear_method_npu_under_test", _SRC_PATH
)
lmn_mod = importlib.util.module_from_spec(_spec)
sys.modules["linear_method_npu_under_test"] = lmn_mod
_spec.loader.exec_module(lmn_mod)

_get_float8_e8m0fnu_dtype = lmn_mod._get_float8_e8m0fnu_dtype
_get_float4_e2m1fn_x2_dtype = lmn_mod._get_float4_e2m1fn_x2_dtype
_NPULinearMethodBase = lmn_mod._NPULinearMethodBase
NPUW8A8Int8LinearMethod = lmn_mod.NPUW8A8Int8LinearMethod
NPUW8A8Int8DynamicLinearMethod = lmn_mod.NPUW8A8Int8DynamicLinearMethod
NPUMXFP8LinearMethod = lmn_mod.NPUMXFP8LinearMethod
NPU_W4A4DynamicLinearMethod = lmn_mod.NPU_W4A4DynamicLinearMethod
NPUMXFP4W4A8LinearMethod = lmn_mod.NPUMXFP4W4A8LinearMethod
NPUMXFP4W4A8OfflineLinearMethod = lmn_mod.NPUMXFP4W4A8OfflineLinearMethod
NPUSingleLevelMXFP4LinearMethod = lmn_mod.NPUSingleLevelMXFP4LinearMethod
NPUSingleLevelMXFP4OfflineLinearMethod = lmn_mod.NPUSingleLevelMXFP4OfflineLinearMethod
NPUDualLevelMXFP4LinearMethod = lmn_mod.NPUDualLevelMXFP4LinearMethod

MXFP8_BLOCK_SIZE = lmn_mod.MXFP8_BLOCK_SIZE  # 32
MXFP4_BLOCK_SIZE = lmn_mod.MXFP4_BLOCK_SIZE  # 32


def _reset_ops():
    for m in (
        _QUANTIZE_MOCK,
        _MATMUL_MOCK,
        _DYN_QUANT_MOCK,
        _MX_QUANT_MOCK,
        _DUAL_MX_QUANT_MOCK,
        _DUAL_MATMUL_MOCK,
    ):
        m.reset_mock()
        m.side_effect = None
        m.return_value = None


# NPUW8A8Int8LinearMethod.process_weights_after_loading calls .to(device="npu")
# on three tensors. On a non-NPU machine that raises, so tests patch
# torch.Tensor.to to pass through for the "npu" device (real NPU hardware is
# still exercised in the NPU CI).
_ORIG_TENSOR_TO = torch.Tensor.to


def _npu_passthrough_to(self, *args, **kwargs):
    dev = kwargs.get("device", args[0] if args else None)
    if dev == "npu":
        return self
    return _ORIG_TENSOR_TO(self, *args, **kwargs)


# MXFP4 online/offline methods guard with `if not weight.is_npu: weight.to(npu)`.
# torch_npu normally adds Tensor.is_npu (absent on CPU); set it True as a class
# attribute to skip the device-move branch entirely (real NPU is exercised in
# the NPU CI). Managed per-test via the mixin below.
class _IsNpuPatchedTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.Tensor.is_npu = True

    def tearDown(self):
        if hasattr(torch.Tensor, "is_npu"):
            del torch.Tensor.is_npu
        super().tearDown()


# RowParallelLinear test layer: subclasses the stubbed RowParallelLinear so
# isinstance(layer, RowParallelLinear) is True, with a settable tp_rank.
class _RowParallelLayer(RowParallelLinear, torch.nn.Module):
    def __init__(self, tp_rank=0):
        torch.nn.Module.__init__(self)
        self.tp_rank = tp_rank


# =============================================================================
# Module-level dtype helpers
# =============================================================================
class TestDtypeHelpers(unittest.TestCase):
    def test_get_float8_e8m0fnu_dtype(self):
        # On CPU (no torch_npu) -> getattr(torch, "float8_e8m0fnu", None)
        result = _get_float8_e8m0fnu_dtype()
        self.assertIs(result, getattr(torch, "float8_e8m0fnu", None))

    def test_get_float4_e2m1fn_x2_dtype_non_npu(self):
        # is_npu() is False -> falls through to getattr(torch, ...)
        result = _get_float4_e2m1fn_x2_dtype()
        self.assertIs(result, getattr(torch, "float4_e2m1fn_x2", None))


# =============================================================================
# _NPULinearMethodBase —__init__
# =============================================================================
class TestLinearMethodBaseInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = SimpleNamespace(weight_bits=8)
        base = _NPULinearMethodBase(cfg)
        self.assertIs(base.quant_config, cfg)

    def test_default_quant_config_none(self):
        base = _NPULinearMethodBase()
        self.assertIsNone(base.quant_config)

    def test_is_linear_method_base_subclass(self):
        self.assertTrue(issubclass(_NPULinearMethodBase, LinearMethodBase))


# =============================================================================
# NPUW8A8Int8DynamicLinearMethod —process_weights_after_loading
# =============================================================================
class TestW8A8DynamicProcess(unittest.TestCase):
    """process: transpose weight, flatten scale/offset (npu_format_cast passthrough)."""

    def setUp(self):
        self.in_, self.out_ = 16, 8
        self.kernel = NPUW8A8Int8DynamicLinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        # weight stored as (out, in) int8; process transposes to (in, out)
        self.layer.weight = Parameter(
            torch.arange(self.out_ * self.in_, dtype=torch.int8).reshape(
                self.out_, self.in_
            ),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.ones(self.out_, 1, dtype=torch.float32), requires_grad=False
        )
        self.layer.weight_offset = Parameter(
            torch.zeros(self.out_, 1, dtype=torch.float32), requires_grad=False
        )

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight.shape, (self.in_, self.out_))

    def test_weight_values_preserved(self):
        original = self.layer.weight.data.clone()
        self.kernel.process_weights_after_loading(self.layer)
        self.assertTrue(
            torch.equal(self.layer.weight.data, original.transpose(0, 1).contiguous())
        )

    def test_weight_scale_flattened(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_scale.shape, (self.out_,))

    def test_weight_offset_flattened(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_offset.shape, (self.out_,))

    def test_no_weight_offset_attr_ok(self):
        # Compressed-tensors format has no weight_offset -> must not crash.
        delattr(self.layer, "weight_offset")
        self.kernel.process_weights_after_loading(self.layer)
        self.assertFalse(hasattr(self.layer, "weight_offset"))
        self.assertEqual(self.layer.weight_scale.shape, (self.out_,))


# =============================================================================
# NPUW8A8Int8DynamicLinearMethod —apply (mocked npu ops)
# =============================================================================
class TestW8A8DynamicApply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 16, 8
        self.kernel = NPUW8A8Int8DynamicLinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_, self.out_, dtype=torch.int8), requires_grad=False
        )
        self.layer.weight_scale = Parameter(
            torch.ones(self.out_, dtype=torch.float32), requires_grad=False
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _DYN_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.ones(rows),
        )
        _MATMUL_MOCK.side_effect = lambda q, w, s, **k: torch.randn(rows, out)

    def test_calls_dynamic_quant_then_matmul(self):
        self._set_mocks(4)
        x = torch.randn(4, self.in_)
        self.kernel.apply(self.layer, x)
        _DYN_QUANT_MOCK.assert_called_once()
        _MATMUL_MOCK.assert_called_once()

    def test_output_shape(self):
        self._set_mocks(4)
        x = torch.randn(4, self.in_)
        out = self.kernel.apply(self.layer, x)
        self.assertEqual(out.shape, (4, self.out_))

    def test_tuple_input_skips_dynamic_quant(self):
        # When x is a (quant_out, dynamic_scale) tuple, npu_dynamic_quant is
        # skipped (computed upstream in a malprolog kernel).
        self._set_mocks(4)
        quant_out = torch.randn(4, self.in_)
        dynamic_scale = torch.ones(4)
        out = self.kernel.apply(self.layer, (quant_out, dynamic_scale))
        _DYN_QUANT_MOCK.assert_not_called()
        _MATMUL_MOCK.assert_called_once()
        self.assertEqual(out.shape, (4, self.out_))

    def test_passes_weight_and_scale_to_matmul(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        args = _MATMUL_MOCK.call_args.args
        self.assertIs(args[1], self.layer.weight)
        self.assertIs(args[2], self.layer.weight_scale)

    def test_pertoken_scale_flattened(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _MATMUL_MOCK.call_args.kwargs
        # dynamic_scale returned as (rows,) already flat; .flatten() is a no-op shape-wise
        self.assertEqual(kw["pertoken_scale"].ndim, 1)


# =============================================================================
# NPUW8A8Int8LinearMethod —apply (mocked; process uses .to("npu") so skipped)
# =============================================================================
class TestW8A8Int8Apply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 16, 8
        self.kernel = NPUW8A8Int8LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_, self.out_, dtype=torch.int8), requires_grad=False
        )
        self.layer.deq_scale = Parameter(
            torch.ones(self.out_, dtype=torch.float32), requires_grad=False
        )
        self.layer.aclnn_input_scale_reciprocal = Parameter(
            torch.ones(self.in_, dtype=torch.float32), requires_grad=False
        )
        self.layer.aclnn_input_offset = Parameter(
            torch.zeros(self.in_, dtype=torch.float32), requires_grad=False
        )
        self.layer.quant_bias = Parameter(
            torch.zeros(self.out_, dtype=torch.float32), requires_grad=False
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _QUANTIZE_MOCK.side_effect = lambda x, *a, **k: torch.randn(rows, self.in_)
        _MATMUL_MOCK.side_effect = lambda x, w, s, **k: torch.randn(rows, out)

    def test_quantizes_when_input_not_int8(self):
        self._set_mocks(4)
        x = torch.randn(4, self.in_)  # float32 -> not int8
        self.kernel.apply(self.layer, x)
        _QUANTIZE_MOCK.assert_called_once()
        _MATMUL_MOCK.assert_called_once()

    def test_skips_quantize_when_input_int8(self):
        self._set_mocks(4)
        x = torch.zeros(4, self.in_, dtype=torch.int8)
        self.kernel.apply(self.layer, x)
        _QUANTIZE_MOCK.assert_not_called()
        _MATMUL_MOCK.assert_called_once()

    def test_passes_quant_bias(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertIs(kw["bias"], self.layer.quant_bias)

    def test_passes_deq_scale_and_weight(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        args = _MATMUL_MOCK.call_args.args
        self.assertIs(args[1], self.layer.weight)
        self.assertIs(args[2], self.layer.deq_scale)

    def _make_rpl_layer(self, tp_rank):
        # RowParallelLinear layer (isinstance True) for the tp_rank branch.
        layer = _RowParallelLayer(tp_rank=tp_rank)
        layer.weight = Parameter(
            torch.zeros(self.in_, self.out_, dtype=torch.int8), requires_grad=False
        )
        layer.deq_scale = Parameter(
            torch.ones(self.out_, dtype=torch.float32), requires_grad=False
        )
        layer.aclnn_input_scale_reciprocal = Parameter(
            torch.ones(self.in_, dtype=torch.float32), requires_grad=False
        )
        layer.aclnn_input_offset = Parameter(
            torch.zeros(self.in_, dtype=torch.float32), requires_grad=False
        )
        layer.quant_bias = Parameter(
            torch.full((self.out_,), 3.0, dtype=torch.float32), requires_grad=False
        )
        return layer

    def test_quant_bias_none_when_tp_rank_gt_0(self):
        # TP>1: only rank 0 fuses bias (avoids double-counting across ranks).
        self._set_mocks(2)
        layer = self._make_rpl_layer(tp_rank=1)
        self.kernel.apply(layer, torch.randn(2, self.in_))
        self.assertIsNone(_MATMUL_MOCK.call_args.kwargs["bias"])

    def test_quant_bias_used_when_tp_rank_zero(self):
        # rank 0 (or non-RowParallel layers) fuse the bias.
        self._set_mocks(2)
        layer = self._make_rpl_layer(tp_rank=0)
        self.kernel.apply(layer, torch.randn(2, self.in_))
        self.assertIs(_MATMUL_MOCK.call_args.kwargs["bias"], layer.quant_bias)


# =============================================================================
# NPUW8A8Int8LinearMethod —process_weights_after_loading
# (.to("npu") patched to passthrough so it runs on CPU)
# =============================================================================
class TestW8A8Int8Process(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 16, 8
        self.kernel = NPUW8A8Int8LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.arange(self.out_ * self.in_, dtype=torch.int8).reshape(
                self.out_, self.in_
            ),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.ones(self.out_, 1, dtype=torch.float32), requires_grad=False
        )
        self.layer.weight_offset = Parameter(
            torch.zeros(self.out_, 1, dtype=torch.float32), requires_grad=False
        )
        self.layer.input_scale = Parameter(
            torch.full((1,), 2.0, dtype=torch.float32), requires_grad=False
        )
        self.layer.input_offset = Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=False
        )
        self._patch = patch.object(torch.Tensor, "to", _npu_passthrough_to)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight.shape, (self.in_, self.out_))

    def test_weight_values_preserved(self):
        original = self.layer.weight.data.clone()
        self.kernel.process_weights_after_loading(self.layer)
        self.assertTrue(
            torch.equal(self.layer.weight.data, original.transpose(0, 1).contiguous())
        )

    def test_weight_scale_flattened(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_scale.shape, (self.out_,))

    def test_weight_offset_flattened(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_offset.shape, (self.out_,))

    def test_no_weight_offset_attr_ok(self):
        delattr(self.layer, "weight_offset")
        self.kernel.process_weights_after_loading(self.layer)
        self.assertFalse(hasattr(self.layer, "weight_offset"))

    def test_aclnn_input_scale_shape(self):
        # expanding_factor = weight.shape[0] = in_ (after transpose); input_scale
        # repeated by that factor.
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.aclnn_input_scale.shape, (self.in_,))
        self.assertIsInstance(self.layer.aclnn_input_scale, torch.nn.Parameter)

    def test_aclnn_input_scale_reciprocal_values(self):
        # aclnn_input_scale_reciprocal = 1 / (input_scale.repeat(in_)) = 1/2 = 0.5
        self.kernel.process_weights_after_loading(self.layer)
        self.assertTrue(
            torch.allclose(
                self.layer.aclnn_input_scale_reciprocal, torch.full((self.in_,), 0.5)
            )
        )

    def test_aclnn_input_offset_shape(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.aclnn_input_offset.shape, (self.in_,))
        self.assertIsInstance(self.layer.aclnn_input_offset, torch.nn.Parameter)

    def test_aclnn_input_offset_values(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertTrue(
            torch.equal(self.layer.aclnn_input_offset.data, torch.zeros(self.in_))
        )


# =============================================================================
# NPUMXFP8LinearMethod —create_weights
# =============================================================================
class TestMXFP8CreateWeights(unittest.TestCase):
    def setUp(self):
        self.kernel = NPUMXFP8LinearMethod(SimpleNamespace())

    def test_registers_weight_parameter(self):
        layer = torch.nn.Module()
        self.kernel.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[4],
            input_size=64,
            output_size=4,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        # weight shape (out, in)
        self.assertEqual(layer.weight.shape, (4, 64))
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_sets_metadata_attrs(self):
        layer = torch.nn.Module()
        self.kernel.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[2, 2],
            input_size=64,
            output_size=4,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
        self.assertEqual(layer.logical_widths, [2, 2])
        self.assertEqual(layer.input_size_per_partition, 64)
        self.assertEqual(layer.output_size_per_partition, 4)
        self.assertEqual(layer.orig_dtype, torch.bfloat16)


# =============================================================================
# NPUMXFP8LinearMethod —process_weights_after_loading (offline path)
# =============================================================================
class TestMXFP8ProcessOffline(unittest.TestCase):
    """Offline path: weight is already float8_e4m3fn -> pure-torch re-layout."""

    def setUp(self):
        self.out_, self.in_ = 4, 64
        self.kernel = NPUMXFP8LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.out_, self.in_, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        # weight_scale [out, in//32]
        self.layer.weight_scale = Parameter(
            torch.zeros(self.out_, self.in_ // 32, dtype=torch.uint8),
            requires_grad=False,
        )

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        # (out, in) -> (in, out)
        self.assertEqual(self.layer.weight.shape, (self.in_, self.out_))

    def test_weight_scale_inv_shape(self):
        self.kernel.process_weights_after_loading(self.layer)
        # [out, in//32] -> reshape(out, in//64, 2) -> transpose(0,1) -> (in//64, out, 2)
        self.assertEqual(
            self.layer.weight_scale_inv.shape, (self.in_ // 64, self.out_, 2)
        )

    def test_weight_scale_dropped(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertFalse(hasattr(self.layer, "weight_scale"))

    def test_bias_cached_as_float32(self):
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.bfloat16), requires_grad=False
        )
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsInstance(self.layer.bias_fp32, torch.nn.Parameter)
        self.assertEqual(self.layer.bias_fp32.dtype, torch.float32)

    def test_no_bias_sets_none(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsNone(self.layer.bias_fp32)

    def test_float32_bias_not_recast(self):
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.float32), requires_grad=False
        )
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsNone(self.layer.bias_fp32)


# =============================================================================
# NPUMXFP8LinearMethod —apply (mocked npu ops)
# =============================================================================
class TestMXFP8Apply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 64, 4
        self.kernel = NPUMXFP8LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_, self.out_, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.layer.weight_scale_inv = Parameter(
            torch.zeros(self.in_ // 64, self.out_, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _MX_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.ones(rows, self.in_ // 32, 2),
        )
        _MATMUL_MOCK.side_effect = lambda qx, w, s, **k: torch.randn(rows, out)

    def test_calls_mx_quant_then_matmul(self):
        self._set_mocks(4)
        self.kernel.apply(self.layer, torch.randn(4, self.in_))
        _MX_QUANT_MOCK.assert_called_once()
        _MATMUL_MOCK.assert_called_once()

    def test_input_cast_to_bfloat16_when_float32(self):
        self._set_mocks(2)
        x = torch.randn(2, self.in_, dtype=torch.float32)
        self.kernel.apply(self.layer, x)
        # the x passed to npu_dynamic_mx_quant should be bfloat16
        passed_x = _MX_QUANT_MOCK.call_args.args[0]
        self.assertEqual(passed_x.dtype, torch.bfloat16)

    def test_bfloat16_input_not_recast(self):
        self._set_mocks(2)
        x = torch.randn(2, self.in_, dtype=torch.bfloat16)
        self.kernel.apply(self.layer, x)
        self.assertEqual(_MX_QUANT_MOCK.call_args.args[0].dtype, torch.bfloat16)

    def test_output_shape_2d(self):
        self._set_mocks(4)
        out = self.kernel.apply(self.layer, torch.randn(4, self.in_))
        self.assertEqual(out.shape, (4, self.out_))

    def test_output_shape_3d(self):
        self._set_mocks(6)
        out = self.kernel.apply(self.layer, torch.randn(2, 3, self.in_))
        self.assertEqual(out.shape, (2, 3, self.out_))

    def test_passes_weight_and_scale_inv(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        args = _MATMUL_MOCK.call_args.args
        self.assertIs(args[1], self.layer.weight)
        self.assertIs(args[2], self.layer.weight_scale_inv)

    def test_group_sizes_passed(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertEqual(kw["group_sizes"], [1, 1, MXFP8_BLOCK_SIZE])

    def test_bias_none_when_not_given(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        self.assertIsNone(_MATMUL_MOCK.call_args.kwargs["bias"])

    def test_cached_bias_fp32_used(self):
        self._set_mocks(2)
        bias_fp32 = Parameter(
            torch.randn(self.out_, dtype=torch.float32), requires_grad=False
        )
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.bfloat16), requires_grad=False
        )
        self.layer.bias_fp32 = bias_fp32
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=self.layer.bias)
        self.assertIs(_MATMUL_MOCK.call_args.kwargs["bias"], bias_fp32)

    def test_dynamic_bias_cast_to_float32(self):
        self._set_mocks(2)
        # A bias not equal to layer.bias -> per-call conversion path
        dyn_bias = torch.randn(self.out_, dtype=torch.bfloat16)
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.bfloat16), requires_grad=False
        )
        self.layer.bias_fp32 = None
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=dyn_bias)
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32)


# =============================================================================
# NPU_W4A4DynamicLinearMethod —process_weights_after_loading (both env paths)
# =============================================================================
class TestW4A4Process(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 16, 8
        self.layer = torch.nn.Module()
        # weight stored as (out, in) int8 (packed int4x2)
        self.layer.weight = Parameter(
            torch.zeros(self.out_, self.in_, dtype=torch.int8), requires_grad=False
        )
        self.layer.weight_scale = Parameter(
            torch.ones(self.out_, 1, dtype=torch.float32), requires_grad=False
        )
        self.layer.weight_offset = Parameter(
            torch.zeros(self.out_, 1, dtype=torch.float32), requires_grad=False
        )
        _envs.SGLANG_NPU_W4A4_NEW_PACKING = _EnvFlag(False)
        self.kernel = NPU_W4A4DynamicLinearMethod(SimpleNamespace())

    def tearDown(self):
        _envs.SGLANG_NPU_W4A4_NEW_PACKING = _EnvFlag(False)

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight.shape, (self.in_, self.out_))

    def test_weight_scale_flattened(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_scale.shape, (self.out_,))

    def test_weight_scale_fp32_created(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_scale_fp32.dtype, torch.float32)
        self.assertEqual(self.layer.weight_scale_fp32.shape, (self.out_,))

    def test_weight_offset_flattened(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_offset.shape, (self.out_,))

    def test_new_packing_false_uses_int4pack_op(self):
        # new_packing=False -> npu_convert_weight_to_int4pack (passthrough) ->
        # weight stays (in, out) int32
        _envs.SGLANG_NPU_W4A4_NEW_PACKING = _EnvFlag(False)
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight.dtype, torch.int32)
        self.assertEqual(self.layer.weight.shape, (self.in_, self.out_))

    def test_new_packing_true_uses_view(self):
        # new_packing=True -> weight.view(int32) -> (in, out//4) int32
        _envs.SGLANG_NPU_W4A4_NEW_PACKING = _EnvFlag(True)
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight.dtype, torch.int32)
        self.assertEqual(self.layer.weight.shape, (self.in_, self.out_ // 4))


# =============================================================================
# NPU_W4A4DynamicLinearMethod —apply (mocked npu ops)
# =============================================================================
class TestW4A4Apply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 16, 8
        self.kernel = NPU_W4A4DynamicLinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_, self.out_ // 4, dtype=torch.int32),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.ones(self.out_, dtype=torch.float32), requires_grad=False
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _DYN_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.ones(rows),
        )
        _MATMUL_MOCK.side_effect = lambda q, w, s, **k: torch.randn(rows, out)

    def test_calls_dynamic_quant_then_matmul(self):
        self._set_mocks(4)
        self.kernel.apply(self.layer, torch.randn(4, self.in_))
        _DYN_QUANT_MOCK.assert_called_once()
        _MATMUL_MOCK.assert_called_once()

    def test_output_shape(self):
        self._set_mocks(4)
        out = self.kernel.apply(self.layer, torch.randn(4, self.in_))
        self.assertEqual(out.shape, (4, self.out_))

    def test_dynamic_quant_uses_quint4x2(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _DYN_QUANT_MOCK.call_args.kwargs
        self.assertEqual(kw["dst_type"], torch.quint4x2)

    def test_passes_weight_scale_and_pertoken_scale(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        args = _MATMUL_MOCK.call_args.args
        self.assertIs(args[1], self.layer.weight)
        self.assertIs(args[2], self.layer.weight_scale)
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertEqual(kw["pertoken_scale"].ndim, 1)


# =============================================================================
# NPUMXFP4W4A8OfflineLinearMethod —process (no is_npu, pure torch + format_cast)
# =============================================================================
class TestMXFP4W4A8OfflineProcess(unittest.TestCase):
    """Offline W4A8 process: packed-FP4 uint8 weight -> (passthrough) format_cast
    -> transpose [in//2, out]; scale [out, in//32] -> [in//64, out, 2]."""

    def setUp(self):
        self.out_, self.in_ = 4, 64
        self.kernel = NPUMXFP4W4A8OfflineLinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        # packed FP4 weight [out, in//2] uint8; scale [out, in//32] uint8
        self.layer.weight = Parameter(
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.zeros(self.out_, self.in_ // 32, dtype=torch.uint8),
            requires_grad=False,
        )

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        # [out, in//2] -> transpose(-1,-2) -> [in//2, out]
        self.assertEqual(self.layer.weight.shape, (self.in_ // 2, self.out_))

    def test_weight_scale_shape(self):
        self.kernel.process_weights_after_loading(self.layer)
        # [out, in//32] -> reshape(out, in//64, 2) -> transpose(-3,-2) -> [in//64, out, 2]
        self.assertEqual(self.layer.weight_scale.shape, (self.in_ // 64, self.out_, 2))


# =============================================================================
# NPUMXFP4W4A8OfflineLinearMethod —apply (mocked npu ops)
# =============================================================================
class TestMXFP4W4A8OfflineApply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 64, 4
        self.kernel = NPUMXFP4W4A8OfflineLinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_ // 2, self.out_, dtype=torch.uint8),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.zeros(self.in_ // 64, self.out_, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _MX_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.ones(rows),
        )
        _MATMUL_MOCK.side_effect = lambda qx, w, s, **k: torch.randn(rows, out)

    def test_calls_mx_quant_then_matmul(self):
        self._set_mocks(4)
        self.kernel.apply(self.layer, torch.randn(4, self.in_))
        _MX_QUANT_MOCK.assert_called_once()
        _MATMUL_MOCK.assert_called_once()

    def test_output_shape_2d(self):
        self._set_mocks(4)
        out = self.kernel.apply(self.layer, torch.randn(4, self.in_))
        self.assertEqual(out.shape, (4, self.out_))

    def test_x2_dtype_and_group_sizes(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertEqual(kw["group_sizes"], [0, 0, MXFP4_BLOCK_SIZE])
        # x2_dtype = float4_e2m1fn_x2 (None on CPU) —just verify it's passed
        self.assertIn("x2_dtype", kw)

    def test_bias_bfloat16_converted_to_float32(self):
        self._set_mocks(2)
        bias = torch.randn(self.out_, dtype=torch.bfloat16)
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=bias)
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32)

    def test_bias_none(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        self.assertIsNone(_MATMUL_MOCK.call_args.kwargs["bias"])

    def test_float32_input_cast_to_bfloat16(self):
        # dtype guard: input not in (fp16, bf16) -> cast to bf16 before mx_quant
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_, dtype=torch.float32))
        self.assertEqual(_MX_QUANT_MOCK.call_args.args[0].dtype, torch.bfloat16)

    def test_bfloat16_input_not_recast(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_, dtype=torch.bfloat16))
        self.assertEqual(_MX_QUANT_MOCK.call_args.args[0].dtype, torch.bfloat16)

    def test_bias_float32_not_recast(self):
        # bias already fp32 -> passed through unchanged (the `and bias.dtype != float32` guard)
        self._set_mocks(2)
        bias = torch.randn(self.out_, dtype=torch.float32)
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=bias)
        self.assertIs(_MATMUL_MOCK.call_args.kwargs["bias"], bias)


# =============================================================================
# NPUMXFP4W4A8LinearMethod (online) —create_weights + process + apply
# =============================================================================
class TestMXFP4W4A8OnlineCreateWeights(unittest.TestCase):
    def setUp(self):
        self.kernel = NPUMXFP4W4A8LinearMethod(SimpleNamespace())

    def test_registers_weight_parameter(self):
        layer = torch.nn.Module()
        self.kernel.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[4],
            input_size=64,
            output_size=4,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertEqual(layer.weight.shape, (4, 64))

    def test_sets_metadata_attrs(self):
        layer = torch.nn.Module()
        self.kernel.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[2, 2],
            input_size=64,
            output_size=4,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
        self.assertEqual(layer.logical_widths, [2, 2])
        self.assertEqual(layer.output_size_per_partition, 4)
        self.assertEqual(layer.orig_dtype, torch.bfloat16)


class TestMXFP4W4A8OnlineProcess(_IsNpuPatchedTestCase):
    """Online W4A8 process: BF16 weight -> mx_quant(mock) -> format_cast(passthrough)
    -> transpose; scale -> [in//64, out, 2]; + bias cache."""

    def setUp(self):
        super().setUp()
        self.out_, self.in_ = 4, 64
        self.kernel = NPUMXFP4W4A8LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.randn(self.out_, self.in_, dtype=torch.bfloat16), requires_grad=False
        )
        _reset_ops()
        # mock npu_dynamic_mx_quant: returns (qw [out, in//2] uint8, w_scale [out, in//64, 2])
        _MX_QUANT_MOCK.side_effect = lambda w, **k: (
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            torch.zeros(self.out_, self.in_ // 64, 2, dtype=torch.uint8),
        )

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        # qw [out, in//2] -> view(uint8) -> format_cast(passthrough) -> transpose -> [in//2, out]
        self.assertEqual(self.layer.weight.shape, (self.in_ // 2, self.out_))

    def test_weight_scale_shape(self):
        self.kernel.process_weights_after_loading(self.layer)
        # w_scale [out, in//64, 2] -> transpose(-3,-2) -> [in//64, out, 2]
        self.assertEqual(self.layer.weight_scale.shape, (self.in_ // 64, self.out_, 2))

    def test_weight_scale_2d_input_reshaped(self):
        # older builds return 2D scale [out, in//32] -> reshape to 3D
        _MX_QUANT_MOCK.side_effect = lambda w, **k: (
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            torch.zeros(self.out_, self.in_ // 32, dtype=torch.uint8),  # 2D
        )
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_scale.shape, (self.in_ // 64, self.out_, 2))

    def test_bias_cached_as_float32(self):
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.bfloat16), requires_grad=False
        )
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsInstance(self.layer.bias_fp32, torch.nn.Parameter)
        self.assertEqual(self.layer.bias_fp32.dtype, torch.float32)

    def test_no_bias_sets_none(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsNone(self.layer.bias_fp32)

    def test_mx_quant_called_with_fp4_dst_and_round_mode(self):
        self.kernel.process_weights_after_loading(self.layer)
        kw = _MX_QUANT_MOCK.call_args.kwargs
        self.assertEqual(kw["round_mode"], "round")
        # dst_type = float4_e2m1fn_x2 (None on CPU) —verify the kwarg is passed
        self.assertIn("dst_type", kw)


class TestMXFP4W4A8OnlineApply(unittest.TestCase):
    """Online W4A8 apply shares the offline apply path (mocked npu ops)."""

    def setUp(self):
        self.in_, self.out_ = 64, 4
        self.kernel = NPUMXFP4W4A8LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_ // 2, self.out_, dtype=torch.uint8),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.zeros(self.in_ // 64, self.out_, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _MX_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.ones(rows),
        )
        _MATMUL_MOCK.side_effect = lambda qx, w, s, **k: torch.randn(rows, out)

    def test_output_shape_3d(self):
        self._set_mocks(6)
        out = self.kernel.apply(self.layer, torch.randn(2, 3, self.in_))
        self.assertEqual(out.shape, (2, 3, self.out_))

    def test_float32_input_cast_to_bfloat16(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_, dtype=torch.float32))
        self.assertEqual(_MX_QUANT_MOCK.call_args.args[0].dtype, torch.bfloat16)

    def test_x2_dtype_fp4_and_group_sizes(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertEqual(kw["group_sizes"], [0, 0, MXFP4_BLOCK_SIZE])
        self.assertIn("x2_dtype", kw)

    # --- 3-branch bias cache (None / cached fp32 / per-call convert) ---
    def test_bias_none_when_not_given(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        self.assertIsNone(_MATMUL_MOCK.call_args.kwargs["bias"])

    def test_cached_bias_fp32_used(self):
        self._set_mocks(2)
        bias_fp32 = Parameter(
            torch.randn(self.out_, dtype=torch.float32), requires_grad=False
        )
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.bfloat16), requires_grad=False
        )
        self.layer.bias_fp32 = bias_fp32
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=self.layer.bias)
        self.assertIs(_MATMUL_MOCK.call_args.kwargs["bias"], bias_fp32)

    def test_dynamic_bias_cast_to_float32(self):
        # bias is NOT layer.bias (or no cache) -> per-call .to(float32)
        self._set_mocks(2)
        dyn_bias = torch.randn(self.out_, dtype=torch.bfloat16)
        self.layer.bias = Parameter(
            torch.randn(self.out_, dtype=torch.bfloat16), requires_grad=False
        )
        self.layer.bias_fp32 = None
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=dyn_bias)
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32)


# =============================================================================
# NPUSingleLevelMXFP4LinearMethod (online W4A4) —create + process + apply
# =============================================================================
class TestSingleLevelMXFP4OnlineProcess(_IsNpuPatchedTestCase):
    def setUp(self):
        super().setUp()
        self.out_, self.in_ = 4, 64
        self.kernel = NPUSingleLevelMXFP4LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.randn(self.out_, self.in_, dtype=torch.bfloat16), requires_grad=False
        )
        _reset_ops()
        _MX_QUANT_MOCK.side_effect = lambda w, **k: (
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            torch.zeros(self.out_, self.in_ // 64, 2, dtype=torch.uint8),
        )

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        # qw [out, in//2] -> Parameter -> .data.transpose(0,1) -> [in//2, out]
        self.assertEqual(self.layer.weight.shape, (self.in_ // 2, self.out_))

    def test_weight_scale_shape(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight_scale.shape, (self.in_ // 64, self.out_, 2))

    def test_mx_quant_uses_fp4_and_round(self):
        self.kernel.process_weights_after_loading(self.layer)
        kw = _MX_QUANT_MOCK.call_args.kwargs
        self.assertEqual(kw["round_mode"], "round")
        self.assertIn("dst_type", kw)


class TestSingleLevelMXFP4OnlineApply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 64, 4
        self.kernel = NPUSingleLevelMXFP4LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.in_ // 2, self.out_, dtype=torch.uint8),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.zeros(self.in_ // 64, self.out_, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _MX_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.ones(rows),
        )
        _MATMUL_MOCK.side_effect = lambda qx, w, s, **k: torch.randn(rows, out)

    def test_output_shape(self):
        self._set_mocks(4)
        out = self.kernel.apply(self.layer, torch.randn(4, self.in_))
        self.assertEqual(out.shape, (4, self.out_))

    def test_x1_x2_dtype_fp4_and_group_sizes(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        kw = _MATMUL_MOCK.call_args.kwargs
        self.assertEqual(kw["group_sizes"], [1, 1, MXFP4_BLOCK_SIZE])
        self.assertIn("x1_dtype", kw)
        self.assertIn("x2_dtype", kw)

    def test_bias_cast_to_float32(self):
        self._set_mocks(2)
        bias = torch.randn(self.out_, dtype=torch.bfloat16)
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=bias)
        self.assertEqual(_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32)

    def test_bias_none(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        self.assertIsNone(_MATMUL_MOCK.call_args.kwargs["bias"])


# =============================================================================
# NPUSingleLevelMXFP4OfflineLinearMethod —process (inherits apply from online)
# =============================================================================
class TestSingleLevelMXFP4OfflineProcess(_IsNpuPatchedTestCase):
    def setUp(self):
        super().setUp()
        self.out_, self.in_ = 4, 64
        self.kernel = NPUSingleLevelMXFP4OfflineLinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.layer.weight_scale = Parameter(
            torch.zeros(self.out_, self.in_ // 32, dtype=torch.uint8),
            requires_grad=False,
        )

    def test_weight_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        # [out, in//2] -> transpose(0,1) -> [in//2, out]
        self.assertEqual(self.layer.weight.shape, (self.in_ // 2, self.out_))

    def test_weight_scale_shape(self):
        self.kernel.process_weights_after_loading(self.layer)
        # [out, in//32] -> reshape(out, in//64, 2) -> transpose(0,1) -> [in//64, out, 2]
        self.assertEqual(self.layer.weight_scale.shape, (self.in_ // 64, self.out_, 2))


# =============================================================================
# NPUDualLevelMXFP4LinearMethod (online dual-level W4A4) —process + apply
# =============================================================================
class TestDualLevelMXFP4Process(_IsNpuPatchedTestCase):
    def setUp(self):
        super().setUp()
        self.out_, self.in_ = 4, 64
        self.kernel = NPUDualLevelMXFP4LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.randn(self.out_, self.in_, dtype=torch.bfloat16), requires_grad=False
        )
        _reset_ops()
        # mock npu_dynamic_dual_level_mx_quant: (qw [out, in//2] uint8, w_l0 [out, in//l0, 1], w_l1 [out])
        _DUAL_MX_QUANT_MOCK.side_effect = lambda w, **k: (
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            torch.zeros(self.out_, self.in_ // 8, 1, dtype=torch.float8_e4m3fn),
            torch.zeros(self.out_, dtype=torch.float8_e4m3fn),
        )

    def test_weight_is_parameter(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsInstance(self.layer.weight, torch.nn.Parameter)

    def test_weight_l0_scale_transposed(self):
        self.kernel.process_weights_after_loading(self.layer)
        # w_l0 [out, in//8, 1] -> squeeze(-1) -> [out, in//8] -> transpose(0,1) -> [in//8, out]
        self.assertEqual(self.layer.weight_l0_scale.shape, (self.in_ // 8, self.out_))

    def test_weight_l1_scale_is_parameter(self):
        self.kernel.process_weights_after_loading(self.layer)
        self.assertIsInstance(self.layer.weight_l1_scale, torch.nn.Parameter)

    def test_dual_mx_quant_called(self):
        self.kernel.process_weights_after_loading(self.layer)
        _DUAL_MX_QUANT_MOCK.assert_called_once()
        self.assertIsNone(_DUAL_MX_QUANT_MOCK.call_args.kwargs.get("smooth_scale"))


class TestDualLevelMXFP4Apply(unittest.TestCase):
    def setUp(self):
        self.in_, self.out_ = 64, 4
        self.kernel = NPUDualLevelMXFP4LinearMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(
            torch.zeros(self.out_, self.in_ // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.layer.weight_l0_scale = Parameter(
            torch.zeros(self.in_ // 8, self.out_, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.layer.weight_l1_scale = Parameter(
            torch.zeros(self.out_, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        _reset_ops()

    def _set_mocks(self, rows):
        out = self.out_
        _DUAL_MX_QUANT_MOCK.side_effect = lambda x, **k: (
            torch.randn(rows, self.in_),
            torch.zeros(rows, self.in_ // 8, dtype=torch.float8_e4m3fn),
            torch.zeros(rows, dtype=torch.float8_e4m3fn),
        )
        _DUAL_MATMUL_MOCK.side_effect = (
            lambda qx, w, al0, wl0, al1, wl1, **k: torch.randn(rows, out)
        )

    def test_calls_dual_mx_quant_then_dual_matmul(self):
        self._set_mocks(4)
        self.kernel.apply(self.layer, torch.randn(4, self.in_))
        _DUAL_MX_QUANT_MOCK.assert_called_once()
        _DUAL_MATMUL_MOCK.assert_called_once()

    def test_output_shape(self):
        self._set_mocks(4)
        out = self.kernel.apply(self.layer, torch.randn(4, self.in_))
        self.assertEqual(out.shape, (4, self.out_))

    def test_matmul_arg_order(self):
        # (act, weight, act_l0, w_l0, act_l1, w_l1)
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        args = _DUAL_MATMUL_MOCK.call_args.args
        self.assertIs(args[1], self.layer.weight)
        self.assertIs(args[3], self.layer.weight_l0_scale)
        self.assertIs(args[5], self.layer.weight_l1_scale)

    def test_bias_cast_to_float32(self):
        self._set_mocks(2)
        bias = torch.randn(self.out_, dtype=torch.bfloat16)
        self.kernel.apply(self.layer, torch.randn(2, self.in_), bias=bias)
        self.assertEqual(
            _DUAL_MATMUL_MOCK.call_args.kwargs["bias"].dtype, torch.float32
        )

    def test_bias_none(self):
        self._set_mocks(2)
        self.kernel.apply(self.layer, torch.randn(2, self.in_))
        self.assertIsNone(_DUAL_MATMUL_MOCK.call_args.kwargs["bias"])


if __name__ == "__main__":
    unittest.main()

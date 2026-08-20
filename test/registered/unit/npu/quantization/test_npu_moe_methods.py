"""
Unit tests for sglang.srt.hardware_backend.npu.quantization.moe_methods.

The module reaches NPU only through ``torch.ops.npu.*`` ops, ``npu_format_cast``,
and the ``is_npu``/``torch_npu`` dtype helpers 鈥?all called at runtime, not import
time. To keep these unit tests runnable on any machine, the heavy sglang
infrastructure (FusedMoEMethodBase / npu utils / envs / moe matmul+quant stubs /
linear_method_npu dtype helpers / is_npu) is stubbed in ``sys.modules`` and the
real ``moe_methods.py`` source is loaded directly by path with importlib. The CI
marker ``register_npu_ci`` is loaded from the real ``ci_register.py`` (by path,
so sglang/__init__.py does not run).

Scope: the CPU-runnable subset 鈥?``_require_e8m0_dtype``, the ``_NPUMoEMethodBase``
helpers, every MoE method ``__init__``, the pure-torch packing helpers
(``_pack_int4``/``_pack_to_int32``/``_unpack_from_int32``/``_update_bias``), and
the ``process_weights_after_loading`` paths that don't call ``.npu()``/
``weight.is_npu`` (offline W4A8 MXFP, W4A16, unquant, MXFP8-offline). The
``apply`` paths that need ``_require_e8m0_dtype`` (which raises on CPU) are
exercised only for the methods that don't call it.
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
# Stub heavy sglang infrastructure BEFORE loading the source module.
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


for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.quantization",
    "sglang.srt.layers.moe",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.quantization",
    "sglang.srt.hardware_backend.npu.moe",
    "sglang.test",
    "sglang.test.ci",
):
    _ensure_pkg(_pkg)

# register_npu_ci: load the REAL marker from sglang's ci_register.py (by path,
# so sglang/__init__.py 鈥?which needs triton 鈥?does not run) and register it in
# sys.modules so the literal import used by the attention-test files resolves.
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
from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

# FusedMoEMethodBase: plain base (no abstractmethods).
_base_mod = ModuleType("sglang.srt.layers.quantization.base_config")


class FusedMoEMethodBase:
    def __init__(self, *args, **kwargs):
        pass


_base_mod.FusedMoEMethodBase = FusedMoEMethodBase
_base_mod.QuantizationConfig = type("QuantizationConfig", (), {})
_install_stub("sglang.srt.layers.quantization.base_config", _base_mod)

# npu utils: npu_format_cast passthrough + NPUACLFormat marker.
_npu_utils = ModuleType("sglang.srt.hardware_backend.npu.utils")


class NPUACLFormat:
    ACL_FORMAT_FRACTAL_NZ = 29


def npu_format_cast(t, *args, **kwargs):
    return t


_npu_utils.NPUACLFormat = NPUACLFormat
_npu_utils.npu_format_cast = npu_format_cast
_install_stub("sglang.srt.hardware_backend.npu.utils", _npu_utils)

# envs: SGLANG_NPU_W4A4_NEW_PACKING (toggleable).
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

# moe.matmul: GroupedMatmul / GroupedMatmulSwigluQuant stubs (record init args).
_matmul_mod = ModuleType("sglang.srt.hardware_backend.npu.moe.matmul")


class GroupedMatmul:
    def __init__(self, *a, **k):
        self._init = (a, k)


class GroupedMatmulSwigluQuant:
    def __init__(self, *a, **k):
        self._init = (a, k)


_matmul_mod.GroupedMatmul = GroupedMatmul
_matmul_mod.GroupedMatmulSwigluQuant = GroupedMatmulSwigluQuant
_install_stub("sglang.srt.hardware_backend.npu.moe.matmul", _matmul_mod)

# moe.quant: HiddenStatesDynamicQuant stub (records quant_dtype).
_quant_mod = ModuleType("sglang.srt.hardware_backend.npu.moe.quant")


class HiddenStatesDynamicQuant:
    def __init__(self, quant_dtype=None, **k):
        self.quant_dtype = quant_dtype


_quant_mod.HiddenStatesDynamicQuant = HiddenStatesDynamicQuant
_install_stub("sglang.srt.hardware_backend.npu.moe.quant", _quant_mod)

# linear_method_npu dtype helpers: return None on CPU (matches the real
# _get_float8_e8m0fnu_dtype / _get_float4_e2m1fn_x2_dtype on a non-NPU build).
_lmn_mod = ModuleType("sglang.srt.hardware_backend.npu.quantization.linear_method_npu")
_lmn_mod._get_float8_e8m0fnu_dtype = lambda: getattr(torch, "float8_e8m0fnu", None)
_lmn_mod._get_float4_e2m1fn_x2_dtype = lambda: getattr(torch, "float4_e2m1fn_x2", None)
_install_stub(
    "sglang.srt.hardware_backend.npu.quantization.linear_method_npu", _lmn_mod
)

# is_npu: always False on CPU (used by _require_e8m0_dtype and _get_float4).
_utils_mod = ModuleType("sglang.srt.utils")


def is_npu():
    return False


_utils_mod.is_npu = is_npu
_install_stub("sglang.srt.utils", _utils_mod)

# get_moe_a2a_backend (lazy-imported in NPUW8A8Int8.maybe_process_fuseep_weights).
_moe_mod = ModuleType("sglang.srt.layers.moe")
_moe_mod.get_moe_a2a_backend = lambda: SimpleNamespace(
    is_ascend_fuseep=lambda: False, is_deepep=lambda: False
)
_install_stub("sglang.srt.layers.moe", _moe_mod)

# torch.ops.npu: shared MagicMock stub (passthrough for npu_convert_weight_to_int4pack).
if "_test_npu_ops_stub" not in sys.modules:
    sys.modules["_test_npu_ops_stub"] = MagicMock()
_NPU = sys.modules["_test_npu_ops_stub"]
torch.ops.npu = _NPU
_NPU.npu_convert_weight_to_int4pack = lambda t, *a, **k: t

register_npu_ci(est_time=6, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Load the real moe_methods.py source file by path.
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _TEST_DIR
for _ in range(5):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
_SRC_PATH = os.path.join(
    _REPO_ROOT,
    "python",
    "sglang",
    "srt",
    "hardware_backend",
    "npu",
    "quantization",
    "moe_methods.py",
)

_spec = importlib.util.spec_from_file_location("moe_methods_under_test", _SRC_PATH)
moe_mod = importlib.util.module_from_spec(_spec)
sys.modules["moe_methods_under_test"] = moe_mod
_spec.loader.exec_module(moe_mod)

_require_e8m0_dtype = moe_mod._require_e8m0_dtype
_NPUMoEMethodBase = moe_mod._NPUMoEMethodBase
NPUW4A8MXFP4MoEMethod = moe_mod.NPUW4A8MXFP4MoEMethod
NPUW4A4Int4MoEMethod = moe_mod.NPUW4A4Int4MoEMethod
NPUW8A8Int8MoEMethod = moe_mod.NPUW8A8Int8MoEMethod
NPUW4A8Int8MoEMethod = moe_mod.NPUW4A8Int8MoEMethod
NPUWNA16Int4MoEMethod = moe_mod.NPUWNA16Int4MoEMethod
NPUUnquantMoEMethod = moe_mod.NPUUnquantMoEMethod
NPUMXFP8MoEMethod = moe_mod.NPUMXFP8MoEMethod


def _reset_e8m0():
    moe_mod._E8M0_DTYPE = None


# =============================================================================
# _require_e8m0_dtype 鈥?raises on CPU (no float8_e8m0fnu without torch_npu)
# =============================================================================
class TestRequireE8m0Dtype(unittest.TestCase):
    def setUp(self):
        _reset_e8m0()

    def test_raises_on_cpu(self):
        with self.assertRaises(RuntimeError):
            _require_e8m0_dtype()

    def test_message_mentions_mxfp8(self):
        try:
            _require_e8m0_dtype()
        except RuntimeError as e:
            self.assertIn("float8_e8m0fnu", str(e))


# =============================================================================
# _NPUMoEMethodBase 鈥?__init__ + static helpers
# =============================================================================
class TestMoEMethodBaseInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = SimpleNamespace()
        base = _NPUMoEMethodBase(cfg)
        self.assertIs(base.quant_config, cfg)

    def test_default_none(self):
        base = _NPUMoEMethodBase()
        self.assertIsNone(base.quant_config)


class TestSetDispatcherOutputDtype(unittest.TestCase):
    def test_no_dispatcher_is_noop(self):
        layer = torch.nn.Module()  # no .dispatcher
        _NPUMoEMethodBase._set_dispatcher_output_dtype(layer, "bf16")  # must not raise

    def test_sets_when_dispatcher_present(self):
        layer = SimpleNamespace()
        dispatcher = MagicMock()
        layer.dispatcher = dispatcher
        _NPUMoEMethodBase._set_dispatcher_output_dtype(layer, "int8")
        dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "int8"}
        )


class TestValidateWeightPrefix(unittest.TestCase):
    def test_raises_when_weight_missing(self):
        layer = torch.nn.Module()  # no w13_weight
        with self.assertRaises(AttributeError):
            _NPUMoEMethodBase._validate_weight_prefix(layer, "w13")

    def test_ok_when_weight_present(self):
        layer = torch.nn.Module()
        layer.w2_weight = Parameter(torch.zeros(1))
        _NPUMoEMethodBase._validate_weight_prefix(layer, "w2")  # must not raise


class TestGetBiasArgs(unittest.TestCase):
    def test_scale_bias_used(self):
        quant_info = SimpleNamespace(w13_scale_bias=MagicMock())
        out = _NPUMoEMethodBase._get_bias_args(quant_info, "w13")
        self.assertEqual(out, {"bias": [quant_info.w13_scale_bias]})

    def test_weight_bias_fallback(self):
        quant_info = SimpleNamespace(w2_weight_bias=MagicMock())
        out = _NPUMoEMethodBase._get_bias_args(quant_info, "w2")
        self.assertEqual(out, {"bias": [quant_info.w2_weight_bias]})

    def test_no_bias_returns_empty(self):
        quant_info = SimpleNamespace()
        self.assertEqual(_NPUMoEMethodBase._get_bias_args(quant_info, "w13"), {})


# =============================================================================
# MoE method __init__s
# =============================================================================
class TestMoEMethodInits(unittest.TestCase):
    def test_w4a8_mxfp4_creates_matmul_and_quantizer(self):
        m = NPUW4A8MXFP4MoEMethod()
        self.assertIsInstance(m.matmul, GroupedMatmul)
        self.assertIsInstance(m.hidden_states_quantizer, HiddenStatesDynamicQuant)
        self.assertEqual(m.hidden_states_quantizer.quant_dtype, torch.float8_e4m3fn)

    def test_w4a4_int4_creates_matmul_and_quantizer(self):
        m = NPUW4A4Int4MoEMethod()
        self.assertIsInstance(m.matmul, GroupedMatmul)
        self.assertEqual(m.hidden_states_quantizer.quant_dtype, torch.quint4x2)

    def test_w8a8_int8_creates_matmul_and_quantizer(self):
        m = NPUW8A8Int8MoEMethod()
        self.assertIsInstance(m.matmul, GroupedMatmul)
        self.assertEqual(m.hidden_states_quantizer.quant_dtype, torch.int8)

    def test_w4a8_int8_default_flags(self):
        m = NPUW4A8Int8MoEMethod()
        self.assertFalse(m.is_per_channel_weight)
        self.assertFalse(m.activation_use_clip)

    def test_w4a8_int8_custom_flags(self):
        m = NPUW4A8Int8MoEMethod(is_per_channel_weight=True, activation_use_clip=True)
        self.assertTrue(m.is_per_channel_weight)
        self.assertTrue(m.activation_use_clip)

    def test_wna16_int4_creates_matmul_only(self):
        m = NPUWNA16Int4MoEMethod()
        self.assertIsInstance(m.matmul, GroupedMatmul)
        self.assertFalse(hasattr(m, "hidden_states_quantizer"))

    def test_unquant_creates_matmul_only(self):
        m = NPUUnquantMoEMethod()
        self.assertIsInstance(m.matmul, GroupedMatmul)
        self.assertFalse(hasattr(m, "hidden_states_quantizer"))

    def test_mxfp8_w13_uses_swiglu_quant(self):
        m = NPUMXFP8MoEMethod("w13")
        self.assertIsInstance(m.matmul, GroupedMatmulSwigluQuant)
        self.assertEqual(m.hidden_states_quantizer.quant_dtype, torch.float8_e4m3fn)

    def test_mxfp8_w2_uses_plain_matmul(self):
        m = NPUMXFP8MoEMethod("w2")
        self.assertIsInstance(m.matmul, GroupedMatmul)
        self.assertIsNone(m.hidden_states_quantizer)


# =============================================================================
# NPUW4A4Int4MoEMethod 鈥?pure-torch packing helpers
# =============================================================================
class TestW4A4Int4PackHelpers(unittest.TestCase):
    def setUp(self):
        self.m = NPUW4A4Int4MoEMethod()

    def test_pack_int4_2d_values(self):
        # pairs (low,high) -> byte = (high<<4)|low. 0x87 overflows int8 to -121.
        w = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)
        out = self.m._pack_int4(w)
        self.assertEqual(out.dtype, torch.int8)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.tolist(), [[33, 67], [101, -121]])

    def test_pack_int4_3d_values(self):
        w = torch.arange(1 * 2 * 4, dtype=torch.int8).reshape(1, 2, 4) + 1
        out = self.m._pack_int4(w)
        self.assertEqual(out.shape, (1, 2, 2))

    def test_pack_int4_odd_n_raises(self):
        w = torch.zeros(2, 3, dtype=torch.int8)
        with self.assertRaises(AssertionError):
            self.m._pack_int4(w)

    def test_w4a4_pack_int4_transposes(self):
        # _w4a4_pack_int4 = transpose -> pack -> transpose back
        # (2,4) -> transpose (4,2) -> pack (4,1) -> transpose back (1,4)
        w = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)
        out = self.m._w4a4_pack_int4(w)
        self.assertEqual(out.shape, (1, 4))

    def test_pack_to_int32_shape(self):
        # 4 int8 -> 1 int32 along last dim
        w = torch.zeros(3, 8, dtype=torch.int8)
        out = self.m._pack_to_int32(w)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, (3, 2))


# =============================================================================
# NPUW4A8Int8MoEMethod 鈥?_pack_to_int32 (assert + view) + _update_bias
# =============================================================================
class TestW4A8Int8PackHelpers(unittest.TestCase):
    def setUp(self):
        self.m = NPUW4A8Int8MoEMethod()

    def test_pack_to_int32_shape(self):
        w = torch.zeros(2, 8, dtype=torch.int8)
        out = self.m._pack_to_int32(w)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, (2, 2))

    def test_pack_to_int32_asserts_indivisible(self):
        w = torch.zeros(2, 7, dtype=torch.int8)  # 7 % 4 != 0
        with self.assertRaises(AssertionError):
            self.m._pack_to_int32(w)

    def test_update_bias_sums_over_intermediate(self):
        # scale_bias [E, intermediate, N] -> transpose(1,2) [E, N, intermediate]
        # -> sum(dim=1) -> [E, intermediate]
        layer = torch.nn.Module()
        bias = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)
        layer.w13_scale_bias = Parameter(bias.clone())
        self.m._update_bias(layer, "w13")
        self.assertEqual(layer.w13_scale_bias.shape, (2, 2))
        self.assertTrue(
            torch.equal(
                layer.w13_scale_bias.data,
                bias.transpose(1, 2).contiguous().sum(dim=1),
            )
        )

    def test_update_bias_no_attr_is_noop(self):
        layer = torch.nn.Module()
        self.m._update_bias(layer, "w13")  # must not raise


# =============================================================================
# NPUWNA16Int4MoEMethod 鈥?_unpack_from_int32 + _pack_to_int32
# =============================================================================
class TestWNA16Int4UnpackPack(unittest.TestCase):
    def setUp(self):
        self.m = NPUWNA16Int4MoEMethod()

    def test_unpack_4bit_dim1_values(self):
        w = torch.tensor([[0x76543210]], dtype=torch.int32)
        out = self.m._unpack_from_int32(w, 4, packed_dim=1)
        self.assertEqual(out.dtype, torch.int8)
        self.assertEqual(out[0].tolist(), [-8, -7, -6, -5, -4, -3, -2, -1])

    def test_unpack_8bit_dim1_values(self):
        w = torch.tensor([[0x04030201]], dtype=torch.int32)
        out = self.m._unpack_from_int32(w, 8, packed_dim=1)
        self.assertEqual(out[0].tolist(), [-127, -126, -125, -124])

    def test_unpack_dim0_values(self):
        w = torch.tensor([[0x76543210]], dtype=torch.int32)
        out = self.m._unpack_from_int32(w, 4, packed_dim=0)
        self.assertEqual(out.shape, (8, 1))
        self.assertEqual(out.reshape(-1).tolist(), [-8, -7, -6, -5, -4, -3, -2, -1])

    def test_unpack_shape_truncation(self):
        # shape truncates the unpacked last dim to original_row_size
        w = torch.tensor([[0x76543210]], dtype=torch.int32)
        out = self.m._unpack_from_int32(w, 4, shape=(1, 5), packed_dim=1)
        self.assertEqual(out.shape, (1, 5))
        self.assertEqual(out[0].tolist(), [-8, -7, -6, -5, -4])

    def test_unpack_asserts_non_int32(self):
        with self.assertRaises(ValueError):
            self.m._unpack_from_int32(torch.zeros(1, 1, dtype=torch.int8), 4)

    def test_unpack_asserts_num_bits_gt_8(self):
        with self.assertRaises(ValueError):
            self.m._unpack_from_int32(torch.zeros(1, 1, dtype=torch.int32), 16)

    def test_pack_to_int32_int8_path(self):
        # int8 -> view(int32): 4 int8 -> 1 int32
        w = torch.zeros(2, 3, 8, dtype=torch.int8)  # (E, K, N), N%4==0
        out = self.m._pack_to_int32(w)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, (2, 3, 2))

    def test_pack_to_int32_int8_asserts_indivisible(self):
        w = torch.zeros(2, 3, 7, dtype=torch.int8)  # 7 % 4 != 0
        with self.assertRaises(AssertionError):
            self.m._pack_to_int32(w)

    def test_pack_to_int32_int32_path_uses_npu_op(self):
        # int32 -> npu_convert_weight_to_int4pack (passthrough) -> view back
        w = torch.zeros(2, 3, 8, dtype=torch.int32)  # last dim %8==0
        out = self.m._pack_to_int32(w)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, (2, 3, 8))

    def test_pack_to_int32_int32_asserts_indivisible(self):
        w = torch.zeros(2, 3, 7, dtype=torch.int32)  # 7 % 8 != 0
        with self.assertRaises(AssertionError):
            self.m._pack_to_int32(w)

    def test_pack_to_int32_bad_dtype_raises(self):
        w = torch.zeros(2, 3, 8, dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.m._pack_to_int32(w)


# =============================================================================
# NPUUnquantMoEMethod 鈥?process_weights_after_loading (format_cast passthrough)
# =============================================================================
class TestUnquantMoEProcess(unittest.TestCase):
    def setUp(self):
        self.E, self.K, self.N = 2, 4, 8
        self.m = NPUUnquantMoEMethod()
        self.layer = torch.nn.Module()
        self.layer.w13_weight = Parameter(
            torch.randn(self.E, self.K, self.N), requires_grad=False
        )

    def test_weight_preserved_by_format_cast(self):
        original = self.layer.w13_weight.data.clone()
        self.m.process_weights_after_loading(self.layer, "w13")
        self.assertTrue(torch.equal(self.layer.w13_weight.data, original))

    def test_w13_sets_dispatcher_bf16(self):
        self.layer.dispatcher = MagicMock()
        self.m.process_weights_after_loading(self.layer, "w13")
        self.layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )

    def test_w2_does_not_set_dispatcher(self):
        self.layer.w2_weight = Parameter(
            torch.randn(self.E, self.K, self.N), requires_grad=False
        )
        self.layer.dispatcher = MagicMock()
        self.m.process_weights_after_loading(self.layer, "w2")
        self.layer.dispatcher.set_quant_config.assert_not_called()

    def test_missing_weight_raises(self):
        layer = torch.nn.Module()  # no w13_weight
        with self.assertRaises(AttributeError):
            self.m.process_weights_after_loading(layer, "w13")


# =============================================================================
# NPUWNA16Int4MoEMethod 鈥?process_weights_after_loading (unpack+repack)
# =============================================================================
class TestWNA16Int4MoEProcess(unittest.TestCase):
    def setUp(self):
        # E=2, K=64, N=8 -> weight [E, K//8, N]=[2,8,8] int32; scale [E, N, 1]
        self.E, self.K, self.N = 2, 64, 8
        self.m = NPUWNA16Int4MoEMethod()
        self.layer = torch.nn.Module()
        self.layer.w13_weight = Parameter(
            torch.zeros(self.E, self.K // 8, self.N, dtype=torch.int32),
            requires_grad=False,
        )
        self.layer.w13_weight_scale = Parameter(
            torch.ones(self.E, self.N, 1, dtype=torch.float32), requires_grad=False
        )
        self.layer.w13_weight_offset = Parameter(
            torch.zeros(self.E, self.N, 1, dtype=torch.float32), requires_grad=False
        )

    def test_scale_transposed(self):
        # [E, N, 1] -> transpose(-1,-2) -> [E, 1, N]
        self.m.process_weights_after_loading(self.layer, "w13")
        self.assertEqual(self.layer.w13_weight_scale.shape, (self.E, 1, self.N))

    def test_offset_transposed(self):
        self.m.process_weights_after_loading(self.layer, "w13")
        self.assertEqual(self.layer.w13_weight_offset.shape, (self.E, 1, self.N))

    def test_offset_optional(self):
        delattr(self.layer, "w13_weight_offset")
        self.m.process_weights_after_loading(self.layer, "w13")  # must not raise
        self.assertFalse(hasattr(self.layer, "w13_weight_offset"))

    def test_weight_is_int32_parameter(self):
        self.m.process_weights_after_loading(self.layer, "w13")
        self.assertIsInstance(self.layer.w13_weight, torch.nn.Parameter)
        self.assertEqual(self.layer.w13_weight.dtype, torch.int32)

    def test_w13_sets_dispatcher_bf16(self):
        self.layer.dispatcher = MagicMock()
        self.m.process_weights_after_loading(self.layer, "w13")
        self.layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )


# =============================================================================
# NPUW4A8MXFP4MoEMethod 鈥?process_weights_after_loading (patch _get_float4)
# =============================================================================
class TestW4A8MXFP4MoEProcess(unittest.TestCase):
    """Offline-style W4A8 MXFP process: format_cast(passthrough)+transpose,
    scale reshape. _get_float4 is patched to a sentinel so the fp4-support guard
    doesn't raise on CPU."""

    def setUp(self):
        self.E, self.K, self.N = 2, 64, 8
        self.m = NPUW4A8MXFP4MoEMethod()
        self.layer = torch.nn.Module()
        # weight [E, K, N]; weight_scale [E, K, N//2] (shape[2] even for reshape)
        self.layer.w13_weight = Parameter(
            torch.zeros(self.E, self.K, self.N, dtype=torch.uint8), requires_grad=False
        )
        self.layer.w13_weight_scale = Parameter(
            torch.zeros(self.E, self.K, self.N, dtype=torch.uint8), requires_grad=False
        )
        self._patch = patch.object(
            moe_mod, "_get_float4_e2m1fn_x2_dtype", return_value=296
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_weight_transposed(self):
        self.m.process_weights_after_loading(self.layer, "w13")
        # [E, K, N] -> format_cast(passthrough) -> transpose(-1,-2) -> [E, N, K]
        self.assertEqual(self.layer.w13_weight.shape, (self.E, self.N, self.K))

    def test_weight_scale_reshaped(self):
        self.m.process_weights_after_loading(self.layer, "w13")
        # [E, K, N] -> reshape(E, K, N//2, 2) -> transpose(1,2) -> [E, N//2, K, 2]
        self.assertEqual(
            self.layer.w13_weight_scale.shape, (self.E, self.N // 2, self.K, 2)
        )

    def test_w13_sets_dispatcher_bf16(self):
        self.layer.dispatcher = MagicMock()
        self.m.process_weights_after_loading(self.layer, "w13")
        self.layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )


# =============================================================================
# NPUMXFP8MoEMethod 鈥?process_weights_after_loading (offline fp8 path)
# =============================================================================
class TestMXFP8MoEProcessOffline(unittest.TestCase):
    """Offline path: weight is already float8_e4m3fn -> pure-torch re-layout
    (scale reshape + transpose). Online path needs is_npu/.to(npu), skipped.
    Note: the MoE MXFP8 process does NOT cache bias (unlike the dense path)."""

    def setUp(self):
        self.E, self.K, self.N = 2, 64, 8
        self.m = NPUMXFP8MoEMethod("w2")
        self.layer = torch.nn.Module()
        # weight [E, K, N] fp8; weight_scale [E, N, K//32] uint8 (per docstring)
        self.layer.w2_weight = Parameter(
            torch.zeros(self.E, self.K, self.N, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.layer.w2_weight_scale = Parameter(
            torch.zeros(self.E, self.N, self.K // 32, dtype=torch.uint8),
            requires_grad=False,
        )

    def test_weight_transposed(self):
        self.m.process_weights_after_loading(self.layer, "w2")
        # [E, K, N] -> format_cast(passthrough) -> transpose(1,2) -> [E, N, K]
        self.assertEqual(self.layer.w2_weight.shape, (self.E, self.N, self.K))

    def test_weight_dtype_preserved(self):
        self.m.process_weights_after_loading(self.layer, "w2")
        self.assertEqual(self.layer.w2_weight.dtype, torch.float8_e4m3fn)

    def test_weight_scale_reshaped(self):
        self.m.process_weights_after_loading(self.layer, "w2")
        # [E, N, K//32] -> reshape(E, N, K//64, 2) -> transpose(1,2) -> [E, K//64, N, 2]
        self.assertEqual(
            self.layer.w2_weight_scale.shape, (self.E, self.K // 64, self.N, 2)
        )


# =============================================================================
# NPUW8A8Int8MoEMethod 鈥?maybe_process_fuseep_weights (non-fuseep path)
# =============================================================================
class TestW8A8Int8FuseepGuard(unittest.TestCase):
    def test_returns_false_when_not_fuseep(self):
        # get_moe_a2a_backend().is_ascend_fuseep() -> False -> skip fuseep layout
        layer = torch.nn.Module()
        result = NPUW8A8Int8MoEMethod.maybe_process_fuseep_weights(layer)
        self.assertFalse(result)

    def test_already_processed_returns_true(self):
        # When fuseep IS active and the layer is already processed, skip & return
        # True (avoids calling process_fuseep_weights a second time).
        _moe_layer_mod = sys.modules["sglang.srt.layers.moe"]
        fake_backend = SimpleNamespace(
            is_ascend_fuseep=lambda: True, is_deepep=lambda: False
        )
        with patch.object(
            _moe_layer_mod, "get_moe_a2a_backend", return_value=fake_backend
        ):
            layer = torch.nn.Module()
            layer._fuseep_weights_processed = True
            result = NPUW8A8Int8MoEMethod.maybe_process_fuseep_weights(layer)
        self.assertTrue(result)


# =============================================================================
# apply wiring (methods that don't call _require_e8m0_dtype) 鈥?mocked matmul
# =============================================================================
class TestWNA16Int4ApplyWiring(unittest.TestCase):
    def setUp(self):
        self.m = NPUWNA16Int4MoEMethod()
        self.m.matmul = MagicMock()
        self.m.matmul.forward.side_effect = lambda *a, **k: torch.zeros(4, 8)

    def test_calls_matmul_forward(self):
        quant_info = SimpleNamespace(
            w13_weight_scale=MagicMock(), w13_weight_offset=MagicMock()
        )
        self.m.apply(
            quant_info,
            torch.randn(4, 8),
            torch.tensor([4]),
            None,
            torch.bfloat16,
            "w13",
            0,
        )
        self.m.matmul.forward.assert_called_once()

    def test_passes_transposed_true(self):
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())
        self.m.apply(
            quant_info,
            torch.randn(4, 8),
            torch.tensor([4]),
            None,
            torch.bfloat16,
            "w2",
            0,
        )
        self.assertTrue(self.m.matmul.forward.call_args.kwargs["transposed"])

    def test_offset_omitted_when_none(self):
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())  # no offset
        self.m.apply(
            quant_info,
            torch.randn(4, 8),
            torch.tensor([4]),
            None,
            torch.bfloat16,
            "w2",
            0,
        )
        self.assertEqual(self.m.matmul.forward.call_args.kwargs["antiquant_offset"], [])


class TestUnquantApplyWiring(unittest.TestCase):
    def setUp(self):
        self.m = NPUUnquantMoEMethod()
        self.m.matmul = MagicMock()
        self.m.matmul.forward.side_effect = lambda *a, **k: torch.zeros(4, 8)

    def test_calls_matmul_forward(self):
        quant_info = SimpleNamespace()
        self.m.apply(
            quant_info,
            torch.randn(4, 8),
            torch.tensor([4]),
            None,
            torch.bfloat16,
            "w2",
            0,
        )
        self.m.matmul.forward.assert_called_once()

    def test_passes_transposed_false(self):
        # unquant path uses transposed=False (weight not transposed at load)
        quant_info = SimpleNamespace()
        self.m.apply(
            quant_info,
            torch.randn(4, 8),
            torch.tensor([4]),
            None,
            torch.bfloat16,
            "w2",
            0,
        )
        self.assertFalse(self.m.matmul.forward.call_args.kwargs["transposed"])


# =============================================================================
# NPUW4A8MXFP4MoEMethod 鈥?apply (distinct: per_token_scale list, weight_dtype=fp4,
# x_dtype=fp8, scale/scale_dtype=None, antiquant_scale, per_token_scale_dtype=e8m0)
# Needs _get_float4 + _require_e8m0 patched (both raise on CPU).
# =============================================================================
class TestW4A8MXFP4MoEApply(unittest.TestCase):
    def setUp(self):
        self.m = NPUW4A8MXFP4MoEMethod()
        self.m.matmul = MagicMock()
        self.m.matmul.forward.side_effect = lambda *a, **k: torch.zeros(4, 8)
        self.m.hidden_states_quantizer = MagicMock(
            return_value=(torch.randn(4, 64), torch.zeros(4, 1, 2))
        )
        self._fp4 = patch.object(
            moe_mod, "_get_float4_e2m1fn_x2_dtype", return_value=296
        )
        self._e8m0 = patch.object(moe_mod, "_require_e8m0_dtype", return_value="e8m0")
        self._fp4.start()
        self._e8m0.start()

    def tearDown(self):
        self._fp4.stop()
        self._e8m0.stop()

    def _apply(self, pertoken_scale=None):
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4], dtype=torch.int32),
            pertoken_scale,
            torch.bfloat16,
            "w2",
            0,
        )

    def test_calls_matmul(self):
        self._apply()
        self.m.matmul.forward.assert_called_once()

    def test_quantizer_called_when_pertoken_scale_none(self):
        self._apply(pertoken_scale=None)
        self.m.hidden_states_quantizer.assert_called_once()

    def test_quantizer_skipped_when_pertoken_scale_given(self):
        # pertoken_scale given -> reshape path (skip quantizer)
        ps = torch.zeros(4, 2)  # hidden=64 -> reshape(4, 64//64=1, 2)
        self._apply(pertoken_scale=ps)
        self.m.hidden_states_quantizer.assert_not_called()

    def test_weight_dtype_fp4_x_dtype_fp8(self):
        self._apply()
        kw = self.m.matmul.forward.call_args.kwargs
        self.assertEqual(kw["weight_dtype"], 296)
        self.assertEqual(kw["x_dtype"], torch.float8_e4m3fn)

    def test_scale_and_scale_dtype_none(self):
        # W4A8 MXFP path uses per_token_scale (not antiquant scale)
        self._apply()
        kw = self.m.matmul.forward.call_args.kwargs
        self.assertIsNone(kw["scale"])
        self.assertIsNone(kw["scale_dtype"])

    def test_antiquant_scale_from_quant_info(self):
        ws = MagicMock()
        quant_info = SimpleNamespace(w2_weight_scale=ws)
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            None,
            torch.bfloat16,
            "w2",
            0,
        )
        self.assertEqual(
            self.m.matmul.forward.call_args.kwargs["antiquant_scale"], [ws]
        )

    def test_per_token_scale_dtype_is_e8m0(self):
        self._apply()
        self.assertEqual(
            self.m.matmul.forward.call_args.kwargs["per_token_scale_dtype"], "e8m0"
        )

    def test_expert_tokens_cast_to_int64(self):
        self._apply()
        args = self.m.matmul.forward.call_args.args
        self.assertEqual(args[3].dtype, torch.int64)

    def test_transposed_true(self):
        self._apply()
        self.assertTrue(self.m.matmul.forward.call_args.kwargs["transposed"])


# =============================================================================
# NPUMXFP8MoEMethod 鈥?apply (w2 path; distinct: scale_dtype=e8m0,
# x_dtype=None, weight_dtype=None). Needs _require_e8m0 patched.
# =============================================================================
class TestMXFP8MoEApplyW2(unittest.TestCase):
    def setUp(self):
        self.m = NPUMXFP8MoEMethod("w2")
        self.m.matmul = MagicMock()
        self.m.matmul.forward.side_effect = lambda *a, **k: torch.zeros(4, 8)
        self._e8m0 = patch.object(moe_mod, "_require_e8m0_dtype", return_value="e8m0")
        self._e8m0.start()

    def tearDown(self):
        self._e8m0.stop()

    def test_w2_calls_matmul(self):
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            torch.zeros(4, 1, 2),
            torch.bfloat16,
            "w2",
            0,
        )
        self.m.matmul.forward.assert_called_once()

    def test_w13_raises_value_error(self):
        # apply() only serves w2; w13 must go through apply_fused_gmm1_swiglu
        quant_info = SimpleNamespace(w13_weight_scale=MagicMock())
        with self.assertRaises(ValueError):
            self.m.apply(
                quant_info,
                torch.randn(4, 64),
                torch.tensor([4]),
                torch.zeros(4, 1, 2),
                torch.bfloat16,
                "w13",
                0,
            )

    def test_scale_and_per_token_scale_dtype_e8m0(self):
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            torch.zeros(4, 1, 2),
            torch.bfloat16,
            "w2",
            0,
        )
        kw = self.m.matmul.forward.call_args.kwargs
        self.assertEqual(kw["scale_dtype"], "e8m0")
        self.assertEqual(kw["per_token_scale_dtype"], "e8m0")

    def test_x_dtype_and_weight_dtype_none(self):
        # MXFP8 path: x/weight dtype is implicit (None), only scale_dtype matters
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            torch.zeros(4, 1, 2),
            torch.bfloat16,
            "w2",
            0,
        )
        kw = self.m.matmul.forward.call_args.kwargs
        self.assertIsNone(kw["x_dtype"])
        self.assertIsNone(kw["weight_dtype"])

    def test_scale_from_quant_info(self):
        ws = MagicMock()
        quant_info = SimpleNamespace(w2_weight_scale=ws)
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            torch.zeros(4, 1, 2),
            torch.bfloat16,
            "w2",
            0,
        )
        self.assertEqual(self.m.matmul.forward.call_args.kwargs["scale"], [ws])


# =============================================================================
# NPUW4A8Int8MoEMethod 鈥?apply (representative: W4A4Int4/W8A8Int8/W4A8Int8 share
# this exact apply body 鈥?scale + per_token_scale + bias_args + transposed=True).
# =============================================================================
class TestW4A8Int8MoEApply(unittest.TestCase):
    def setUp(self):
        self.m = NPUW4A8Int8MoEMethod()
        self.m.matmul = MagicMock()
        self.m.matmul.forward.side_effect = lambda *a, **k: torch.zeros(4, 8)

    def _apply(self, pertoken_scale=None):
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock())
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            pertoken_scale,
            torch.bfloat16,
            "w2",
            0,
        )

    def test_calls_matmul(self):
        self._apply(pertoken_scale=torch.zeros(4, 1, 2))
        self.m.matmul.forward.assert_called_once()

    def test_quantizer_called_when_pertoken_scale_none(self):
        self.m.hidden_states_quantizer = MagicMock(
            return_value=(torch.randn(4, 64), torch.zeros(4, 1, 2))
        )
        self._apply(pertoken_scale=None)
        self.m.hidden_states_quantizer.assert_called_once()

    def test_scale_and_per_token_scale_passed(self):
        ws = MagicMock()
        ps = torch.zeros(4, 1, 2)
        quant_info = SimpleNamespace(w2_weight_scale=ws)
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            ps,
            torch.bfloat16,
            "w2",
            0,
        )
        kw = self.m.matmul.forward.call_args.kwargs
        self.assertEqual(kw["scale"], [ws])
        self.assertEqual(kw["per_token_scale"], [ps])

    def test_transposed_true(self):
        self._apply(pertoken_scale=torch.zeros(4, 1, 2))
        self.assertTrue(self.m.matmul.forward.call_args.kwargs["transposed"])

    def test_bias_args_merged(self):
        # _get_bias_args adds bias=[...] when a scale_bias attr is present
        sb = MagicMock()
        quant_info = SimpleNamespace(w2_weight_scale=MagicMock(), w2_scale_bias=sb)
        self.m.apply(
            quant_info,
            torch.randn(4, 64),
            torch.tensor([4]),
            torch.zeros(4, 1, 2),
            torch.bfloat16,
            "w2",
            0,
        )
        self.assertEqual(self.m.matmul.forward.call_args.kwargs["bias"], [sb])


if __name__ == "__main__":
    unittest.main()

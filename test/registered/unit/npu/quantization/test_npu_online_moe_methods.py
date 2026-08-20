"""
Unit tests for sglang.srt.hardware_backend.npu.quantization.online_moe_methods.

``online_moe_methods.py`` is the ``--quantization mxfp8`` online entry point:
``NPUMXFP8OnlineMoEMethod`` subclasses ``UnquantizedFusedMoEMethod`` and only
overrides ``create_moe_runner`` (validates the moe backend, then attaches
``NPUMXFP8MoEMethod`` w13/w2 kernels and builds the runner). The heavy sglang
deps (``UnquantizedFusedMoEMethod`` / ``MoeRunner`` / ``MoeRunnerBackend`` /
``get_moe_runner_backend`` / ``NPUMXFP8MoEMethod``) are stubbed and the real
``online_moe_methods.py`` source is loaded directly by path with importlib.
"""

import importlib.util
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch

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
    "sglang.srt.layers.moe.moe_runner",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.quantization",
    "sglang.test",
    "sglang.test.ci",
):
    _ensure_pkg(_pkg)

# register_npu_ci: load the REAL marker from sglang's ci_register.py (by path).
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

# UnquantizedFusedMoEMethod: plain base (NPUMXFP8OnlineMoEMethod subclasses it).
_unquant_mod = ModuleType("sglang.srt.layers.quantization.unquant")


class UnquantizedFusedMoEMethod:
    def __init__(self):
        pass


_unquant_mod.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
_install_stub("sglang.srt.layers.quantization.unquant", _unquant_mod)

# moe_runner: MoeRunner stub (records backend + config).
_moe_runner_mod = ModuleType("sglang.srt.layers.moe.moe_runner")


class MoeRunner:
    def __init__(self, backend, config):
        self._backend = backend
        self._config = config


_moe_runner_mod.MoeRunner = MoeRunner
_install_stub("sglang.srt.layers.moe.moe_runner", _moe_runner_mod)

# moe.utils: MoeRunnerBackend enum-like + get_moe_runner_backend (configurable).
_moe_utils_mod = ModuleType("sglang.srt.layers.moe.utils")


class MoeRunnerBackend:
    ASCEND = "ascend"
    AUTO = "auto"
    FLASHINFER = "flashinfer"


_get_backend_mock = MagicMock()  # return a backend SimpleNamespace per test
_moe_utils_mod.MoeRunnerBackend = MoeRunnerBackend
_moe_utils_mod.get_moe_runner_backend = _get_backend_mock
_install_stub("sglang.srt.layers.moe.utils", _moe_utils_mod)

# moe_methods.NPUMXFP8MoEMethod: opaque stub (records weight_prefix). The real
# class is tested in test_npu_moe_methods.py; here it is only instantiated.
_moe_methods_mod = ModuleType(
    "sglang.srt.hardware_backend.npu.quantization.moe_methods"
)


class NPUMXFP8MoEMethod:
    def __init__(self, weight_prefix):
        self.weight_prefix = weight_prefix


_moe_methods_mod.NPUMXFP8MoEMethod = NPUMXFP8MoEMethod
_install_stub(
    "sglang.srt.hardware_backend.npu.quantization.moe_methods", _moe_methods_mod
)

register_npu_ci(est_time=4, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Load the real online_moe_methods.py source file by path.
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
    "online_moe_methods.py",
)

_spec = importlib.util.spec_from_file_location(
    "online_moe_methods_under_test", _SRC_PATH
)
online_mod = importlib.util.module_from_spec(_spec)
sys.modules["online_moe_methods_under_test"] = online_mod
_spec.loader.exec_module(online_mod)

NPUMXFP8OnlineMoEMethod = online_mod.NPUMXFP8OnlineMoEMethod


def _backend(kind):
    """A backend SimpleNamespace with is_auto/is_ascend matching `kind`."""
    return SimpleNamespace(
        is_auto=lambda: kind == "auto",
        is_ascend=lambda: kind == "ascend",
        value=kind,
    )


# =============================================================================
# NPUMXFP8OnlineMoEMethod 鈥?__init__
# =============================================================================
class TestInit(unittest.TestCase):
    def test_stores_quant_config(self):
        cfg = SimpleNamespace()
        m = NPUMXFP8OnlineMoEMethod(cfg)
        self.assertIs(m.quant_config, cfg)

    def test_default_quant_config_none(self):
        m = NPUMXFP8OnlineMoEMethod()
        self.assertIsNone(m.quant_config)

    def test_is_unquant_subclass(self):
        self.assertTrue(issubclass(NPUMXFP8OnlineMoEMethod, UnquantizedFusedMoEMethod))


# =============================================================================
# NPUMXFP8OnlineMoEMethod 鈥?create_moe_runner (backend validation + kernel attach)
# =============================================================================
class TestCreateMoeRunner(unittest.TestCase):
    def setUp(self):
        self.m = NPUMXFP8OnlineMoEMethod(SimpleNamespace())
        self.layer = torch.nn.Module()
        self.config = SimpleNamespace()
        _get_backend_mock.reset_mock()

    def _run(self):
        self.m.create_moe_runner(self.layer, self.config)

    def test_auto_backend_accepted(self):
        _get_backend_mock.return_value = _backend("auto")
        self._run()  # must not raise

    def test_ascend_backend_accepted(self):
        _get_backend_mock.return_value = _backend("ascend")
        self._run()

    def test_non_ascend_backend_raises(self):
        _get_backend_mock.return_value = _backend("flashinfer")
        with self.assertRaises(ValueError) as ctx:
            self._run()
        self.assertIn("--moe-runner-backend", str(ctx.exception))

    def test_attaches_w13_kernel(self):
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIsInstance(self.layer.w13_kernel, NPUMXFP8MoEMethod)
        self.assertEqual(self.layer.w13_kernel.weight_prefix, "w13")

    def test_attaches_w2_kernel(self):
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIsInstance(self.layer.w2_kernel, NPUMXFP8MoEMethod)
        self.assertEqual(self.layer.w2_kernel.weight_prefix, "w2")

    def test_w13_and_w2_are_distinct(self):
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIsNot(self.layer.w13_kernel, self.layer.w2_kernel)

    def test_config_layer_set_to_layer(self):
        # moe_runner_config.layer = layer (AscendRunnerCore reads layer.w2_kernel)
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIs(self.config.layer, self.layer)

    def test_moe_runner_config_stored(self):
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIs(self.m.moe_runner_config, self.config)

    def test_runner_built_with_ascend_backend(self):
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIsInstance(self.m.runner, MoeRunner)
        self.assertEqual(self.m.runner._backend, MoeRunnerBackend.ASCEND)
        self.assertIs(self.m.runner._config, self.config)

    def test_aiter_runner_none(self):
        # inherited apply() consults _aiter_runner; aiter is CUDA/ROCm-only
        _get_backend_mock.return_value = _backend("auto")
        self._run()
        self.assertIsNone(self.m._aiter_runner)

    def test_kernels_attached_before_runner_built(self):
        # AscendRunnerCore.__init__ reads layer.w2_kernel, so the kernels must
        # exist on the layer before MoeRunner(...) is constructed.
        _get_backend_mock.return_value = _backend("auto")
        # Use a MoeRunner that asserts layer.w2_kernel is set at construction time
        _orig_runner = online_mod.MoeRunner

        class _CheckingRunner:
            def __init__(self, backend, config):
                assert hasattr(config.layer, "w2_kernel"), "w2_kernel missing"
                self._backend = backend
                self._config = config

        online_mod.MoeRunner = _CheckingRunner
        try:
            self._run()
        finally:
            online_mod.MoeRunner = _orig_runner
        self.assertIsInstance(self.m.runner, _CheckingRunner)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Pure-Python contracts that the optional FlashInfer MLA value gate is really
optional: capability detection, clean fallback and a visible warning.

The production modules import torch / the sglang package tree at import time.
This test loads them by file path under temporary ``sys.modules`` stubs and
injects a missing ``flashinfer.cake_mla_value_gate`` dependency, so both the
"dependency present" and "dependency missing" branches can be exercised on a
CPU-only box with nothing but the standard library::

    python3 test/registered/unit/kernel_plugins/nvidia/kimi_k3/test_mla_value_gate_optional.py

No production code is changed: this pins the fallback contract that is already
in the branch (find_spec guard + _refuse() + opt-in slot) against regression.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MISSING = object()
SLOT = "k3_flashinfer_mla_value"


def _locate(relative: str) -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "python" / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate python/{relative} from {__file__}")


def _ensure_torch_stub() -> None:
    """A CPU box has no torch; keep a minimal stub for the duration of the test.

    CI has the real torch, so nothing is injected there and the real package is
    never shadowed. mla_value_gate imports torch lazily inside _device_supported.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.Tensor = object
        stub.cuda = SimpleNamespace(get_device_capability=Mock(return_value=(10, 3)))
        sys.modules["torch"] = stub


@contextlib.contextmanager
def _temporary_modules(stubs):
    saved = {}
    for name, module in stubs.items():
        saved[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_module(name: str, path: Path, stubs=None):
    """Load a source file under temporary stubs, then restore sys.modules."""
    with _temporary_modules(stubs or {}):
        module_spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class _Slots:
    def __init__(self, value: str = ""):
        self.value = value

    def get(self) -> str:
        return self.value


def _sglang_stubs(envs, can_use, prepare):
    """Stub just the two leaves flashinfer_value.py imports from sglang."""
    stubs = {
        "sglang": _package("sglang"),
        "sglang.kernels": _package("sglang.kernels"),
        "sglang.kernels.ops": _package("sglang.kernels.ops"),
        "sglang.srt": _package("sglang.srt"),
    }
    attention = types.ModuleType("sglang.kernels.ops.attention")
    attention.can_use_cake_mla_value_gate = can_use
    attention.prepare_cake_mla_value_gate = prepare
    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = envs
    stubs["sglang.kernels.ops.attention"] = attention
    stubs["sglang.srt.environ"] = environ
    stubs["sglang.kernels.ops"].attention = attention
    stubs["sglang.srt"].environ = environ
    return stubs, attention


_ensure_torch_stub()
_TORCH = sys.modules["torch"]

_ENV = SimpleNamespace(SGLANG_NVIDIA_MEGAKERNEL_SLOTS=_Slots())
_CAN_USE = Mock(return_value=True)
_PREPARE = Mock(return_value=SimpleNamespace(program="program", arch="sm_103a"))
_STUBS, _ATTENTION = _sglang_stubs(_ENV, _CAN_USE, _PREPARE)

# Real source, loaded under stubs; no production module is duplicated or edited.
_CAKE = _load_module(
    "_cake_mla_value_gate_contract",
    _locate("sglang/kernels/cake_kernels/mla_value_gate.py"),
)
_FI = _load_module(
    "_flashinfer_value_contract",
    _locate("sglang/srt/kernel_plugins/nvidia/kimi_k3/cutedsl/mla/flashinfer_value.py"),
    _STUBS,
)


def _load_can_use_facade(cake_module):
    """Extract the public ops.attention facade by AST (its package is heavy).

    can_use_cake_mla_value_gate() imports the cake module lazily *inside the
    function body*, so the sys.modules stubs must be present both when the
    function is defined and on every call. The returned wrapper installs them
    for the duration of each call and restores the real packages afterwards.
    """
    import ast
    import functools

    path = _locate("sglang/kernels/ops/attention/__init__.py")
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "can_use_cake_mla_value_gate"
    )
    body = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    facade = types.ModuleType("sglang.kernels.cake_kernels.mla_value_gate")
    facade.supports_mla_value_gate = cake_module.supports_mla_value_gate
    stubs = {
        "sglang": _package("sglang"),
        "sglang.kernels": _package("sglang.kernels"),
        "sglang.kernels.cake_kernels": _package("sglang.kernels.cake_kernels"),
        "sglang.kernels.cake_kernels.mla_value_gate": facade,
    }
    scope = {"torch": _TORCH}
    with _temporary_modules(stubs):
        exec(compile(body, str(path), "exec"), scope)
    resolved = scope["can_use_cake_mla_value_gate"]

    @functools.wraps(resolved)
    def call(*args, **kwargs):
        with _temporary_modules(stubs):
            return resolved(*args, **kwargs)

    return call


_CAN_USE_FACADE = _load_can_use_facade(_CAKE)

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except Exception:  # no package / no torch on a bare CPU box
    pass
else:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _w_uv(cuda=True):
    return SimpleNamespace(is_cuda=cuda, device=SimpleNamespace(index=0))


class TestCapabilityDetection(unittest.TestCase):
    """The dependency-present / dependency-missing branches of the gate probe."""

    def setUp(self):
        _CAKE._device_supported.cache_clear()

    def tearDown(self):
        _CAKE._device_supported.cache_clear()

    def test_dependency_present_on_supported_sm_is_available(self):
        with (
            patch.object(_CAKE, "find_spec", return_value=object()) as find,
            patch.object(_TORCH.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            self.assertTrue(_CAKE.supports_mla_value_gate(_w_uv()))
        find.assert_called_once_with("flashinfer.cake_mla_value_gate")

    def test_missing_flashinfer_module_is_unavailable(self):
        """sys.modules / find_spec report no module -> clean False, no exception."""
        with (
            patch.object(_CAKE, "find_spec", return_value=None),
            patch.object(_TORCH.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            self.assertFalse(_CAKE.supports_mla_value_gate(_w_uv()))

    def test_public_facade_returns_false_when_dependency_is_missing(self):
        """can_use_cake_mla_value_gate(): missing module -> False, no raise."""
        with (
            patch.object(_CAKE, "find_spec", return_value=None),
            patch.object(_TORCH.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            self.assertFalse(_CAN_USE_FACADE(_w_uv()))

    def test_public_facade_returns_true_when_dependency_is_present(self):
        with (
            patch.object(_CAKE, "find_spec", return_value=object()),
            patch.object(_TORCH.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            self.assertTrue(_CAN_USE_FACADE(_w_uv()))

    def test_unsupported_sm_is_unavailable_even_with_the_module(self):
        with (
            patch.object(_CAKE, "find_spec", return_value=object()),
            patch.object(_TORCH.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            self.assertFalse(_CAKE.supports_mla_value_gate(_w_uv()))

    def test_cpu_tensor_is_unavailable_without_a_device_probe(self):
        with patch.object(_CAKE, "_device_supported") as probe:
            self.assertFalse(_CAKE.supports_mla_value_gate(_w_uv(cuda=False)))
        probe.assert_not_called()


class TestOptionalFallback(unittest.TestCase):
    """prepare_value_gate(): opt-in, visible warning, clean MlaValue fallback."""

    def setUp(self):
        _FI._REFUSED.clear()
        _ENV.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.value = SLOT
        _CAN_USE.reset_mock()
        _CAN_USE.side_effect = None
        _CAN_USE.return_value = True
        _PREPARE.reset_mock()
        _PREPARE.side_effect = None
        _PREPARE.return_value = SimpleNamespace(program="program", arch="sm_103a")
        self.logger = MagicMock()
        self.logger_patch = patch.object(_FI, "logger", self.logger)
        self.logger_patch.start()
        self.addCleanup(self.logger_patch.stop)
        self.addCleanup(_FI._REFUSED.clear)

    def test_flag_off_never_probes_flashinfer(self):
        """Backward compatible: the gate is off unless its slot is listed."""
        _ENV.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.value = "k3_native_mla"
        self.assertIsNone(_FI.prepare_value_gate(_w_uv(), rows=8, q=1))
        _CAN_USE.assert_not_called()
        _PREPARE.assert_not_called()
        self.logger.warning.assert_not_called()

    def test_dependency_present_returns_the_runner(self):
        runner = _FI.prepare_value_gate(_w_uv(), rows=8, q=1)
        self.assertEqual(runner.program, "program")
        _PREPARE.assert_called_once()
        self.logger.warning.assert_not_called()
        self.logger.warning_once.assert_not_called()

    def test_capability_refusal_falls_back_with_a_visible_warning(self):
        """Dependency missing -> False -> None + warning, never an exception."""
        _CAN_USE.return_value = False
        self.assertIsNone(_FI.prepare_value_gate(_w_uv(), rows=8, q=1))
        self.assertIn(0, _FI._REFUSED)
        self.logger.warning.assert_called_once()
        (message,), _ = self.logger.warning.call_args
        self.assertIn("using MlaValue", message)
        self.assertIn("flashinfer.cake_mla_value_gate", message)

    def test_importerror_falls_back_with_a_visible_warning(self):
        """A pinned wheel without the module raises ImportError at prepare()."""
        _PREPARE.side_effect = ImportError(
            "No module named 'flashinfer.cake_mla_value_gate'"
        )
        self.assertIsNone(_FI.prepare_value_gate(_w_uv(), rows=8, q=1))
        self.logger.warning.assert_called_once()
        (message,), _ = self.logger.warning.call_args
        self.assertIn("using MlaValue", message)
        self.assertIn("ImportError", message)

    def test_unclassified_failure_is_a_bucket_local_warning(self):
        _PREPARE.side_effect = RuntimeError("program load failed")
        self.assertIsNone(_FI.prepare_value_gate(_w_uv(), rows=8, q=1))
        self.logger.warning_once.assert_called_once()
        (message,), _ = self.logger.warning_once.call_args
        self.assertIn("using MlaValue", message)
        self.assertEqual(_FI._REFUSED, {})  # not a capability verdict

    def test_a_capability_refusal_is_remembered_so_buckets_do_not_retry(self):
        _CAN_USE.return_value = False
        for rows in (1, 8, 64):
            self.assertIsNone(_FI.prepare_value_gate(_w_uv(), rows=rows, q=1))
        self.assertEqual(_CAN_USE.call_count, 1)
        self.logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

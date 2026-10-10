import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.srt import runtime_context
from sglang.srt import utils as srt_utils
from sglang.srt.mem_cache.pool_host import _kvcacheio
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ALL_OPS = sorted(name for name in vars(_kvcacheio) if name.startswith("transfer_kv_"))

_MEM_CACHE = Path(_kvcacheio.__file__).resolve().parents[1]
_POOL_FILES = [
    _MEM_CACHE / "memory_pool_host.py",
    *(_MEM_CACHE / "pool_host").glob("*.py"),
]
# Pools that call kvcacheio ops but refuse unsupported devices on their own terms.
_OWN_DEVICE_GATE = {
    "MambaPoolHost": "refuses every device but CUDA/ROCm/NPU at construction",
    "UnifiedPageEnvelopeHostPool": "raises on non-CUDA/ROCm at first transfer",
}


class _PoolClass:
    def __init__(self, node: ast.ClassDef):
        self.bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        self.calls = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ALL_OPS
        }
        self.own_table = None
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "_kvcacheio_ops"
            ):
                self.own_table = ast.literal_eval(stmt.value)


def _parse_pool_classes() -> dict[str, _PoolClass]:
    classes = {}
    for path in _POOL_FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = _PoolClass(node)
    return classes


_POOLS = _parse_pool_classes()


def _table(name: str):
    """The `_kvcacheio_ops` a pool class declares or inherits, else None."""
    cls = _POOLS[name]
    if cls.own_table is not None:
        return cls.own_table
    for base in cls.bases:
        if base in _POOLS and (table := _table(base)) is not None:
            return table
    return None


def _table_ops(table) -> set[str]:
    return {op for layouts in table.values() for ops in layouts.values() for op in ops}


class TestHostPoolOpTables(unittest.TestCase):
    """Each host pool's `_kvcacheio_ops` must match the ops its dispatch calls."""

    def test_every_called_op_is_declared(self):
        # A pool calling an op its table omits skips the construction check for it.
        checked = 0
        for name, cls in _POOLS.items():
            if not cls.calls or name in _OWN_DEVICE_GATE:
                continue
            table = _table(name)
            with self.subTest(pool=name):
                self.assertIsNotNone(table, "calls kvcacheio ops but has no table")
                self.assertLessEqual(cls.calls, _table_ops(table))
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_no_declared_op_is_stale(self):
        # A table entry the dispatch never calls refuses a pool that would work.
        for name, cls in _POOLS.items():
            if cls.own_table is None:
                continue
            with self.subTest(pool=name):
                self.assertLessEqual(_table_ops(cls.own_table), cls.calls)


class TestKvcacheioGate(unittest.TestCase):
    """The gate binds or refuses per op, as the device's sgl-kernel build allows."""

    PAIR = _table("MHATokenToKVPoolHost")
    SINGLE = _table("MLATokenToKVPoolHost")
    K_ONLY = _table("MHATokenToKOnlyPoolHost")

    def tearDown(self):
        # Rebind against the real device and sgl_kernel for the tests after this one.
        importlib.reload(_kvcacheio)

    def _reload(self, *, device, kvcacheio_ops):
        """Reload the gate as if on `device` with a kvcacheio exposing `kvcacheio_ops`.

        `kvcacheio_ops=None` makes `import sgl_kernel.kvcacheio` fail.
        """
        if kvcacheio_ops is None:
            modules = {"sgl_kernel.kvcacheio": None}
        else:
            fake = types.ModuleType("sgl_kernel.kvcacheio")
            for name in kvcacheio_ops:
                setattr(fake, name, mock.Mock(name=name))
            parent = types.ModuleType("sgl_kernel")
            parent.kvcacheio = fake
            modules = {"sgl_kernel": parent, "sgl_kernel.kvcacheio": fake}
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.multiple(
                srt_utils,
                is_cuda=lambda: device == "cuda",
                is_hip=lambda: device == "hip",
                is_xpu=lambda: device == "xpu",
            ),
        ):
            importlib.reload(_kvcacheio)

    def _require(self, *, io_backend, layout, ops):
        memory = SimpleNamespace(hicache_io_backend=io_backend)
        with mock.patch.object(runtime_context, "get_memory", return_value=memory):
            _kvcacheio.require_kvcacheio_ops(pool="TestPool", layout=layout, ops=ops)

    def _assert_never_reads_server_args(self):
        with mock.patch.object(
            runtime_context, "get_memory", side_effect=AssertionError("read")
        ):
            for table in (self.PAIR, self.SINGLE):
                _kvcacheio.require_kvcacheio_ops(
                    pool="TestPool", layout="layer_first", ops=table
                )

    def test_xpu_complete_build_binds_all_ops(self):
        self._reload(device="xpu", kvcacheio_ops=ALL_OPS)
        self.assertEqual(_kvcacheio.missing_ops, {})
        self.assertIsNone(_kvcacheio.unavailable_reason)
        self._assert_never_reads_server_args()

    def test_xpu_missing_op_refuses_only_pools_that_use_it(self):
        self._reload(
            device="xpu",
            kvcacheio_ops=[n for n in ALL_OPS if n != "transfer_kv_all_layer_mla"],
        )
        self.assertEqual(list(_kvcacheio.missing_ops), ["transfer_kv_all_layer_mla"])
        self.assertIsNone(_kvcacheio.unavailable_reason)

        for io_backend, layouts in self.PAIR.items():
            for layout in layouts:
                self._require(io_backend=io_backend, layout=layout, ops=self.PAIR)
        self._require(io_backend="kernel", layout="page_first", ops=self.SINGLE)
        # K-only backs up layer_first per layer, so it never calls the all-layer op.
        self._require(io_backend="kernel", layout="layer_first", ops=self.K_ONLY)
        with self.assertRaisesRegex(ValueError, "Pick another io backend or layout"):
            self._require(io_backend="kernel", layout="layer_first", ops=self.SINGLE)
        with self.assertRaisesRegex(RuntimeError, "transfer_kv_all_layer_mla"):
            _kvcacheio.transfer_kv_all_layer_mla()

    def test_xpu_without_kvcacheio_refuses_every_pool(self):
        self._reload(device="xpu", kvcacheio_ops=None)
        self.assertEqual(sorted(_kvcacheio.missing_ops), ALL_OPS)
        self.assertIsNotNone(_kvcacheio.unavailable_reason)
        for ops in (self.PAIR, self.SINGLE):
            with self.assertRaisesRegex(ValueError, "Launch without HiCache"):
                self._require(io_backend="direct", layout="layer_first", ops=ops)

    def test_cuda_missing_op_fails_at_import(self):
        with self.assertRaises(ImportError):
            self._reload(
                device="cuda",
                kvcacheio_ops=[n for n in ALL_OPS if n != "transfer_kv_all_layer_mla"],
            )

    def test_device_without_kvcacheio_stubs_without_refusing(self):
        self._reload(device="cpu", kvcacheio_ops=None)
        self.assertEqual(_kvcacheio.missing_ops, {})
        self.assertIsNotNone(_kvcacheio.unavailable_reason)
        self._assert_never_reads_server_args()
        with self.assertRaisesRegex(RuntimeError, "transfer_kv_per_layer"):
            _kvcacheio.transfer_kv_per_layer()


if __name__ == "__main__":
    unittest.main()

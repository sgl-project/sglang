"""CPU-only failure and process-lifetime tests for MHC prewarm coordination."""

import __future__

import ast
import errno
import fcntl
import functools
import logging
import multiprocessing
import os
import queue
import sys
import tempfile
import time
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, patch

# Load the actual functions without importing the GPU kernel module and its
# CUDA/TileLang dependencies. Child processes use the same source with spawn.
_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = _ROOT / "python/sglang/kernels/ops/layernorm/mhc.py"
_NAMES = {"_warn_prewarm_lock_fallback", "_claim_prewarm_bucket", "prewarm_mhc_pre"}
_tree = ast.parse(_SOURCE.read_text())
_tree.body = [
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef) and node.name in _NAMES
]
_namespace = dict(
    functools=functools,
    logger=logging.getLogger("mhc_prewarm_test"),
    contextmanager=contextmanager,
    Path=Path,
    Iterator=Iterator,
    time=time,
)
# Postponed annotations also allow executing the prewarm function without torch.
exec(
    compile(_tree, str(_SOURCE), "exec", flags=__future__.annotations.compiler_flag),
    _namespace,
)
claim = _namespace["_claim_prewarm_bucket"]

# CI registration is loaded directly to keep this test free of GPU dependencies.
_ci_namespace = {"__name__": "_mhc_ci_register"}
_ci_path = _ROOT / "python/sglang/test/ci/ci_register.py"
_ci_module = types.ModuleType(_ci_namespace["__name__"])
sys.modules[_ci_module.__name__] = _ci_module
exec(compile(_ci_path.read_text(), str(_ci_path), "exec"), _ci_module.__dict__)
register_cpu_ci = _ci_module.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def cache_namespace(root):
    module = types.ModuleType("tilelang.env")
    module.env = types.SimpleNamespace(TILELANG_CACHE_DIR=root)
    previous = sys.modules.get("tilelang.env")
    sys.modules["tilelang.env"] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("tilelang.env", None)
        else:
            sys.modules["tilelang.env"] = previous


def contender(root, ready, release, results):
    with cache_namespace(root):
        ready.wait(10)
        with claim("bucket", wait=False) as own:
            results.put(own)
            if own:
                release.wait(10)


def waiter(root, started, results):
    with cache_namespace(root):
        started.set()
        with claim("bucket", wait=True, timeout=5) as own:
            results.put(own)


class TestMhcPrewarmLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.enterContext(cache_namespace(self.root))
        _namespace["_warn_prewarm_lock_fallback"].cache_clear()
        self.ctx = multiprocessing.get_context("spawn")

    def start(self, target, *args):
        process = self.ctx.Process(target=target, args=args)
        process.start()

        def cleanup():
            if process.is_alive():
                process.kill()
            process.join(10)

        self.addCleanup(cleanup)
        return process

    def test_normal_claim_has_one_owner(self):
        ready, release, results = self.ctx.Event(), self.ctx.Event(), self.ctx.Queue()
        processes = [
            self.start(contender, self.root, ready, release, results) for _ in range(4)
        ]
        ready.set()
        self.assertEqual(sum(results.get(timeout=10) for _ in processes), 1)
        release.set()
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)

    def test_waiter_proceeds_after_release(self):
        started, results = self.ctx.Event(), self.ctx.Queue()
        with claim("bucket", wait=False) as own:
            self.assertTrue(own)
            process = self.start(waiter, self.root, started, results)
            self.assertTrue(started.wait(10))
            with self.assertRaises(queue.Empty):
                results.get(timeout=0.2)
        self.assertTrue(results.get(timeout=10))
        process.join(10)
        self.assertEqual(process.exitcode, 0)

    def test_owner_death_releases_lock(self):
        ready, release, results = self.ctx.Event(), self.ctx.Event(), self.ctx.Queue()
        owner = self.start(contender, self.root, ready, release, results)
        ready.set()
        self.assertTrue(results.get(timeout=10))
        started, peer_results = self.ctx.Event(), self.ctx.Queue()
        peer = self.start(waiter, self.root, started, peer_results)
        self.assertTrue(started.wait(10))
        with self.assertRaises(queue.Empty):
            peer_results.get(timeout=0.2)
        owner.kill()
        owner.join(10)
        self.assertTrue(peer_results.get(timeout=10))
        peer.join(10)
        self.assertEqual(peer.exitcode, 0)

    def test_live_owner_wait_is_bounded(self):
        with claim("bucket", wait=False):
            with self.assertLogs("mhc_prewarm_test", level="WARNING"):
                with claim("bucket", wait=True, timeout=0) as own:
                    self.assertTrue(own)

    def assert_fallback(self):
        for wait in (False, True):
            with claim("bucket", wait=wait) as own:
                self.assertTrue(own)

    def test_mkdir_and_open_errors_fail_open_and_warn_once(self):
        for operation in ("mkdir", "open"):
            for code in (
                errno.EROFS,
                errno.EACCES,
                errno.ENOSPC,
                errno.EDQUOT,
                errno.EIO,
            ):
                with self.subTest(operation=operation, errno=code):
                    _namespace["_warn_prewarm_lock_fallback"].cache_clear()
                    target = (
                        "pathlib.Path.mkdir"
                        if operation == "mkdir"
                        else "builtins.open"
                    )
                    with patch(target, side_effect=OSError(code, os.strerror(code))):
                        with self.assertLogs(
                            "mhc_prewarm_test", level="WARNING"
                        ) as logs:
                            self.assert_fallback()
                    self.assertEqual(len(logs.output), 1)

    def test_unwritable_directory(self):
        self.root.chmod(0o555)
        self.addCleanup(self.root.chmod, 0o755)
        if os.access(self.root, os.W_OK):
            self.skipTest("Current user bypasses directory permissions")
        with self.assertLogs("mhc_prewarm_test", level="WARNING"):
            self.assert_fallback()

    def test_flock_error(self):
        with patch.object(
            fcntl, "flock", side_effect=OSError(errno.ENOSYS, "unsupported")
        ):
            with self.assertLogs("mhc_prewarm_test", level="WARNING"):
                self.assert_fallback()

    def test_missing_fcntl(self):
        with patch.dict(sys.modules, {"fcntl": None}):
            with self.assertLogs("mhc_prewarm_test", level="WARNING"):
                self.assert_fallback()

    def test_unlock_and_close_errors_preserve_replay_error(self):
        real_flock, real_open = fcntl.flock, open

        def unlock_error(file, flags):
            if flags == fcntl.LOCK_UN:
                raise OSError(errno.EIO, "unlock failed")
            return real_flock(file, flags)

        def open_with_close_error(*args, **kwargs):
            file = real_open(*args, **kwargs)
            wrapper = Mock(wraps=file)

            def close():
                file.close()
                raise OSError(errno.EIO, "close failed")

            wrapper.close.side_effect = close
            return wrapper

        with patch.object(fcntl, "flock", side_effect=unlock_error):
            with patch("builtins.open", side_effect=open_with_close_error):
                with self.assertLogs("mhc_prewarm_test", level="WARNING") as logs:
                    with claim("bucket", wait=False) as own:
                        self.assertTrue(own)
                    with self.assertRaisesRegex(OSError, "replay failed"):
                        with claim("bucket", wait=False) as own:
                            self.assertTrue(own)
                            raise OSError("replay failed")
                self.assertEqual(len(logs.output), 1)
        with claim("bucket", wait=False) as own:
            self.assertTrue(own)

    def test_separate_tilelang_caches_do_not_contend(self):
        with claim("bucket", wait=False):
            with cache_namespace(self.root / "other"):
                with claim("bucket", wait=False) as own:
                    self.assertTrue(own)

    def test_sglang_cache_does_not_change_lock_scope(self):
        with claim("bucket", wait=False):
            with patch.dict(os.environ, SGLANG_JIT_CACHE_DIR=str(self.root / "other")):
                with claim("bucket", wait=False) as own:
                    self.assertFalse(own)

    def test_prewarm_materializes_all_buckets_after_readonly_failure(self):
        runtime = types.ModuleType("sglang.srt.runtime_context")
        runtime.get_schedule = lambda: types.SimpleNamespace(chunked_prefill_size=128)
        replay = Mock()
        torch = types.SimpleNamespace(
            inference_mode=nullcontext,
            version=types.SimpleNamespace(hip=None),
            cuda=types.SimpleNamespace(
                get_device_properties=lambda device: types.SimpleNamespace(
                    major=10, minor=0
                )
            ),
        )
        residual = Mock(shape=(128, 4, 4096), device="cuda:0")
        residual.new_zeros.side_effect = lambda tokens, *args: tokens
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}):
            with patch.dict(
                _namespace,
                torch=torch,
                mhc_pre=replay,
                get_mhc_pre_token_count_representatives=lambda *args: (64, 128),
            ):
                with patch.object(
                    Path, "mkdir", side_effect=OSError(errno.EROFS, "read-only")
                ):
                    with self.assertLogs("mhc_prewarm_test", level="WARNING"):
                        _namespace["prewarm_mhc_pre"](
                            residual,
                            None,
                            None,
                            None,
                            1e-6,
                            1e-6,
                            1e-6,
                            2.0,
                            20,
                            16,
                            16,
                            None,
                            None,
                        )
        self.assertEqual([call.args[0] for call in replay.call_args_list], [64, 128])


if __name__ == "__main__":
    unittest.main()

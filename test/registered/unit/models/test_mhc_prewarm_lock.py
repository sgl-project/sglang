"""CPU-only tests for MHC prewarm locking and fail-open replay."""

import __future__

import ast
import errno
import fcntl
import functools
import logging
import multiprocessing
import os
import runpy
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

# Execute production functions without importing GPU kernel dependencies.
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/kernels/ops/layernorm/mhc.py"
tree = ast.parse(SOURCE.read_text())
names = {"_warn_prewarm_lock_fallback", "_claim_prewarm_bucket", "prewarm_mhc_pre"}
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
logger = logging.getLogger("mhc_prewarm_test")
scope = dict(
    logger=logger,
    functools=functools,
    time=time,
    contextmanager=contextmanager,
    Path=Path,
)
exec(compile(tree, str(SOURCE), "exec", __future__.annotations.compiler_flag), scope)
claim = scope["_claim_prewarm_bucket"]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMhcPrewarmLock(unittest.TestCase):
    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(root)
        self.enterContext(
            patch.dict(
                sys.modules, {"tilelang.env": Mock(env=Mock(TILELANG_CACHE_DIR=root))}
            )
        )
        scope["_warn_prewarm_lock_fallback"].cache_clear()
        # These CPU-only workers inherit only the test's lightweight imports.
        self.ctx = multiprocessing.get_context("fork")

    def worker(self, wait, release=None):
        reader, writer = self.ctx.Pipe(duplex=False)

        def run():
            writer.send("started")
            with claim("bucket", wait=wait, timeout=5) as own:
                writer.send(own)
                if release is not None:
                    release.wait(10)

        process = self.ctx.Process(target=run)
        process.start()
        writer.close()

        def cleanup():
            if process.is_alive():
                process.kill()
            process.join(10)
            reader.close()

        self.addCleanup(cleanup)
        self.assertTrue(reader.poll(10))
        self.assertEqual(reader.recv(), "started")
        return process, reader

    def test_claim_wait_and_owner_exit(self):
        for killed in (False, True):
            with self.subTest(killed=killed):
                release = self.ctx.Event()
                owner, result = self.worker(False, release)
                self.assertTrue(result.poll(10))
                self.assertTrue(result.recv())
                with claim("bucket", wait=False) as own:
                    self.assertFalse(own)
                peer, result = self.worker(True)
                self.assertFalse(result.poll(0.2))
                if killed:
                    owner.kill()
                else:
                    release.set()
                owner.join(10)
                self.assertFalse(owner.is_alive())
                self.assertTrue(result.poll(10))
                self.assertTrue(result.recv())
                peer.join(10)
                self.assertEqual(peer.exitcode, 0)

    def test_wait_timeout(self):
        with claim("bucket", wait=False):
            with self.assertLogs(logger, level="WARNING"):
                with claim("bucket", wait=True, timeout=0) as own:
                    self.assertTrue(own)

    def test_unwritable_directory(self):
        self.root.chmod(0o555)
        self.addCleanup(self.root.chmod, 0o755)
        if os.access(self.root, os.W_OK):
            self.skipTest("Current user bypasses directory permissions")
        with self.assertLogs(logger, level="WARNING"):
            with claim("bucket", wait=False) as own:
                self.assertTrue(own)

    def test_lock_failures_preserve_full_replay(self):
        replay = Mock()
        residual = Mock(shape=(128, 4, 4096), device="cuda:0")
        residual.new_zeros.side_effect = lambda tokens, *args: tokens
        torch = Mock(inference_mode=nullcontext, version=Mock(hip=None))
        torch.cuda.get_device_properties.return_value = Mock(major=10, minor=0)
        runtime = Mock(get_schedule=lambda: Mock(chunked_prefill_size=128))
        self.enterContext(
            patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime})
        )
        self.enterContext(
            patch.dict(
                scope,
                torch=torch,
                mhc_pre=replay,
                get_mhc_pre_token_count_representatives=lambda *args: (64, 128),
            )
        )

        error = OSError(errno.EROFS, "read-only filesystem")
        file = Mock()
        failures = {
            "mkdir": patch.object(Path, "mkdir", side_effect=error),
            "open": patch("builtins.open", side_effect=error),
            "flock": patch.object(fcntl, "flock", side_effect=error),
            "unlock": patch.object(fcntl, "flock", side_effect=[None, error] * 3),
            "close": patch.object(file, "close", side_effect=error),
            "fcntl": patch.dict(sys.modules, {"fcntl": None}),
        }
        for name, failure in failures.items():
            with (
                self.subTest(failure=name),
                patch("builtins.open", return_value=file),
                patch.object(fcntl, "flock"),
            ):
                scope["_warn_prewarm_lock_fallback"].cache_clear()
                replay.reset_mock()
                with failure, self.assertLogs(logger, level="WARNING") as logs:
                    scope["prewarm_mhc_pre"](residual, *([None] * 12))
                    self.assertEqual(
                        [c.args[0] for c in replay.call_args_list], [64, 128]
                    )
                    with self.assertRaisesRegex(OSError, "replay failed"):
                        with claim("bucket", wait=True):
                            raise OSError("replay failed")
                self.assertEqual(len(logs.output), 1)


if __name__ == "__main__":
    unittest.main()

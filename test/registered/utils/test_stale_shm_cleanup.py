import os
import subprocess
import sys
import unittest
from multiprocessing import shared_memory
from unittest.mock import patch

from sglang.srt.utils.stale_shm_cleanup import (
    _creator_pid,
    cleanup_stale_shm,
    make_shm_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _spawn_dead_pid() -> int:
    """Return a pid that is guaranteed dead (already reaped)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class TestMakeShmName(unittest.TestCase):
    def test_embeds_pid_and_is_unique(self):
        a, b = make_shm_name("mm"), make_shm_name("mm")
        self.assertNotEqual(a, b)
        self.assertEqual(_creator_pid(a), os.getpid())

    def test_creator_pid_parsing(self):
        self.assertEqual(_creator_pid("sgl_shm_mq_1234_abcd1234"), 1234)
        self.assertEqual(_creator_pid("multi_tokenizer_args_5678"), 5678)
        self.assertIsNone(_creator_pid("psm_deadbeef"))
        self.assertIsNone(_creator_pid("sgl_shm_garbage"))
        self.assertIsNone(_creator_pid("multi_tokenizer_args_notanint"))
        # Non-positive pids would make os.kill probe process groups.
        self.assertIsNone(_creator_pid("sgl_shm_mm_-1_abcd1234"))
        self.assertIsNone(_creator_pid("sgl_shm_mm_0_abcd1234"))


@unittest.skipUnless(os.path.isdir("/dev/shm"), "requires /dev/shm")
class TestCleanupStaleShm(unittest.TestCase):
    def _make_segment(self, name: str) -> str:
        shm = shared_memory.SharedMemory(create=True, size=4096, name=name)
        shm.close()
        self.addCleanup(self._unlink_quiet, name)
        return name

    @staticmethod
    def _unlink_quiet(name: str):
        try:
            shared_memory.SharedMemory(name=name).unlink()
        except FileNotFoundError:
            pass

    def test_removes_dead_creator_keeps_live_and_foreign(self):
        dead_pid = _spawn_dead_pid()
        stale = self._make_segment(f"sgl_shm_mm_{dead_pid}_aaaa0000")
        live = self._make_segment(f"sgl_shm_mm_{os.getpid()}_bbbb0000")
        # Anonymous segments from other processes get psm_* names; the sweep
        # must never touch them even when their creator is dead.
        foreign = self._make_segment("psm_testforeign")

        with patch.dict(os.environ, {"SGLANG_IS_IN_CI": "true"}):
            cleanup_stale_shm()

        self.assertFalse(os.path.exists(f"/dev/shm/{stale}"))
        self.assertTrue(os.path.exists(f"/dev/shm/{live}"))
        self.assertTrue(os.path.exists(f"/dev/shm/{foreign}"))

    def test_noop_outside_ci(self):
        dead_pid = _spawn_dead_pid()
        stale = self._make_segment(f"sgl_shm_mq_{dead_pid}_cccc0000")

        with patch.dict(os.environ, {"SGLANG_IS_IN_CI": "false"}):
            cleanup_stale_shm()

        self.assertTrue(os.path.exists(f"/dev/shm/{stale}"))

    def test_shm_ring_buffer_uses_reclaimable_name(self):
        """Bind the production call site: ShmRingBuffer must emit a
        pid-stamped name, or the leak this module fixes silently returns."""
        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            ShmRingBuffer,
        )

        buf = ShmRingBuffer(1, 64, 1)
        try:
            self.assertEqual(_creator_pid(buf.shared_memory.name), os.getpid())
        finally:
            buf.shared_memory.close()
            buf.shared_memory.unlink()

    def test_run_by_path_without_sglang_importable(self):
        """ci_install_dependency.sh runs the module by file path before
        sglang is installed; it must work with an empty PYTHONPATH."""
        import sglang.srt.utils.stale_shm_cleanup as mod

        dead_pid = _spawn_dead_pid()
        stale = self._make_segment(f"sgl_shm_mm_{dead_pid}_eeee0000")

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["SGLANG_IS_IN_CI"] = "true"
        result = subprocess.run(
            [sys.executable, mod.__file__],
            env=env,
            capture_output=True,
            text=True,
            cwd="/",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(os.path.exists(f"/dev/shm/{stale}"))

    def test_multi_tokenizer_args_cleanup(self):
        dead_pid = _spawn_dead_pid()
        stale = self._make_segment(f"multi_tokenizer_args_{dead_pid}")

        with patch.dict(os.environ, {"SGLANG_IS_IN_CI": "true"}):
            cleanup_stale_shm()

        self.assertFalse(os.path.exists(f"/dev/shm/{stale}"))


if __name__ == "__main__":
    unittest.main()

import mmap
import os
import subprocess
import sys
import time
import unittest
from multiprocessing import shared_memory
from unittest.mock import patch

from sglang.srt.utils.stale_shm_cleanup import (
    _ORPHAN_MIN_AGE_S,
    _creator_pid,
    _live_shm_paths,
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
        self.assertEqual(_creator_pid("torch_2952_1126906123"), 2952)
        self.assertIsNone(_creator_pid("psm_deadbeef"))
        self.assertIsNone(_creator_pid("sgl_shm_garbage"))
        self.assertIsNone(_creator_pid("multi_tokenizer_args_notanint"))
        self.assertIsNone(_creator_pid("torch_notanint_1"))
        self.assertIsNone(_creator_pid("torch_1234"))
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

    def _make_raw_file(self, name: str, age_s: int = 0) -> str:
        """Create a plain /dev/shm file (orphan families are not created via
        shared_memory), optionally backdating its mtime."""
        path = f"/dev/shm/{name}"
        with open(path, "wb") as f:
            f.write(b"\0" * 4096)
        if age_s:
            old = time.time() - age_s
            os.utime(path, (old, old))
        self.addCleanup(self._unlink_path_quiet, path)
        return path

    @staticmethod
    def _unlink_path_quiet(path: str):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def test_orphan_family_sweep(self):
        """One sweep must remove aged unreferenced orphan-family files while
        keeping recent, still-mapped, and unknown-prefix ones."""
        if _live_shm_paths() is None:
            self.skipTest("cannot inspect every same-uid process")
        old = _ORPHAN_MIN_AGE_S + 60
        stale = [
            self._make_raw_file("sglang_loads_test_deadbeef.shm", age_s=old),
            self._make_raw_file("cuda.shm.0.deadbeef.1", age_s=old),
            self._make_raw_file("nccl-testonly", age_s=old),
            self._make_raw_file("sem.loky-0-testonly", age_s=old),
        ]
        recent = self._make_raw_file("sglang_loads_test_cafecafe.shm")
        unknown = self._make_raw_file("unknown_family_file", age_s=old)
        # Held via mmap only (fd closed) so the /proc/*/maps scan is what
        # must protect it.
        mapped = self._make_raw_file("sglang_loads_test_beefbeef.shm", age_s=old)
        fd = os.open(mapped, os.O_RDWR)
        try:
            held = mmap.mmap(fd, 4096)
        finally:
            os.close(fd)

        try:
            with patch.dict(os.environ, {"SGLANG_IS_IN_CI": "true"}):
                cleanup_stale_shm()
            for path in stale:
                self.assertFalse(os.path.exists(path), path)
            for path in (recent, unknown, mapped):
                self.assertTrue(os.path.exists(path), path)
        finally:
            held.close()


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from multiprocessing import shared_memory

from sglang.srt.utils.stale_shm_cleanup import (
    _creator_pid,
    make_shm_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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
        self.addCleanup(self._unlink_quiet, f"/dev/shm/{name}")
        return name

    @staticmethod
    def _unlink_quiet(path: str):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

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

    def _make_raw_file(self, name: str) -> str:
        """Orphan families are plain files, not shared_memory segments."""
        path = f"/dev/shm/{name}"
        with open(path, "wb") as f:
            f.write(b"\0" * 4096)
        self.addCleanup(self._unlink_quiet, path)
        return path


if __name__ == "__main__":
    unittest.main()

import errno
import os
import tempfile
import unittest

from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestIoUringReader(CustomTestCase):
    def test_reads_pages_through_python_binding(self):
        try:
            reader_type = load_rust_extension(
                "sglang.srt.rust_extensions._storage"
            ).IoUringReader
            reader = reader_type(1, 2, 4096)
        except OSError as error:
            if error.errno in (errno.ENOSYS, errno.EPERM):
                raise unittest.SkipTest(f"io_uring is unavailable: {error}") from error
            raise

        with tempfile.NamedTemporaryFile() as file:
            file.write(bytes([0x15]) * 4096)
            file.write(bytes([0xA6]) * 4096)
            file.flush()
            os.fsync(file.fileno())
            pages = reader.read_pages([file.fileno(), file.fileno()], [0, 4096])

        self.assertEqual(pages[0], bytes([0x15]) * 4096)
        self.assertEqual(pages[1], bytes([0xA6]) * 4096)


if __name__ == "__main__":
    unittest.main()

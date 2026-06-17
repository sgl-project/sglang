import shutil
import tempfile
import unittest
import uuid

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.mem_cache.memory_pool_host import SharedMemoryHostTensorAllocator
from sglang.test.test_utils import CustomTestCase


class TestSharedMemoryHostTensorAllocator(CustomTestCase):
    def test_same_prefix_maps_same_storage(self):
        directory = tempfile.mkdtemp()
        prefix = f"allocator_unit_{uuid.uuid4().hex}"
        try:
            writer = SharedMemoryHostTensorAllocator(
                group=None,
                name_prefix=prefix,
                is_writer=True,
                directory=directory,
                unlink_after_attach=False,
            )
            reader = SharedMemoryHostTensorAllocator(
                group=None,
                name_prefix=prefix,
                is_writer=False,
                directory=directory,
                unlink_after_attach=False,
            )

            a = writer.allocate((8,), torch.int32, "cpu")
            b = reader.allocate((8,), torch.int32, "cpu")

            a.copy_(torch.arange(8, dtype=torch.int32))
            self.assertTrue(torch.equal(b, torch.arange(8, dtype=torch.int32)))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

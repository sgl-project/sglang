"""GPU storage registration must not inherit MoRI's CPU location default."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def buffer(ptr, size, device=0, kind="cuda"):
    storage = SimpleNamespace(data_ptr=lambda: ptr, nbytes=lambda: size)
    return SimpleNamespace(
        device=SimpleNamespace(type=kind, index=device),
        untyped_storage=lambda: storage,
    )


class TestDeviceRegistration(CustomTestCase):
    def register(self, buffers, client):
        native = ModuleType("mori.io")
        native.MemoryLocationType = SimpleNamespace(GPU=object())
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker.storage = SimpleNamespace(client=client)
        linker.pools = {"kv": SimpleNamespace(get_hybrid_pool_buffer=lambda: buffers)}
        with patch.dict(sys.modules, {"mori.io": native}):
            linker._register_buffers()
        return linker, native.MemoryLocationType.GPU

    def test_allocation_views_register_once_with_gpu_location_and_tensor_device(self):
        client = SimpleNamespace(register_memory=Mock(return_value=True))
        linker, gpu = self.register(
            [buffer(4096, 8192, 3), buffer(4096, 8192, 3), buffer(16384, 1024, 1)],
            client,
        )
        self.assertEqual(
            client.register_memory.call_args_list,
            [unittest.mock.call(4096, 8192, gpu, 3), unittest.mock.call(16384, 1024, gpu, 1)],
        )
        self.assertEqual(linker._registered, [(4096, 8192), (16384, 1024)])

    def test_failure_is_not_retried_with_cpu_defaults(self):
        client = SimpleNamespace(register_memory=Mock(return_value=False))
        with self.assertRaisesRegex(RuntimeError, "Failed to register"):
            self.register([buffer(4096, 8192, 2)], client)
        self.assertEqual(client.register_memory.call_count, 1)
        self.assertEqual(len(client.register_memory.call_args.args), 4)

    def test_host_buffer_cannot_be_registered_as_device(self):
        client = SimpleNamespace(register_memory=Mock(return_value=True))
        with self.assertRaisesRegex(ValueError, "device KV buffers"):
            self.register([buffer(4096, 8192, None, "cpu")], client)
        client.register_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _install_kvcacheio_stub():
    kvcacheio_stub = types.ModuleType("sgl_kernel.kvcacheio")
    for name in [
        "transfer_kv_all_layer",
        "transfer_kv_all_layer_direct_lf_pf",
        "transfer_kv_all_layer_lf_pf",
        "transfer_kv_all_layer_lf_ph",
        "transfer_kv_all_layer_mla",
        "transfer_kv_all_layer_mla_lf_pf",
        "transfer_kv_direct",
        "transfer_kv_per_layer",
        "transfer_kv_per_layer_direct_pf_lf",
        "transfer_kv_per_layer_mla",
        "transfer_kv_per_layer_mla_pf_lf",
        "transfer_kv_per_layer_pf_lf",
        "transfer_kv_per_layer_ph_lf",
    ]:
        setattr(kvcacheio_stub, name, lambda *args, **kwargs: None)
    sgl_kernel_stub = types.ModuleType("sgl_kernel")
    sgl_kernel_stub.kvcacheio = kvcacheio_stub
    sys.modules.setdefault("sgl_kernel", sgl_kernel_stub)
    sys.modules.setdefault("sgl_kernel.kvcacheio", kvcacheio_stub)


def _memory_pool_host_modules():
    _install_kvcacheio_stub()
    from sglang.srt.mem_cache.memory_pool_host import (  # noqa: PLC0415
        HostKVCache,
        MHATokenToKVPoolHost,
        bounded_pretouch_host_tensor,
        make_prealloc_host_kv_pool,
    )

    return (
        HostKVCache,
        MHATokenToKVPoolHost,
        bounded_pretouch_host_tensor,
        make_prealloc_host_kv_pool,
    )


def _fake_mha_pool():
    pool = object.__new__(MHATokenToKVPool)
    pool.store_dtype = torch.float32
    pool.dtype = torch.float32
    pool.size = 4
    pool.start_layer = 0
    pool.end_layer = 1
    pool.layer_num = 2
    pool.head_num = 2
    pool.head_dim = 4
    pool.device = "cpu"
    return pool


def _fake_mla_pool():
    pool = object.__new__(MLATokenToKVPool)
    pool.store_dtype = torch.float32
    pool.dtype = torch.float32
    pool.size = 4
    pool.start_layer = 0
    pool.end_layer = 1
    pool.layer_num = 2
    pool.kv_lora_rank = 2
    pool.qk_rope_head_dim = 2
    pool.device = "cpu"
    return pool


class TestHiCacheHostPrealloc(unittest.TestCase):
    def test_default_path_allocates_eagerly(self):
        _, MHATokenToKVPoolHost, _, _ = _memory_pool_host_modules()
        host_pool = MHATokenToKVPoolHost(
            _fake_mha_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
        )

        self.assertTrue(host_pool._buffer_ready)
        self.assertEqual(tuple(host_pool.kv_buffer.shape), (2, 2, 9, 2, 4))
        self.assertEqual(host_pool.k_data_ptrs.numel(), 2)
        self.assertEqual(host_pool.v_data_ptrs.numel(), 2)

    def test_mha_host_pool_can_allocate_in_background(self):
        _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        host_pool = make_prealloc_host_kv_pool(
            _fake_mha_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="page_first",
        )
        host_pool.pin_memory = False

        self.assertIsNone(host_pool.kv_buffer)
        host_pool.start_kv_buffer_allocation()
        host_pool.wait_kv_buffer_ready()

        self.assertTrue(host_pool._buffer_ready)
        self.assertEqual(tuple(host_pool.kv_buffer.shape), (2, 9, 2, 2, 4))
        self.assertEqual(len(host_pool.k_data_refs), 2)

    def test_mla_host_pool_can_allocate_in_background(self):
        _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        host_pool = make_prealloc_host_kv_pool(
            _fake_mla_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
        )
        host_pool.pin_memory = False

        host_pool.start_kv_buffer_allocation()
        host_pool.wait_kv_buffer_ready()

        self.assertTrue(host_pool._buffer_ready)
        self.assertEqual(tuple(host_pool.kv_buffer.shape), (2, 9, 1, 4))
        self.assertEqual(len(host_pool.data_refs), 2)

    def test_background_allocation_exception_is_raised_from_wait(self):
        HostKVCache, _, _, _ = _memory_pool_host_modules()

        class FailingHostPool(HostKVCache):
            def get_size_per_token(self):
                return 1

            def init_kv_buffer(self):
                raise RuntimeError("intentional allocation failure")

            def load_to_device_per_layer(self, *args, **kwargs):
                raise NotImplementedError

            def backup_from_device_all_layer(self, *args, **kwargs):
                raise NotImplementedError

            def get_data_page(self, *args, **kwargs):
                raise NotImplementedError

            def get_dummy_flat_data_page(self):
                raise NotImplementedError

            def set_from_flat_data_page(self, *args, **kwargs):
                raise NotImplementedError

        device_pool = SimpleNamespace(
            store_dtype=torch.uint8,
            size=1,
            start_layer=0,
            end_layer=0,
            device="cpu",
        )
        host_pool = FailingHostPool(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            defer_alloc=True,
        )

        host_pool.start_kv_buffer_allocation()
        with self.assertRaisesRegex(RuntimeError, "intentional allocation failure"):
            host_pool.wait_kv_buffer_ready()

    def test_prealloc_factory_rejects_unsupported_pool(self):
        _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        dsa_pool = object.__new__(DSATokenToKVPool)

        with self.assertRaisesRegex(ValueError, "DSA host pools"):
            make_prealloc_host_kv_pool(
                dsa_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=1,
                layout="layer_first",
            )

    def test_bounded_pretouch_touches_buffer(self):
        _, _, bounded_pretouch_host_tensor, _ = _memory_pool_host_modules()
        buffer = torch.ones((1024,), dtype=torch.float32)

        bounded_pretouch_host_tensor(buffer, threading.Event())

        self.assertEqual(buffer[0].item(), 0.0)

    def test_wait_stops_background_pretouch_before_join(self):
        _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        host_pool = make_prealloc_host_kv_pool(
            _fake_mha_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="page_first",
        )
        host_pool.pin_memory = True
        host_pool.device_pool.device = "cuda"

        observed_stop = []

        def wait_for_stop(buffer, stop_event, tag=""):
            observed_stop.append(stop_event.wait(timeout=2.0))

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.bounded_pretouch_host_tensor",
            side_effect=wait_for_stop,
        ), patch(
            "sglang.srt.mem_cache.memory_pool_host.cuda_host_register_tensor",
            return_value=None,
        ):
            host_pool.start_kv_buffer_allocation()
            host_pool.device_pool.device = "cpu"
            host_pool.wait_kv_buffer_ready()

        self.assertEqual(observed_stop, [True])
        self.assertTrue(host_pool._buffer_ready)


if __name__ == "__main__":
    unittest.main()

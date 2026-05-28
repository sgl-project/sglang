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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _install_kvcacheio_stub():
    # memory_pool_host imports kvcacheio kernels at module import time. These
    # tests do not exercise transfer kernels, so install a small stub to keep
    # CPU CI and no-driver containers focused on the host-pool lifecycle.
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
        setattr(
            kvcacheio_stub,
            name,
            lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError),
        )
    sgl_kernel_stub = types.ModuleType("sgl_kernel")
    sgl_kernel_stub.kvcacheio = kvcacheio_stub
    sys.modules.setdefault("sgl_kernel", sgl_kernel_stub)
    sys.modules.setdefault("sgl_kernel.kvcacheio", kvcacheio_stub)


def _memory_pool_host_modules():
    _install_kvcacheio_stub()
    from sglang.srt.mem_cache.memory_pool_host import (  # noqa: PLC0415
        HostKVCache,
        HostTensorAllocator,
        MHATokenToKVPoolHost,
        alloc_with_host_register,
        defer_cuda_host_register,
        make_prealloc_host_kv_pool,
    )

    return (
        HostKVCache,
        HostTensorAllocator,
        MHATokenToKVPoolHost,
        alloc_with_host_register,
        defer_cuda_host_register,
        make_prealloc_host_kv_pool,
    )


def _host_pool_host_classes():
    _install_kvcacheio_stub()
    from sglang.srt.mem_cache.memory_pool_host import (  # noqa: PLC0415
        MHATokenToKVPoolHost,
        MLATokenToKVPoolHost,
    )

    return MHATokenToKVPoolHost, MLATokenToKVPoolHost


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
    def test_mha_host_pool_default_path_allocates_eagerly(self):
        _, _, MHATokenToKVPoolHost, _, _, _ = _memory_pool_host_modules()
        host_pool = MHATokenToKVPoolHost(
            _fake_mha_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
        )

        self.assertTrue(host_pool._buffer_ready)
        self.assertIsNotNone(host_pool.kv_buffer)
        self.assertEqual(tuple(host_pool.kv_buffer.shape), (2, 2, 9, 2, 4))
        self.assertEqual(host_pool.k_data_ptrs.numel(), 2)
        self.assertEqual(host_pool.v_data_ptrs.numel(), 2)

        # Idempotent after success.
        kv_buffer = host_pool.kv_buffer
        host_pool.wait_kv_buffer_ready()
        self.assertIs(host_pool.kv_buffer, kv_buffer)

    def test_mha_host_pool_can_allocate_in_background(self):
        _, _, _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        host_pool = make_prealloc_host_kv_pool(
            _fake_mha_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="page_first",
        )
        host_pool.pin_memory = False

        self.assertIsNone(host_pool.kv_buffer)
        self.assertFalse(host_pool._buffer_ready)

        host_pool.start_kv_buffer_allocation()
        host_pool.wait_kv_buffer_ready()

        self.assertTrue(host_pool._buffer_ready)
        self.assertIsNotNone(host_pool.kv_buffer)
        self.assertEqual(tuple(host_pool.kv_buffer.shape), (2, 9, 2, 2, 4))
        self.assertEqual(len(host_pool.k_data_refs), 2)
        self.assertEqual(host_pool.k_data_ptrs.device.type, "cpu")

    def test_mla_host_pool_can_allocate_in_background(self):
        # RFC v1 explicitly supports MLA host pools; mirror the MHA async path
        # to catch breakage in MLATokenToKVPoolHost.init_kv_buffer /
        # _post_alloc_setup that would otherwise slip past the MHA-only tests.
        _, _, _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        host_pool = make_prealloc_host_kv_pool(
            _fake_mla_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout="layer_first",
        )
        host_pool.pin_memory = False

        self.assertIsNone(host_pool.kv_buffer)
        self.assertFalse(host_pool._buffer_ready)

        host_pool.start_kv_buffer_allocation()
        host_pool.wait_kv_buffer_ready()

        self.assertTrue(host_pool._buffer_ready)
        self.assertIsNotNone(host_pool.kv_buffer)
        # layer_first MLA dims: (layer_num, size, 1, kv_lora_rank + qk_rope_head_dim).
        self.assertEqual(tuple(host_pool.kv_buffer.shape), (2, 9, 1, 4))
        self.assertEqual(len(host_pool.data_refs), 2)
        self.assertEqual(host_pool.data_ptrs.device.type, "cpu")

    def test_background_allocation_exception_is_raised_from_wait(self):
        HostKVCache, _, _, _, _, _ = _memory_pool_host_modules()

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
        self.assertFalse(host_pool._buffer_ready)

    def test_deferred_host_register_pretouches_host_tensor(self):
        (
            _,
            HostTensorAllocator,
            _,
            alloc_with_host_register,
            defer_cuda_host_register,
            _,
        ) = _memory_pool_host_modules()

        stop_event = threading.Event()
        touched = []
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.bounded_pretouch_host_tensor",
            side_effect=lambda buffer: touched.append(buffer),
        ):
            with defer_cuda_host_register(stop_event):
                buffer = alloc_with_host_register(
                    (8,), torch.float32, "cpu", True, HostTensorAllocator()
                )

        self.assertEqual(touched, [buffer])

    def test_wait_stops_background_pretouch_before_join(self):
        HostKVCache, _, _, _, _, _ = _memory_pool_host_modules()

        class StopAwareHostPool(HostKVCache):
            def __init__(self, *args, **kwargs):
                self.stop_observed = False
                super().__init__(*args, **kwargs)

            def get_size_per_token(self):
                return 1

            def init_kv_buffer(self):
                stop_event = self._alloc_stop_event
                self.stop_observed = stop_event.wait(timeout=2.0)
                return torch.empty((1,), dtype=torch.uint8)

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
        host_pool = StopAwareHostPool(
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
        host_pool.wait_kv_buffer_ready()

        self.assertTrue(host_pool.stop_observed)
        self.assertTrue(host_pool._buffer_ready)

    def test_prealloc_factory_rejects_dsa_for_first_pr(self):
        _, _, _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        dsa_pool = object.__new__(DSATokenToKVPool)
        with self.assertRaisesRegex(ValueError, "DSA host pools"):
            make_prealloc_host_kv_pool(
                dsa_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=1,
                layout="layer_first",
            )

    def test_prealloc_factory_rejects_unknown_pool_type(self):
        # Cover the generic fallthrough in make_prealloc_host_kv_pool that
        # rejects anything not MHA/MLA/DSA (Mamba, V4 paged, sidecars, etc.).
        _, _, _, _, _, make_prealloc_host_kv_pool = _memory_pool_host_modules()
        unknown_pool = SimpleNamespace(size=1, page_size=1)
        with self.assertRaisesRegex(ValueError, "does not support host pool type"):
            make_prealloc_host_kv_pool(
                unknown_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=1,
                layout="layer_first",
            )

    def test_hiradix_rejects_mismatched_mha_prealloc_pool(self):
        _install_kvcacheio_stub()
        _, MLATokenToKVPoolHost = _host_pool_host_classes()
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache  # noqa: PLC0415

        params = SimpleNamespace(
            enable_metrics=False,
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: _fake_mha_pool()
            ),
            prealloc_host_kv_pool=object.__new__(MLATokenToKVPoolHost),
        )

        with self.assertRaisesRegex(
            ValueError, "Mismatched HiCache prealloc host pool .* for MHA KV pool"
        ):
            HiRadixCache(params, SimpleNamespace())

    def test_hiradix_rejects_mismatched_mla_prealloc_pool(self):
        _install_kvcacheio_stub()
        MHATokenToKVPoolHost, _ = _host_pool_host_classes()
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache  # noqa: PLC0415

        params = SimpleNamespace(
            enable_metrics=False,
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: _fake_mla_pool()
            ),
            prealloc_host_kv_pool=object.__new__(MHATokenToKVPoolHost),
        )

        with self.assertRaisesRegex(
            ValueError, "Mismatched HiCache prealloc host pool .* for MLA KV pool"
        ):
            HiRadixCache(params, SimpleNamespace())

    def test_unified_kv_only_rejects_mismatched_prealloc_pool(self):
        _install_kvcacheio_stub()
        _, MLATokenToKVPoolHost = _host_pool_host_classes()
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (  # noqa: PLC0415
            build_kv_only_stack,
        )

        params = SimpleNamespace(
            prealloc_host_kv_pool=object.__new__(MLATokenToKVPoolHost),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Mismatched HiCache prealloc host pool .* for unified KV-only path",
        ):
            build_kv_only_stack(
                params=params,
                server_args=SimpleNamespace(),
                kv_pool=_fake_mha_pool(),
                full_layer_mapping={0: 0},
                page_size=1,
                tp_group=None,
                load_cache_event=None,
                storage_backend=None,
                use_mla=False,
            )

    def test_bounded_pretouch_skips_when_stop_event_missing(self):
        # Eager-path callers never enter defer_cuda_host_register, so tls has
        # no stop_event and pre-touch must be a no-op rather than crashing.
        _install_kvcacheio_stub()
        import sglang.srt.mem_cache.memory_pool_host as mph  # noqa: PLC0415

        buffer = torch.ones((16,), dtype=torch.float32)
        mph.bounded_pretouch_host_tensor(buffer)
        self.assertTrue(torch.all(buffer == 1.0))

    def test_bounded_pretouch_handles_empty_buffer(self):
        _install_kvcacheio_stub()
        import sglang.srt.mem_cache.memory_pool_host as mph  # noqa: PLC0415

        buffer = torch.empty((0,), dtype=torch.float32)
        with mph.defer_cuda_host_register(threading.Event()):
            # Must not raise or attempt strided indexing on a zero-element tensor.
            mph.bounded_pretouch_host_tensor(buffer)
        self.assertEqual(buffer.numel(), 0)

    def test_bounded_pretouch_stops_on_event(self):
        # If wait_kv_buffer_ready sets the stop_event before the worker reaches
        # its next chunk, the loop must break before zeroing anything.
        _install_kvcacheio_stub()
        import sglang.srt.mem_cache.memory_pool_host as mph  # noqa: PLC0415

        buffer = torch.ones((1024,), dtype=torch.float32)
        stop = threading.Event()
        stop.set()
        with mph.defer_cuda_host_register(stop):
            mph.bounded_pretouch_host_tensor(buffer)
        self.assertTrue(torch.all(buffer == 1.0))

    def test_bounded_pretouch_touches_buffer_when_event_unset(self):
        # Control case: without a stop signal, the worker must actually fault
        # in pages (at least the first stride element gets zeroed).
        _install_kvcacheio_stub()
        import sglang.srt.mem_cache.memory_pool_host as mph  # noqa: PLC0415

        buffer = torch.ones((1024,), dtype=torch.float32)
        with mph.defer_cuda_host_register(threading.Event()):
            mph.bounded_pretouch_host_tensor(buffer)
        self.assertEqual(buffer[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()

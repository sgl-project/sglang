"""Startup and runtime support boundaries for DCP L3 storage."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    resolve_hicache_dcp_compatibility,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _args(**changes):
    options = dict(
        model_path="dummy",
        tp_size=4,
        dcp_size=2,
        enable_hierarchical_cache=True,
        hicache_ratio=2,
        hicache_storage_backend="file",
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        dtype="bfloat16",
        kv_cache_dtype="auto",
        hicache_write_policy="write_through",
        hicache_storage_prefetch_policy="wait_complete",
    )
    options.update(changes)
    return ServerArgs(**options)


class TestDcpStorageGuards(unittest.TestCase):
    def test_mooncake_startup_preserves_dcp_layer_first(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for io in ("kernel", "direct"):
                for backend in ("mooncake", None):
                    with self.subTest(io=io, backend=backend):
                        args = _args(
                            hicache_storage_backend=backend,
                            hicache_mem_layout="layer_first",
                            hicache_io_backend=io,
                        )
                        handle_hicache(args)
                        self.assertEqual(
                            resolving_view(args).hicache_mem_layout, "layer_first"
                        )

    def test_other_storage_layout_defaults_are_unchanged(self):
        for backend, dcp in (("mooncake", 1), ("npu_memcache", 2)):
            for io, expected in (
                ("kernel", "page_first"),
                ("direct", "page_first_direct"),
            ):
                with (
                    self.subTest(backend=backend, dcp=dcp, io=io),
                    mock.patch(
                        "sglang.srt.arg_groups.hicache_hook.use_mla_backend",
                        return_value=True,
                    ),
                ):
                    args = _args(
                        dcp_size=dcp,
                        hicache_storage_backend=backend,
                        hicache_mem_layout="layer_first",
                        hicache_io_backend=io,
                    )
                    handle_hicache(args)
                    self.assertEqual(resolving_view(args).hicache_mem_layout, expected)

    def test_supported_topologies(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for tp, dcp in ((2, 2), (4, 2), (4, 4)):
                with self.subTest(tp=tp, dcp=dcp):
                    resolve_hicache_dcp_compatibility(_args(tp_size=tp, dcp_size=dcp))
            for dtype in ("bf16", "bfloat16"):
                with self.subTest(kv_cache_dtype=dtype):
                    resolve_hicache_dcp_compatibility(_args(kv_cache_dtype=dtype))

    def test_inherits_mla_hicache_options(self):
        cases = (
            dict(hicache_mem_layout="layer_first"),
            dict(hicache_mem_layout="page_first_direct", hicache_io_backend="direct"),
            dict(dtype="float16"),
            dict(kv_cache_dtype="fp8_e4m3"),
            dict(hicache_write_policy="write_back"),
            dict(hicache_write_policy="write_through_selective"),
            dict(hicache_storage_prefetch_policy="best_effort"),
            dict(hicache_storage_prefetch_policy="timeout"),
            dict(hicache_host_memory_mode="buffer_only"),
            dict(pp_size=2),
            dict(dp_size=2),
            dict(attn_cp_size=2),
            dict(dp_size=2, enable_dp_attention=True),
            dict(speculative_algorithm="DSPARK"),
            dict(disaggregation_mode="prefill"),
        )
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in cases:
                with self.subTest(options=options):
                    resolve_hicache_dcp_compatibility(_args(**options))

    def test_keeps_existing_dcp_constraints(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in (
                dict(enable_hisparse=True),
                dict(enable_lmcache=True),
                dict(speculative_algorithm="EAGLE"),
            ):
                with (
                    self.subTest(options=options),
                    self.assertRaises(NotImplementedError),
                ):
                    resolve_hicache_dcp_compatibility(_args(**options))

    def test_keeps_existing_mla_constraint(self):
        with (
            mock.patch(
                "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=False
            ),
            self.assertRaisesRegex(NotImplementedError, "only supported for MLA"),
        ):
            resolve_hicache_dcp_compatibility(_args())

    def test_startup_and_runtime_share_controller_guards(self):
        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.kv_buffer = object()
        host.layout = "page_first"
        host.device_pool = SimpleNamespace()
        device = SimpleNamespace(
            device="cpu", layer_num=1, register_layer_transfer_counter=mock.Mock()
        )
        allocator = mock.Mock()
        allocator.get_kvcache.return_value = device
        for backend, pool, message in (
            ("hf3fs", host, "requires file or Mooncake storage"),
            ("file", object(), "requires one materialized MLA host pool"),
            ("mooncake", object(), "requires one materialized MLA host pool"),
        ):
            entry = SimpleNamespace(host_pool=pool)
            group = SimpleNamespace(anchor_entry=entry, entries=[entry])
            with (
                self.subTest(backend=backend, message=message),
                mock.patch(
                    "sglang.srt.managers.cache_controller.get_parallel",
                    return_value=SimpleNamespace(attn_dcp_size=2),
                ),
                mock.patch("sglang.srt.managers.cache_controller.LayerDoneCounter"),
                mock.patch("sglang.srt.managers.cache_controller.L2TransferEngine"),
                mock.patch.object(
                    HybridCacheController, "_start_storage_threads"
                ) as start,
                mock.patch.object(
                    HybridCacheController, "_stop_storage_threads"
                ) as stop,
            ):
                # Startup reaches the same controller attach used by the API.
                with self.assertRaisesRegex(NotImplementedError, message):
                    HybridCacheController(
                        allocator,
                        group,
                        128,
                        None,
                        threading.Event(),
                        storage_backend=backend,
                    )
                controller = HybridCacheController(
                    allocator, group, 128, None, threading.Event()
                )
                cache = SimpleNamespace(
                    cache_controller=controller,
                    enable_storage=False,
                    sliding_window_size=None,
                )
                ok, reason = StorageAttachment(cache).attach(backend)
                self.assertFalse(ok)
                self.assertIn(message, reason)
                self.assertFalse(controller.enable_storage)
                start.assert_not_called()
                stop.assert_not_called()

    def test_runtime_policy_updates_use_existing_validation(self):
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(storage_backend_type="file"),
            enable_storage=True,
        )
        attachment = StorageAttachment(cache)
        attachment._apply_policies = mock.Mock()
        for write in ("write_back", "write_through", "write_through_selective"):
            for prefetch in ("best_effort", "timeout", "wait_complete"):
                with self.subTest(write=write, prefetch=prefetch):
                    ok, reason = attachment.attach(
                        "file",
                        hicache_write_policy=write,
                        hicache_storage_prefetch_policy=prefetch,
                    )
                    self.assertTrue(ok, reason)
                    attachment._apply_policies.assert_called_with(prefetch, write)
        attachment._apply_policies.reset_mock()
        ok, _ = attachment.attach("file", hicache_write_policy="invalid")
        self.assertFalse(ok)
        attachment._apply_policies.assert_not_called()


if __name__ == "__main__":
    unittest.main()

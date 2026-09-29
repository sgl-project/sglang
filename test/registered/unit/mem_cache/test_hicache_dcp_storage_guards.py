"""Startup and runtime support boundaries for DCP storage."""

import unittest
from types import SimpleNamespace
from unittest import mock

from test_hicache_dcp_host_pool import _make_host_pool

from sglang.srt.arg_groups.hicache_hook import (
    resolve_hicache_dcp_compatibility,
    validate_hicache_dcp_storage,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _args(**changes):
    options = dict(
        model_path="dummy",
        tp_size=4,
        dcp_size=2,
        enable_hierarchical_cache=True,
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


class TestDcpStorageGuards(CustomTestCase):
    def test_pd_storage_stays_gated_until_transfer_support(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for role in ("prefill", "decode"):
                with (
                    self.subTest(role=role),
                    self.assertRaisesRegex(NotImplementedError, "aggregated serving"),
                ):
                    resolve_hicache_dcp_compatibility(_args(disaggregation_mode=role))

    def test_dynamic_sidecar_cannot_bypass_single_pool_storage_guard(self):
        for dcp in (1, 2):
            with self.subTest(dcp=dcp):
                pool = _make_host_pool(0, dcp_size=dcp, layout="page_first")
                anchor = PoolEntry(
                    PoolName.KV,
                    pool,
                    pool.device_pool,
                    lambda x: x,
                    is_primary_index_anchor=True,
                )
                sidecar = PoolEntry(PoolName.DRAFT, pool, pool.device_pool, lambda x: x)
                controller = HybridCacheController.__new__(HybridCacheController)
                controller.mem_pool_host = HostPoolGroup([anchor])
                controller.enable_storage = True
                controller.storage_config = SimpleNamespace(dcp_size=dcp)
                controller.storage_backend = mock.Mock()
                controller.extra_host_mem_release_queues = {}
                if dcp > 1:
                    with self.assertRaisesRegex(
                        NotImplementedError, "one materialized MLA"
                    ):
                        controller.register_host_pool_entry(sidecar)
                    self.assertEqual(controller.mem_pool_host.entries, [anchor])
                    self.assertEqual(controller.extra_host_mem_release_queues, {})
                else:
                    controller.register_host_pool_entry(sidecar)
                    self.assertEqual(
                        controller.mem_pool_host.entries, [anchor, sidecar]
                    )
                    self.assertIn(
                        PoolName.DRAFT, controller.extra_host_mem_release_queues
                    )

    def test_startup_and_attach_preserve_supported_options(self):
        cases = (
            {},
            dict(hicache_storage_backend="mooncake"),
            dict(tp_size=2, dcp_size=2),
            dict(tp_size=4, dcp_size=4),
            dict(kv_cache_dtype="bf16"),
            dict(kv_cache_dtype="bfloat16"),
            dict(hicache_mem_layout="layer_first"),
            dict(hicache_mem_layout="page_first_direct", hicache_io_backend="direct"),
            dict(dtype="float16"),
            dict(kv_cache_dtype="fp8_e4m3"),
            dict(hicache_write_policy="write_back"),
            dict(hicache_write_policy="write_through_selective"),
            dict(hicache_storage_prefetch_policy="best_effort"),
            dict(hicache_storage_prefetch_policy="timeout"),
            dict(pp_size=2),
            dict(dp_size=2),
            dict(attn_cp_size=2),
            dict(dp_size=2, enable_dp_attention=True),
        )
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in cases:
                with self.subTest(options=options):
                    args = _args(**options)
                    resolve_hicache_dcp_compatibility(args)
                    validate_hicache_dcp_storage(
                        args, storage_backend=args.hicache_storage_backend
                    )

    def test_keeps_existing_dcp_constraints(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in (
                dict(enable_hisparse=True),
                dict(enable_lmcache=True),
                dict(speculative_algorithm="EAGLE"),
                dict(speculative_algorithm="DSPARK"),
                # Buffer-mode staging budgets count physical host rows.
                dict(hicache_host_memory_mode="buffer_only"),
            ):
                with (
                    self.subTest(options=options),
                    self.assertRaises(NotImplementedError),
                ):
                    resolve_hicache_dcp_compatibility(_args(**options))

    def test_dspark_without_storage_keeps_dcp_host_cache_support(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            resolve_hicache_dcp_compatibility(
                _args(speculative_algorithm="DSPARK", hicache_storage_backend=None)
            )

    def test_runtime_rejection_has_no_side_effects(self):
        for attached in (False, True):
            for mla, backend, message in (
                (False, "file", "MLA"),
                (True, "nixl", "file or Mooncake storage"),
            ):
                cache = SimpleNamespace(
                    cache_controller=SimpleNamespace(write_policy="write_back"),
                    enable_storage=attached,
                    prefetch_stop_policy="timeout",
                    write_through_threshold=2,
                    is_write_back=True,
                )
                attachment = StorageAttachment(cache)
                with (
                    self.subTest(attached=attached, mla=mla, backend=backend),
                    mock.patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=SimpleNamespace(attn_dcp_size=2),
                    ),
                    mock.patch(
                        "sglang.srt.runtime_context.get_server_args",
                        return_value=_args(),
                    ),
                    mock.patch(
                        "sglang.srt.arg_groups.hicache_hook.use_mla_backend",
                        return_value=mla,
                    ),
                ):
                    ok, reason = attachment.attach(
                        backend,
                        hicache_storage_prefetch_policy="wait_complete",
                        hicache_write_policy="write_through",
                    )
                    self.assertFalse(ok)
                    self.assertIn(message, reason)
                    with self.assertRaisesRegex(NotImplementedError, message):
                        validate_hicache_dcp_storage(_args(), storage_backend=backend)
                self.assertEqual(cache.prefetch_stop_policy, "timeout")
                self.assertEqual(cache.cache_controller.write_policy, "write_back")
                self.assertEqual(cache.write_through_threshold, 2)
                self.assertTrue(cache.is_write_back)
                self.assertEqual(cache.enable_storage, attached)

    def test_rejected_pool_attach_restores_policies(self):
        def reject(**_):
            raise NotImplementedError(
                "HiCache L3 with DCP requires one materialized MLA host pool."
            )

        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                write_policy="write_back",
                attach_storage_backend=reject,
                mem_pool_host=SimpleNamespace(entries=None),
            ),
            enable_storage=False,
            prefetch_stop_policy="timeout",
            write_through_threshold=2,
            is_write_back=True,
            sliding_window_size=None,
        )
        with (
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args", return_value=_args()
            ),
            mock.patch(
                "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
            ),
        ):
            for config, error in (
                (None, "one materialized MLA host pool"),
                ("{", "Failed to parse"),
            ):
                with self.subTest(config=config):
                    ok, reason = StorageAttachment(cache).attach(
                        "file",
                        storage_backend_extra_config_json=config,
                        hicache_storage_prefetch_policy="wait_complete",
                        hicache_write_policy="write_through",
                    )
                    self.assertFalse(ok)
                    self.assertIn(error, reason)
                    self.assertEqual(cache.prefetch_stop_policy, "timeout")
                    self.assertEqual(cache.cache_controller.write_policy, "write_back")
                    self.assertEqual(cache.write_through_threshold, 2)
                    self.assertTrue(cache.is_write_back)


if __name__ == "__main__":
    unittest.main()

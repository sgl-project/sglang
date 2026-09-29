"""Startup and runtime support boundaries for DCP storage."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from test_hicache_dcp_host_pool import _make_host_pool
from test_mooncake_dcp_storage import _mamba_pool

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
    def test_materialized_mamba_registration_preserves_independent_slot_pool(self):
        pool = _make_host_pool(0, dcp_size=2, layout="page_first")
        state = _mamba_pool()
        self.addCleanup(pool.destroy)
        self.addCleanup(state.destroy)
        anchor = PoolEntry(
            PoolName.KV,
            pool,
            pool.device_pool,
            lambda x: x,
            is_primary_index_anchor=True,
        )
        sidecar = PoolEntry(PoolName.MAMBA, state, state.device_pool, lambda x: x)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor])
        controller.enable_storage = True
        controller.storage_config = SimpleNamespace(dcp_size=2)
        controller.storage_backend = mock.Mock()
        controller.extra_host_mem_release_queues = {}
        available = pool.available_size()
        controller.register_host_pool_entry(sidecar)
        self.assertIs(controller.mem_pool_host.anchor_entry, anchor)
        self.assertIs(controller.mem_pool_host.get_pool(PoolName.MAMBA), state)
        slots = controller.mem_pool_host.alloc(1, pool=PoolName.MAMBA)
        self.assertEqual(len(slots), 1)
        self.assertEqual(pool.available_size(), available)
        controller.mem_pool_host.free(slots, pool=PoolName.MAMBA)
        self.assertIn(PoolName.MAMBA, controller.extra_host_mem_release_queues)

    def test_mamba_label_cannot_register_a_token_pool_as_state(self):
        pool = _make_host_pool(0, dcp_size=2, layout="page_first")
        self.addCleanup(pool.destroy)
        anchor = PoolEntry(
            PoolName.KV,
            pool,
            pool.device_pool,
            lambda x: x,
            is_primary_index_anchor=True,
        )
        disguised = PoolEntry(PoolName.MAMBA, pool, pool.device_pool, lambda x: x)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor])
        controller.enable_storage = True
        controller.storage_config = SimpleNamespace(dcp_size=2)
        controller.extra_host_mem_release_queues = {}
        with self.assertRaisesRegex(NotImplementedError, "materialized Mamba"):
            controller.register_host_pool_entry(disguised)
        self.assertEqual(controller.mem_pool_host.entries, [anchor])
        self.assertEqual(controller.extra_host_mem_release_queues, {})

    def test_mamba_registration_cannot_enable_packed_draft_storage(self):
        pool = _make_host_pool(0, dcp_size=2, layout="page_first")
        state = _mamba_pool()
        self.addCleanup(pool.destroy)
        self.addCleanup(state.destroy)
        anchor = PoolEntry(
            PoolName.KV,
            pool,
            pool.device_pool,
            lambda x: x,
            is_primary_index_anchor=True,
        )
        sidecar = PoolEntry(PoolName.MAMBA, state, state.device_pool, lambda x: x)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor])
        # Model a draft packed into the primary pool: it is not a separate
        # entry, so checking only the sidecar would accidentally admit it.
        controller.mem_pool_host.entries[0] = replace(
            anchor, packed_draft_device_pools=(pool.device_pool,)
        )
        controller.enable_storage = True
        controller.storage_config = SimpleNamespace(dcp_size=2)
        controller.extra_host_mem_release_queues = {}
        controller.storage_backend = mock.Mock()
        with self.assertRaisesRegex(NotImplementedError, "materialized Mamba"):
            controller.register_host_pool_entry(sidecar)
        self.assertEqual(len(controller.mem_pool_host.entries), 1)
        self.assertEqual(controller.extra_host_mem_release_queues, {})

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
            dict(disaggregation_mode="prefill"),
            dict(disaggregation_mode="decode"),
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

    def test_decode_offload_cannot_bypass_guard_without_hicache(self):
        with (
            mock.patch(
                "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
            ),
            self.assertRaisesRegex(NotImplementedError, "decode offload"),
        ):
            resolve_hicache_dcp_compatibility(
                _args(
                    enable_hierarchical_cache=False,
                    disaggregation_mode="decode",
                    disaggregation_decode_enable_offload_kvcache=True,
                )
            )

    def test_decode_runtime_uses_effective_prefetch_policy(self):
        for attached in (False, True):
            for startup, current, requested, accepted in (
                ("wait_complete", "wait_complete", "best_effort", False),
                ("wait_complete", "wait_complete", "timeout", False),
                ("wait_complete", "timeout", None, False),
                ("timeout", "timeout", "wait_complete", True),
                ("timeout", "wait_complete", None, True),
            ):
                controller = SimpleNamespace(
                    storage_backend_type="file",
                    write_policy="write_back",
                    attach_storage_backend=mock.Mock(),
                    mem_pool_host=SimpleNamespace(entries=[]),
                )
                cache = SimpleNamespace(
                    cache_controller=controller,
                    enable_storage=attached,
                    prefetch_stop_policy=current,
                    write_through_threshold=2,
                    is_write_back=True,
                    sliding_window_size=None,
                    _enable_metrics_flag=False,
                    extra_metric_labels=None,
                )
                attachment = StorageAttachment(cache)
                with (
                    self.subTest(
                        attached=attached, current=current, requested=requested
                    ),
                    mock.patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=SimpleNamespace(attn_dcp_size=2),
                    ),
                    mock.patch(
                        "sglang.srt.runtime_context.get_server_args",
                        return_value=_args(
                            disaggregation_mode="decode",
                            hicache_storage_prefetch_policy=startup,
                        ),
                    ),
                    mock.patch(
                        "sglang.srt.arg_groups.hicache_hook.use_mla_backend",
                        return_value=True,
                    ),
                    mock.patch.object(attachment, "apply_runtime_config"),
                ):
                    ok, reason = attachment.attach(
                        "file",
                        hicache_storage_prefetch_policy=requested,
                        hicache_write_policy="write_through",
                    )
                    self.assertEqual(ok, accepted, reason)
                    if accepted:
                        self.assertEqual(cache.prefetch_stop_policy, "wait_complete")
                    else:
                        self.assertIn("wait_complete", reason)
                        self.assertEqual(cache.prefetch_stop_policy, current)
                        self.assertEqual(controller.write_policy, "write_back")
                        self.assertEqual(cache.write_through_threshold, 2)
                        self.assertTrue(cache.is_write_back)
                        controller.attach_storage_backend.assert_not_called()

    def test_decode_storage_keeps_promised_prefetches_complete(self):
        # Decode promises the probed L3 span to prefill before the read runs;
        # early-stopping policies turn a short read into an aborted request.
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in (
                dict(hicache_storage_prefetch_policy="best_effort"),
                dict(hicache_storage_prefetch_policy="timeout"),
                dict(disaggregation_decode_enable_offload_kvcache=True),
            ):
                with (
                    self.subTest(options=options),
                    self.assertRaises(NotImplementedError),
                ):
                    validate_hicache_dcp_storage(
                        _args(disaggregation_mode="decode", **options)
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

"""Startup and runtime support boundaries for DCP file storage."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.hicache_hook import (
    resolve_hicache_dcp_compatibility,
    validate_hicache_dcp_storage,
)
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

    def test_runtime_rejection_has_no_side_effects(self):
        for attached in (False, True):
            for mla, backend, message in (
                (False, "file", "MLA"),
                (True, "mooncake", "file storage"),
            ):
                cache = SimpleNamespace(
                    cache_controller=SimpleNamespace(), enable_storage=attached
                )
                attachment = StorageAttachment(cache)
                attachment._apply_policies = mock.Mock()
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
                    ok, reason = attachment.attach(backend)
                    self.assertFalse(ok)
                    self.assertIn(message, reason)
                    with self.assertRaisesRegex(NotImplementedError, message):
                        validate_hicache_dcp_storage(_args(), storage_backend=backend)
                attachment._apply_policies.assert_not_called()

    def test_runtime_policy_updates_use_existing_validation(self):
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(storage_backend_type="file"),
            enable_storage=True,
        )
        attachment = StorageAttachment(cache)
        attachment._apply_policies = mock.Mock()
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

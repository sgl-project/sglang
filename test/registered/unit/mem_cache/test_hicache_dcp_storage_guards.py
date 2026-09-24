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

    def test_unsupported_options(self):
        cases = dict(
            hicache_storage_backend="mooncake",
            hicache_mem_layout="layer_first",
            hicache_io_backend="direct",
            dtype="float16",
            kv_cache_dtype="fp8_e4m3",
            hicache_write_policy="write_back",
            hicache_storage_prefetch_policy="timeout",
            hicache_host_memory_mode="buffer_only",
            pp_size=2,
            dp_size=2,
            attn_cp_size=2,
            enable_dp_attention=True,
            speculative_algorithm="DSPARK",
            enable_hisparse=True,
            enable_lmcache=True,
            disaggregation_mode="prefill",
        )
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for name, value in cases.items():
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        NotImplementedError, "HiCache L3 with DCP requires"
                    ),
                ):
                    resolve_hicache_dcp_compatibility(_args(**{name: value}))
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=False
        ):
            with self.assertRaisesRegex(NotImplementedError, "dense MLA"):
                validate_hicache_dcp_storage(_args())

    def test_runtime_policy_rejection_has_no_side_effects(self):
        for attached in (False, True):
            cache = SimpleNamespace(
                cache_controller=SimpleNamespace(write_policy="write_through"),
                prefetch_stop_policy="wait_complete",
                enable_storage=attached,
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
                    "sglang.srt.arg_groups.hicache_hook.use_mla_backend",
                    return_value=True,
                ),
            ):
                for policy in (
                    dict(hicache_write_policy="write_back"),
                    dict(hicache_storage_prefetch_policy="timeout"),
                ):
                    ok, reason = attachment.attach("file", **policy)
                    self.assertFalse(ok)
                    self.assertIn("requires", reason)
            attachment._apply_policies.assert_not_called()


if __name__ == "__main__":
    unittest.main()

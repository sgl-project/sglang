"""Context-first mutation.

``get_context().override(source, **fields)`` is the business mutation entry: it
writes the resolved config bags (the single source of truth) and never touches
``server_args`` (the pristine startup record). Routing is by NS metadata; a bad
field aborts before any write; provenance is recorded.
"""

import unittest

from sglang.srt import runtime_context as rc
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestContextOverride(CustomTestCase):
    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _publish(self):
        sa = ServerArgs(model_path="dummy")
        rc.get_context().set_server_args(sa)
        return sa

    def test_override_writes_bag_not_server_args(self):
        sa = self._publish()
        before = sa.hicache_ratio
        rc.get_context().override("test", hicache_ratio=before + 1.0)
        self.assertEqual(rc.get_memory().hicache_ratio, before + 1.0)
        # server_args stays the pristine startup record.
        self.assertEqual(sa.hicache_ratio, before)

    def test_override_routes_across_namespaces(self):
        self._publish()
        rc.get_context().override(
            "test",
            moe_runner_backend="triton",
            page_size=64,
            disaggregation_mode="decode",
        )
        self.assertEqual(rc.get_exec().moe.moe_runner_backend, "triton")
        self.assertEqual(rc.get_schedule().page_size, 64)
        self.assertEqual(rc.get_disagg().disaggregation_mode, "decode")

    def test_override_unknown_field_raises_and_is_atomic(self):
        self._publish()
        before = rc.get_memory().hicache_ratio
        with self.assertRaises(ValueError):
            rc.get_context().override(
                "test", hicache_ratio=before + 5.0, not_a_real_field=1
            )
        # No partial write: the valid field was not applied.
        self.assertEqual(rc.get_memory().hicache_ratio, before)

    def test_override_before_publish_raises(self):
        with self.assertRaises(ValueError):
            rc.get_context().override("test", page_size=32)

    def test_override_provenance_recorded(self):
        self._publish()
        rc.get_context().override("srcA", page_size=16)
        log = rc.get_context().overrides_log()
        self.assertEqual(log[-1], ("srcA", {"page_size": 16}))

    def test_republish_resets_provenance(self):
        self._publish()
        rc.get_context().override("srcA", page_size=16)
        self.assertTrue(rc.get_context().overrides_log())
        self._publish()
        self.assertEqual(rc.get_context().overrides_log(), [])

    def test_set_internal_state_fields_reach_parallel_and_spec(self):
        # The fields /set_internal_state overrides must reach the accessors the
        # (1e) flipped readers now use: pp via get_parallel(), thresholds via
        # get_spec().
        self._publish()
        rc.get_context().override(
            "update_server_args",
            pp_max_micro_batch_size=8,
            speculative_accept_threshold_single=0.5,
            speculative_accept_threshold_acc=0.9,
        )
        self.assertEqual(rc.get_parallel().pp_max_micro_batch_size, 8)
        self.assertEqual(rc.get_spec().speculative_accept_threshold_single, 0.5)
        self.assertEqual(rc.get_spec().speculative_accept_threshold_acc, 0.9)

    def test_kv_cache_dtype_override_reaches_get_model_not_server_args(self):
        # Load-time resolution: the resolved kv-cache dtype is written
        # to the model bag; server_args stays the RAW resolver input.
        sa = self._publish()
        raw = sa.kv_cache_dtype
        rc.get_context().override(
            "ModelRunner.configure_kv_cache_dtype", kv_cache_dtype="fp8_e4m3"
        )
        self.assertEqual(rc.get_model().kv_cache_dtype, "fp8_e4m3")
        self.assertEqual(sa.kv_cache_dtype, raw)

    def test_bare_server_args_write_raises_after_resolution(self):
        # server_args is read-only after resolution regardless of the
        # SGLANG_STRICT_CONFIG_MUTATION env; write via override instead.
        sa = ServerArgs(model_path="dummy")
        object.__setattr__(sa, "_declarations_materialized", True)
        with self.assertRaises(AttributeError):
            sa.page_size = 999


if __name__ == "__main__":
    unittest.main()

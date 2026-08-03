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

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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

    def test_preserve_config_keeps_post_publish_overrides(self):
        # A nested build (e.g. a draft worker) publishes its own private copy;
        # on exit the target's resolved bags — including post-publish
        # overrides — must be reinstated verbatim, not re-projected from the
        # pristine record (which would silently drop the overrides).
        target = self._publish()
        rc.get_context().override(
            "ModelRunner.configure_kv_cache_dtype", kv_cache_dtype="fp8_e4m3"
        )
        draft = ServerArgs(model_path="dummy")
        draft.override(source="draft-build", kv_cache_dtype="bf16")
        with rc.get_context().preserve_config():
            rc.get_context().set_server_args(draft)
            # Inside the scope the draft's bags are live...
            self.assertEqual(rc.get_model().kv_cache_dtype, "bf16")
            # ...and its own post-publish overrides work as usual.
            rc.get_context().override("draft-load", kv_cache_dtype="fp8_e5m2")
            self.assertEqual(rc.get_model().kv_cache_dtype, "fp8_e5m2")
        # Target slot, bags, and provenance restored verbatim.
        self.assertIs(rc.get_context().server_args, target)
        self.assertEqual(rc.get_model().kv_cache_dtype, "fp8_e4m3")
        self.assertEqual(
            rc.get_context().overrides_log(),
            [
                (
                    "ModelRunner.configure_kv_cache_dtype",
                    {"kv_cache_dtype": "fp8_e4m3"},
                )
            ],
        )

    def test_preserve_config_restores_in_scope_override_without_republish(self):
        # An override inside the scope (no republish) mutates the live bags
        # and provenance log in place; the scope must restore entry VALUES,
        # not just reassign the aliased objects.
        self._publish()
        rc.get_context().override("srcA", page_size=16)
        with rc.get_context().preserve_config():
            rc.get_context().override("in-scope", page_size=64)
            self.assertEqual(rc.get_schedule().page_size, 64)
        self.assertEqual(rc.get_schedule().page_size, 16)
        self.assertEqual(
            rc.get_context().overrides_log(), [("srcA", {"page_size": 16})]
        )

    def test_preserve_config_restores_on_exception(self):
        target = self._publish()
        rc.get_context().override("srcA", page_size=16)
        with self.assertRaises(RuntimeError):
            with rc.get_context().preserve_config():
                rc.get_context().set_server_args(ServerArgs(model_path="dummy"))
                raise RuntimeError("nested build failed")
        self.assertIs(rc.get_context().server_args, target)
        self.assertEqual(rc.get_schedule().page_size, 16)

    def test_publish_records_role(self):
        rc.publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.assertEqual(rc.publish_role(), "scheduler")

    def test_legacy_shims_record_roles(self):
        # Unit 2a: the legacy setters publish with their process role.
        from sglang.srt.server_args import (
            set_global_server_args_for_scheduler,
            set_global_server_args_for_tokenizer,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.assertEqual(rc.publish_role(), "scheduler")
        set_global_server_args_for_tokenizer(ServerArgs(model_path="dummy"))
        self.assertEqual(rc.publish_role(), "tokenizer")

    def test_reset_clears_role(self):
        rc.publish(ServerArgs(model_path="dummy"), role="test")
        rc.reset_context()
        self.assertIsNone(rc.publish_role())

    def test_direct_install_clears_role(self):
        # A role-less set_server_args (test overrides, draft-worker builds)
        # must not inherit the previous lifecycle's role.
        rc.publish(ServerArgs(model_path="dummy"), role="scheduler")
        rc.get_context().set_server_args(ServerArgs(model_path="dummy"))
        self.assertIsNone(rc.publish_role())


if __name__ == "__main__":
    unittest.main()

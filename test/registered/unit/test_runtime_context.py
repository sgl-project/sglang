"""Unit tests for runtime_context: delegation, singletons, and override()."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups.arg_utils import A, Arg
from sglang.srt.runtime_context import (
    Flags,
    ParallelContext,
    RuntimeContext,
    _FlagGroupBase,
    get_context,
    get_flags,
    get_parallel,
    get_server_args,
    reset_context,
)
from sglang.test.test_utils import CustomTestCase

_PS = "sglang.srt.distributed.parallel_state"
_DP = "sglang.srt.layers.dp_attention"

SIZE_RANK_DELEGATIONS = [
    ("world_size", f"{_PS}.get_world_size"),
    ("world_rank", f"{_PS}.get_world_rank"),
    ("tp_size", f"{_PS}.get_tensor_model_parallel_world_size"),
    ("tp_rank", f"{_PS}.get_tensor_model_parallel_rank"),
    ("pp_size", f"{_PS}.get_pipeline_model_parallel_world_size"),
    ("pp_rank", f"{_PS}.get_pipeline_model_parallel_rank"),
    ("moe_ep_size", f"{_PS}.get_moe_expert_parallel_world_size"),
    ("moe_ep_rank", f"{_PS}.get_moe_expert_parallel_rank"),
    ("moe_dp_size", f"{_PS}.get_moe_data_parallel_world_size"),
    ("moe_dp_rank", f"{_PS}.get_moe_data_parallel_rank"),
    ("moe_tp_size", f"{_PS}.get_moe_tensor_parallel_world_size"),
    ("moe_tp_rank", f"{_PS}.get_moe_tensor_parallel_rank"),
    ("attn_tp_size", f"{_PS}.get_attn_tensor_model_parallel_world_size"),
    ("attn_tp_rank", f"{_PS}.get_attn_tensor_model_parallel_rank"),
    ("attn_cp_size", f"{_PS}.get_attn_context_model_parallel_world_size"),
    ("attn_cp_rank", f"{_PS}.get_attn_context_model_parallel_rank"),
    ("attn_dp_size", f"{_DP}.get_attention_dp_size"),
    ("attn_dp_rank", f"{_DP}.get_attention_dp_rank"),
]

GROUP_DELEGATIONS = [
    ("world_group", f"{_PS}.get_world_group"),
    ("tp_group", f"{_PS}.get_tp_group"),
    ("pp_group", f"{_PS}.get_pp_group"),
    ("moe_ep_group", f"{_PS}.get_moe_ep_group"),
    ("moe_dp_group", f"{_PS}.get_moe_dp_group"),
    ("moe_tp_group", f"{_PS}.get_moe_tp_group"),
    ("attn_tp_group", f"{_PS}.get_attn_tp_group"),
    ("attn_cp_group", f"{_PS}.get_attn_cp_group"),
]


class TestRuntimeContextSingletons(CustomTestCase):
    def test_singletons(self):
        self.assertIs(get_parallel(), get_parallel())
        self.assertIsInstance(get_parallel(), ParallelContext)
        self.assertIsInstance(get_context(), RuntimeContext)
        self.assertIs(get_context().parallel, get_parallel())


class _IsolatedOverrides(CustomTestCase):
    """Give each test a clean override map, restoring afterward only the overrides
    installed outside it (e.g. by another test file sharing the process)."""

    def setUp(self):
        super().setUp()
        p = get_parallel()
        self._saved_overrides = dict(p._overrides)
        p._overrides.clear()

    def tearDown(self):
        p = get_parallel()
        p._overrides.clear()
        p._overrides.update(self._saved_overrides)
        super().tearDown()


class TestParallelDelegation(_IsolatedOverrides):
    def test_size_rank_delegate_to_canonical_getters(self):
        # Patch each getter to a distinct sentinel: a miswired attribute would read
        # a different (unpatched) getter and fail.
        for i, (attr, target) in enumerate(SIZE_RANK_DELEGATIONS):
            sentinel = 1000 + i
            with patch(target, return_value=sentinel):
                self.assertEqual(
                    getattr(get_parallel(), attr),
                    sentinel,
                    msg=f"{attr} must delegate to {target}",
                )

    def test_groups_delegate_to_canonical_getters(self):
        for attr, target in GROUP_DELEGATIONS:
            sentinel = object()
            with patch(target, return_value=sentinel):
                self.assertIs(
                    getattr(get_parallel(), attr),
                    sentinel,
                    msg=f"{attr} must delegate to {target}",
                )

    def test_wrapper_holds_no_resolved_state(self):
        # __slots__: no __dict__; the only instance state is the override hook.
        self.assertFalse(hasattr(get_parallel(), "__dict__"))
        # tp_group IS exposed: live delegation handles PD-multiplexing / the tp patch.
        self.assertTrue(hasattr(ParallelContext, "tp_group"))
        # local_attn_dp is intentionally not part of the wrapper surface.
        self.assertFalse(hasattr(ParallelContext, "local_attn_dp_size"))


class TestParallelOverride(_IsolatedOverrides):
    def test_override_takes_precedence(self):
        p = get_parallel()
        with p.override(tp_size=99, tp_rank=3, attn_dp_size=8):
            self.assertEqual(p.tp_size, 99)
            self.assertEqual(p.tp_rank, 3)
            self.assertEqual(p.attn_dp_size, 8)
            # same singleton: a fresh get_parallel() sees the override too
            self.assertEqual(get_parallel().tp_size, 99)
        self.assertEqual(p._overrides, {})

    def test_override_can_force_groups(self):
        sentinel = object()
        with get_parallel().override(tp_group=sentinel):
            self.assertIs(get_parallel().tp_group, sentinel)

    def test_override_nests_and_restores(self):
        p = get_parallel()
        with p.override(tp_size=2):
            self.assertEqual(p.tp_size, 2)
            with p.override(tp_size=4, pp_size=2):
                self.assertEqual(p.tp_size, 4)
                self.assertEqual(p.pp_size, 2)
            self.assertEqual(p.tp_size, 2)
            self.assertNotIn("pp_size", p._overrides)

    def test_override_unknown_key_raises_and_does_not_mutate(self):
        p = get_parallel()
        with self.assertRaises(ValueError):
            with p.override(tp_sizee=1):  # typo
                pass
        self.assertEqual(p._overrides, {})


class _IsolatedServerArgs(CustomTestCase):
    """Save/restore the published ServerArgs around each test (the slot is
    process-global; another test file sharing the process may have published)."""

    def setUp(self):
        super().setUp()
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)
        super().tearDown()


class TestServerArgsOwnership(_IsolatedServerArgs):
    """V2b: the context owns the slot; the legacy getters are identity shims."""

    def test_legacy_setter_publishes_into_context(self):
        # Identity (not equality) is the contract; publish accepts any object.
        sentinel = object()
        server_args_module.set_global_server_args_for_scheduler(sentinel)
        self.assertIs(server_args_module.get_global_server_args(), sentinel)
        self.assertIs(get_server_args(), sentinel)
        self.assertIs(get_context().server_args, sentinel)

    def test_context_publish_visible_through_legacy_getter(self):
        sentinel = object()
        get_context().set_server_args(sentinel)
        self.assertIs(server_args_module.get_global_server_args(), sentinel)

    def test_tokenizer_alias_is_same_function(self):
        self.assertIs(
            server_args_module.set_global_server_args_for_tokenizer,
            server_args_module.set_global_server_args_for_scheduler,
        )

    def test_pre_publish_error_verbatim(self):
        reset_context()
        for accessor in (get_server_args, server_args_module.get_global_server_args):
            with self.assertRaises(ValueError) as cm:
                accessor()
            self.assertEqual(str(cm.exception), "Global server args is not set yet!")

    def test_republish_overwrite_allowed(self):
        first, second = object(), object()
        server_args_module.set_global_server_args_for_scheduler(first)
        server_args_module.set_global_server_args_for_scheduler(second)
        self.assertIs(get_server_args(), second)

    def test_reset_context_clears_owned_store(self):
        server_args_module.set_global_server_args_for_scheduler(object())
        reset_context()
        with self.assertRaises(ValueError):
            get_server_args()

    def test_module_global_removed(self):
        # The legacy storage must not survive: a stale _global_server_args would
        # silently fork the config into two objects.
        self.assertFalse(hasattr(server_args_module, "_global_server_args"))


@dataclasses.dataclass
class _FakeCaptureGroup(_FlagGroupBase):
    gamma: int = 0


class TestFlagsTier(_IsolatedServerArgs):
    """Runtime-flags tier: typed groups, typo-safe writes, override primitive.

    Resolved configuration lives on server_args fields (materialized at the
    end of __post_init__); the flags tier only carries runtime state
    (today: the capture lifecycle)."""

    def test_wiring_and_groups(self):
        flags = get_flags()
        self.assertIs(flags, get_context().flags)
        self.assertIsInstance(flags, Flags)
        self.assertTrue(hasattr(flags, "capture"))

    def test_typo_safety(self):
        group = _FakeCaptureGroup()
        with self.assertRaises(AttributeError):
            group.gamma_misspelled = 2  # undeclared leaf
        with self.assertRaises(AttributeError):
            get_flags().not_a_flag = 1

    def test_override_is_transactional(self):
        group = _FakeCaptureGroup()
        with group.override(gamma=99):
            self.assertEqual(group.gamma, 99)
        self.assertEqual(group.gamma, 0)
        with self.assertRaises(ValueError):
            with group.override(gamma=2, delta=3):  # delta undeclared
                pass
        self.assertEqual(group.gamma, 0)  # validated before any write

    def test_reset_context_installs_fresh_flags(self):
        old = get_flags()
        old.capture.enable_torch_compile = True
        reset_context()
        self.assertIsNot(get_flags(), old)
        self.assertFalse(get_flags().capture.enable_torch_compile)


@dataclasses.dataclass
class _FakeResolvedArgs:
    """Publishable fixture with a resolvable whitelist (real flat leaves)."""

    page_size: A[int | None, Arg(help="p", resolvable=True)] = None
    sampling_backend: A[str | None, Arg(help="s", resolvable=True)] = None
    _resolved_overrides: list = dataclasses.field(default_factory=list)


class TestPublishLifecycle(_IsolatedServerArgs):
    """Publish installs the resolved server_args and seeds the capture tier."""

    def _publish(self, **kw):
        args = _FakeResolvedArgs(**kw)
        get_context().set_server_args(args)
        return args

    def test_capture_tier_seeded_at_publish(self):
        args = self._publish(page_size=1)
        args.enable_torch_compile = True
        get_context().set_server_args(args)  # re-publish picks up the value
        self.assertTrue(get_flags().capture.enable_torch_compile)
        # capture-time write (B4) targets the capture leaf
        get_flags().capture.enable_torch_compile = False
        self.assertFalse(get_flags().capture.enable_torch_compile)

    def test_capture_tier_defaults_for_sentinel_publish(self):
        get_context().set_server_args(object())
        self.assertFalse(get_flags().capture.enable_torch_compile)

    def test_declare_load_time_override_writes_through(self):
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        args = self._publish(page_size=1)
        declare_load_time_override("model.load_time", {"page_size": 64})
        self.assertEqual(args.page_size, 64)

    def test_declare_load_time_override_validates_whitelist(self):
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        args = self._publish(page_size=1)
        with self.assertRaises(ValueError):
            declare_load_time_override("bad", {"nope": 1})
        self.assertEqual(args.page_size, 1)

    def test_declare_load_time_override_records_provenance(self):
        from sglang.srt.arg_groups.overrides import declare_load_time_override
        from sglang.srt.server_args import ServerArgs

        class _Args(_FakeResolvedArgs):
            override = ServerArgs.override

        args = _Args(page_size=1)
        get_context().set_server_args(args)
        declare_load_time_override("model.load_time", {"page_size": 64})
        self.assertEqual(args.page_size, 64)
        self.assertIn(("model.load_time", {"page_size": 64}), args._resolved_overrides)


if __name__ == "__main__":
    unittest.main()

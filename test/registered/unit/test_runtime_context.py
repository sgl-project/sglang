"""Unit tests for runtime_context: delegation, singletons, and override()."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups.arg_utils import NS, A, Arg
from sglang.srt.runtime_context import (
    Flags,
    ParallelContext,
    RuntimeContext,
    _FlagGroupBase,
    assert_published,
    get_context,
    get_exec,
    get_flags,
    get_parallel,
    get_server_args,
    max_speculative_num_draft_tokens,
    publish,
    publish_role,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_PS = "sglang.srt.distributed.parallel_state"
_DP = "sglang.srt.layers.dp_attention"

SIZE_RANK_DELEGATIONS = [
    ("world_size", f"{_PS}.get_world_size"),
    ("world_rank", f"{_PS}.get_world_rank"),
    ("tp_size", f"{_PS}.get_tensor_model_parallel_world_size"),
    ("tp_rank", f"{_PS}.get_tensor_model_parallel_rank"),
    ("dcp_size", f"{_PS}.get_dcp_world_size"),
    ("dcp_rank", f"{_PS}.get_dcp_rank"),
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
    ("dcp_group", f"{_PS}.get_dcp_group"),
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


class TestParallelDCP(_IsolatedOverrides):
    def test_attn_dcp_defaults_when_group_is_uninitialized(self):
        with (
            patch(f"{_PS}.get_dcp_group_no_assert", return_value=None),
            patch(f"{_PS}.get_dcp_world_size", side_effect=AssertionError),
            patch(f"{_PS}.get_dcp_rank", side_effect=AssertionError),
        ):
            self.assertFalse(get_parallel().dcp_enabled)
            self.assertEqual(get_parallel().attn_dcp_size, 1)
            self.assertEqual(get_parallel().attn_dcp_rank, 0)

    def test_attn_dcp_delegates_when_enabled(self):
        with (
            patch(f"{_PS}.get_dcp_group_no_assert", return_value=object()),
            patch(f"{_PS}.get_dcp_world_size", return_value=8),
            patch(f"{_PS}.get_dcp_rank", return_value=3),
        ):
            self.assertTrue(get_parallel().dcp_enabled)
            self.assertEqual(get_parallel().attn_dcp_size, 8)
            self.assertEqual(get_parallel().attn_dcp_rank, 3)

    def test_dcp_enablement_is_platform_agnostic(self):
        with (
            patch(f"{_PS}.get_dcp_group_no_assert", return_value=object()),
            patch("sglang.srt.utils.is_cuda", return_value=False) as is_cuda,
            patch(f"{_PS}.get_dcp_world_size", return_value=8),
            patch(f"{_PS}.get_dcp_rank", return_value=3),
        ):
            self.assertTrue(get_parallel().dcp_enabled)
            self.assertEqual(get_parallel().attn_dcp_size, 8)
            self.assertEqual(get_parallel().attn_dcp_rank, 3)
            is_cuda.assert_not_called()


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
        # Identity, not equality: the slot holds the very object published.
        sentinel = ServerArgs(model_path="dummy")
        server_args_module.set_global_server_args_for_scheduler(sentinel)
        self.assertIs(server_args_module.get_global_server_args(), sentinel)
        self.assertIs(get_server_args(), sentinel)
        self.assertIs(get_context().server_args, sentinel)

    def test_tokenizer_alias_is_distinct_role_shim(self):
        # Deliberately NOT an alias: the two legacy setters publish with
        # different process roles (scheduler vs tokenizer).
        self.assertIsNot(
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
        first = ServerArgs(model_path="dummy")
        second = ServerArgs(model_path="dummy")
        server_args_module.set_global_server_args_for_scheduler(first)
        server_args_module.set_global_server_args_for_scheduler(second)
        self.assertIs(get_server_args(), second)

    def test_reset_context_clears_owned_store(self):
        server_args_module.set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy")
        )
        reset_context()
        with self.assertRaises(ValueError):
            get_server_args()


class TestAssertPublished(_IsolatedServerArgs):
    """Publishing is the process entry's job; the constructors only check.

    `ModelRunner`, `TokenizerManager` and `MMEncoder` assert. A publish inside
    a process that has already published re-projects the bags, discarding every
    `override()` taken since and the provenance log with it, so a constructor
    that finds nothing published fails loud.
    """

    def _record(self, **fields):
        return ServerArgs(model_path="dummy", **fields)

    def test_the_check_leaves_a_live_process_alone(self):
        record = self._record(grammar_backend="xgrammar")
        publish(record, role="scheduler")
        get_context().override("grammar.import_fallback", grammar_backend="none")

        assert_published(record, role="scheduler")

        self.assertEqual(
            get_exec().kernel.grammar_backend,
            "none",
            "the check re-projected the bags, so the import fallback was "
            "discarded and the process reports a backend it is not using",
        )
        self.assertEqual(
            len(get_context().overrides_log()),
            1,
            "the provenance of the override went with it",
        )

    def test_a_different_record_fails(self):
        first = self._record(grammar_backend="xgrammar")
        publish(first, role="scheduler")
        second = self._record(grammar_backend="llguidance")

        with self.assertRaisesRegex(RuntimeError, "a different record is published"):
            assert_published(second, role="scheduler")

        self.assertIs(
            get_server_args(),
            first,
            "the failing check published anyway",
        )

    def test_an_empty_slot_fails(self):
        """An empty slot fails."""
        reset_context()
        record = self._record(grammar_backend="xgrammar")

        with self.assertRaisesRegex(
            RuntimeError, "nothing is published in this process"
        ):
            assert_published(record, role="scheduler")

    def test_the_same_record_under_a_different_role_fails(self):
        """The role decides which namespaces this process may read."""
        record = self._record()
        publish(record, role="tokenizer")

        with self.assertRaisesRegex(RuntimeError, "published under role 'tokenizer'"):
            assert_published(record, role="scheduler")

        self.assertEqual(publish_role(), "tokenizer")

    def test_no_constructor_publishes_outside_the_two_entries(self):
        """Publishing from an `__init__` is an entry's job or a bug.

        It is right when the constructor *is* the entry -- an `Engine` being
        (re)built, the Ray actor that stands in for `run_scheduler_process`,
        where resetting the bags is the point. It is wrong anywhere else,
        because the process is already live with a record and re-projecting
        drops its overrides. The census is pinned, so a new constructor publish
        fails here until it is one of the two.

        Both the publisher set and "which `__init__` reaches one" come from
        `sglang.test.config_publishers`, which derives them from the code --
        a hand-written spelling list here missed a constructor that publishes
        one hop away through a helper. The derivation follows helpers defined
        in the same module; a constructor that publishes through a helper in
        *another* module is not seen, which is the one hole left here.
        """
        import pathlib

        import sglang
        from sglang.test.config_publishers import constructor_publishers

        srt = pathlib.Path(sglang.__file__).resolve().parent / "srt"
        self.assertEqual(
            constructor_publishers(srt),
            {
                ("entrypoints/engine.py", "Engine", "publish"),
                ("ray/scheduler_actor.py", "SchedulerActor", "publish"),
            },
            "a constructor publishes and it is not one of the two entries; "
            "publish at the process entry and let the constructor assert",
        )


class TestServerArgsScopedOverride(_IsolatedServerArgs):
    """ctx.override_server_args: the config tier's scoped test override —
    tests force execution paths by overriding the context, not by
    hand-building and publishing config objects."""

    def test_install_publishes_fresh_config_with_fields(self):
        reset_context()
        override = get_context().override_server_args(
            attention_backend="triton", chunked_prefill_size=-1
        )
        published = override.install()
        self.assertIs(get_server_args(), published)
        self.assertEqual(published.attention_backend, "triton")
        self.assertEqual(published.chunked_prefill_size, -1)
        # unnamed fields keep their dataclass defaults
        self.assertEqual(published.tp_size, 1)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            get_context().override_server_args(not_a_config_field=1).install()

    def test_restore_reinstates_previous_publish(self):
        previous = object()
        get_context().set_server_args(previous)
        override = get_context().override_server_args(tp_size=8)
        override.install()
        self.assertEqual(get_server_args().tp_size, 8)
        override.restore()
        self.assertIs(get_server_args(), previous)

    def test_restore_reinstates_the_empty_slot(self):
        reset_context()
        with get_context().override_server_args():
            get_server_args()  # published inside the scope
        with self.assertRaises(ValueError):
            get_server_args()

    def test_nesting_restores_in_order(self):
        reset_context()
        with get_context().override_server_args(tp_size=2) as outer:
            with get_context().override_server_args(tp_size=4):
                self.assertEqual(get_server_args().tp_size, 4)
            self.assertIs(get_server_args(), outer)
            self.assertEqual(get_server_args().tp_size, 2)

    def test_private_attribute_seeding(self):
        # Property caches (e.g. _mamba_cache_chunk_size) are seeded through
        # the same call; the strict guard exempts underscore names.
        published = (
            get_context().override_server_args(_mamba_cache_chunk_size=64).install()
        )
        self.assertEqual(published.mamba_cache_chunk_size, 64)

    def test_installed_config_arms_the_strict_guard(self):
        # The published dummy must behave like a resolved config: bare writes
        # raise.
        published = get_context().override_server_args(tp_size=2).install()
        with self.assertRaises(AttributeError):
            published.tp_size = 4
        self.assertEqual(published.tp_size, 2)

    def test_restore_resets_the_capture_seed(self):
        # install() seeds flags.capture from the published dummy; restore()
        # must put back the pre-install runtime state on both restore paths.
        reset_context()
        self.assertFalse(get_flags().capture.enable_torch_compile)
        override = get_context().override_server_args(enable_torch_compile=True)
        override.install()
        self.assertTrue(get_flags().capture.enable_torch_compile)
        override.restore()
        self.assertFalse(get_flags().capture.enable_torch_compile)

    def test_double_install_rejected(self):
        override = get_context().override_server_args()
        override.install()
        with self.assertRaises(AssertionError):
            override.install()

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

    page_size: A[int | None, Arg(help="p", resolvable=True), NS("schedule")] = None
    sampling_backend: A[
        str | None, Arg(help="s", resolvable=True), NS("exec.kernel")
    ] = None
    attention_backend: A[str | None, Arg(help="ab"), NS("exec.kernel")] = None
    prefill_attention_backend: A[str | None, Arg(help="pab"), NS("exec.kernel")] = None
    decode_attention_backend: A[str | None, Arg(help="dab"), NS("exec.kernel")] = None
    disable_radix_cache: A[bool, Arg(help="drc"), NS("memory")] = False
    mamba_radix_cache_strategy: A[str, Arg(help="mrcs"), NS("exec.mamba")] = "auto"
    speculative_num_draft_tokens: A[int | None, Arg(help="d"), NS("spec")] = None
    speculative_adaptive: A[bool, Arg(help="a"), NS("spec")] = False
    speculative_adaptive_config: A[str | None, Arg(help="c"), NS("spec")] = None
    load_format: A[str, Arg(help="lf"), NS("model")] = "auto"
    remote_instance_weight_loader_backend: A[str, Arg(help="rb"), NS("model")] = "nccl"
    remote_instance_weight_loader_start_seed_via_transfer_engine: A[
        bool, Arg(help="rs"), NS("model")
    ] = False
    modelexpress_config: A[str | None, Arg(help="mx"), NS("model")] = None
    disaggregation_mode: A[str, Arg(help="dm"), NS("disagg")] = "null"
    max_running_requests: A[int | None, Arg(help="mrr"), NS("schedule")] = None
    chunked_prefill_size: A[int, Arg(help="cps"), NS("schedule")] = -1
    max_prefill_tokens: A[int, Arg(help="mpt"), NS("schedule")] = 16384
    enable_dynamic_chunking: A[bool, Arg(help="edc"), NS("schedule")] = False
    cuda_graph_config: A[object | None, Arg(help="cgc"), NS("exec.graph")] = None
    tp_size: A[int, Arg(help="tp"), NS("parallel")] = 1
    pp_size: A[int, Arg(help="pp"), NS("parallel")] = 1
    _resolved_overrides: list = dataclasses.field(default_factory=list)


class TestMoeFlagsGroup(_IsolatedServerArgs):
    """flags.moe: materialized by initialize_moe_config; the ACTIVE backends
    swap under the speculative contexts and restore on exit."""

    def _init(self, **kw):
        from sglang.srt.layers.moe.utils import initialize_moe_config

        defaults = dict(
            moe_a2a_backend="none",
            moe_runner_backend="auto",
            speculative_moe_runner_backend=None,
            speculative_moe_a2a_backend=None,
            deepep_mode="auto",
            deepep_config=None,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
            tbo_token_distribution_threshold=0.48,
            disable_flashinfer_cutlass_moe_fp4_allgather=False,
            quantization=None,
            disable_shared_experts_fusion=False,
        )
        defaults.update(kw)
        # The flags are seeded from the bags, so the test publishes a config
        # carrying these values.
        override = get_context().override_server_args(**defaults)
        override.install()
        self.addCleanup(override.restore)
        initialize_moe_config()

    def test_lazy_defaults_before_initialize(self):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
            is_tbo_enabled,
        )

        reset_context()
        self.assertTrue(get_moe_a2a_backend().is_none())
        self.assertEqual(get_moe_runner_backend().name, "AUTO")
        self.assertFalse(is_tbo_enabled())

    def test_initialize_materializes_group(self):
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend, is_tbo_enabled

        self._init(moe_a2a_backend="deepep", enable_two_batch_overlap=True)
        self.assertTrue(get_moe_a2a_backend().is_deepep())
        self.assertTrue(is_tbo_enabled())
        self.assertEqual(get_flags().moe.deepep_config, "")

    def test_speculative_swap_and_restore(self):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
            speculative_moe_a2a_backend_context,
            speculative_moe_backend_context,
        )

        self._init(
            moe_a2a_backend="deepep",
            moe_runner_backend="triton",
            speculative_moe_runner_backend="auto",
            speculative_moe_a2a_backend="none",
        )
        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertEqual(get_moe_runner_backend().name, "AUTO")
            self.assertTrue(get_moe_a2a_backend().is_none())
            # MTP layers are unquantized: fp4 allgather is forced off
            self.assertTrue(get_flags().moe.disable_fp4_allgather)
        self.assertEqual(get_moe_runner_backend().name, "TRITON")
        self.assertTrue(get_moe_a2a_backend().is_deepep())
        self.assertFalse(get_flags().moe.disable_fp4_allgather)

    def test_swap_restores_on_exception(self):
        from sglang.srt.layers.moe.utils import (
            get_moe_runner_backend,
            speculative_moe_backend_context,
        )

        self._init(moe_runner_backend="triton", speculative_moe_runner_backend="auto")
        with self.assertRaises(RuntimeError):
            with speculative_moe_backend_context():
                raise RuntimeError("boom")
        self.assertEqual(get_moe_runner_backend().name, "TRITON")


class TestDpFlagsGroup(_IsolatedServerArgs):
    """flags.dp: the DP-attention runtime flags; is_dp_attention_enabled is a
    thin shim over the group leaf."""

    def test_shim_reads_the_leaf(self):
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        reset_context()
        self.assertFalse(is_dp_attention_enabled())
        get_flags().dp.enabled = True
        self.assertTrue(is_dp_attention_enabled())

    def test_scoped_override_forces_the_predicate(self):
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        reset_context()
        with get_flags().dp.override(enabled=True):
            self.assertTrue(is_dp_attention_enabled())
        self.assertFalse(is_dp_attention_enabled())


class TestResources(_IsolatedServerArgs):
    """ctx.resources: named slots for process-level resource handles with one
    reset lifecycle; owning accessors keep their creation/publish semantics."""

    def test_graph_pool_lazy_create_and_reuse(self):
        from types import SimpleNamespace

        from sglang.srt.model_executor.runner_utils.pool import (
            get_global_graph_memory_pool,
            get_or_create_global_graph_memory_pool,
        )

        reset_context()
        self.assertIsNone(get_global_graph_memory_pool())
        dev = SimpleNamespace(graph_pool_handle=lambda: object())
        handle = get_or_create_global_graph_memory_pool(dev)
        self.assertIs(get_or_create_global_graph_memory_pool(dev), handle)

    def test_expert_recorder_noop_default_and_injection(self):
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.runtime_context import get_resources

        reset_context()
        self.assertEqual(
            type(get_global_expert_distribution_recorder()).__name__,
            "_ExpertDistributionRecorderNoop",
        )
        with get_resources().override(expert_distribution_recorder="mock"):
            self.assertEqual(get_global_expert_distribution_recorder(), "mock")

    def test_expert_location_metadata_publish_once_until_reset(self):
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
            set_global_expert_location_metadata,
        )

        reset_context()
        self.assertIsNone(get_global_expert_location_metadata())
        set_global_expert_location_metadata("meta")
        with self.assertRaises(AssertionError):
            set_global_expert_location_metadata("again")
        reset_context()
        self.assertIsNone(get_global_expert_location_metadata())


class TestNamedStreams(_IsolatedServerArgs):
    """ctx.get_stream(name): keyed get-or-create (the persistent-buffer
    pattern); set_stream installs explicitly."""

    def test_get_or_create_shares_by_name(self):
        from unittest.mock import patch

        reset_context()
        created = []

        class _FakeStream:
            def __init__(self):
                created.append(self)

        with patch("torch.cuda.Stream", _FakeStream):
            a = get_context().get_stream("alt")
            b = get_context().get_stream("alt")
            c = get_context().get_stream("other")
        self.assertIs(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(len(created), 2)

    def test_get_buffer_keyed_lazy(self):
        reset_context()
        created = []

        def factory():
            created.append(object())
            return created[-1]

        a = get_context().get_buffer("ws", factory)
        b = get_context().get_buffer("ws", factory)
        self.assertIs(a, b)
        self.assertEqual(len(created), 1)
        self.assertIsNot(get_context().get_buffer("other", factory), a)

    def test_set_stream_installs_explicitly(self):
        reset_context()
        sentinel = object()
        get_context().set_stream("alt", sentinel)
        self.assertIs(get_context().get_stream("alt"), sentinel)

    def test_reset_clears_the_registry(self):
        reset_context()
        get_context().set_stream("alt", object())
        reset_context()
        self.assertEqual(get_context().resources.streams, {})

    def test_capturer_slots_roundtrip_and_reset(self):
        from sglang.srt.state_capturer.indexer_topk import (
            get_global_indexer_capturer,
            set_global_indexer_capturer,
        )
        from sglang.srt.state_capturer.routed_experts import (
            get_global_experts_capturer,
            set_global_experts_capturer,
        )

        reset_context()
        self.assertIsNone(get_global_indexer_capturer())
        self.assertIsNone(get_global_experts_capturer())
        indexer, experts = object(), object()
        set_global_indexer_capturer(indexer)
        set_global_experts_capturer(experts)
        self.assertIs(get_global_indexer_capturer(), indexer)
        self.assertIs(get_global_experts_capturer(), experts)
        reset_context()
        self.assertIsNone(get_global_indexer_capturer())
        self.assertIsNone(get_global_experts_capturer())

    def test_tcp_store_slot_roundtrip_and_reset(self):
        from sglang.srt.distributed.utils import (
            get_global_tcp_store,
            set_global_tcp_store,
        )

        reset_context()
        self.assertIsNone(get_global_tcp_store())
        store = object()
        set_global_tcp_store(store)
        self.assertIs(get_global_tcp_store(), store)
        reset_context()
        self.assertIsNone(get_global_tcp_store())

    def test_trace_level_env_seeded_lazy_default(self):
        from sglang.srt.observability.trace import (
            get_global_trace_level,
            set_global_trace_level,
        )

        reset_context()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TRACE_LEVEL", None)
            self.assertEqual(get_global_trace_level(), 3)
        set_global_trace_level(5)
        self.assertEqual(get_global_trace_level(), 5)
        reset_context()
        with patch.dict(os.environ, {"SGLANG_TRACE_LEVEL": "1"}):
            self.assertEqual(get_global_trace_level(), 1)


class TestEpBufferState(_IsolatedServerArgs):
    """EP dispatcher buffer managers: state lives on ctx.resources; the
    facade keeps the mode-transition and clean semantics."""

    def test_deepep_dispatch_mode_transitions_and_reset(self):
        try:
            from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
        except ImportError:
            self.skipTest("deep_ep not installed")

        reset_context()
        cleans = []

        class _FakeBuffer:
            low_latency_mode = True

            def clean_low_latency_buffer(self, *args):
                cleans.append(args)

        state = DeepEPBuffer._state()
        state.buffer = _FakeBuffer()
        state.hidden_size = 7168
        state.num_max_dispatch_tokens_per_rank = 128
        state.num_experts = 256

        DeepEPBuffer.set_dispatch_mode_as_normal()
        # NORMAL -> LOW_LATENCY must clean the low-latency buffer once.
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        self.assertEqual(cleans, [(128, 7168, 256)])
        # LOW_LATENCY -> LOW_LATENCY must not clean again.
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        self.assertEqual(len(cleans), 1)

        reset_context()
        self.assertIsNone(DeepEPBuffer._state().buffer)


class TestForwardFlags(_IsolatedServerArgs):
    """ctx.forward: contextvar-backed per-forward flags; scoped() restores,
    threads see defaults."""

    def test_scoped_set_restore_and_nesting(self):
        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        self.assertFalse(fwd.multi_stream)
        with fwd.scoped(multi_stream=True):
            self.assertTrue(fwd.multi_stream)
            with fwd.scoped(multi_stream=False):
                self.assertFalse(fwd.multi_stream)
            self.assertTrue(fwd.multi_stream)
        self.assertFalse(fwd.multi_stream)

    def test_scoped_restores_on_exception_and_validates_keys(self):
        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        with self.assertRaises(RuntimeError):
            with fwd.scoped(moe_output_buffer="buf"):
                raise RuntimeError("boom")
        self.assertIsNone(fwd.moe_output_buffer)
        with self.assertRaises(ValueError):
            with fwd.scoped(nope=1):
                pass
        with self.assertRaises(AttributeError):
            fwd.multi_stream = True  # attribute writes are rejected

    def test_threads_see_defaults(self):
        import threading

        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        seen = {}
        with fwd.scoped(multi_stream=True):

            def probe():
                seen["value"] = get_forward().multi_stream

            worker = threading.Thread(target=probe)
            worker.start()
            worker.join()
        self.assertFalse(seen["value"])  # a new thread sees the default

    def test_graph_visible_flags_trace_under_torch_compile(self):
        # Regression: dynamo cannot trace ContextVar.get, and these flags are
        # read inside compiled model code (vocab embedding, communicator, DP
        # gather) — they must stay plain-slot backed. fullgraph=True turns
        # any graph break back into a failure.
        import torch

        from sglang.srt.runtime_context import get_forward

        reset_context()

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def probe(x):
            fwd = get_forward()
            if fwd.attn_input_scattered:
                x = x + 1
            if fwd.is_extend_in_batch:
                x = x + 2
            if fwd.fuse_mlp_allreduce:
                x = x + 4
            if fwd.mlp_reduce_scatter:
                x = x + 8
            if fwd.flashinfer_trtllm_bypass:
                x = x + 16
            return x

        self.assertEqual(probe(torch.zeros(())).item(), 0)
        with get_forward().scoped(attn_input_scattered=True):
            self.assertEqual(probe(torch.zeros(())).item(), 1)
        get_forward().set("is_extend_in_batch", True)
        self.assertEqual(probe(torch.zeros(())).item(), 2)
        get_forward().set("is_extend_in_batch", False)
        with get_forward().scoped(
            fuse_mlp_allreduce=True,
            mlp_reduce_scatter=True,
            flashinfer_trtllm_bypass=True,
        ):
            self.assertEqual(probe(torch.zeros(())).item(), 28)
        self.assertEqual(probe(torch.zeros(())).item(), 0)

    def test_parallel_config_leaves_trace_under_torch_compile(self):
        # Regression: gate helpers such as ``enable_moe_dense_fully_dp()`` read
        # parallel config leaves inside compiled model forwards through the
        # `config` property, which must stay dynamo-traceable
        # (``object.__getattribute__`` graph-breaks).
        # fullgraph=True turns any graph break back into a failure.
        import torch

        from sglang.srt.runtime_context import get_parallel

        reset_context()
        with get_context().override_server_args(moe_dense_tp_size=1, dwdp_size=4):

            @torch.compile(fullgraph=True, backend="eager", dynamic=False)
            def probe(x):
                par = get_parallel()
                if par.enable_prefill_context_parallel:
                    x = x + 1
                if par.moe_dense_tp_size == 1:
                    x = x + 2
                if par.dwdp_size > 1:
                    x = x + 4
                return x

            self.assertEqual(probe(torch.zeros(())).item(), 6)

    def test_graph_visible_flags_are_process_visible_across_threads(self):
        # Documented divergence from the contextvar-backed flags: plain slots
        # are process-global (the storage form these flags had before the
        # tier), so another thread sees the current value, not the default.
        import threading

        from sglang.srt.runtime_context import get_forward

        reset_context()
        seen = {}
        with get_forward().scoped(attn_input_scattered=True):

            def probe():
                seen["value"] = get_forward().attn_input_scattered

            worker = threading.Thread(target=probe)
            worker.start()
            worker.join()
        self.assertTrue(seen["value"])
        self.assertFalse(get_forward().attn_input_scattered)

    def test_multi_stream_shims(self):
        from sglang.srt.utils.multi_stream_utils import (
            do_multi_stream,
            with_multi_stream,
        )

        reset_context()
        self.assertFalse(do_multi_stream())
        with with_multi_stream(True):
            self.assertTrue(do_multi_stream())
        self.assertFalse(do_multi_stream())

    def test_attn_tp_context_per_forward_slots(self):
        from types import SimpleNamespace

        from sglang.srt.layers.communicator import get_attn_tp_context
        from sglang.srt.runtime_context import get_forward

        reset_context()
        ctx = get_attn_tp_context()
        self.assertFalse(ctx.input_scattered)
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False, is_target_verify=lambda: False
            ),
            input_ids=None,
            can_run_tbo=False,
        )
        sentinel = SimpleNamespace(fetch_qkv_latent=lambda: "qkv")
        with ctx.maybe_input_scattered(fb):
            ctx.set_attn_inputs(sentinel)
            self.assertEqual(ctx.fetch_qkv_latent(), "qkv")
        # attn inputs are cleared at scope exit, flag restored
        self.assertIsNone(get_forward().attn_inputs)
        self.assertFalse(ctx.input_scattered)

    def test_dp_buffer_state_split(self):
        import torch

        from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as wrapper
        from sglang.srt.layers.dp_attention import (
            get_dp_dtype,
            get_dp_global_num_tokens,
            get_global_dp_buffer_len,
            is_dp_max_padding,
            set_dp_buffer_len,
        )

        reset_context()
        # metadata is init-static (flags.dp); sizing is per-forward sticky
        wrapper.set_metadata(64, torch.float16, torch.device("cpu"))
        self.assertEqual(get_dp_dtype(), torch.float16)
        set_dp_buffer_len(128, 32, True, [64, 64])
        self.assertEqual(get_global_dp_buffer_len(), 128)
        self.assertTrue(is_dp_max_padding())
        self.assertEqual(get_dp_global_num_tokens(), [64, 64])
        set_dp_buffer_len(256, 64, False)  # sticky until the next write
        self.assertEqual(get_global_dp_buffer_len(), 256)
        self.assertFalse(is_dp_max_padding())
        self.assertIsNone(get_dp_global_num_tokens())
        reset_context()
        self.assertIsNone(get_dp_dtype())

    def test_is_extend_in_batch_sticky_within_thread(self):
        from sglang.srt.layers.dp_attention import (
            get_is_extend_in_batch,
            set_is_extend_in_batch,
        )

        reset_context()
        self.assertFalse(get_is_extend_in_batch())
        set_is_extend_in_batch(True)
        self.assertTrue(get_is_extend_in_batch())  # sticky until next write
        set_is_extend_in_batch(False)
        self.assertFalse(get_is_extend_in_batch())

    def test_moe_output_buffer_ctx(self):
        from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx
        from sglang.srt.runtime_context import get_forward

        reset_context()
        sentinel = object()
        with moe_output_buffer_ctx(sentinel):
            self.assertIs(get_forward().moe_output_buffer, sentinel)
        self.assertIsNone(get_forward().moe_output_buffer)

    def test_mlp_comm_forward_flags(self):
        """Decoder-published MLP collective flags: scoped restore + skip helpers."""
        from sglang.srt.layers.moe.utils import (
            should_skip_mlp_all_reduce,
            should_skip_post_experts_all_reduce,
        )
        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        self.assertFalse(fwd.fuse_mlp_allreduce)
        self.assertFalse(fwd.mlp_reduce_scatter)
        self.assertFalse(fwd.flashinfer_trtllm_bypass)
        self.assertFalse(should_skip_mlp_all_reduce())

        with fwd.scoped(fuse_mlp_allreduce=True):
            self.assertTrue(fwd.fuse_mlp_allreduce)
            self.assertTrue(should_skip_mlp_all_reduce())
            # Fusion alone is enough to skip post-experts AR.
            self.assertTrue(should_skip_post_experts_all_reduce(is_tp_path=True))
        self.assertFalse(fwd.fuse_mlp_allreduce)
        self.assertFalse(should_skip_mlp_all_reduce())

        with fwd.scoped(mlp_reduce_scatter=True):
            self.assertTrue(fwd.mlp_reduce_scatter)
            self.assertTrue(should_skip_mlp_all_reduce())
        self.assertFalse(fwd.mlp_reduce_scatter)

        with fwd.scoped(flashinfer_trtllm_bypass=True):
            self.assertTrue(fwd.flashinfer_trtllm_bypass)
        self.assertFalse(fwd.flashinfer_trtllm_bypass)

    def test_dp_reduce_scatterv_requires_single_rank_attention_dp_shards(self):
        from sglang.srt.layers.moe.utils import should_use_dp_reduce_scatterv

        reset_context()
        with patch(
            "sglang.srt.layers.moe.utils.is_dp_attention_enabled",
            return_value=True,
        ):
            # The optimized path is valid when the collective group and the
            # variable-split list have the same number of entries.
            with get_parallel().override(tp_size=8, attn_dp_size=8, moe_ep_size=8):
                self.assertTrue(should_use_dp_reduce_scatterv())

            # Otherwise the standard all-reduce plus scatter path must be used.
            with get_parallel().override(tp_size=8, attn_dp_size=2, moe_ep_size=2):
                self.assertFalse(should_use_dp_reduce_scatterv())


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


class TestDerivedPredicatesAgreeAcrossTiers(_IsolatedServerArgs):
    """One definition per predicate, checked rather than asserted in prose.

    Each of these exists twice by construction -- once over a config-shaped
    object (the resolution pipeline's `*_of` helper, which `ServerArgs`
    delegates to) and once over the published bags. The pair must agree on
    every input, or a decision made before publish differs from the same
    decision made after it.
    """

    _STRATEGIES = ("auto", "no_buffer", "extra_buffer", "extra_buffer_lazy")

    def test_mamba_extra_buffer_matches_the_member(self):
        from sglang.srt.runtime_context import (
            mamba_extra_buffer_enabled,
            mamba_extra_buffer_lazy_enabled,
        )

        for disable_radix_cache in (False, True):
            for strategy in self._STRATEGIES:
                with self.subTest(radix=disable_radix_cache, strategy=strategy):
                    args = _FakeResolvedArgs(
                        disable_radix_cache=disable_radix_cache,
                        mamba_radix_cache_strategy=strategy,
                    )
                    get_context().set_server_args(args)
                    self.assertEqual(
                        ServerArgs.enable_mamba_extra_buffer(args),
                        mamba_extra_buffer_enabled(),
                    )
                    self.assertEqual(
                        ServerArgs.enable_mamba_extra_buffer_lazy(args),
                        mamba_extra_buffer_lazy_enabled(),
                    )

    def test_prefill_buffer_ceiling_matches_the_member(self):
        from sglang.srt.runtime_context import max_prefill_buffer_tokens

        for chunked in (-1, 0, 1024, 8192):
            for dynamic in (False, True):
                for pp in (1, 4):
                    for max_prefill in (0, 2048, 16384):
                        with self.subTest(
                            chunked=chunked,
                            dynamic=dynamic,
                            pp=pp,
                            max_prefill=max_prefill,
                        ):
                            args = _FakeResolvedArgs(
                                chunked_prefill_size=chunked,
                                enable_dynamic_chunking=dynamic,
                                pp_size=pp,
                                max_prefill_tokens=max_prefill,
                            )
                            get_context().set_server_args(args)
                            self.assertEqual(
                                ServerArgs.max_prefill_buffer_tokens(args),
                                max_prefill_buffer_tokens(),
                            )

    def test_activation_reserve_matches_the_member(self):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import pre_capture_activation_reserve_mb

        graph = SimpleNamespace(decode=SimpleNamespace(max_bs=64))
        cases = (
            dict(disaggregation_mode="null", chunked_prefill_size=8192),
            dict(disaggregation_mode="null", chunked_prefill_size=-1),
            dict(
                disaggregation_mode="null",
                chunked_prefill_size=-1,
                max_prefill_tokens=1024,
            ),
            dict(disaggregation_mode="decode", max_running_requests=32),
            dict(disaggregation_mode="decode", max_running_requests=None),
            dict(
                disaggregation_mode="decode",
                max_running_requests=None,
                speculative_num_draft_tokens=4,
            ),
            dict(
                disaggregation_mode="null",
                chunked_prefill_size=8192,
                tp_size=8,
                pp_size=2,
            ),
        )
        for case in cases:
            for gpu_mem in (None, 20 * 1024, 80 * 1024):
                with self.subTest(gpu_mem=gpu_mem, **case):
                    args = _FakeResolvedArgs(cuda_graph_config=graph, **case)
                    get_context().set_server_args(args)
                    self.assertEqual(
                        ServerArgs.pre_capture_activation_reserve_mb(args, gpu_mem),
                        pre_capture_activation_reserve_mb(gpu_mem),
                    )

    def test_remote_instance_transfer_engine_matches_the_member(self):
        from sglang.srt.runtime_context import remote_instance_transfer_engine_enabled

        backends = ("nccl", "transfer_engine", "modelexpress")
        transports = (None, '{"transport": "transfer_engine"}', '{"transport": "nixl"}')
        for seed_via_te in (False, True):
            for load_format in ("auto", "remote_instance"):
                for backend in backends:
                    for mx in transports:
                        with self.subTest(
                            seed=seed_via_te,
                            load_format=load_format,
                            backend=backend,
                            modelexpress=mx,
                        ):
                            args = _FakeResolvedArgs(
                                load_format=load_format,
                                remote_instance_weight_loader_backend=backend,
                                remote_instance_weight_loader_start_seed_via_transfer_engine=seed_via_te,
                                modelexpress_config=mx,
                            )
                            get_context().set_server_args(args)
                            for override in (None, "remote_instance", "auto"):
                                self.assertEqual(
                                    ServerArgs.remote_instance_weight_loader_use_transfer_engine(
                                        args, override
                                    ),
                                    remote_instance_transfer_engine_enabled(override),
                                )

    def test_attention_backends_match_the_member(self):
        from sglang.srt.runtime_context import attention_backends

        backends = (None, "fa3", "triton")
        for base in backends:
            for prefill in backends:
                for decode in backends:
                    with self.subTest(base=base, prefill=prefill, decode=decode):
                        args = _FakeResolvedArgs(
                            attention_backend=base,
                            prefill_attention_backend=prefill,
                            decode_attention_backend=decode,
                        )
                        get_context().set_server_args(args)
                        self.assertEqual(
                            ServerArgs.get_attention_backends(args),
                            attention_backends(),
                        )


class TestAdaptiveDraftBoundLifecycle(_IsolatedServerArgs):
    """The adaptive draft-token bound is memoized on the config path, so the
    memo has to end with the publication it was computed under.

    Without that, a process that republishes with the same adaptive-config path
    -- the file having been rewritten in between -- keeps the previous bound and
    under-allocates the draft-token buffers sized from it.
    """

    def _write_config(self, steps):
        path = os.path.join(tempfile.mkdtemp(prefix="adaptive_cfg_"), "adaptive.json")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), ignore_errors=True)
        with open(path, "w") as handle:
            json.dump({"1": {"candidate_steps": steps}}, handle)
        return path

    def test_republishing_recomputes_the_bound(self):
        path = self._write_config([2])
        get_context().set_server_args(
            _FakeResolvedArgs(
                speculative_num_draft_tokens=3,
                speculative_adaptive=True,
                speculative_adaptive_config=path,
            )
        )
        self.assertEqual(max_speculative_num_draft_tokens(), 3)

        with open(path, "w") as handle:
            json.dump({"1": {"candidate_steps": [4]}}, handle)
        # Same path, new contents: the memo must not survive the republish.
        get_context().set_server_args(
            _FakeResolvedArgs(
                speculative_num_draft_tokens=3,
                speculative_adaptive=True,
                speculative_adaptive_config=path,
            )
        )
        self.assertEqual(max_speculative_num_draft_tokens(), 5)

    def test_reset_clears_the_bound(self):
        path = self._write_config([2])
        get_context().set_server_args(
            _FakeResolvedArgs(
                speculative_num_draft_tokens=3,
                speculative_adaptive=True,
                speculative_adaptive_config=path,
            )
        )
        self.assertEqual(max_speculative_num_draft_tokens(), 3)
        reset_context()
        with open(path, "w") as handle:
            json.dump({"1": {"candidate_steps": [6]}}, handle)
        get_context().set_server_args(
            _FakeResolvedArgs(
                speculative_num_draft_tokens=3,
                speculative_adaptive=True,
                speculative_adaptive_config=path,
            )
        )
        self.assertEqual(max_speculative_num_draft_tokens(), 7)


class TestNamedAccessorsCallWhatTheyWrap(CustomTestCase):
    """A named accessor must *call* a member that is a method.

    `return get_server_args().x` hands back a bound method when `x` is defined
    with `def`; the failure then lands far away, in whatever arithmetic the
    caller does with it. Checked statically so accessors that need a real model
    config are covered too.
    """

    def test_accessors_that_wrap_methods_call_them(self):
        import ast
        import functools
        import inspect

        import sglang.srt.runtime_context as rc
        from sglang.srt.server_args import ServerArgs

        tree = ast.parse(inspect.getsource(rc))
        wrong = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if not (isinstance(inner, ast.Return) and inner.value is not None):
                    continue
                value = inner.value
                called = isinstance(value, ast.Call)
                target = value.func if called else value
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Call)
                    and isinstance(target.value.func, ast.Name)
                    and target.value.func.id == "get_server_args"
                ):
                    continue
                member = getattr(ServerArgs, target.attr, None)
                # A `property` / `functools.cached_property` member is already
                # evaluated by the attribute access, so it is named here to keep
                # the failure message from calling it "not a method" -- the fix
                # for those is the opposite one.
                kind = (
                    "a property"
                    if isinstance(member, (property, functools.cached_property))
                    else "not a method"
                )
                if inspect.isfunction(member) and not called:
                    wrong.append(
                        f"{node.name}(): returns ServerArgs.{target.attr} without "
                        "calling it, so callers get a bound method"
                    )
                if not inspect.isfunction(member) and called:
                    wrong.append(
                        f"{node.name}(): calls ServerArgs.{target.attr}, which is "
                        f"{kind} -- the attribute access already produced the value"
                    )
        self.assertEqual([], wrong, "\n".join(wrong))


if __name__ == "__main__":
    unittest.main()

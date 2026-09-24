"""Unit tests for runtime configuration, process placement, and overrides."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=33, suite="base-a-test-cpu")

import dataclasses
import json
import os
import pathlib as _pathlib
import shutil
import tempfile
import types
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import msgspec.structs

import sglang as _sglang
import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups import prefill_buffer_ceiling
from sglang.srt.arg_groups.arg_utils import NS, A, Arg
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
)
from sglang.srt.arg_groups.overrides import (
    mamba_cache_chunk_size as mamba_cache_chunk_size_of,
)
from sglang.srt.arg_groups.overrides import (
    max_prefill_buffer_tokens as max_prefill_buffer_tokens_of,
)
from sglang.srt.arg_groups.overrides import (
    resolution_result,
    resolved_view,
)
from sglang.srt.runtime_context import (
    Flags,
    ParallelContext,
    RuntimeContext,
    SpawnRanks,
    _FlagGroupBase,
    assert_published,
    derive_parallel_widths,
    get_context,
    get_device,
    get_exec,
    get_flags,
    get_parallel,
    get_schedule,
    get_server_args,
    max_prefill_buffer_tokens,
    max_speculative_num_draft_tokens,
    publish,
    publish_role,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_SRT = _pathlib.Path(next(iter(_sglang.__path__))).resolve() / "srt"
_PACKAGE = _pathlib.Path(next(iter(_sglang.__path__))).resolve()


def _sources():
    """Yield Python sources in the package and checkout siblings.

    Installed packages without a checkout scan only the package.
    """
    roots = [_PACKAGE]
    checkout = _PACKAGE.parents[1]
    roots += [
        checkout / name
        for name in ("benchmark", "examples", "scripts", "test")
        if (checkout / name).is_dir()
    ]
    for root in roots:
        for path in root.rglob("*.py"):
            yield path


_PS = "sglang.srt.distributed.parallel_state"


def _parallel_state():
    from sglang.srt.distributed import parallel_state

    return parallel_state


_DP = "sglang.srt.layers.dp_attention"

# Model-parallel group names and their module globals. WORLD is initialized separately.
GROUP_STAMPS = {
    "tp_group": "_TP",
    "dcp_group": "_DCP",
    "pp_group": "_PP",
    "moe_ep_group": "_MOE_EP",
    "moe_dp_group": "_MOE_DP",
    "moe_tp_group": "_MOE_TP",
    "attn_tp_group": "_ATTN_TP",
    "attn_cp_group": "_ATTN_CP",
    "shared_experts_tp_group": "_SHARED_EXPERTS_TP",
}


def _groups_the_build_states() -> dict:
    """Extract context-to-module group mappings from initialization source."""
    import ast
    import inspect
    import textwrap

    from sglang.srt.distributed import parallel_state

    body = textwrap.dedent(inspect.getsource(parallel_state.initialize_model_parallel))
    for node in ast.walk(ast.parse(body)):
        keys = getattr(node, "keys", None)
        if (
            isinstance(node, ast.Dict)
            and keys
            and all(
                isinstance(k, ast.Constant) and str(k.value).endswith("_group")
                for k in keys
            )
        ):
            return {k.value: v.id for k, v in zip(node.keys, node.values)}
    raise AssertionError("initialize_model_parallel states no group at all")


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


class TestTheBuildStatesEveryGroup(_IsolatedOverrides):
    """Group initialization publishes every available handle to the context."""

    def test_the_build_states_every_group_the_context_declares(self):
        from sglang.srt.runtime_context import _parallel_fields

        declared = {name for name in _parallel_fields() if name.endswith("_group")}
        self.assertEqual(declared, set(GROUP_STAMPS) | {"world_group"})

    def test_the_world_group_is_stated_where_it_is_built(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.distributed import parallel_state

        body = textwrap.dedent(
            inspect.getsource(parallel_state.init_distributed_environment)
        )
        stated = {
            kw.arg
            for node in ast.walk(ast.parse(body))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "override_permanently"
            for kw in node.keywords
        }
        self.assertIn("world_group", stated)

    def test_nothing_reads_a_name_this_build_has_not_stated_yet(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.distributed import parallel_state

        body = textwrap.dedent(
            inspect.getsource(parallel_state.initialize_model_parallel)
        )
        read = set()
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = parallel_state._CONTEXT_NAME_OF.get(node.func.id)
                if name is not None:
                    read.add(name)
        self.assertTrue(read, "no getter is called here; this proves nothing")
        self.assertEqual(read - {"world_group"}, set())

    def test_each_group_is_stated_from_the_global_it_was_built_into(self):
        self.assertEqual(_groups_the_build_states(), GROUP_STAMPS)

    def test_a_dimension_the_configuration_has_not_got_is_left_unstated(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.distributed import parallel_state

        body = textwrap.dedent(
            inspect.getsource(parallel_state.initialize_model_parallel)
        )
        stamp = next(
            node
            for node in ast.walk(ast.parse(body))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "override_permanently"
        )
        self.assertEqual([keyword.arg for keyword in stamp.keywords], [None])
        mapping = stamp.keywords[0].value
        self.assertIsInstance(mapping, ast.DictComp)
        self.assertEqual(len(mapping.generators), 1)
        guards = mapping.generators[0].ifs
        self.assertEqual(len(guards), 1)
        guard = guards[0]
        self.assertIsInstance(guard, ast.Compare)
        self.assertEqual([type(op) for op in guard.ops], [ast.IsNot])
        self.assertEqual([c.value for c in guard.comparators], [None])
        self.assertEqual(ast.unparse(guard.left), ast.unparse(mapping.value))


class TestParallelDelegation(_IsolatedOverrides):
    def test_wrapper_holds_no_resolved_state(self):
        # __slots__: no __dict__; the only instance state is the override hook.
        self.assertFalse(hasattr(get_parallel(), "__dict__"))
        # tp_group IS exposed: live delegation handles PD-multiplexing / the tp patch.
        self.assertTrue(hasattr(ParallelContext, "tp_group"))
        # local_attn_dp is intentionally not part of the wrapper surface.
        self.assertFalse(hasattr(ParallelContext, "local_attn_dp_size"))


class TestTheTwoWorldWidths(_IsolatedOverrides):
    """Launch width and WORLD capacity are distinct configuration values."""

    def _published(self, **fields):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", **fields), role="test")
        return get_parallel()

    def test_the_launch_width_is_a_rank_per_stage_of_each_group(self):
        self.assertEqual(self._published(tp_size=4, pp_size=2).launch_world_size, 8)

    def test_the_launch_width_spans_the_ranks_a_joiner_came_in_above(self):
        parallel = self._published(tp_size=4, pp_size=1, ep_join_rank_offset=8)
        self.assertEqual(parallel.launch_world_size, 12)

    def test_the_ceiling_is_the_configured_one_when_there_is_one(self):
        self.assertEqual(self._published(tp_size=4, max_ep_size=32).max_world_size, 32)

    def test_without_a_configured_ceiling_the_room_is_the_launch_width(self):
        parallel = self._published(tp_size=8)
        self.assertEqual(parallel.launch_world_size, 8)
        self.assertEqual(parallel.max_world_size, 8)

    def test_each_width_can_be_stated_on_its_own(self):
        parallel = self._published(tp_size=8)
        with parallel.override(launch_world_size=2):
            self.assertEqual(parallel.launch_world_size, 2)
            self.assertEqual(parallel.max_world_size, 8)
            with parallel.override(max_world_size=6):
                self.assertEqual(parallel.max_world_size, 6)
                self.assertEqual(parallel.launch_world_size, 2)


class TestSpawnIdentities(_IsolatedOverrides):
    """Publish records the launcher-assigned ranks and device."""

    def setUp(self):
        super().setUp()
        parallel = get_parallel()
        self._saved_stamp = dict(parallel._stamp)
        self.addCleanup(
            lambda: (
                parallel.clear_stamp(),
                parallel.override_permanently(**self._saved_stamp),
            )
        )
        reset_context()
        self.addCleanup(reset_context)

    def test_one_rank_fixes_the_rest(self):
        publish(
            ServerArgs(model_path="dummy", tp_size=4, pp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=5, dp_rank=2),
        )
        parallel = get_parallel()
        self.assertEqual(parallel.launch_world_rank, 5)
        self.assertEqual(parallel.tp_rank, 1)
        self.assertEqual(parallel.pp_rank, 1)
        self.assertEqual(parallel.dp_rank, 2)

    def test_the_spawn_states_the_device_and_the_record_stays_clean(self):
        server_args = ServerArgs(model_path="dummy")
        publish(
            server_args,
            role="test",
            ranks=SpawnRanks(world_rank=0, gpu_id=3),
        )
        self.assertEqual(get_device().gpu_id, 3)
        # `gpu_id` is a runtime field, not a `ServerArgs` input.
        self.assertNotIn(
            "gpu_id", {f.name for f in msgspec.structs.fields(type(server_args))}
        )

    def test_a_process_on_no_device_is_told_nothing(self):
        publish(
            ServerArgs(model_path="dummy"),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        self.assertIsNone(get_device().gpu_id)

    def test_no_controller_is_an_answer_not_a_failure(self):
        publish(
            ServerArgs(model_path="dummy", tp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=0, dp_rank=None),
        )
        self.assertIsNone(get_parallel().dp_rank)

    def test_publishing_without_a_bundle_names_what_is_missing(self):
        publish(ServerArgs(model_path="dummy", tp_size=2), role="test")
        with self.assertRaises(RuntimeError) as caught:
            get_parallel().dp_rank
        self.assertIn("rank bundle", str(caught.exception))

    def test_the_attention_rank_keeps_its_own_explanation(self):
        publish(ServerArgs(model_path="dummy", tp_size=2), role="test")
        with self.assertRaises(RuntimeError) as caught:
            get_parallel().attn_dp_rank
        self.assertIn("initialize_dp_attention", str(caught.exception))


class TestAttentionRanksComeFromPublish(_IsolatedOverrides):
    """Published ranks are available before distributed initialization."""

    def setUp(self):
        super().setUp()
        parallel = get_parallel()
        self._saved_stamp = dict(parallel._stamp)
        self.addCleanup(
            lambda: (
                parallel.clear_stamp(),
                parallel.override_permanently(**self._saved_stamp),
            )
        )
        reset_context()
        self.addCleanup(reset_context)

    def test_it_matches_the_topology_init_for_every_shape(self):
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

        shapes = [
            (8, 1, 1, False),
            (8, 2, 1, True),
            (8, 4, 1, True),
            (8, 2, 2, True),
            (16, 4, 2, True),
        ]
        for tp_size, dp_size, attn_cp_size, dp_attn in shapes:
            for tp_rank in range(tp_size):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy",
                        tp_size=tp_size,
                        dp_size=dp_size,
                        attn_cp_size=attn_cp_size,
                        enable_dp_attention=dp_attn,
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=tp_rank),
                )
                want_tp, _, want_dp, _ = compute_dp_attention_world_info(
                    dp_attn, tp_rank, tp_size, dp_size, attn_cp_size
                )
                msg = f"tp={tp_size} dp={dp_size} cp={attn_cp_size} rank={tp_rank}"
                self.assertEqual(get_parallel().attn_tp_rank, want_tp, msg)
                self.assertEqual(get_parallel().attn_dp_rank, want_dp, msg)

    def test_the_rank_reads_without_a_process_group(self):
        publish(
            ServerArgs(
                model_path="dummy", tp_size=8, dp_size=2, enable_dp_attention=True
            ),
            role="test",
            ranks=SpawnRanks(world_rank=5),
        )
        with patch.object(_parallel_state(), "_ATTN_TP", None):
            self.assertEqual(get_parallel().attn_tp_rank, 1)
            self.assertEqual(get_parallel().attn_dp_rank, 1)

    def test_without_a_bundle_a_rank_read_says_what_is_missing(self):
        publish(ServerArgs(model_path="dummy", tp_size=8), role="test")
        with self.assertRaises(RuntimeError) as caught:
            get_parallel().attn_tp_rank
        message = str(caught.exception)
        self.assertIn("has not been written in this process", message)
        self.assertIn("override(attn_tp_rank=...)", message)


class TestStampedRanks(_IsolatedOverrides):
    """Attention-DP rank reads require an explicit runtime value."""

    def setUp(self):
        super().setUp()
        parallel = get_parallel()
        self._saved_derived = dict(parallel._stamp)
        parallel.clear_stamp()
        self.addCleanup(
            lambda: (
                parallel.clear_stamp(),
                parallel.override_permanently(**self._saved_derived),
            )
        )

    def test_the_stamp_is_the_answer(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_dp_rank=3)
        self.assertEqual(parallel.attn_dp_rank, 3)
        parallel.override_permanently(attn_dp_rank=9)
        self.assertEqual(parallel.attn_dp_rank, 9)

    def test_a_scope_still_wins_over_the_stamp(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_dp_rank=3)
        with parallel.override(attn_dp_rank=0):
            self.assertEqual(parallel.attn_dp_rank, 0)
        self.assertEqual(parallel.attn_dp_rank, 3)

    def test_unstamped_names_the_cause(self):
        with self.assertRaises(RuntimeError) as caught:
            get_parallel().attn_dp_rank
        self.assertIn("initialize_dp_attention", str(caught.exception))

    def test_a_stated_width_reaches_the_padding_mode(self):
        from sglang.srt.layers.dp_attention import DpPaddingMode

        with get_parallel().override(attn_dp_size=1):
            mode = DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=True, global_num_tokens=[3, 5]
            )
        self.assertIs(mode, DpPaddingMode.MAX_LEN)

        with get_parallel().override(attn_dp_size=2):
            mode = DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=True, global_num_tokens=[3, 5]
            )
        self.assertIs(mode, DpPaddingMode.SUM_LEN)

    def test_the_gather_slot_follows_the_list_that_was_gathered(self):
        from sglang.srt.layers.dp_attention import dp_gather_slot

        self.addCleanup(reset_context)
        dp_flags = get_flags().dp
        saved = (
            dp_flags.use_world_group_for_gather,
            dp_flags.joiner_skip_all_gather,
        )

        def restore():
            (
                dp_flags.use_world_group_for_gather,
                dp_flags.joiner_skip_all_gather,
            ) = saved

        self.addCleanup(restore)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=8, dp_size=8, enable_dp_attention=True
            ),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        parallel = get_parallel()
        dp_flags.use_world_group_for_gather = False
        self.assertEqual(dp_gather_slot(), parallel.attn_dp_rank)

        # After a scale-up the gather spans the expanded WORLD, and the joining
        # cohort is numbered from its offset.
        dp_flags.use_world_group_for_gather = True
        dp_flags.joiner_skip_all_gather = False
        parallel.override_permanently(ep_join_rank_offset=8)
        self.assertEqual(dp_gather_slot(), 8 + parallel.tp_rank)
        self.assertEqual(parallel.attn_dp_size, 8)
        self.assertEqual(parallel.tp_size, 8)

    def test_a_scale_up_writes_no_width(self):
        from sglang.srt.layers.dp_attention import update_dp_attention_post_scale

        dp_flags = get_flags().dp
        saved_gather = dp_flags.use_world_group_for_gather
        self.addCleanup(setattr, dp_flags, "use_world_group_for_gather", saved_gather)
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=8, dp_size=8, enable_dp_attention=True
            ),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        parallel = get_parallel()
        before = (parallel.attn_dp_size, parallel.attn_dp_rank, parallel.tp_size)
        update_dp_attention_post_scale(new_dp_size=16, new_dp_rank=11)
        self.assertTrue(dp_flags.use_world_group_for_gather)
        self.assertEqual(
            (parallel.attn_dp_size, parallel.attn_dp_rank, parallel.tp_size), before
        )


class TestEveryDeclaredParallelNameIsStatable(_IsolatedOverrides):
    """Overrides accept exactly the declared parallel fields."""

    def test_every_declared_name_can_be_stated_and_reads_back(self):
        from sglang.srt.runtime_context import _parallel_fields

        names = sorted(_parallel_fields())
        # Sizes, ranks, groups and the configured leaves of the namespace.
        self.assertGreater(len(names), 30)
        parallel = get_parallel()
        for name in names:
            sentinel = object()
            with parallel.override(**{name: sentinel}):
                self.assertIs(getattr(parallel, name), sentinel, msg=name)

    def test_every_name_the_class_answers_for_is_in_the_set(self):
        from sglang.srt.runtime_context import _parallel_fields

        answered = {
            name
            for name, value in vars(ParallelContext).items()
            if isinstance(value, property)
        }
        self.assertTrue(answered)
        self.assertEqual(answered - _parallel_fields(), set())

    def test_a_live_name_is_never_also_answered_from_the_bag(self):
        from sglang.srt.runtime_context import (
            _derived_widths,
            _parallel_config_leaves,
        )

        self.assertEqual(set(_derived_widths()) & _parallel_config_leaves(), set())

    def test_an_undeclared_name_is_refused(self):
        with self.assertRaises(ValueError):
            with get_parallel().override(not_a_parallel_name=1):
                pass


class TestReadsWithoutAPublishedConfig(_IsolatedOverrides):
    """Overrides support shared SRT layers without published SRT configuration.

    Multimodal generation supplies its own TP group and widths this way.
    """

    def setUp(self):
        super().setUp()
        parallel = get_parallel()
        self._saved_stamp = dict(parallel._stamp)
        self.addCleanup(
            lambda: (
                parallel.clear_stamp(),
                parallel.override_permanently(**self._saved_stamp),
            )
        )
        reset_context()
        self.addCleanup(reset_context)

    def test_a_stamped_width_reads_with_nothing_published(self):
        parallel = get_parallel()
        self.assertIsNone(parallel._config)
        parallel.override_permanently(
            **derive_parallel_widths(
                tp_size=2,
                attn_cp_size=1,
                attn_dp_size=1,
                moe_ep_size=1,
                moe_dp_size=1,
                dcp_size=1,
                dcp_enabled=False,
            )
        )
        self.assertEqual(parallel.attn_tp_size, 2)
        self.assertEqual(parallel.moe_tp_size, 2)

    def test_an_unstamped_width_still_names_the_cause(self):
        with self.assertRaisesRegex(RuntimeError, r"not available"):
            get_parallel().attn_tp_size


class TestPrivateAttributeProbing(_IsolatedOverrides):
    def test_probing_a_private_name_does_not_recurse(self):
        fresh = ParallelContext.__new__(ParallelContext)  # slots unset
        for probe in ("_config", "_stamp", "_overrides", "__deepcopy__"):
            with self.assertRaises(AttributeError, msg=probe):
                getattr(fresh, probe)

    def test_a_built_context_survives_a_copy(self):
        import copy

        self.assertIsInstance(copy.copy(get_parallel()), ParallelContext)


class TestAWidthReadStaysTraceable(_IsolatedOverrides):
    """Parallel width reads must trace under ``torch.compile(fullgraph=True)``."""

    def test_a_width_read_compiles_into_the_graph(self):
        import torch

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=8, dp_size=2, enable_dp_attention=True
            ),
            role="test",
        )

        def read(x):
            return x * get_parallel().attn_tp_size

        # backend="eager": this pins tracing, not code generation, and stays
        # runnable on a box with no inductor toolchain.
        compiled = torch.compile(read, fullgraph=True, backend="eager")
        self.assertEqual(compiled(torch.ones(3)).tolist(), [4.0, 4.0, 4.0])


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
    """DCP widths come from configuration; active DCP ranks come from publication."""

    def _published(self, **fields):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", **fields), role="test")
        return get_parallel()

    def test_attn_dcp_is_one_when_dcp_is_off(self):
        parallel = self._published(tp_size=8, dcp_size=1)
        self.assertFalse(parallel.dcp_enabled)
        self.assertEqual(parallel.attn_dcp_size, 1)

    def test_attn_dcp_is_the_configured_width_when_on(self):
        parallel = self._published(tp_size=8, dcp_size=8)
        self.assertTrue(parallel.dcp_enabled)
        self.assertEqual(parallel.attn_dcp_size, 8)

    def _placed(self, world_rank, **fields):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", **fields),
            role="test",
            ranks=SpawnRanks(world_rank=world_rank),
        )
        return get_parallel()

    def test_the_dcp_rank_is_where_the_tp_rank_falls_in_its_slice(self):
        parallel = self._placed(5, tp_size=8, dcp_size=4)
        self.assertEqual(parallel.dcp_rank, 1)
        self.assertEqual(parallel.attn_dcp_rank, 1)

    def test_the_gated_off_rank_answers_without_a_spawn_bundle(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", tp_size=8), role="test")
        self.assertEqual(get_parallel().attn_dcp_rank, 0)

    def test_the_dcp_rank_is_gated_on_a_width_the_configuration_carries(self):
        parallel = self._placed(5, tp_size=8, dcp_size=1)
        self.assertFalse(parallel.dcp_enabled)
        self.assertEqual(parallel.attn_dcp_rank, 0)
        with self.assertRaises(RuntimeError):
            parallel.dcp_rank

    def test_the_width_does_not_consult_the_platform(self):
        with patch("sglang.srt.utils.is_cuda", return_value=False) as is_cuda:
            parallel = self._published(tp_size=8, dcp_size=8)
            self.assertTrue(parallel.dcp_enabled)
            self.assertEqual(parallel.attn_dcp_size, 8)
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
    """The context owns ServerArgs; supported legacy accessors share its storage."""

    def test_legacy_setter_publishes_into_context(self):
        # Identity, not equality: the slot holds the very object published.
        sentinel = ServerArgs(model_path="dummy")
        server_args_module.set_global_server_args_for_scheduler(sentinel)
        self.assertIs(get_server_args(), sentinel)
        self.assertIs(get_context().server_args, sentinel)

    def test_the_retired_accessor_raises_and_names_the_replacement(self):
        """`get_global_server_args` is retired: it answered with the record,
        so a caller reading a field resolution had decided got a stale value
        and no error at all.

        `RuntimeError` unconditionally, not a warning first: a
        `DeprecationWarning` is filtered by default outside `__main__`, so no
        production caller would have seen it, and under
        `-W error::DeprecationWarning` it would have changed the exception a
        caller catches. The message has to name where to read instead, since
        the answer differs by what the caller wanted.
        """
        with self.assertRaises(RuntimeError) as cm:
            server_args_module.get_global_server_args()
        message = str(cm.exception)
        self.assertIn("runtime_context", message)
        self.assertIn("get_server_args()", message)

        # And the type does not change when warnings are errors.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(RuntimeError):
                server_args_module.get_global_server_args()

    def test_tokenizer_alias_is_distinct_role_shim(self):
        # Deliberately NOT an alias: the two legacy setters publish with
        # different process roles (scheduler vs tokenizer).
        self.assertIsNot(
            server_args_module.set_global_server_args_for_tokenizer,
            server_args_module.set_global_server_args_for_scheduler,
        )

    def test_pre_publish_error_verbatim(self):
        reset_context()
        with self.assertRaises(ValueError) as cm:
            get_server_args()
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
        # The hook declares; the record keeps the operator's input, so the
        # values are read where resolution puts them.
        self.assertEqual(resolution_result(published, "attention_backend"), "triton")
        self.assertEqual(resolution_result(published, "chunked_prefill_size"), -1)
        # unnamed fields keep their dataclass defaults
        self.assertEqual(resolution_result(published, "tp_size"), 1)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            get_context().override_server_args(not_a_config_field=1).install()

    def test_restore_reinstates_previous_publish(self):
        previous = object()
        get_context().set_server_args(previous)
        override = get_context().override_server_args(tp_size=8)
        override.install()
        self.assertEqual(get_parallel().tp_size, 8)
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
                self.assertEqual(get_parallel().tp_size, 4)
            self.assertIs(get_server_args(), outer)
            self.assertEqual(get_parallel().tp_size, 2)

    def test_private_attribute_seeding(self):
        # Property caches (e.g. _mamba_cache_chunk_size) are seeded through
        # the same call; the strict guard exempts underscore names.
        published = (
            get_context().override_server_args(_mamba_cache_chunk_size=64).install()
        )
        self.assertEqual(mamba_cache_chunk_size_of(published), 64)

    def test_an_underscore_field_is_declared_like_any_other(self):
        """The split is fields vs not-fields, not the leading underscore.

        `_speculative_draft_quantization_explicitly_set` is a real field
        published under `spec`. Seeding it as a raw attribute instead of
        declaring it would leave the earlier resolution authoritative, so both
        the resolution and the bag would keep answering the pre-override value
        while the record said otherwise.
        """
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.runtime_context import get_spec

        name = "_speculative_draft_quantization_explicitly_set"
        self.assertIn(name, ServerArgs.__struct_fields__)

        published = get_context().override_server_args(**{name: True}).install()
        # The record keeps the operator's input, as it does for every other
        # field; the override travels as a declaration.
        self.assertIsNone(getattr(published, name))
        self.assertIs(resolution_result(published, name), True)
        self.assertIs(getattr(get_spec(), name), True)

    def test_installed_config_arms_the_strict_guard(self):
        # The published dummy must behave like a resolved config: bare writes
        # raise.
        published = get_context().override_server_args(tp_size=2).install()
        with self.assertRaises(AttributeError):
            published.tp_size = 4
        self.assertEqual(resolution_result(published, "tp_size"), 2)

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
    speculative_algorithm: A[str | None, Arg(help="sa"), NS("spec")] = None
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
            self.assertTrue(get_flags().moe.speculative_context)
        self.assertEqual(get_moe_runner_backend().name, "TRITON")
        self.assertTrue(get_moe_a2a_backend().is_deepep())
        self.assertFalse(get_flags().moe.disable_fp4_allgather)
        self.assertFalse(get_flags().moe.speculative_context)

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
        # parallel config leaves inside compiled model forwards, which must
        # stay dynamo-traceable (``object.__getattribute__`` graph-breaks).
        # fullgraph=True turns any graph break back into a failure.
        import torch

        from sglang.srt.runtime_context import get_parallel

        reset_context()
        with get_context().override_server_args(moe_dense_tp_size=1, dwdp_size=4):

            @torch.compile(fullgraph=True, backend="eager", dynamic=False)
            def probe(x):
                par = get_parallel()
                if par.enable_prefill_cp:
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

    def test_the_mamba_extra_buffer_predicate_has_one_answer(self):
        """It used to be asserted that two spellings agreed. There is one now:
        the declaration computes it at publish, and the bag carries it."""
        for disable_radix_cache in (False, True):
            for strategy in self._STRATEGIES:
                with self.subTest(radix=disable_radix_cache, strategy=strategy):
                    reset_context()
                    publish(
                        ServerArgs(
                            model_path="dummy",
                            disable_radix_cache=disable_radix_cache,
                            mamba_radix_cache_strategy=strategy,
                        ),
                        role="test",
                    )
                    expected = disable_radix_cache is False and strategy in (
                        "extra_buffer",
                        "extra_buffer_lazy",
                    )
                    self.assertEqual(
                        get_exec().mamba.enable_mamba_extra_buffer, expected
                    )
                    self.assertEqual(
                        get_exec().mamba.enable_mamba_extra_buffer_lazy,
                        disable_radix_cache is False
                        and strategy == "extra_buffer_lazy",
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
                                max_prefill_buffer_tokens_of(args),
                                max_prefill_buffer_tokens(),
                            )

    def test_prefill_buffer_ceiling_hook_honored_across_tiers(self):
        args = _FakeResolvedArgs(
            chunked_prefill_size=8192,
            enable_dynamic_chunking=True,
            pp_size=4,
            max_prefill_tokens=16384,
        )

        def provider(record, default_ceiling):
            self.assertIs(record, args)
            return default_ceiling + 5

        with patch.object(prefill_buffer_ceiling, "_prefill_buffer_ceiling_fn", None):
            register = prefill_buffer_ceiling.register_prefill_buffer_ceiling
            self.assertEqual(max_prefill_buffer_tokens_of(args), 16384)
            self.assertIs(register(provider), provider)
            register(provider)
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                register(lambda record, default_ceiling: default_ceiling)
            for record_or_view in (args, resolving_view(args), resolved_view(args)):
                self.assertEqual(max_prefill_buffer_tokens_of(record_or_view), 16389)
            get_context().set_server_args(args)
            self.assertEqual(max_prefill_buffer_tokens(), 16389)
            with get_schedule().override(max_prefill_tokens=32768):
                self.assertEqual(max_prefill_buffer_tokens(), 32773)
            self.assertEqual(args.max_prefill_tokens, 16384)

    def test_prefill_buffer_ceiling_provider_can_preserve_defaults(self):
        args = _FakeResolvedArgs(chunked_prefill_size=4096)

        def provider(record, default_ceiling):
            return default_ceiling

        with patch.object(prefill_buffer_ceiling, "_prefill_buffer_ceiling_fn", None):
            prefill_buffer_ceiling.register_prefill_buffer_ceiling(provider)
            for record_or_view in (args, resolving_view(args), resolved_view(args)):
                self.assertEqual(max_prefill_buffer_tokens_of(record_or_view), 4096)
            get_context().set_server_args(args)
            self.assertEqual(max_prefill_buffer_tokens(), 4096)

    def test_activation_reserve_matches_the_member(self):
        from types import SimpleNamespace

        from sglang.srt.arg_groups.overrides import (
            pre_capture_activation_reserve_mb_of,
        )
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
                        pre_capture_activation_reserve_mb_of(args, gpu_mem),
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
                            attention_backends_of(resolved_view(args)),
                            attention_backends(),
                        )


class TestAdaptiveDraftBoundLifecycle(_IsolatedServerArgs):
    """The adaptive draft-token bound is snapshotted at each publication."""

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
        # The new publication must not retain the previous capacity.
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


class TestParallelLeafReads(_IsolatedServerArgs):
    """The contract ``ParallelContext.__getattr__`` answers a parallel leaf on."""

    def test_a_leaf_answers_what_resolution_decided(self):
        from sglang.srt.arg_groups.overrides import resolution_result

        with get_context().override_server_args() as server_args:
            self.assertEqual(
                resolution_result(server_args, "nccl_port"),
                get_parallel().nccl_port,
                "a parallel leaf read off the context disagreed with what "
                "resolution decided",
            )

    def test_before_publish_the_error_names_the_namespace(self):
        with self.assertRaisesRegex(ValueError, r"'parallel' not published"):
            getattr(ParallelContext(), "nccl_port")

    def test_an_unknown_name_is_still_an_attribute_error(self):
        with self.assertRaisesRegex(AttributeError, r"has no 'not_a_leaf'"):
            getattr(ParallelContext(), "not_a_leaf")


class TestDerivedWidths(_IsolatedOverrides):
    """Derived widths use configuration unless explicitly overridden."""

    def setUp(self):
        super().setUp()
        parallel = get_parallel()
        self._saved_derived = dict(parallel._stamp)
        parallel.clear_stamp()
        self.addCleanup(
            lambda: (
                parallel.clear_stamp(),
                parallel.override_permanently(**self._saved_derived),
            )
        )

    def test_the_published_configuration_decides_the_widths(self):
        """The quotients are computed once, at publish, from the leaves.

        Every input is a record field, so there is nothing to recompute on a
        read: `publish` fills the bag and the bag is the answer.
        """
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=8, dp_size=2, enable_dp_attention=True
            ),
            role="test",
        )
        self.assertEqual(get_parallel().attn_tp_size, 4)
        self.assertEqual(get_parallel().attn_dp_size, 2)
        self.assertEqual(get_parallel().moe_tp_size, 8)

        reset_context()
        publish(
            ServerArgs(model_path="dummy", tp_size=8, ep_size=4, moe_dp_size=2),
            role="test",
        )
        self.assertEqual(get_parallel().moe_tp_size, 1)

    def test_a_topology_is_stated_by_naming_the_width(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", tp_size=8), role="test")
        self.assertEqual(get_parallel().attn_tp_size, 8)

        with self.assertRaises(ValueError) as caught:
            with get_parallel().override(tp_size=2):
                pass
        self.assertIn(
            "tp_size == attn_tp_size * attn_dp_size * attn_cp_size",
            str(caught.exception),
        )

        with get_parallel().override(tp_size=2, attn_tp_size=2, moe_tp_size=2):
            self.assertEqual(get_parallel().tp_size, 2)
            self.assertEqual(get_parallel().attn_tp_size, 2)
        with get_parallel().override(attn_tp_size=4, tp_size=4, moe_tp_size=4):
            self.assertEqual(get_parallel().attn_tp_size, 4)

    def test_an_unstated_topology_still_fails(self):
        """Neutral leaves are for the dimensions a caller is not using, not for
        a caller that stated nothing: every width would come back 1, which is a
        plausible-looking number invented out of nothing."""
        with self.assertRaises(RuntimeError) as caught:
            get_parallel().attn_tp_size
        self.assertIn("not available", str(caught.exception))

    def test_a_permanent_override_and_a_live_group_both_win_over_the_leaves(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_tp_size=7)
        self.addCleanup(parallel.clear_stamp)
        with parallel.override(tp_size=8, attn_dp_size=2):
            self.assertEqual(parallel.attn_tp_size, 7)

    def test_the_quotients_come_from_the_leaves(self):
        widths = derive_parallel_widths(
            tp_size=8,
            attn_cp_size=1,
            attn_dp_size=2,
            moe_ep_size=4,
            moe_dp_size=2,
            dcp_size=1,
            dcp_enabled=False,
        )
        self.assertEqual(widths["attn_tp_size"], 8 // 2 // 1)
        self.assertEqual(widths["moe_tp_size"], 8 // 4 // 2)
        self.assertEqual(widths["attn_dcp_size"], 1)

    def test_no_world_width_is_a_quotient_of_the_leaves(self):
        widths = derive_parallel_widths(
            tp_size=4,
            attn_cp_size=1,
            attn_dp_size=1,
            moe_ep_size=1,
            moe_dp_size=1,
            dcp_size=1,
            dcp_enabled=False,
        )
        self.assertEqual(
            {name for name in widths if "world" in name},
            set(),
        )
        # WORLD includes the ranks below the joining cohort.
        from sglang.srt.runtime_context import launch_world_size_of

        self.assertEqual(
            launch_world_size_of(
                SimpleNamespace(ep_join_rank_offset=8, tp_size=4, pp_size=1)
            ),
            12,
        )

    def test_the_bare_name_is_gone(self):
        with self.assertRaisesRegex(AttributeError, r"has no 'world_size'"):
            get_parallel().world_size
        with self.assertRaisesRegex(ValueError, r"unknown parallel field"):
            with get_parallel().override(world_size=4):
                pass

    def test_a_permanently_overridden_width_is_what_the_reader_answers_with(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_tp_size=4, moe_tp_size=1)
        with patch(
            f"{_PS}.get_attn_tensor_model_parallel_world_size",
            side_effect=AssertionError("the group must not be asked"),
        ):
            self.assertEqual(parallel.attn_tp_size, 4)

    def test_a_scoped_override_still_wins_over_the_permanent_one(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_tp_size=4)
        with parallel.override(attn_tp_size=1):
            self.assertEqual(parallel.attn_tp_size, 1)
        self.assertEqual(parallel.attn_tp_size, 4)

    def test_the_group_is_never_consulted(self):
        """There is no third source. A quotient comes from a scoped override, a
        permanent override, or the published leaf -- never from a group
        coordinator."""
        reset_context()
        self.addCleanup(reset_context)
        with patch(
            f"{_PS}.get_attn_tensor_model_parallel_world_size",
            side_effect=AssertionError("the group must not be consulted"),
        ):
            publish(
                ServerArgs(
                    model_path="dummy", tp_size=8, dp_size=2, enable_dp_attention=True
                ),
                role="test",
            )
            self.assertEqual(get_parallel().attn_tp_size, 4)

    def test_with_neither_the_failure_names_the_cause(self):
        with patch(
            f"{_PS}.get_attn_tensor_model_parallel_world_size",
            side_effect=AssertionError("attention tp group is not initialized"),
        ):
            with self.assertRaisesRegex(RuntimeError, r"derived parallel width"):
                get_parallel().attn_tp_size

    def test_the_permanent_override_is_cleared_and_reset(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_dp_size=2)
        self.assertEqual(parallel.attn_dp_size, 2)
        # Elastic scaling overrides again where it updates the live width.
        parallel.override_permanently(attn_dp_size=4)
        self.assertEqual(parallel.attn_dp_size, 4)
        parallel.clear_stamp()
        with parallel.override(tp_size=8, attn_dp_size=1):
            self.assertEqual(parallel.attn_dp_size, 1)

    def test_reset_context_drops_the_permanent_override(self):
        parallel = get_parallel()
        parallel.override_permanently(attn_tp_size=4)
        self.assertEqual(parallel.attn_tp_size, 4)
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", tp_size=1), role="test")
        self.assertEqual(get_parallel().attn_tp_size, 1)

    def test_the_rank_helper_agrees_with_the_override(self):
        """`compute_dp_attention_world_info` keeps the ranks and takes the
        widths from the same derivation `override_permanently`'s callers use."""
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

        for tp_size, dp_size, attn_cp_size in ((8, 2, 1), (8, 2, 2), (16, 4, 2)):
            _, attn_tp_size, _, attn_dp_size = compute_dp_attention_world_info(
                True, 0, tp_size, dp_size, attn_cp_size
            )
            widths = derive_parallel_widths(
                tp_size=tp_size,
                attn_cp_size=attn_cp_size,
                attn_dp_size=attn_dp_size,
                moe_ep_size=1,
                moe_dp_size=1,
                dcp_size=1,
                dcp_enabled=False,
            )
            self.assertEqual(attn_tp_size, widths["attn_tp_size"])
            self.assertEqual(attn_dp_size, widths["attn_dp_size"])

    def test_recomputing_from_published_leaves_matches_the_publish_bag(self):
        shapes = (
            dict(tp_size=8),
            dict(tp_size=8, dp_size=2, enable_dp_attention=True),
            dict(tp_size=8, ep_size=4, moe_dp_size=2),
            dict(tp_size=8, dcp_size=8),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                reset_context()
                self.addCleanup(reset_context)
                publish(ServerArgs(model_path="dummy", **shape), role="test")
                parallel = get_parallel()
                published = {
                    "attn_tp_size": parallel.attn_tp_size,
                    "attn_dp_size": parallel.attn_dp_size,
                    "moe_ep_size": parallel.moe_ep_size,
                    "moe_tp_size": parallel.moe_tp_size,
                    "dcp_enabled": parallel.dcp_enabled,
                    "attn_dcp_size": parallel.attn_dcp_size,
                }
                # What every real `initialize_model_parallel` caller forwards:
                # its own already-published leaves, through the same two
                # functions the bag was projected with.
                recomputed = derive_parallel_widths(
                    tp_size=parallel.tp_size,
                    attn_cp_size=parallel.attn_cp_size,
                    attn_dp_size=(
                        parallel.dp_size if parallel.enable_dp_attention else 1
                    ),
                    moe_ep_size=parallel.ep_size,
                    moe_dp_size=parallel.moe_dp_size,
                    dcp_size=parallel.dcp_size,
                    dcp_enabled=parallel.dcp_size > 1,
                )
                self.assertEqual(published, recomputed)

    def test_initialize_model_parallel_builds_at_the_published_widths(self):
        from unittest.mock import Mock

        from sglang.srt.distributed import parallel_state

        reset_context()
        self.addCleanup(reset_context)
        world_size = 8
        publish(ServerArgs(model_path="dummy", tp_size=world_size), role="test")
        self.assertEqual(get_parallel().attn_tp_size, world_size)

        built_at = []
        with (
            patch.object(parallel_state, "_WORLD", None),
            patch.object(parallel_state, "_TP", None),
            patch.object(parallel_state, "_DCP", None),
            patch.object(parallel_state, "_ATTN_CP", None),
            patch.object(parallel_state, "_ATTN_TP", None),
            patch.object(parallel_state, "_MOE_DP", None),
            patch.object(parallel_state, "_MOE_EP", None),
            patch.object(parallel_state, "_MOE_TP", None),
            patch.object(parallel_state, "_PP", None),
            patch.object(parallel_state, "_SELF_PP", None),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_backend", return_value="nccl"),
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=lambda group_ranks, *a, **k: (
                    built_at.append(group_ranks),
                    Mock(device_group=Mock()),
                )[1],
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):
            mock_world_group.return_value = Mock(device_group=Mock(), local_rank=0)
            parallel_state.initialize_model_parallel()
        self.addCleanup(parallel_state.destroy_model_parallel)

        self.assertEqual(built_at[0], [list(range(world_size))])
        self.assertEqual(get_parallel().attn_tp_size, world_size)


class TestTheDerivedHalfIsDeclared(CustomTestCase):
    """The quotients are declared beside the leaves, in the same class.

    A namespace is one file and one class. `Parallel` says both what an
    operator can set and what that decides; the quotients are unannotated, so
    they are not dataclass fields and never reach the record.
    `ParallelContext` installs a property per declaration rather than carrying
    its own list, so the two cannot drift.
    """

    def test_every_declared_quotient_has_a_property(self):
        from sglang.srt.arg_groups.arg_utils import Derived
        from sglang.srt.arg_groups.fields.parallel import Parallel

        declared = {
            name for name, value in vars(Parallel).items() if isinstance(value, Derived)
        }
        self.assertTrue(declared, "the derived half is empty")
        for name in declared:
            self.assertIsInstance(
                getattr(type(get_context().parallel), name, None),
                property,
                f"{name} is declared but no property was installed",
            )

    def test_a_computed_name_names_the_function_that_computes_it(self):
        import importlib

        from sglang.srt.runtime_context import _derived_widths

        computed = {n: d.fn for n, d in _derived_widths().items() if d.fn}
        self.assertTrue(computed, "nothing is computed from the leaves")
        for name, fn in computed.items():
            module, _, attr = fn.rpartition(".")
            self.assertEqual(attr, f"{name}_of", f"{name} is computed by {attr}")
            self.assertTrue(callable(getattr(importlib.import_module(module), attr)))

    def test_the_arithmetic_produces_nothing_that_is_not_declared(self):
        from sglang.srt.runtime_context import _derived_widths

        produced = set(
            derive_parallel_widths(
                tp_size=8,
                attn_cp_size=1,
                attn_dp_size=2,
                moe_ep_size=1,
                moe_dp_size=1,
                dcp_size=1,
                dcp_enabled=False,
            )
        )
        self.assertEqual(produced - set(_derived_widths()), set())

    def test_a_declared_quotient_is_not_a_record_field(self):
        """It has no operator input to preserve, and the record is what crosses
        a process boundary."""

        from sglang.srt.arg_groups.arg_utils import Derived
        from sglang.srt.arg_groups.fields.parallel import Parallel
        from sglang.srt.server_args import ServerArgs

        fields = {f.name for f in msgspec.structs.fields(ServerArgs)}
        for name, value in vars(Parallel).items():
            if isinstance(value, Derived):
                self.assertNotIn(name, fields)


class TestAnEntryThatBuildsARunnerHandsOverItsPlacement(CustomTestCase):
    """Runner entry points must publish launcher placement."""

    def test_every_publisher_that_builds_a_runner_passes_a_bundle(self):
        import ast as _ast

        offenders = []
        for path in _sources():
            text = path.read_text(encoding="utf-8-sig")
            if "ModelRunner(" not in text or "publish(" not in text:
                continue
            tree = _ast.parse(text)
            builds = any(
                isinstance(n, _ast.Call)
                and getattr(n.func, "id", getattr(n.func, "attr", None))
                == "ModelRunner"
                for n in _ast.walk(tree)
            )
            if not builds:
                continue
            for node in _ast.walk(tree):
                if (
                    isinstance(node, _ast.Call)
                    and getattr(node.func, "id", None) == "publish"
                    and not any(kw.arg == "ranks" for kw in node.keywords)
                ):
                    offenders.append(f"{path}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "these publish without a spawn bundle and then build a ModelRunner, "
            "whose construction reads a recorded identity:\n  "
            + "\n  ".join(offenders),
        )


class TestTheAccessorsHaveNoCallersOutsideTheirPackage(CustomTestCase):
    """SRT callers outside ``distributed`` use the runtime context.

    Multimodal generation has its own parallel state and is excluded.
    """

    # Group constructors and non-topology helpers may have callers.
    ALLOWED = {
        "get_self_pp_group",
        "get_default_distributed_backend",
        "get_mooncake_transfer_engine",
    }

    def _accessors(self):
        """Find public getters defined in the parallel-state source."""
        from sglang.srt.distributed import parallel_state as parallel_state_module

        source = _pathlib.Path(parallel_state_module.__file__).read_text().splitlines()
        return {
            line[len("def ") : line.index("(")]
            for line in source
            if line.startswith("def get_") or line.startswith("def is_")
        }

    def _callers(self, name):
        """Find getter calls outside the defining package, including import aliases."""
        import re

        from sglang.srt.distributed import parallel_state as parallel_state_module

        root = _pathlib.Path(parallel_state_module.__file__).parents[2]
        hits = []
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if rel.startswith(("srt/distributed/", "multimodal_gen/", "test/")):
                continue
            text = path.read_text()
            spellings = (
                {name}
                | set(re.findall(rf"import\s+{re.escape(name)}\s+as\s+(\w+)", text))
                | set(re.findall(rf"^\s*{re.escape(name)}\s+as\s+(\w+),?$", text, re.M))
            )
            pattern = re.compile(
                r"(?<![.\w])(?:" + "|".join(re.escape(s) for s in spellings) + r")\("
            )
            for number, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith(("def ", "#")):
                    continue
                if pattern.search(line):
                    hits.append(f"{rel}:{number}")
        return hits

    def test_no_business_code_calls_them(self):
        offenders = {}
        for name in sorted(self._accessors() - self.ALLOWED):
            callers = self._callers(name)
            if callers:
                offenders[name] = callers
        self.assertEqual(
            offenders,
            {},
            "read these through get_parallel() instead, or say here why the "
            "context cannot answer them",
        )

    def test_calling_one_from_outside_the_package_is_deprecated(self):
        import warnings

        from sglang.srt.distributed import parallel_state

        parallel_state._ALREADY_WARNED.discard("get_tensor_model_parallel_rank")
        self.addCleanup(
            parallel_state._ALREADY_WARNED.discard, "get_tensor_model_parallel_rank"
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                parallel_state.get_tensor_model_parallel_rank()
            except Exception:
                pass
        messages = [str(w.message) for w in seen]
        self.assertTrue(
            any("get_parallel().tp_rank" in m for m in messages),
            f"expected the replacement to be named, got {messages}",
        )

    def test_a_scope_reaches_callers_that_went_straight_to_the_getter(self):
        from sglang.srt.distributed import parallel_state

        stand_in = SimpleNamespace(world_size=1, rank_in_group=0)
        with get_parallel().override(tp_group=stand_in):
            self.assertIs(get_parallel().tp_group, stand_in)
            self.assertIs(parallel_state.get_tp_group(), stand_in)

    def test_the_package_that_defines_them_is_not_warned_at(self):
        import warnings

        from sglang.srt.distributed import parallel_state

        parallel_state._ALREADY_WARNED.discard("get_tensor_model_parallel_rank")
        self.addCleanup(
            parallel_state._ALREADY_WARNED.discard, "get_tensor_model_parallel_rank"
        )
        caller = types.ModuleType("sglang.srt.distributed.pretend_internal")
        caller.__dict__["call"] = lambda: (
            parallel_state.get_tensor_model_parallel_rank()
        )
        exec(
            "def call():\n    from sglang.srt.distributed import parallel_state\n"
            "    return parallel_state.get_tp_group()",
            caller.__dict__,
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                caller.call()
            except Exception:
                pass
        self.assertEqual([str(w.message) for w in seen], [])

    def test_the_guard_would_notice_a_caller(self):
        self.assertTrue(self._callers("get_self_pp_group"))


class TestTheTopologyIdentities(CustomTestCase):
    """Validate topology consistency at publication and override boundaries."""

    def _publish_square(self):
        """tp=4 over two attention-DP replicas of two: every identity holds."""
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=4, dp_size=2, enable_dp_attention=True
            ),
            role="scheduler",
            ranks=SpawnRanks(world_rank=3, dp_rank=1),
        )

    def test_a_published_topology_is_consistent(self):
        self._publish_square()
        parallel = get_parallel()
        self.assertEqual(parallel.tp_size, 4)
        self.assertEqual(parallel.attn_dp_size, 2)
        self.assertEqual(parallel.attn_tp_size, 2)
        self.assertEqual(parallel.tp_rank, 3)

    def test_a_width_that_does_not_factor_is_refused(self):
        self._publish_square()
        with self.assertRaises(ValueError) as caught:
            with get_parallel().override(
                tp_size=4, attn_tp_size=3, attn_dp_size=1, attn_cp_size=1
            ):
                pass
        message = str(caught.exception)
        self.assertIn("set by override", message)
        self.assertIn("attn_tp_size * attn_dp_size * attn_cp_size", message)
        self.assertIn("4 != 3 * 1 * 1", message)

    def test_a_rank_at_its_width_is_refused(self):
        self._publish_square()
        with self.assertRaises(ValueError) as caught:
            with get_parallel().override(tp_rank=4, tp_size=4):
                pass
        self.assertIn("0 <= tp_rank < tp_size", str(caught.exception))

    def test_a_moe_width_that_does_not_factor_is_refused(self):
        self._publish_square()
        with self.assertRaises(ValueError) as caught:
            with get_parallel().override(
                tp_size=4, moe_ep_size=1, moe_dp_size=1, moe_tp_size=3
            ):
                pass
        message = str(caught.exception)
        self.assertIn("moe_ep_size * moe_dp_size * moe_tp_size", message)
        self.assertIn("4 != 1 * 1 * 3", message)

    def test_a_rank_the_attention_layout_cannot_produce_is_refused(self):
        self._publish_square()
        with self.assertRaises(ValueError) as caught:
            with get_parallel().override(
                tp_rank=0,
                attn_dp_rank=1,
                attn_cp_rank=0,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_tp_size=2,
            ):
                pass
        self.assertIn("attn_tp_size + attn_tp_rank", str(caught.exception))

    def test_the_published_ranks_satisfy_the_layout(self):
        self._publish_square()
        parallel = get_parallel()
        self.assertEqual(
            parallel.tp_rank,
            (parallel.attn_dp_rank * parallel.attn_cp_size + parallel.attn_cp_rank)
            * parallel.attn_tp_size
            + parallel.attn_tp_rank,
        )

    def test_a_refused_write_leaves_nothing_behind(self):
        self._publish_square()
        written = dict(tp_size=4, attn_tp_size=3, attn_dp_size=1, attn_cp_size=1)
        before = {name: getattr(get_parallel(), name) for name in written}
        with self.assertRaises(ValueError):
            with get_parallel().override(**written):
                pass
        self.assertEqual(
            {name: getattr(get_parallel(), name) for name in written}, before
        )
        self.assertEqual(before["attn_tp_size"], 2)

    def test_a_group_built_at_another_width_is_refused(self):
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        from sglang.srt.runtime_context import _WIDTH_AND_GROUP

        exercised = set()
        for size_name, group_name in _WIDTH_AND_GROUP:
            with self.subTest(group=group_name):
                self._publish_square()
                configured = getattr(get_parallel(), size_name)
                wrong = GroupCoordinator.__new__(GroupCoordinator)
                wrong.world_size = configured + 4
                wrong.rank_in_group = 0
                with self.assertRaises(ValueError) as caught:
                    get_parallel().override_permanently(**{group_name: wrong})
                message = str(caught.exception)
                self.assertIn(f"{group_name}.world_size == {size_name}", message)
                self.assertIn(
                    f"built {configured + 4}, configured {configured}", message
                )
                with self.assertRaises(RuntimeError):
                    getattr(get_parallel(), group_name)
                exercised.add((size_name, group_name))
        self.assertEqual(exercised, set(_WIDTH_AND_GROUP))

    def test_a_group_built_at_the_configured_width_is_quiet(self):
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        self._publish_square()
        right = GroupCoordinator.__new__(GroupCoordinator)
        right.world_size = 4
        right.rank_in_group = 3
        get_parallel().override_permanently(tp_group=right)
        self.assertIs(get_parallel().tp_group, right)

    def test_a_draft_scope_states_a_consistent_topology(self):
        from sglang.srt.distributed import parallel_state
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        self._publish_square()
        group = GroupCoordinator.__new__(GroupCoordinator)
        group.world_size = 2
        group.rank_in_group = 1
        with parallel_state.patch_tensor_parallel_group(group, owns_attention=True):
            self.assertEqual(get_parallel().attn_tp_size, 2)


class TestTheParallelPhase(CustomTestCase):
    """Entry points initialize parallel groups before constructing runners."""

    def test_the_layer_phase_leaves_a_stated_placement_alone(self):
        """A server that runs its own WORLD states its attention placement, and
        the layer phase reads the model's shape only."""
        import torch

        from sglang.srt.distributed import bootstrap
        from sglang.srt.layers.dp_attention import initialize_dp_attention_flags

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                tp_size=4,
                dp_size=2,
                enable_dp_attention=True,
                device="cpu",
            ),
            role="test",
            ranks=SpawnRanks(world_rank=3, dp_rank=1),
        )
        get_parallel().override_permanently(
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_tp_size=4,
            attn_tp_rank=3,
            attn_cp_size=1,
            attn_cp_rank=0,
        )

        initialize_dp_attention_flags(get_server_args())
        bootstrap.init_layer_runtime(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
                hidden_size=8,
                dtype=torch.float16,
            )
        )

        self.assertTrue(get_flags().dp.enabled)
        self.assertEqual(get_parallel().attn_dp_size, 1)
        self.assertEqual(get_parallel().attn_dp_rank, 0)

    def test_encoder_cp_mlp_reduces_over_attention_tp(self):
        from unittest.mock import Mock

        import torch

        from sglang.srt.layers.dp_attention import (
            init_dp_gathered_buffer,
            initialize_dp_attention_flags,
        )
        from sglang.srt.models.qwen3_vl import Qwen3_VisionMLP

        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy",
            device="cpu",
            tp_size=4,
            attn_cp_size=2,
            enable_dp_attention=True,
        )
        publish(server_args, role="test", ranks=SpawnRanks(world_rank=0))
        # Identical shards contribute equally; reducing across CP replicas
        # would double the output even though the layer uses attention TP.
        tp_group = SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            all_reduce=Mock(side_effect=lambda x: x * 4),
        )
        attn_tp_group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_reduce=Mock(side_effect=lambda x: x * 2),
        )
        get_parallel().override_permanently(
            tp_group=tp_group, attn_tp_group=attn_tp_group
        )
        initialize_dp_attention_flags(server_args)
        init_dp_gathered_buffer(
            SimpleNamespace(
                hf_config=SimpleNamespace(), hidden_size=4, dtype=torch.float32
            )
        )
        mlp = Qwen3_VisionMLP(4, 4, bias=False, hidden_act="relu")
        with torch.no_grad():
            for parameter in mlp.parameters():
                parameter.fill_(1)
            output = mlp(torch.ones(1, 4))

        torch.testing.assert_close(output, torch.full((1, 4), 16.0))
        attn_tp_group.all_reduce.assert_called_once()
        tp_group.all_reduce.assert_not_called()

    def test_building_twice_is_refused(self):
        from sglang.srt.distributed import bootstrap

        bootstrap.reset_parallel_initialised()
        self.addCleanup(bootstrap.reset_parallel_initialised)
        with (
            patch.object(bootstrap, "_resolve_backend", return_value="gloo"),
            patch.object(bootstrap, "_resolve_dist_init_method", return_value="env://"),
            patch.object(bootstrap, "_set_all_reduce_flags"),
            patch.object(bootstrap, "_init_parallel_groups"),
            patch.object(bootstrap, "monkey_patch_p2p_access_check"),
            patch.object(bootstrap, "_init_cpu_threads_env"),
            patch.object(bootstrap, "_bind_threads_if_cpu", return_value=None),
            patch.object(bootstrap, "maybe_init_shared_mooncake_transfer_engine"),
        ):
            reset_context()
            self.addCleanup(reset_context)
            publish(
                ServerArgs(model_path="dummy"),
                role="test",
                ranks=SpawnRanks(world_rank=0),
            )
            kwargs = dict(
                server_args=ServerArgs(model_path="dummy"),
                device="cpu",
                dist_port=12345,
            )
            bootstrap.init_parallel_runtime(**kwargs)
            with self.assertRaises(RuntimeError) as caught:
                bootstrap.init_parallel_runtime(**kwargs)
        self.assertIn("ran twice", str(caught.exception))

    def test_every_publisher_that_builds_a_runner_runs_the_phase(self):
        import ast as _ast

        offenders = []
        for path in _sources():
            text = path.read_text(encoding="utf-8-sig")
            if "ModelRunner(" not in text or "publish(" not in text:
                continue
            tree = _ast.parse(text)
            builds = any(
                isinstance(n, _ast.Call)
                and getattr(n.func, "id", getattr(n.func, "attr", None))
                == "ModelRunner"
                for n in _ast.walk(tree)
            )
            if not builds:
                continue
            for phase in ("init_parallel_runtime(", "init_layer_runtime("):
                if phase not in text:
                    offenders.append(f"{path} ({phase[:-1]})")
        self.assertEqual(
            offenders,
            [],
            "these publish and then build a ModelRunner without running both "
            "phases first -- the group build derives the topology, and the "
            "layer phase sizes what the model's shape decides:\n  "
            + "\n  ".join(offenders),
        )


class TestWhoAnswersDuringADraftScope(CustomTestCase):
    """Draft scopes expose a consistent topology.

    Objects constructed in a draft scope retain their placement after it exits.
    """

    def _single_member_group(self):
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        group = GroupCoordinator.__new__(GroupCoordinator)
        group.world_size = 1
        group.rank_in_group = 0
        return group

    def _two_stage_pipeline(self):
        """This process is stage 1 of 2, published the way a spawn states it."""
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", pp_size=2),
            role="scheduler",
            ranks=SpawnRanks(world_rank=1),
        )

    def test_the_pipeline_swap_states_every_member_it_installs(self):
        from sglang.srt.distributed import parallel_state

        group = self._single_member_group()
        self._two_stage_pipeline()
        self.assertEqual(get_parallel().pp_size, 2)
        with parallel_state.patch_pipeline_parallel_group(group):
            self.assertEqual(get_parallel().pp_size, 1)
            self.assertEqual(get_parallel().pp_rank, 0)
            self.assertIs(get_parallel().pp_group, group)
        self.assertEqual(get_parallel().pp_size, 2)
        self.assertEqual(get_parallel().pp_rank, 1)

    def _group(self, world_size, rank):
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        group = GroupCoordinator.__new__(GroupCoordinator)
        group.world_size = world_size
        group.rank_in_group = rank
        return group

    def test_the_tensor_swap_states_the_draft_has_no_attention_replica(self):
        from sglang.srt.distributed import parallel_state

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=4, dp_size=2, enable_dp_attention=True
            ),
            role="scheduler",
            ranks=SpawnRanks(world_rank=0, dp_rank=0),
        )
        self.assertEqual(get_parallel().attn_dp_size, 2)
        self.assertEqual(get_parallel().attn_tp_size, 2)

        group = self._group(world_size=2, rank=1)
        with parallel_state.patch_tensor_parallel_group(group, owns_attention=True):
            parallel = get_parallel()
            self.assertEqual(parallel.tp_size, 2)
            self.assertEqual(parallel.attn_tp_size, 2)
            self.assertEqual(parallel.attn_tp_rank, 1)
            self.assertEqual(parallel.attn_dp_size, 1)
            self.assertEqual(parallel.attn_dp_rank, 0)
            self.assertEqual(parallel.attn_cp_size, 1)
            self.assertEqual(parallel.attn_cp_rank, 0)
            # The scope leaves the deployment's replica count alone.
            self.assertEqual(parallel.dp_size, 2)
            self.assertEqual(
                parallel.tp_size,
                parallel.attn_tp_size * parallel.attn_dp_size * parallel.attn_cp_size,
            )
        self.assertEqual(get_parallel().attn_dp_size, 2)
        self.assertEqual(get_parallel().dp_size, 2)

    def test_a_full_width_swap_leaves_the_attention_layout_alone(self):
        from sglang.srt.distributed import parallel_state

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", tp_size=4, dp_size=2, enable_dp_attention=True
            ),
            role="scheduler",
            ranks=SpawnRanks(world_rank=0, dp_rank=0),
        )
        whole_tp = self._group(world_size=4, rank=0)
        with patch.object(parallel_state, "_TP", whole_tp):
            with parallel_state.patch_tensor_parallel_group(
                whole_tp, owns_attention=False
            ):
                parallel = get_parallel()
                self.assertEqual(parallel.tp_size, 4)
                self.assertEqual(parallel.attn_dp_size, 2)
                self.assertEqual(parallel.attn_tp_size, 2)
                self.assertEqual(parallel.dp_size, 2)

    def test_a_report_built_for_a_runner_follows_that_runner(self):
        from sglang.srt.distributed import parallel_state
        from sglang.srt.utils.weight_checker import WeightChecker

        self._two_stage_pipeline()
        self.assertEqual(get_parallel().pp_size, 2)
        group = self._single_member_group()
        with parallel_state.patch_pipeline_parallel_group(group):
            checker = WeightChecker(get_model=lambda: None)

        self.assertEqual(get_parallel().pp_size, 2)
        info = checker._parallelism_info(role="target")
        self.assertEqual((info.pp_rank, info.pp_size), (0, 1))


class TestTheRetiredNamesAreGoneEverywhere(CustomTestCase):
    """Classify parallel getters and reject retired package imports."""

    #: Getters that answer something other than a place in the topology.
    NOT_A_PLACEMENT = {
        "get_default_distributed_backend",
        "get_mooncake_transfer_engine",
        "get_torch_distributed_pg_options",
    }
    #: Widths whose group is not in ``_WIDTH_AND_GROUP``.
    WIDTH_WITHOUT_A_CHECKED_GROUP = {
        "get_dcp_world_size",
        "get_moe_data_parallel_world_size",
        "get_moe_tensor_parallel_world_size",
    }
    #: Variants that answer a group the map already covers.
    VARIANT_OF_A_MAPPED_GROUP = {
        "get_dcp_group_no_assert",
        "get_self_pp_group",
    }

    def _retired(self):
        from sglang.srt.distributed.parallel_state import _CONTEXT_NAME_OF

        return set(_CONTEXT_NAME_OF)

    def test_every_getter_the_module_defines_is_classified(self):
        """Every getter the module defines is deprecated or classified here."""
        from sglang.srt.distributed import parallel_state

        defined = {
            name
            for name in dir(parallel_state)
            if name.startswith("get_")
            and callable(getattr(parallel_state, name))
            and getattr(getattr(parallel_state, name), "__module__", None)
            == parallel_state.__name__
        }
        self.assertTrue(defined, "no getters found; this proves nothing")
        unclassified = (
            defined
            - self._retired()
            - self.NOT_A_PLACEMENT
            - self.WIDTH_WITHOUT_A_CHECKED_GROUP
            - self.VARIANT_OF_A_MAPPED_GROUP
        )
        self.assertEqual(
            unclassified,
            set(),
            "these getters are neither deprecated nor classified; say which "
            "kind each one is, or route it through get_parallel():\n  "
            + "\n  ".join(sorted(unclassified)),
        )
        stale = (
            self.NOT_A_PLACEMENT
            | self.WIDTH_WITHOUT_A_CHECKED_GROUP
            | self.VARIANT_OF_A_MAPPED_GROUP
        ) - defined
        self.assertEqual(
            stale, set(), f"these are named here but no longer defined: {stale}"
        )

    def test_nothing_imports_a_retired_name_from_the_package(self):
        import ast as _ast

        retired = self._retired()
        offenders = []
        for path in _sources():
            for node in _ast.walk(_ast.parse(path.read_text(encoding="utf-8-sig"))):
                if (
                    isinstance(node, _ast.ImportFrom)
                    and node.module == "sglang.srt.distributed"
                ):
                    for alias in node.names:
                        if alias.name in retired:
                            offenders.append(f"{path}:{node.lineno} {alias.name}")
        self.assertEqual(
            offenders,
            [],
            "these import a name the package no longer re-exports; import it "
            "from parallel_state, or read get_parallel():\n  " + "\n  ".join(offenders),
        )


class TestNothingReadsThePlacementBeforeItIsFrozen(CustomTestCase):
    """Runner initialization must set placement attributes before reading them."""

    def _model_runner(self):
        import ast as _ast

        source = (_SRT / "model_executor" / "model_runner.py").read_text()
        for node in _ast.parse(source).body:
            if isinstance(node, _ast.ClassDef) and node.name == "ModelRunner":
                return {m.name: m for m in node.body if isinstance(m, _ast.FunctionDef)}
        raise AssertionError("ModelRunner not found")

    def _frozen_names(self, methods):
        import ast as _ast

        return {
            target.attr
            for node in _ast.walk(methods["init_torch_distributed"])
            if isinstance(node, _ast.Assign)
            for target in node.targets
            if isinstance(target, _ast.Attribute)
            and isinstance(target.value, _ast.Name)
            and target.value.id == "self"
        }

    def _reads(self, methods, fn, frozen, depth=0):
        import ast as _ast

        found = set()
        for node in _ast.walk(fn):
            if (
                isinstance(node, _ast.Attribute)
                and isinstance(node.value, _ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, _ast.Load)
                and node.attr in frozen
            ):
                found.add(node.attr)
            if (
                depth < 2
                and isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and isinstance(node.func.value, _ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in methods
                and node.func.attr != "init_torch_distributed"
            ):
                found |= self._reads(
                    methods, methods[node.func.attr], frozen, depth + 1
                )
        return found

    def _calls_before_the_freeze(self, methods):
        import ast as _ast

        calls = []
        for statement in methods["__init__"].body:
            for node in _ast.walk(statement):
                if (
                    isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and isinstance(node.func.value, _ast.Name)
                    and node.func.value.id == "self"
                ):
                    calls.append((node.lineno, node.func.attr))
        calls.sort()
        names = [name for _, name in calls]
        self.assertIn(
            "init_torch_distributed",
            names,
            "the freeze moved; this census is keyed on where it happens",
        )
        return calls[: names.index("init_torch_distributed")]

    def test_no_method_called_before_the_freeze_reads_what_it_freezes(self):
        methods = self._model_runner()
        frozen = self._frozen_names(methods)
        self.assertGreater(len(frozen), 5, "found no frozen names; census is broken")
        offenders = []
        for lineno, name in self._calls_before_the_freeze(methods):
            fn = methods.get(name)
            if fn is None:
                continue
            read = self._reads(methods, fn, frozen)
            if read:
                offenders.append(
                    f"__init__:{lineno} self.{name}() reads {sorted(read)}"
                )
        self.assertEqual(
            offenders,
            [],
            "these run before init_torch_distributed and read what it sets; "
            "ask get_parallel() there, or move the call after the freeze:\n  "
            + "\n  ".join(offenders),
        )

    def test_the_census_would_notice_one(self):
        import ast as _ast
        import textwrap

        methods = {
            m.name: m
            for m in _ast.parse(
                textwrap.dedent(
                    """
                    class R:
                        def init_torch_distributed(self):
                            self.tp_rank = 0

                        def early(self):
                            return self.tp_rank
                    """
                )
            )
            .body[0]
            .body
        }
        frozen = self._frozen_names(methods)
        self.assertEqual(frozen, {"tp_rank"})
        self.assertEqual(self._reads(methods, methods["early"], frozen), {"tp_rank"})


if __name__ == "__main__":
    unittest.main()

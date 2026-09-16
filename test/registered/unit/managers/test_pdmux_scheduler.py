import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.parallel_state import (
    is_pdmux_enabled,
    is_pdmux_prefill_enabled,
)
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCHEDULER_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"
)
TP_WORKER_PATH = SCHEDULER_PATH.parent / "tp_worker.py"
DP_ATTN_PATH = SCHEDULER_PATH.parent / "scheduler_components" / "dp_attn.py"


def _init_call_order(class_name, targets):
    """Where each target call lands in `__init__`'s execution.

    Resolves one level of `self.init_*()` indirection, so a target that moves
    into (or out of) a helper is still placed at the point `__init__` runs it.
    `__init__` is straight-line code, so source order is execution order.
    """
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }

    def self_calls(node):
        return sorted(
            (call.lineno, call.func.attr)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        )

    positions = {}
    for step, (_lineno, name) in enumerate(self_calls(methods["__init__"])):
        if name in targets:
            positions.setdefault(name, step)
        elif name in methods:
            for _, nested in self_calls(methods[name]):
                if nested in targets:
                    positions.setdefault(nested, step)
    return positions


class _Batch:
    def __init__(self, empty):
        self._empty = empty

    def is_empty(self):
        return self._empty


class _ChunkedReq:
    """Hashable stand-in for Req (the merge path stores it in a set)."""

    def __init__(self, *, extend_end, prefix_len):
        self.extend_range = SimpleNamespace(end=extend_end)
        self.prefix_indices = [0] * prefix_len


def _make_chunked_req(*, extend_end, prefix_len):
    return _ChunkedReq(extend_end=extend_end, prefix_len=prefix_len)


class TestPDMuxScheduler(unittest.TestCase):
    def test_dp_attn_adapter_uses_active_pdmux_tp_group(self):
        tree = ast.parse(DP_ATTN_PATH.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SchedulerDPAttnAdapter"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_mlp_sync_batch"
        )
        prepare_call = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_mlp_sync_batch_raw"
        )
        tp_group = next(
            keyword.value
            for keyword in prepare_call.keywords
            if keyword.arg == "tp_group"
        )

        self.assertIsInstance(tp_group, ast.Call)
        self.assertIsInstance(tp_group.func, ast.Name)
        self.assertEqual(tp_group.func.id, "get_tp_group")

    def tearDown(self):
        # The prefill lane's TP binding is scoped, so nothing global to clear.
        pass

    def _make_scheduler(
        self,
        *,
        decode_empty,
        split_index=0,
        extend_num_tokens=128000,
        scheduler_global_num_tokens=None,
        token_budget=65536,
    ):
        return SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=61),
            pdmux_config=SimpleNamespace(split_forward_token_budget=token_budget),
            running_batch=_Batch(decode_empty),
            split_prefill_batch=SimpleNamespace(
                split_index=split_index,
                extend_num_tokens=extend_num_tokens,
                scheduler_global_num_tokens=scheduler_global_num_tokens,
            ),
        )

    def test_prefill_runs_remaining_layers_without_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=True, split_index=7)

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler, decode_batch=None
        )

        self.assertEqual(count, 54)

    def test_prefill_uses_token_budget_with_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=False)

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[1]),
        )

        self.assertEqual(count, 1)

    def test_prefill_count_is_clamped_to_remaining_layers(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            split_index=59,
            extend_num_tokens=8192,
            token_budget=65536,
        )

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[1]),
        )

        self.assertEqual(count, 2)

    def test_prefill_split_count_matches_on_active_and_idle_dp_ranks(self):
        active_scheduler = self._make_scheduler(
            decode_empty=False,
            extend_num_tokens=16384,
            scheduler_global_num_tokens=[16384, 0],
        )
        idle_scheduler = self._make_scheduler(
            decode_empty=True,
            extend_num_tokens=0,
            scheduler_global_num_tokens=[16384, 0],
        )

        active_count = SchedulerMultiplexMixin._get_split_forward_count(
            active_scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[0, 3]),
        )
        idle_count = SchedulerMultiplexMixin._get_split_forward_count(
            idle_scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[0, 3]),
        )

        self.assertEqual(active_count, 4)
        self.assertEqual(idle_count, active_count)

    def test_idle_decode_participants_do_not_force_prefill_splitting(self):
        scheduler = self._make_scheduler(
            decode_empty=True,
            split_index=7,
            extend_num_tokens=16384,
            scheduler_global_num_tokens=[16384, 0],
        )

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[0, 0]),
        )

        self.assertEqual(count, 54)

    def test_dsv4_prefill_admission_uses_planner_hard_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 65520)
        self.assertTrue(enforce)

    def test_non_dsv4_prefill_admission_preserves_soft_budget(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=None,
            page_size=16,
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 131072)
        self.assertFalse(enforce)

    def test_chunked_prefill_admission_preserves_soft_budget(self):
        """Chunked prefill enforces the planner limit per chunk, so the hard
        admission clamp must deactivate — keeping it would re-reject the long
        requests chunking exists to serve."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
            chunked_prefill_size=16384,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 131072)
        self.assertFalse(enforce)

    def test_dsv4_request_length_stays_within_planner_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            max_prefill_tokens=131072,
            page_size=16,
            chunked_prefill_size=None,
        )

        max_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
            scheduler, 1048576
        )

        self.assertEqual(max_input_len, 65521)

    def test_dsv4_request_limit_matches_smaller_prefill_budget(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            max_prefill_tokens=32767,
            page_size=16,
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, scheduler.max_prefill_tokens
        )
        max_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
            scheduler, 1048576
        )

        self.assertEqual(budget, 32752)
        self.assertTrue(enforce)
        self.assertEqual(max_input_len, budget + 1)

    def test_init_rejects_chunked_prefill_size_over_plan_limit(self):
        """65520 is the page-aligned uint16 compressor-plan cap; a larger
        chunk budget would overflow a single prefill plan at runtime."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=65536,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        with self.assertRaisesRegex(ValueError, "65520"):
            SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
                scheduler, attn_backend=attn_backend
            )

    def test_init_accepts_chunked_prefill_size_at_plan_limit(self):
        """With a valid chunk budget the per-request length clamp must stay
        off: chunking is what serves requests beyond the planner limit."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=65520,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
            scheduler, attn_backend=attn_backend
        )

        self.assertEqual(scheduler.pdmux_max_prefill_plan_tokens, (1 << 16) - 1)
        self.assertEqual(scheduler.max_req_input_len, 1048576)

    def test_init_tightens_request_length_without_chunked_prefill(self):
        """Without chunking PDMux cannot split an oversized request, so init
        must clamp request validation to the planner limit."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=None,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
            scheduler, attn_backend=attn_backend
        )

        self.assertEqual(scheduler.max_req_input_len, 65521)

    @staticmethod
    @contextmanager
    def _stubbed_stream_idx():
        """Stand in for the module-level stream-index state.

        The real setter validates against `STREAM_GROUPS`, which only
        `initialize_stream_groups` fills and which needs a GPU.
        """
        state = {"idx": 0}
        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.set_current_stream_idx",
                lambda idx: state.update(idx=idx),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_current_stream_idx",
                lambda: state["idx"],
            ),
        ):
            yield

    def _make_stream_group_scheduler(self, *, manual_divisions, group_num):
        model_runner = SimpleNamespace(update_decode_attn_backend=lambda _idx: None)
        return SimpleNamespace(
            split_prefill_batch=object(),
            pdmux_standard=False,
            draft_worker=None,
            pdmux_config=SimpleNamespace(
                manual_divisions=manual_divisions, decode_bs_divisor=36
            ),
            real_sm_group_num=group_num,
            tp_worker=SimpleNamespace(model_runner=model_runner),
            stream_groups=[(f"p{i}", f"d{i}") for i in range(group_num)],
        )

    def test_manual_division_below_every_threshold_uses_first_shared_group(self):
        """A decode batch under every configured threshold still needs a group.

        The selection loop only assigns a stream index on a threshold it meets,
        so a batch below all of them left the index unbound -- an
        UnboundLocalError raised from the scheduler loop. A single-division
        config (`--sm-group-num 3`) whose threshold is above 1 hits this for
        every small decode batch that overlaps a split prefill.
        """
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[128, 0, 8]], group_num=3
        )
        running_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        with self._stubbed_stream_idx():
            stream_idx, stream_group = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, running_batch, has_prefill=True
            )

        self.assertEqual(stream_idx, 1)
        self.assertEqual(stream_group, ("p1", "d1"))

    def test_manual_division_picks_the_highest_met_threshold(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1], [64, 0, 8]], group_num=4
        )
        running_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 12)

        with self._stubbed_stream_idx():
            stream_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, running_batch, has_prefill=True
            )

        self.assertEqual(stream_idx, 2)

    def test_stream_switch_uses_speculative_worker_backend_hook(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1]], group_num=3
        )
        scheduler.model_worker = SimpleNamespace(
            update_pdmux_decode_attn_backend=Mock()
        )
        target_update = scheduler.tp_worker.model_runner.update_decode_attn_backend
        running_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        with self._stubbed_stream_idx():
            SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, running_batch, has_prefill=True
            )

        scheduler.model_worker.update_pdmux_decode_attn_backend.assert_called_once_with(
            1
        )
        target_update.assert_not_called()

    def test_split_prefill_forward_installs_hicache_consumer_first(self):
        """Every split-prefill segment must install the HiCache consumer index
        before running the model.

        `set_hicache_consumer` selects which layer-transfer event set the KV
        pool waits on before reading loaded-back pages, and the decode forward
        that runs between segments resets it to the decode batch's -1 --
        which disables the wait entirely. A segment that skips the install
        therefore reads host-loaded KV while the transfer stream is still
        copying: garbage indices out of the DSV4 top-k indexer and a
        device-side IndexKernel assert under load (the PDMux + HiCache
        benchmark crash of 2026-08-26). This is the only forward entry point
        besides `forward_batch_generation`, which does install it.
        """
        tree = ast.parse(TP_WORKER_PATH.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_batch_split_prefill"
        )
        calls = [
            call.func.attr
            for call in ast.walk(method)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

        self.assertIn("set_hicache_consumer", calls)
        self.assertLess(calls.index("set_hicache_consumer"), calls.index("forward"))

    def test_plan_limit_resolves_after_chunked_prefill_size(self):
        """Scheduler init must resolve the chunk size before the plan limit.

        `init_pdmux_prefill_plan_limit` branches on `self.chunked_prefill_size`,
        which only `init_chunked_prefill` sets. Resolving the limit first raises
        AttributeError during startup for every attention backend that declares
        a plan limit -- which is exactly the PDMux configurations that need the
        limit, so the crash is not hypothetical.
        """
        targets = ("init_chunked_prefill", "init_pdmux_prefill_plan_limit")
        positions = _init_call_order("Scheduler", targets)

        for target in targets:
            self.assertIn(target, positions, f"{target} is not reached from __init__")
        self.assertLess(
            positions["init_chunked_prefill"],
            positions["init_pdmux_prefill_plan_limit"],
        )

    def test_pdmux_initialization_uses_parallel_state_gpu_id(self):
        config = object()
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(gpu_id=3),
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.load_pdmux_config",
                return_value=config,
            ) as load_pdmux_config,
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_disagg",
                return_value=SimpleNamespace(pdmux_config_path="pdmux.yaml"),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.initialize_stream_groups"
            ) as initialize_stream_groups,
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_stream_groups",
                return_value=[object(), object(), object()],
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_sm_counts",
                return_value=[(1, 0), (1, 1), (0, 1)],
            ),
        ):
            SchedulerMultiplexMixin.init_pdmux(scheduler)

        load_pdmux_config.assert_called_once_with("pdmux.yaml")
        initialize_stream_groups.assert_called_once_with(3, config)
        self.assertEqual(scheduler.real_sm_group_num, 3)

    def test_pdmux_prefill_status_is_observable(self):
        sentinel = object()
        with patch.object(parallel_state, "_PDMUX_PREFILL_TP_GROUP", sentinel):
            with patch.object(parallel_state, "get_parallel") as get_parallel:
                get_parallel.return_value.tp_group = sentinel
                self.assertTrue(is_pdmux_prefill_enabled())
            with patch.object(parallel_state, "get_parallel") as get_parallel:
                get_parallel.return_value.tp_group = object()
                self.assertFalse(is_pdmux_prefill_enabled())

    def test_pdmux_process_status_does_not_follow_prefill_phase(self):
        with patch.object(parallel_state, "_PDMUX_PREFILL_TP_GROUP", object()):
            with patch.object(parallel_state, "get_parallel") as get_parallel:
                get_parallel.return_value.tp_group = object()

                self.assertTrue(is_pdmux_enabled())
                self.assertFalse(is_pdmux_prefill_enabled())

    def _make_merge_streams(self, operations):
        prefill_stream = Mock()
        merge_done = object()
        prefill_stream.record_event.side_effect = lambda: (
            operations.append(("record", None)) or merge_done
        )
        decode_stream = Mock()
        decode_stream.wait_event.side_effect = lambda event: operations.append(
            ("wait", event)
        )
        return prefill_stream, decode_stream, merge_done

    def test_finished_prefill_merge_publishes_decode_dependency(self):
        operations = []
        split_batch = Mock()
        split_batch.chunked_req = None
        split_batch.is_empty.return_value = False
        # The unconditional filter drops nothing here: same size before/after.
        split_batch.batch_size.side_effect = [2, 2]
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.batch_is_full = True
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=None,
            process_batch_result=Mock(),
        )
        prefill_result = object()

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            prefill_result,
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.process_batch_result.assert_called_once_with(
            split_batch, prefill_result
        )
        split_batch.filter_batch.assert_called_once_with(chunked_req_to_exclude=[])
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(
            operations,
            [("merge", split_batch), ("record", None), ("wait", merge_done)],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertIsNone(scheduler.split_prefill_batch)

    def test_finished_prefill_releases_persistent_forward_batch(self):
        split_forward_batch = object()
        split_batch = Mock()
        split_batch.chunked_req = None
        split_batch.split_forward_batch = split_forward_batch
        split_batch.split_index = 61
        split_batch.split_forward_count = 4
        split_batch.split_prefill_finished = True
        split_batch.batch_size.side_effect = [1, 1]
        split_batch.is_empty.return_value = False
        running_batch = Mock()
        running_batch.is_empty.return_value = True
        running_batch.batch_is_full = True
        prefill_stream, decode_stream, merge_done = self._make_merge_streams([])
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=None,
            process_batch_result=Mock(),
        )

        returned = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            prefill_result=object(),
            prefill_stream=prefill_stream,
            decode_stream=decode_stream,
            running_batch=running_batch,
        )

        self.assertIs(returned, split_batch)
        self.assertIsNone(split_batch.split_forward_batch)
        self.assertEqual(split_batch.split_index, 0)
        self.assertEqual(split_batch.split_forward_count, 1)
        self.assertFalse(split_batch.split_prefill_finished)
        self.assertIsNone(scheduler.split_prefill_batch)
        decode_stream.wait_event.assert_called_once_with(merge_done)

    def test_merge_excludes_and_stashes_unfinished_chunked_request(self):
        """A request that only finished a middle chunk must be stashed and
        kept out of the decode batch; merging it would start decoding with a
        partial prefill."""
        operations = []
        chunked_req = _make_chunked_req(extend_end=32, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.split_prefill_finished = True
        split_batch.batch_size.side_effect = [2, 1]
        split_batch.is_empty.return_value = False
        split_batch.filter_batch.side_effect = lambda **kwargs: operations.append(
            ("filter", kwargs)
        )
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.batch_is_full = True
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(
                side_effect=lambda req: operations.append(("stash", req))
            ),
        )

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.stash_chunked_request.assert_called_once_with(chunked_req)
        (filter_op,) = [op for op in operations if op[0] == "filter"]
        self.assertEqual(filter_op[1]["chunked_req_to_exclude"], [chunked_req])
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(
            [op[0] for op in operations],
            ["stash", "filter", "merge", "record", "wait"],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIsNone(scheduler.split_prefill_batch)

    def test_merge_of_pure_middle_chunk_keeps_decode_batch(self):
        """A batch holding only a middle chunk merges nothing into decode, but
        the dependency event must still be published: the stash frees
        deduplicated KV pages on the prefill stream that decode may
        reallocate right after."""
        operations = []
        chunked_req = _make_chunked_req(extend_end=32, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.split_prefill_finished = True
        split_batch.batch_size.side_effect = [1, 0]
        split_batch.is_empty.return_value = True
        running_batch = Mock()
        running_batch.batch_is_full = True
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
        )

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        running_batch.merge_batch.assert_not_called()
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(
            [op[0] for op in operations],
            ["record", "wait"],
        )

    def test_merge_skips_stash_for_parked_chunk(self):
        """A parked chunk (no new KV beyond the cached prefix) must be
        excluded from the merge without being stashed — stashing it would be
        a no-op insert that still pays radix-cache work."""
        chunked_req = _make_chunked_req(extend_end=16, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.batch_size.side_effect = [1, 0]
        split_batch.is_empty.return_value = True
        running_batch = Mock()
        prefill_stream, decode_stream, _ = self._make_merge_streams([])
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
        )

        SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.stash_chunked_request.assert_not_called()
        split_batch.filter_batch.assert_called_once_with(
            chunked_req_to_exclude=[chunked_req]
        )

    def test_update_split_prefill_batch_processes_pending_chunked_abort(self):
        """PDMux never calls get_next_batch_to_run, so the mixin must drain
        pending chunked aborts itself; without this an aborted chunked request
        leaks its KV forever."""
        running_batch = _Batch(empty=True)
        scheduler = SimpleNamespace(
            split_prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=None, running_batch=running_batch
                )
            ),
            dp_attn_adapter=SimpleNamespace(
                maybe_prepare_mlp_sync_batch=Mock(return_value=None)
            ),
        )

        created, returned = SchedulerMultiplexMixin.update_split_prefill_batch(
            scheduler, 1, running_batch
        )

        scheduler.process_pending_chunked_abort.assert_called_once_with()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(
            None
        )
        self.assertFalse(created)
        self.assertIs(returned, running_batch)

    def test_update_split_prefill_batch_accepts_peer_dp_idle_batch(self):
        running_batch = _Batch(empty=True)
        idle_mode = SimpleNamespace(is_idle=lambda: True)
        idle_batch = _Batch(empty=True)
        idle_batch.forward_mode = idle_mode
        adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=Mock(return_value=idle_batch)
        )
        scheduler = SimpleNamespace(
            split_prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=None, running_batch=running_batch
                )
            ),
            dp_attn_adapter=adapter,
        )

        created, returned = SchedulerMultiplexMixin.update_split_prefill_batch(
            scheduler, 1, running_batch
        )

        self.assertTrue(created)
        self.assertIs(returned, running_batch)
        self.assertIs(scheduler.split_prefill_batch, idle_batch)
        self.assertIs(idle_batch.forward_mode, idle_mode)
        self.assertEqual(idle_batch.split_index, 0)
        self.assertFalse(idle_batch.split_prefill_finished)
        self.assertEqual(idle_batch.split_forward_count, 1)
        self.assertIsNone(idle_batch.split_forward_batch)
        adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(None)

    def test_update_split_prefill_batch_defers_abort_while_chunk_in_flight(self):
        """Tearing down a chunked request while its split forward is running
        is unsafe; the abort must wait for the between-chunks safe point."""
        running_batch = _Batch(empty=True)
        scheduler = SimpleNamespace(
            split_prefill_batch=Mock(),
            process_pending_chunked_abort=Mock(),
        )

        created, returned = SchedulerMultiplexMixin.update_split_prefill_batch(
            scheduler, 1, running_batch
        )

        scheduler.process_pending_chunked_abort.assert_not_called()
        self.assertFalse(created)
        self.assertIs(returned, running_batch)


if __name__ == "__main__":
    unittest.main()

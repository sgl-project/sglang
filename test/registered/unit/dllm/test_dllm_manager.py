"""The dLLM owner's request lists: which list holds a request at each stage,
and the requests it rejects at intake."""

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.manager import DllmManager
from sglang.srt.dllm.req_state import DllmReqPhase, DllmReqState
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_request_lists import (
    make_request_registry,
    make_request_retirer,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RowBudget:
    """The admission gate of BeamCoordinator with row_budget() free rows."""

    num_rows = staticmethod(BeamCoordinator.num_rows)

    def __init__(self, row_budget):
        self.row_budget = row_budget

    def has_room(self, num_reqs, num_rows):
        return num_rows <= self.row_budget()


def _manager(**deps):
    kwargs = dict(
        dllm_config=DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=_BLOCK_SIZE,
            mask_id=_MASK_ID,
            max_running_requests=8,
            first_done_first_out_mode=True,
            requires_separate_context_encoding=False,
        ),
        kv=SimpleNamespace(
            tree_cache=None,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=Mock(), free_group_end=Mock()
            ),
        ),
        services=SimpleNamespace(
            metrics_sink=SimpleNamespace(
                num_generated_tokens=0, report_prefill_stats=Mock()
            ),
            output_streamer=SimpleNamespace(stream_output=Mock()),
        ),
        policy=SimpleNamespace(calc_priority=Mock()),
        new_token_ratio_tracker=None,
        page_size=1,
        max_prefill_tokens=0,
        chunked_prefill_size=None,
        priority_scheduling_preemption_threshold=0,
        enable_priority_preemption=False,
        enable_priority_scheduling=False,
        truncation_align_size=None,
        context_len=128,
        beam_coordinator=_RowBudget(lambda: 64),
        step_builder=SimpleNamespace(
            build_extend_step=_Step, build_sampling_info=Mock()
        ),
    )
    kwargs.update(deps)
    if "request_retirer" not in kwargs:
        kwargs["request_retirer"] = make_request_retirer()
        kwargs["request_retirer"].send_abort_output = Mock()
    return DllmManager(**kwargs)


_BLOCK_SIZE = 4
_MASK_ID = 99


def _req(rid, prompt_len=8, max_new_tokens=64):
    sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
    sampling_params.normalize(None)
    return Req(rid, "", array("q", range(1, prompt_len + 1)), sampling_params)


class _Adder:
    """Admits every request with one block."""

    def __init__(self):
        self.can_run_list = []
        self.preempt_list = []

    def _admit(self, req):
        req.extend_end = req.prefix_len + _BLOCK_SIZE
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE

    def add_dllm_staging_req(self, req):
        return self._admit(req)

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        return self._admit(req)


class _NoTokenAdder(_Adder):
    """Admits nothing: the first request finds no KV."""

    def _admit(self, req):
        return AddReqResult.NO_TOKEN


class _DecodeOnlyAdder(_Adder):
    """Finds no KV for a prefill block; a decode block fits."""

    def _admit(self, req):
        if req.dllm.is_prefill():
            return AddReqResult.NO_TOKEN
        return super()._admit(req)


class _Step:
    def __init__(self, reqs, kv_reuse_lens):
        self.reqs = list(reqs)
        self.kv_reuse_lens = kv_reuse_lens
        self.req_pool_indices = None
        self.return_logprob = False
        self.prefill_stats = None
        self.dp_cooperation_info = None

    def batch_size(self):
        return len(self.reqs)


def _registry_over(manager):
    """A registry whose only live requests are the manager's."""
    return make_request_registry(mode_owners=ModeOwners(dllm=manager))


def _fdfo_result(blocks, accept_lengths, algo_states):
    return SimpleNamespace(
        copy_done=None,
        next_token_ids=blocks,
        accept_length_per_req_cpu=accept_lengths,
        dllm_algo_state=algo_states,
        can_run_cuda_graph=False,
    )


class TestDllmManagerLists(unittest.TestCase):
    def setUp(self):
        self.checkpoint = self._patch("sglang.srt.dllm.manager.checkpoint_kv_cache")
        self._patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        )
        self._patch(
            "sglang.srt.dllm.manager.DllmManager._create_dllm_prefill_adder",
            side_effect=_Adder,
        )
        self._patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter."
            "PrefillStats.from_adder",
            return_value=None,
        )

    def _patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _assert_lists(self, manager, waiting, staging, between_blocks=()):
        self.assertEqual([r.rid for r in manager.waiting], [r.rid for r in waiting])
        self.assertEqual([r.rid for r in manager.staging], [r.rid for r in staging])
        self.assertEqual(
            [r.rid for r in manager.between_blocks], [r.rid for r in between_blocks]
        )

    def _run_one_block(self, manager, reqs):
        """Take reqs, run one block and leave it unresolved: the requests are
        then between blocks."""
        step, _ = manager.schedule(list(reqs))
        manager.on_launch(step)
        n = len(step.reqs)
        manager.commit_extend(
            step, _fdfo_result([[_MASK_ID] * _BLOCK_SIZE] * n, [0] * n, [None] * n)
        )

    def test_each_request_is_in_one_list_from_schedule_to_commit(self):
        manager = _manager()
        first, second = _req("first"), _req("second")

        step, rest = manager.schedule([first, second])

        self.assertEqual(rest, [])
        self.assertEqual(step.reqs, [first, second])
        self._assert_lists(manager, waiting=[first, second], staging=[])

        manager.on_launch(step)
        self._assert_lists(manager, waiting=[], staging=[first, second])

        # FDFO: the first block resolves, the second stays unresolved.
        state = object()
        unresolved = [7, 7, _MASK_ID, 7]
        manager.commit_extend(
            step,
            _fdfo_result([[5, 6, 7, 8], unresolved], [_BLOCK_SIZE, 0], [None, state]),
        )

        self._assert_lists(
            manager, waiting=[], staging=[], between_blocks=[first, second]
        )
        self.checkpoint.assert_called_once_with(first, manager.tree_cache)
        # The unresolved block keeps its KV and state, and runs again as is.
        self.assertEqual(second.dllm.incomplete_ids.tolist(), unresolved)
        self.assertIs(second.dllm.algo_state, state)
        self.assertEqual(second.dllm.phase, DllmReqPhase.STAGING_DECODE)
        self.assertEqual(
            second.full_untruncated_fill_ids[second.prefix_len :].tolist(), unresolved
        )
        # The resolved one moved to its next block.
        self.assertEqual(first.dllm.block_offset, _BLOCK_SIZE)
        self.assertEqual(
            first.full_untruncated_fill_ids[-_BLOCK_SIZE:].tolist(),
            [_MASK_ID] * _BLOCK_SIZE,
        )

    def test_commit_returns_requests_in_the_order_they_were_taken(self):
        manager = _manager()
        reqs = [_req("a"), _req("b"), _req("c")]
        manager.schedule(list(reqs))
        launched = _Step([reqs[2], reqs[0]], None)
        manager.on_launch(launched)
        self._assert_lists(manager, waiting=[reqs[1]], staging=[reqs[2], reqs[0]])

        manager.commit_extend(
            launched,
            _fdfo_result([[_MASK_ID] * _BLOCK_SIZE] * 2, [0, 0], [None, None]),
        )

        self._assert_lists(
            manager, waiting=[reqs[1]], staging=[], between_blocks=[reqs[0], reqs[2]]
        )

    def test_a_step_selects_across_both_lists_in_the_order_requests_were_taken(
        self,
    ):
        manager = _manager()
        a, b, c = _req("a"), _req("b"), _req("c")
        for seq, req in enumerate((a, b, c)):
            req.dllm = DllmReqState.for_req(req, manager.dllm_config, intake_seq=seq)
            req.dllm.phase = DllmReqPhase.STAGING_DECODE
        manager.waiting = [b]
        manager.between_blocks = [a, c]

        self.assertEqual(manager.get_decode_requests(), [a, b, c])

    def test_a_request_finished_at_commit_leaves_both_lists(self):
        manager = _manager()
        # A prompt shorter than a block: its first block is output.
        done = _req("done", prompt_len=2, max_new_tokens=2)

        step, _ = manager.schedule([done])
        manager.on_launch(step)
        manager.commit_extend(step, _fdfo_result([[1, 2, 3, 4]], [_BLOCK_SIZE], [None]))

        self.assertTrue(done.finished())
        self._assert_lists(manager, waiting=[], staging=[])
        self.checkpoint.assert_not_called()

    def test_no_request_slot_with_an_empty_manager_does_not_block_later_requests(
        self,
    ):
        allocatable = [0]
        manager = _manager(beam_coordinator=_RowBudget(lambda: allocatable[0]))

        step, rest = manager.schedule([_req("a")])
        self.assertIsNone(step)

        allocatable[0] = 64
        step, rest = manager.schedule(rest)

        self.assertEqual([r.rid for r in step.reqs], ["a"])

    def test_abort_after_no_token_does_not_block_the_next_request(self):
        manager = _manager()
        self._patch(
            "sglang.srt.dllm.manager.DllmManager._create_dllm_prefill_adder",
            side_effect=_NoTokenAdder,
        )

        step, rest = manager.schedule([_req("a")])
        self.assertIsNone(step)
        manager.abort_waiting(AbortReq(rid="a"))
        self.assertTrue(manager.is_empty())

        self._patch(
            "sglang.srt.dllm.manager.DllmManager._create_dllm_prefill_adder",
            side_effect=_Adder,
        )
        step, rest = manager.schedule([_req("b")])

        self.assertEqual([r.rid for r in step.reqs], ["b"])

    def test_no_token_with_a_request_in_flight_does_not_wait_for_its_commit(self):
        manager = _manager()
        step, _ = manager.schedule([_req("a")])
        manager.on_launch(step)
        self._patch(
            "sglang.srt.dllm.manager.DllmManager._create_dllm_prefill_adder",
            side_effect=_NoTokenAdder,
        )

        blocked, rest = manager.schedule([_req("b")])
        self.assertIsNone(blocked)
        self.assertEqual([r.rid for r in manager.staging], ["a"])

        # The next pass, before the in-flight step commits, admits b.
        self._patch(
            "sglang.srt.dllm.manager.DllmManager._create_dllm_prefill_adder",
            side_effect=_Adder,
        )
        step, rest = manager.schedule(rest)

        self.assertEqual([r.rid for r in step.reqs], ["b"])

    def test_a_prefill_that_does_not_fit_lets_a_started_request_run(self):
        # The started request holds the KV between its blocks; the new
        # request's prefill finds no room until the started one finishes.
        manager = _manager()
        started, new = _req("started"), _req("new")
        self._run_one_block(manager, [started])
        self._patch(
            "sglang.srt.dllm.manager.DllmManager._create_dllm_prefill_adder",
            side_effect=_DecodeOnlyAdder,
        )

        step, rest = manager.schedule([new])

        self.assertEqual(rest, [])
        self.assertEqual([r.rid for r in step.reqs], ["started"])
        self.assertTrue(step.forward_mode.is_dllm_extend())
        self._assert_lists(manager, waiting=[new], staging=[], between_blocks=[started])

    def test_abort_pops_and_releases_the_waiting_requests_it_names(self):
        manager = _manager()
        named, holding_kv, kept = _req("abort-1"), _req("abort-2"), _req("keep")
        holding_kv.kv.req_pool_idx = 3
        manager.waiting = [named, kept, holding_kv]

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            manager.abort_waiting(AbortReq(rid="abort"))

        self.assertEqual(manager.waiting, [kept])
        self.assertEqual(
            [
                c.args[0]
                for c in manager.request_retirer.send_abort_output.call_args_list
            ],
            [named, holding_kv],
        )
        release.assert_called_once_with(
            holding_kv, manager.request_retirer.tree_cache, checkpoint=False
        )

    def test_scheduler_abort_reaches_each_list(self):
        manager = _manager()
        waiting, staging, between = _req("waiting"), _req("staging"), _req("between")
        manager.waiting = [waiting]
        manager.staging = [staging]
        manager.between_blocks = [between]

        _registry_over(manager).abort_request(AbortReq(abort_all=True))

        # Waiting and between blocks: popped and released. Staging: finished
        # at its commit.
        self.assertEqual(manager.waiting, [])
        self.assertEqual(manager.between_blocks, [])
        self.assertEqual(
            [
                c.args[0]
                for c in manager.request_retirer.send_abort_output.call_args_list
            ],
            [waiting, between],
        )
        self.assertEqual(manager.staging, [staging])
        self.assertIsNotNone(staging.to_finish)
        self.assertIsNone(waiting.to_finish)
        self.assertIsNone(between.to_finish)

    def test_running_timeout_reaches_a_request_between_blocks(self):
        manager = _manager()
        started, taken = _req("started"), _req("taken")
        self._run_one_block(manager, [started])
        manager.waiting.append(taken)
        stale = time.perf_counter() - 10
        started.time_stats.forward_entry_time = stale
        taken.time_stats.forward_entry_time = stale
        registry = _registry_over(manager)

        self.assertEqual(registry.collect_inflight_reqs(), [started])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = registry.poll_timeout_aborts()
        self.assertEqual([abort.rid for abort in aborts], ["started"])

        registry.abort_request(aborts[0])

        self._assert_lists(manager, waiting=[taken], staging=[])
        manager.request_retirer.send_abort_output.assert_called_once_with(started)
        self.assertIsNone(started.to_finish)


class TestDllmRequestValidation(unittest.TestCase):
    def test_a_grammar_request_is_rejected(self):
        # A dLLM step applies no vocab mask and never takes a request back
        # from the grammar queue.
        manager = _manager()
        self.assertIsNone(manager.validate_request(_req("plain")))
        for name, value in (
            ("json_schema", '{"type": "object"}'),
            ("regex", "a+"),
            ("ebnf", 'root ::= "a"'),
            ("structural_tag", "{}"),
        ):
            with self.subTest(name=name):
                req = _req(name)
                setattr(req.sampling_params, name, value)

                error = manager.validate_request(req)

                self.assertIsNotNone(error)
                self.assertIn(name, error)


if __name__ == "__main__":
    unittest.main()

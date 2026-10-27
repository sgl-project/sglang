"""Exits that skipped a release part, now retired through RequestRetirer."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.managers.scheduler_components.engine_state import LaunchCounters
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.scheduler_request_intake import make_request_intake
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

_RETIRER = "sglang.srt.managers.scheduler_components.request_retirer"
_SCHEDULER = "sglang.srt.managers.scheduler"

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _retirer(**deps) -> RequestRetirer:
    args = dict(
        kv=SimpleNamespace(tree_cache=MagicMock()),
        beam_coordinator=MagicMock(),
        hisparse_coordinator=None,
        decode_offload_manager=None,
        pending=PendingSteps(),
        output_streamer=MagicMock(),
        ipc_channels=MagicMock(),
    )
    args.update(deps)
    return RequestRetirer(**args)


def _image_item() -> MultimodalDataItem:
    return MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(2))


class TestBootstrapFailureReleasesMedia(CustomTestCase):
    """A PD prefill bootstrap failure releases the request's media like every
    other abort."""

    def _fail_bootstrap(self, req: Req) -> None:
        retirer = _retirer()
        owner = prefill_owner_shell(
            tree_cache=retirer.tree_cache,
            request_retirer=retirer,
            req_to_metadata_buffer_idx_allocator=Mock(),
            output_streamer=Mock(),
            metrics_sink=SimpleNamespace(enable_metrics=False),
        )
        owner.kv_senders[req.rid] = SimpleNamespace(failure_exception=Mock())
        released = []
        with (
            get_parallel().override(tp_rank=0),
            patch.object(
                MultimodalDataItem,
                "release_transport_proxies",
                autospec=True,
                side_effect=released.append,
            ),
        ):
            owner.handle_bootstrap_failure(req)
        self.assertNotIn(req.rid, owner.kv_senders)
        self.assertIsNotNone(req.finished_reason)
        return released

    def test_releases_this_turns_media(self):
        item = _image_item()
        req = Req("r", "", [1, 2], SamplingParams())
        req.disagg = DisaggReqState()
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])

        released = self._fail_bootstrap(req)

        self.assertEqual(released, [item])
        self.assertIsNone(item.feature)
        self.assertIsNone(req.multimodal_inputs)

    def test_session_turn_releases_its_media_not_the_history(self):
        session = Session(100, streaming=True)
        session._inflight = True
        history = _image_item()
        item = _image_item()
        req = Req("s", "", [1, 2], SamplingParams(), session=session)
        req.disagg = DisaggReqState()
        req.multimodal_inputs = MultimodalInputs(mm_items=[history])
        req.extend_image_inputs(MultimodalInputs(mm_items=[item]))

        released = self._fail_bootstrap(req)

        self.assertEqual(released, [item])
        self.assertIsNone(item.feature)
        self.assertTrue(torch.equal(history.feature, torch.ones(2)))
        self.assertFalse(session._inflight)


def _beam_coordinator(core_lists) -> BeamCoordinator:
    return BeamCoordinator(
        model_config=None,
        spec_algorithm=None,
        max_req_len=0,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        future_map=None,
        core_lists=core_lists,
    )


def _beam_leader(beam: BeamCoordinator, rid: str, **attrs) -> Req:
    """A request whose beam group validate_and_init has counted in."""
    req = Req(rid, "", [1, 2], SamplingParams(), **attrs)
    req.beam_group = BeamGroup(
        beam_width=2,
        stop_token_ids=[],
        max_new_tokens=4,
        num_return=1,
        device="cpu",
    )
    req.beam_group.leader = req
    beam._num_live_groups += 1
    return req


class TestBeamGroupLeavesWithItsLeader(CustomTestCase):
    """A beam leader that leaves outside the beam commit retires its group, so
    the live-group count returns to zero."""

    def setUp(self):
        enter_scope(self, published_topology())

    def _intake(self, **deps):
        core_lists = CoreLists()
        beam = _beam_coordinator(core_lists)
        intake = make_request_intake(
            core_lists=core_lists,
            beam_coordinator=beam,
            request_retirer=_retirer(beam_coordinator=beam),
            **deps,
        )
        return intake, beam

    def test_queue_limit_rejection(self):
        intake, beam = self._intake(max_queued_requests=1)
        intake.lists.waiting_queue = [Req("queued", "", [1], SamplingParams())]
        leader = _beam_leader(beam, "beam")

        intake.add_request_to_queue(leader)

        self.assertNotIn(leader, intake.lists.waiting_queue)
        self.assertEqual(beam._num_live_groups, 0)

    def test_priority_rejection(self):
        get_context().override("test", enable_priority_scheduling=False)
        get_context().override("test", abort_on_priority_when_disabled=True)
        intake, beam = self._intake()
        leader = _beam_leader(beam, "beam", priority=1)

        intake.add_request_to_queue(leader)

        self.assertEqual(intake.lists.waiting_queue, [])
        self.assertEqual(beam._num_live_groups, 0)

    def test_mid_chunk_abort(self):
        sched = partial_scheduler()
        sched.parts.mode_owners = ModeOwners()
        sched.parts.lists = CoreLists()
        beam = _beam_coordinator(sched.parts.lists)
        leader = _beam_leader(beam, "beam")
        leader.kv.req_pool_idx = 1
        sched.parts.lists.chunked_req = leader
        sched.parts.lists._pending_chunked_abort_req = leader
        sched.parts.request_retirer = _retirer(beam_coordinator=beam)
        sched.parts.kv.tree_cache = sched.parts.request_retirer.tree_cache
        sched.parts.ipc_channels = Mock()

        with (
            patch(f"{_RETIRER}.release_kv_cache") as release,
            # Where the scheduler released the KV itself.
            patch(f"{_SCHEDULER}.release_kv_cache", release, create=True),
        ):
            sched.process_pending_chunked_abort()

        release.assert_called_once_with(
            leader, sched.parts.request_retirer.tree_cache, checkpoint=False
        )
        self.assertIsNone(sched.parts.lists.chunked_req)
        self.assertEqual(beam._num_live_groups, 0)


class TestRetractBacksUpBeforeHiSparse(CustomTestCase):
    """A PD decode retraction copies the request's KV to host before HiSparse
    frees the rows and device buffers the copy reads."""

    def test_backup_runs_before_the_hisparse_release(self):
        enter_scope(
            self,
            published_topology(
                disaggregation_mode="decode",
                disaggregation_decode_retraction_backup="host_pool",
            ),
        )
        events = []
        hisparse = SimpleNamespace(
            retract_req=lambda req: events.append("hisparse"),
            holds_rows=lambda req: True,
            request_finished=lambda req: events.append("hisparse"),
        )
        retirer = _retirer(
            kv=SimpleNamespace(
                tree_cache=MagicMock(),
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            ),
            hisparse_coordinator=hisparse,
        )
        req = Req("r", "", [1, 2], SamplingParams())
        req.kv.req_pool_idx = 1

        def release_kv(req, tree_cache, *, checkpoint):
            events.append("kv")
            req.kv.req_pool_idx = None

        with (
            patch(
                f"{_RETIRER}.backup_kv_cache",
                side_effect=lambda *args: events.append("backup") or True,
            ),
            patch(f"{_RETIRER}.release_kv_cache", side_effect=release_kv),
            patch(f"{_RETIRER}.evict_from_tree_cache"),
        ):
            self.assertTrue(retirer.retract(req, num_remaining_reqs=1))

        self.assertEqual(events, ["backup", "hisparse", "kv"])


class TestOverLengthEmbeddingIsRejected(CustomTestCase):
    """An embedding prompt over the input limit, without auto-truncation, is
    rejected like a generate prompt: queued only as a stub that finishes with
    a 400 abort."""

    def test_over_length_embedding_finishes_with_an_abort(self):
        enter_scope(self, published_topology(allow_auto_truncate=False))
        intake = make_request_intake(
            request_retirer=_retirer(), max_req_input_len=4, tokenizer=None
        )
        intake._maybe_namespace_elastic_radix_cache = Mock()
        intake._prefetch_kvcache = Mock()
        recv_req = SimpleNamespace(
            rid="long-embedding",
            input_text="",
            input_ids=list(range(8)),
            sampling_params=SamplingParams(max_new_tokens=0),
            positional_embed_overrides=None,
            token_type_ids=None,
            routed_dp_rank=None,
            priority=None,
            dimensions=None,
            lora_id=None,
            lora_name=None,
            http_worker_ipc=None,
            time_stats=None,
            return_pooled_hidden_states=False,
            multi_item_delimiter_indices=None,
            token_indices_to_pool=None,
            decision_layout=None,
            mm_inputs=None,
        )

        intake.handle_embedding_request(recv_req)

        (req,) = intake.lists.waiting_queue
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("exceeds the maximum allowed length", req.to_finish.message)


class TestOptimisticRequeueKeepsMedia(CustomTestCase):
    """A PD prefill request requeued for another optimistic attempt ends this
    attempt's cache request and frees its KV, but keeps its media for the
    retry."""

    def test_requeued_request_keeps_its_media(self):
        enter_scope(self, published_topology(optimistic_prefill_attempts=4))
        retirer = _retirer()
        owner = prefill_owner_shell(
            tree_cache=retirer.tree_cache,
            request_retirer=retirer,
            req_to_metadata_buffer_idx_allocator=Mock(),
            metrics_sink=SimpleNamespace(enable_metrics=False),
            counters=LaunchCounters(),
        )
        item = _image_item()
        req = Req("retry", "", [1, 2, 3], SamplingParams())
        req.disagg = DisaggReqState(metadata_buffer_index=0)
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])
        req.kv.req_pool_idx = 1
        handle = req.cache_request_handle

        def release_kv(req, tree_cache, *, checkpoint):
            self.assertFalse(checkpoint)
            req.kv.req_pool_idx = None

        with (
            patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
            patch(f"{_RETIRER}.release_kv_cache", side_effect=release_kv),
        ):
            waiting = owner.optimistic_release_and_requeue(req, [])

        self.assertEqual(waiting, [req])
        retirer.tree_cache.finish.assert_called_once_with(
            handle, CacheRequestOutcome.ABORT
        )
        self.assertIs(req.multimodal_inputs.mm_items[0], item)
        self.assertTrue(torch.equal(item.feature, torch.ones(2)))


class TestPreallocAbortLeavesThroughTheRetirer(CustomTestCase):
    """A PD decode request aborted before preallocation releases its media and
    ends its cache request like every other abort."""

    def test_aborted_prealloc_request_releases_media_and_cache_request(self):
        retirer = _retirer()
        item = _image_item()
        req = Req("aborted", "", [1, 2], SamplingParams())
        req.finished_reason = FINISH_ABORT("aborted")
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])
        handle = req.cache_request_handle
        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = Mock()
        queue._update_handshake_waiters = Mock()
        queue._uses_swa_tail_prealloc = Mock(return_value=False)
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._swa_aware_allocatable_token_budgets = Mock(return_value=(0, None))
        queue._hicache_pending_restore_tokens = Mock(return_value=0)
        queue.enable_priority_scheduling = False
        queue.enable_hisparse = False
        queue.enable_lora = False
        queue.output_streamer = Mock()
        queue.tree_cache = retirer.tree_cache
        queue.request_retirer = retirer

        _, failed = queue.pop_preallocated([], lora_ids_in_use=set)

        self.assertEqual(failed, [decode_req])
        retirer.tree_cache.finish.assert_called_once_with(
            handle, CacheRequestOutcome.ABORT
        )
        self.assertIsNone(item.feature)
        self.assertIsNone(req.multimodal_inputs)


class _RecordingTree:
    """A prefix cache that records the row ranges it frees, over a real slot
    pool."""

    def __init__(self, page_size: int = 1):
        self.req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=32, device="cpu", enable_memory_saver=False
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
        self.frees = []
        self.unlock = Mock()
        self.on_release = Mock()
        self.finish = Mock()

    def free_kv_row(self, kv, ranges):
        self.frees.append((kv.req_pool_idx, list(ranges)))

    def claim_kv_row(self, req):
        return False

    def supports_mamba(self):
        return False


class TestDeferredFree(CustomTestCase):
    """A released request's slot and the KV pages a pending step may still
    write go back to their pools at that step's host commit, not before."""

    def setUp(self):
        enter_scope(self, published_topology())

    def _setup(self, page_size=1):
        self.tree = _RecordingTree(page_size)
        self.pool = self.tree.req_to_token_pool
        self.pending = PendingSteps()
        self.retirer = _retirer(
            kv=SimpleNamespace(
                tree_cache=self.tree,
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.tree.token_to_kv_pool_allocator,
            ),
            pending=self.pending,
        )

    def _req(self, rid="r", *, committed=5, allocated=6):
        req = Req(rid, "", list(range(8)), SamplingParams())
        self.pool.alloc([req])
        req.kv.kv_committed_len = committed
        req.kv.kv_allocated_len = allocated
        return req

    def _launch(self, *reqs):
        step = SimpleNamespace(reqs=list(reqs))
        self.pending.push(
            PendingStep(step=step, result=None, launch_ts=0.0, after_idle_gap=False)
        )
        return step

    def _commit(self, step):
        record = self.pending.popleft()
        self.assertIs(record.step, step)
        self.retirer.release_held(step)

    def test_no_pending_step_frees_at_once(self):
        self._setup()
        req = self._req()
        slot = req.kv.req_pool_idx

        self.retirer.retire_aborted(req)

        self.assertEqual(self.tree.frees, [(slot, [(0, 5)]), (slot, [(5, 6)])])
        self.assertIn(slot, self.pool.free_slots)
        self.assertFalse(req.kv.holds_kv)

    def test_the_last_pending_step_holds_the_slot_and_written_pages(self):
        self._setup()
        req = self._req()
        slot = req.kv.req_pool_idx
        other = self._launch(self._req("other"))
        step = self._launch(req)

        self.retirer.retire_aborted(req)

        # The committed pages go now; the written token and the slot wait.
        self.assertEqual(self.tree.frees, [(slot, [(0, 5)])])
        self.assertNotIn(slot, self.pool.free_slots)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(len(list(self.pending.held())), 1)

        self._commit(other)
        self.assertNotIn(slot, self.pool.free_slots)
        self._commit(step)
        self.assertEqual(self.tree.frees[1:], [(slot, [(5, 6)])])
        self.assertIn(slot, self.pool.free_slots)
        self.assertEqual(list(self.pending.held()), [])

    def test_two_pending_steps_hold_until_the_newer_commits(self):
        self._setup()
        req = self._req()
        slot = req.kv.req_pool_idx
        older, newer = self._launch(req), self._launch(req)

        self.retirer.retire_finished(req, checkpoint=False)

        self._commit(older)
        self.assertNotIn(slot, self.pool.free_slots)
        self._commit(newer)
        self.assertIn(slot, self.pool.free_slots)

    def test_retract_while_held_takes_a_new_slot(self):
        self._setup()
        req = self._req()
        slot = req.kv.req_pool_idx
        step = self._launch(req)

        with patch(f"{_RETIRER}.evict_from_tree_cache"):
            self.retirer.retract(req, num_remaining_reqs=0)
        # The requeued request is admitted again before the step commits.
        self.pool.alloc([req])
        new_slot = req.kv.req_pool_idx
        self.assertNotEqual(new_slot, slot)
        # An abort of the requeued request releases only what it holds now.
        self.retirer.retire_aborted_waiting(req)

        self._commit(step)
        self.assertIn(slot, self.pool.free_slots)
        self.assertNotIn(new_slot, self.pool.free_slots)
        self.assertEqual(self.tree.frees, [(slot, [(0, 5)]), (slot, [(5, 6)])])

    def test_abort_while_held_frees_once(self):
        self._setup()
        req = self._req()
        slot = req.kv.req_pool_idx
        step = self._launch(req)

        self.retirer.retire_finished(req, checkpoint=False)
        # An abort reaching the request after it left holds nothing more.
        self.retirer.retire_aborted(req)

        self._commit(step)
        self.assertEqual(self.pool.free_slots.count(slot), 1)
        self.assertEqual(self.tree.frees, [(slot, [(0, 5)]), (slot, [(5, 6)])])

    def test_pp_slots_hold_by_the_step_that_carries_the_request(self):
        self._setup()
        req = self._req()
        slot = req.kv.req_pool_idx
        slot0 = self._launch(self._req("slot0"))
        slot1 = self._launch(req)
        slot0_next = self._launch(self._req("slot0-next"))

        self.retirer.retire_aborted(req)

        self._commit(slot0)
        self.assertNotIn(slot, self.pool.free_slots)
        self._commit(slot1)
        self.assertIn(slot, self.pool.free_slots)
        self._commit(slot0_next)

    def test_paged_hold_starts_at_the_committed_page(self):
        self._setup(page_size=4)
        req = self._req(committed=6, allocated=7)
        slot = req.kv.req_pool_idx
        step = self._launch(req)

        self.retirer.retire_aborted(req)

        self.assertEqual(self.tree.frees, [(slot, [(0, 4)])])
        self._commit(step)
        # The page holding the committed tail and the written token; the
        # over-allocated range starts on the next page, past the allocation.
        self.assertEqual(self.tree.frees[1:], [(slot, [(4, 6)])])


if __name__ == "__main__":
    unittest.main()

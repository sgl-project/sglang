import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
    commit_step_counters,
    set_mamba_track_indices_from_reqs,
)
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.forward_batch_view import worker_forward_batch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    prepare_mamba_track_for_verify,
    spec_prepare_for_decode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import enter_scope, published_topology

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


# The decode checkpoint grid is lcm(mamba_cache_chunk_size, tree page,
# interval); keeping all three equal leaves it at the interval under test.
TRACK_INTERVAL = 4


def _make_batch() -> tuple[Req, ScheduleBatch]:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.output_ids.append(3)
    req.kv.kv_committed_len = 2

    batch = ScheduleBatch(reqs=[req])
    batch.tree_cache = SimpleNamespace(page_size=TRACK_INTERVAL)
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False, vocab_size=128)
    batch.enable_overlap = True
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.seq_lens = torch.tensor([2], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2], dtype=torch.int32)
    return req, batch


def _make_processor(tree_cache=None) -> SchedulerBatchResultProcessor:
    metrics_sink = MagicMock()
    metrics_sink.num_generated_tokens = 0
    metrics_sink.forward_ct_decode = 0
    kv = SimpleNamespace(
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=tree_cache or SimpleNamespace(page_size=TRACK_INTERVAL),
        req_to_token_pool=None,
    )
    return SchedulerBatchResultProcessor(
        is_generation=True,
        model_config=SimpleNamespace(think_end_ids=None),
        kv=kv,
        services=SimpleNamespace(
            metrics_sink=metrics_sink,
            output_streamer=MagicMock(),
        ),
        beam_coordinator=MagicMock(),
        decode_offload_manager=None,
        metrics_collector=None,
        workers=SimpleNamespace(draft_worker=None, model_worker=MagicMock()),
        logprob_result_processor=None,
        pending=PendingSteps(),
        request_retirer=make_request_retirer(kv=kv),
    )


def _make_result():
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=[4],
        speculative_num_draft_tokens=0,
    )


class TestMambaBoundaryCommit(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_overlapped_boundary_moves_req_state_at_its_own_commit(self):
        """Under overlap, the step after a boundary step is built before the
        boundary commits. The build reads the Req; the boundary's checkpoint
        reaches the Req only at its own commit, from the step it ran."""
        req, _ = _make_batch()
        req.origin_input_ids = array("q", [1, 2, 3])
        req.kv.req_pool_idx = 0
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 3
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([10, 11])
        req.kv.mamba_next_track_idx = 0
        req.kv.mamba_last_track_idx = 1
        processor = _make_processor()
        result = _make_result()

        scheduler = partial_scheduler()
        scheduler.parts.mode_owners = ModeOwners()
        scheduler.parts.counters = LaunchCounters()
        scheduler.parts.flags = EngineFlags()
        scheduler.mode.disaggregation_mode = DisaggregationMode.NULL
        scheduler.parts.mode_owners.disagg_prefill = None
        scheduler.parts.mode_owners.disagg_decode = None
        scheduler.parts.mode_owners.dllm = None
        scheduler.parts.mode_owners.hisparse_coordinator = None
        scheduler.parts.flags.gracefully_exit = False
        scheduler.flat_slot = ScheduleSlot()
        scheduler.ingest_requests = MagicMock(side_effect=[[], [], [], StopIteration])
        scheduler.parts.flags._engine_paused = False
        scheduler._sched_idled = False
        scheduler.parts.pending = PendingSteps()
        scheduler.parts.request_retirer = make_request_retirer(
            pending=scheduler.parts.pending
        )
        scheduler.parts.recv_skipper = None
        scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)
        scheduler.parts.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.mode.step_launcher = SimpleNamespace(
            run_batch=MagicMock(return_value=result)
        )
        scheduler._apply_war_barrier = MagicMock()
        scheduler.parts.step_counters = MagicMock()
        scheduler.parts.is_generation = False

        pool = SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor([[10, 11]]),
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx,
        )
        steps = []

        def schedule_step(slot):
            # Two decode steps of req: seq lens 4 (a boundary) and 5.
            if len(steps) == 2:
                return None
            _, step = _make_batch()
            step.reqs = [req]
            step.req_to_token_pool = pool
            step.prepare_for_decode()
            steps.append(step)
            return step

        def alloc_for_decode(step, token_per_req):
            req.kv.kv_allocated_len += token_per_req
            return torch.tensor([3], dtype=torch.int64)

        scheduler.schedule_step = schedule_step
        state_at_commit = []

        def process_batch_result(result_batch, batch_result):
            state_at_commit.append(
                (req.kv.mamba_next_track_idx, req.kv.mamba_last_track_seqlen)
            )
            commit_step_counters(result_batch)
            processor.process_batch_result_decode(result_batch, batch_result)

        scheduler.process_batch_result = process_batch_result
        scheduler.parts.batch_result_processor = processor
        scheduler.mode.stages = init_mode_stages(
            scheduler.parts, scheduler.loop_schedules(), lora_ids_in_use=set
        )

        with (
            # The mamba predicates and the track interval read the published
            # bags, so publish the configuration under test (non-lazy extra
            # buffer, interval 4); observability and disagg reads are served
            # by the same publish at their defaults.
            get_context().override_server_args(
                mamba_radix_cache_strategy="extra_buffer",
                mamba_track_interval=TRACK_INTERVAL,
                _mamba_cache_chunk_size=TRACK_INTERVAL,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                side_effect=alloc_for_decode,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs",
                side_effect=_set_mamba_track_indices_on_cpu,
            ),
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
        ):
            with self.assertRaises(StopIteration):
                scheduler.event_loop_overlap()

        self.assertEqual([s.mamba_track_mask_cpu for s in steps], [[True], [False]])
        # The second step was built before the boundary committed.
        self.assertEqual([s.mamba_track_buffer_indices for s in steps], [[0], [0]])
        self.assertEqual(state_at_commit, [(0, None), (1, TRACK_INTERVAL)])
        self.assertEqual(req.kv.mamba_last_track_idx, 0)
        self.assertEqual(req.kv.mamba_next_track_idx, 1)
        self.assertEqual(req.kv.mamba_last_track_seqlen, TRACK_INTERVAL)
        self.assertEqual(req.decode_batch_idx, 2)
        self.assertEqual(req.kv.kv_committed_len, req.kv.kv_allocated_len)


class TestMambaSpecBoundaryCommit(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_spec_step_carries_the_slot_its_verify_tracks(self):
        """A spec decode step that crosses a track boundary: the step's plan
        names the slot its verify tracks into, and the host commit makes that
        slot the latest checkpoint, also for a request finishing in that step."""
        req, step = _make_batch()
        req.sampling_params.max_new_tokens = 3
        req.origin_input_ids = array("q", [1, 2, 3])
        req.kv.req_pool_idx = 0
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 3
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([10, 11])
        req.kv.mamba_next_track_idx = 0
        req.kv.mamba_last_track_idx = 1
        step.spec_algorithm = SpeculativeAlgorithm.EAGLE
        step.req_to_token_pool = SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor([[10, 11]]),
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx,
        )
        # The verify accepts one draft plus the bonus: the request goes from
        # length 4 to 6, crossing the boundary at 4, and reaches max_new_tokens.
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.tensor([5, 6, 0, 0]),
            accept_lens=torch.tensor([2]),
            speculative_num_draft_tokens=4,
        )
        processor = _make_processor()

        with (
            get_context().override_server_args(
                mamba_radix_cache_strategy="extra_buffer",
                mamba_track_interval=TRACK_INTERVAL,
                _mamba_cache_chunk_size=TRACK_INTERVAL,
            ),
            patch("sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode"),
            patch(
                "sglang.srt.speculative.spec_utils.set_mamba_track_indices_from_reqs",
                side_effect=_set_mamba_track_indices_on_cpu,
            ),
            patch(
                "sglang.srt.managers.scheduler_components."
                "request_retirer.release_kv_cache"
            ),
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
        ):
            # StepBuilder.new_decode_step, then the verify on its forward view.
            step.prepare_for_decode()
            spec_prepare_for_decode(step)
            forward_batch = worker_forward_batch(step)
            prepare_mamba_track_for_verify(forward_batch)
            processor.process_batch_result_decode(step, result)

        self.assertEqual(step.mamba_track_buffer_indices, [0])
        self.assertEqual(forward_batch.mamba_track_indices.tolist(), [10])
        self.assertTrue(req.finished())
        self.assertEqual(req.kv.mamba_last_track_idx, 0)
        self.assertEqual(req.kv.mamba_next_track_idx, 1)
        self.assertEqual(req.kv.mamba_last_track_seqlen, TRACK_INTERVAL)


# The old checkpoint at the first boundary, held in slot 10 at position 0.
_LAZY_CHECKPOINT_SLOT = 10


def _lazy_overrides():
    return get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy",
        mamba_track_interval=TRACK_INTERVAL,
        _mamba_cache_chunk_size=TRACK_INTERVAL,
    )


def _make_lazy_step(req: Req, *, spec: bool, other_slot: int = -1) -> ScheduleBatch:
    """A step of req, which holds one checkpoint slot (and other_slot at
    position 1, -1 for none), over a mamba pool that has no free slot."""
    _, step = _make_batch()
    step.reqs = [req]
    req.kv.req_pool_idx = 0
    req.kv.mamba_ping_pong_track_buffer = torch.tensor(
        [_LAZY_CHECKPOINT_SLOT, other_slot]
    )
    req.kv.mamba_next_track_idx = 0
    req.kv.mamba_last_track_idx = 0
    req.kv.mamba_last_track_seqlen = TRACK_INTERVAL
    if spec:
        step.spec_algorithm = SpeculativeAlgorithm.EAGLE
    step.req_to_token_pool = SimpleNamespace(
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor(
            [[_LAZY_CHECKPOINT_SLOT, other_slot]]
        ),
        mamba_allocator=SimpleNamespace(alloc=lambda n: None, free=MagicMock()),
        get_mamba_ping_pong_other_idx=lambda idx: 1 - idx,
    )
    return step


def _make_lazy_req(seq_len: int) -> Req:
    """A decoding request of seq_len tokens, one of them an output."""
    req, _ = _make_batch()
    req.origin_input_ids = array("q", range(seq_len - 1))
    req.kv.kv_allocated_len = req.kv.kv_committed_len = seq_len
    return req


def _spec_result(num_accepted: int) -> GenerationBatchResult:
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=torch.tensor([5] * num_accepted),
        accept_lens=torch.tensor([num_accepted]),
        speculative_num_draft_tokens=2,
    )


def _plan_spec_step(step: ScheduleBatch) -> None:
    # Two draft tokens: a verify from length 7 can cross the boundary at 8.
    step.mamba_lazy_spec_prepare(TRACK_INTERVAL, max_draft_tokens=2)


class TestMambaLazyInPlaceTrack(unittest.TestCase):
    """A lazy request that cannot get a second slot at a boundary tracks in
    place into the slot of its checkpoint. The build only plans it; the host
    commit of a step that crossed names the new checkpoint."""

    def setUp(self):
        enter_scope(self, published_topology())

    def test_spec_step_plans_the_checkpoint_slot_and_its_commit_names_a_crossing(
        self,
    ):
        for num_accepted, crossed in ((2, True), (1, False)):
            with self.subTest(crossed=crossed), _lazy_overrides():
                req = _make_lazy_req(7)
                step = _make_lazy_step(req, spec=True)
                _plan_spec_step(step)
                self.assertEqual(step.mamba_lazy_spec_track_positions_cpu, [0])
                self.assertEqual(step.mamba_track_write_position(0), 0)
                self.assertEqual(req.kv.mamba_last_track_seqlen, TRACK_INTERVAL)

                # The verify accepts num_accepted tokens from length 7.
                req.output_ids.extend([5] * num_accepted)
                _make_processor()._mamba_prefix_cache_update(
                    req, step, _spec_result(num_accepted), 0
                )
                self.assertEqual(
                    req.kv.mamba_last_track_seqlen,
                    2 * TRACK_INTERVAL if crossed else TRACK_INTERVAL,
                )
                self.assertEqual(req.kv.mamba_last_track_idx, 0)
                self.assertEqual(req.kv.mamba_next_track_idx, 0)
                self.assertEqual(
                    req.kv.mamba_ping_pong_track_buffer.tolist(),
                    [_LAZY_CHECKPOINT_SLOT, -1],
                )
                step.req_to_token_pool.mamba_allocator.free.assert_not_called()

    def test_decode_step_at_a_boundary_tracks_in_place(self):
        req = _make_lazy_req(7)
        step = _make_lazy_step(req, spec=False)

        def alloc_for_decode(step, token_per_req):
            req.kv.kv_allocated_len += token_per_req
            return torch.tensor([7], dtype=torch.int64)

        with (
            _lazy_overrides(),
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                side_effect=alloc_for_decode,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs",
                side_effect=_set_mamba_track_indices_on_cpu,
            ),
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
        ):
            step.prepare_for_decode()
            self.assertEqual(step.mamba_track_mask_cpu, [True])
            self.assertEqual(step.mamba_track_indices.tolist(), [_LAZY_CHECKPOINT_SLOT])
            self.assertEqual(req.kv.mamba_last_track_seqlen, TRACK_INTERVAL)
            commit_step_counters(step)
            _make_processor()._mamba_prefix_cache_update(req, step, _make_result(), 0)

        self.assertEqual(req.kv.mamba_last_track_seqlen, 2 * TRACK_INTERVAL)
        self.assertEqual(req.kv.mamba_last_track_idx, 0)
        self.assertEqual(req.kv.mamba_next_track_idx, 0)


class TestMambaLazyFinishDonation(unittest.TestCase):
    """A finishing lazy request donates its checkpoint slot only when no
    forward writes it in place: not this step's crossing, and not a pending
    step's plan."""

    def setUp(self):
        enter_scope(self, published_topology())

    def _checkpoint_at_finish(self, step, result, pending_step=None) -> bool:
        req = step.reqs[0]
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        processor = _make_processor()
        if pending_step is not None:
            processor.pending.push(
                PendingStep(
                    step=pending_step,
                    result=None,
                    launch_ts=0.0,
                    after_idle_gap=False,
                )
            )
        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            processor._handle_finish_state_updated_req(
                req, step, result, 0, LogitsProcessorOutput(next_token_logits=None)
            )
        release.assert_called_once()
        return release.call_args.kwargs["checkpoint"]

    def test_pending_step_that_writes_the_checkpoint_slot_blocks_the_donation(self):
        # (spec, the pending step's second slot, whether it may track, donated)
        cases = (
            (False, -1, True, False),
            (False, 11, True, True),
            (False, -1, False, True),
            (True, -1, True, False),
            (True, 11, True, True),
            (True, -1, False, True),
        )
        for spec, other_slot, tracks, donated in cases:
            with (
                self.subTest(spec=spec, other_slot=other_slot, tracks=tracks),
                _lazy_overrides(),
            ):
                # The committing step crosses nothing and finishes the request
                # at length 7.
                req = _make_lazy_req(6)
                step = _make_lazy_step(req, spec=spec, other_slot=other_slot)
                step.mamba_track_mask_cpu = [False]
                pending_step = _make_lazy_step(req, spec=spec, other_slot=other_slot)
                if spec:
                    # Planned from length 7: two draft tokens reach the
                    # boundary at 8, one does not.
                    req.kv.kv_committed_len = 7
                    pending_step.mamba_lazy_spec_prepare(
                        TRACK_INTERVAL, max_draft_tokens=2 if tracks else 0
                    )
                    req.kv.kv_committed_len = 6
                    step.mamba_lazy_spec_track_positions_cpu = [0]
                    result = _spec_result(1)
                else:
                    pending_step.mamba_track_mask_cpu = [tracks]
                    pending_step.mamba_track_buffer_indices = [
                        0 if other_slot == -1 else 1
                    ]
                    step.mamba_track_buffer_indices = [0]
                    result = _make_result()
                req.output_ids.append(5)
                self.assertEqual(
                    self._checkpoint_at_finish(step, result, pending_step), donated
                )
                self.assertTrue(self._checkpoint_at_finish(step, result))

    def test_finishing_spec_step_that_crossed_in_place_skips_the_donation(self):
        for num_accepted, donated in ((2, False), (1, True)):
            with self.subTest(num_accepted=num_accepted), _lazy_overrides():
                req = _make_lazy_req(7)
                step = _make_lazy_step(req, spec=True)
                _plan_spec_step(step)
                req.output_ids.extend([5] * num_accepted)
                self.assertEqual(
                    self._checkpoint_at_finish(step, _spec_result(num_accepted)),
                    donated,
                )
                self.assertEqual(req.kv.mamba_last_track_seqlen, TRACK_INTERVAL)


class TestMambaLazyPrefillCommitPublish(unittest.TestCase):
    """Under overlap a prefill commits after the request's first decode
    launched. When that decode tracks in place into the slot of the prefill's
    checkpoint, the commit publishes no checkpoint of that slot."""

    def setUp(self):
        enter_scope(self, published_topology())

    def _commit_prefill(self, max_new_tokens: int, pending_tracks: bool):
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
        sampling_params.normalize(None)
        req = Req(
            rid="req",
            origin_input_text="",
            origin_input_ids=array("q", range(2 * TRACK_INTERVAL - 1)),
            sampling_params=sampling_params,
            vocab_size=128,
        )
        req.prefix_len = 0
        req.extend_end = 2 * TRACK_INTERVAL - 1
        # The first decode, at length 2 * TRACK_INTERVAL, crosses a boundary
        # and finds no free slot: it plans the checkpoint slot.
        decode_step = _make_lazy_step(req, spec=False)
        decode_step.mamba_track_mask_cpu = [pending_tracks]
        decode_step.mamba_track_buffer_indices = [0]
        processor = _make_processor(tree_cache=MagicMock())
        processor.pending.push(
            PendingStep(
                step=decode_step,
                result=None,
                launch_ts=0.0,
                after_idle_gap=False,
            )
        )
        prefill_step = SimpleNamespace(
            reqs=[req],
            chunked_req=None,
            num_decode_rows=0,
            return_logprob=False,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            extend_lens=[req.extend_end],
            extend_logprob_start_lens=None,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=None,
            next_token_ids=torch.tensor([5]),
            grammar_advanced=False,
            can_run_cuda_graph=False,
        )
        with (
            _lazy_overrides(),
            patch(
                "sglang.srt.managers.scheduler_components."
                "request_retirer.release_kv_cache"
            ) as release,
        ):
            processor.process_batch_result_prefill(prefill_step, result)
        return req, processor.kv.tree_cache, release

    def test_running_request_checkpoint(self):
        for pending_tracks in (True, False):
            with self.subTest(pending_tracks=pending_tracks):
                req, tree_cache, release = self._commit_prefill(32, pending_tracks)
                self.assertFalse(req.finished())
                release.assert_not_called()
                self.assertEqual(tree_cache.checkpoint.called, not pending_tracks)
                self.assertEqual(req.prefix_len, req.extend_end)

    def test_request_finishing_at_prefill(self):
        for pending_tracks in (True, False):
            with self.subTest(pending_tracks=pending_tracks):
                req, _, release = self._commit_prefill(1, pending_tracks)
                self.assertTrue(req.finished())
                release.assert_called_once()
                self.assertEqual(
                    release.call_args.kwargs["checkpoint"], not pending_tracks
                )


class TestMambaExtendAdmission(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_next_chunk_built_before_commit_tracks_into_the_other_slot(self):
        """Under overlap the next chunk is built, and the chunked request
        stashed, before a chunk commits; admission, not the commit, hands the
        next chunk the other ping-pong slot."""
        req, batch = _make_batch()
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([10, 11])
        req.kv.mamba_next_track_idx = 0
        req.kv.mamba_last_track_idx = 1
        batch.forward_mode = ForwardMode.EXTEND
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=TRACK_INTERVAL)
        )
        batch.req_to_token_pool = SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx
        )
        positions = []
        with get_context().override_server_args(
            mamba_radix_cache_strategy="extra_buffer",
            mamba_track_interval=TRACK_INTERVAL,
            _mamba_cache_chunk_size=TRACK_INTERVAL,
        ):
            for prefix_len in (0, 2 * TRACK_INTERVAL):
                req.prefix_len = prefix_len
                req.extend_end = prefix_len + 2 * TRACK_INTERVAL
                entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
                batch.mamba_prefill_track_mask_cpu = [entry.track_mask]
                batch.mamba_track_buffer_indices = [entry.track_position]
                batch.mamba_prefill_checkpoint_seqlens_cpu = [entry.checkpoint_seqlen]
                batch.admit_mamba_track()
                positions.append(entry.track_position)
        self.assertEqual(positions, [0, 1])
        self.assertEqual(req.kv.mamba_last_track_idx, 1)
        self.assertEqual(req.kv.mamba_next_track_idx, 0)
        self.assertEqual(req.kv.mamba_last_track_seqlen, 4 * TRACK_INTERVAL)


class TestMambaPrevTrackSeqlen(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_flip_names_the_checkpoint_in_the_other_slot(self):
        req, batch = _make_batch()
        processor = _make_processor()
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([10, 11])
        req.kv.mamba_ping_pong_empty = [False, False]
        req.kv.mamba_next_track_idx = 0
        req.kv.mamba_last_track_idx = 1
        batch.req_to_token_pool = SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx
        )
        with (
            get_context().override_server_args(
                mamba_radix_cache_strategy="extra_buffer",
                mamba_track_interval=TRACK_INTERVAL,
                _mamba_cache_chunk_size=TRACK_INTERVAL,
            ),
            patch.object(
                SchedulerBatchResultProcessor,
                "_mamba_check_track_boundary",
                side_effect=[(True, TRACK_INTERVAL), (True, 2 * TRACK_INTERVAL)],
            ),
        ):
            batch.mamba_track_buffer_indices = [0]
            processor._mamba_prefix_cache_update(req, batch, _make_result(), 0)
            self.assertIsNone(req.kv.mamba_prev_track_seqlen)
            batch.mamba_track_buffer_indices = [1]
            processor._mamba_prefix_cache_update(req, batch, _make_result(), 0)
        self.assertEqual(req.kv.mamba_last_track_seqlen, 2 * TRACK_INTERVAL)
        self.assertEqual(req.kv.mamba_prev_track_seqlen, TRACK_INTERVAL)


class TestMambaFreedTrackIndicesCPU(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_freed_rows_are_inert_and_live_rows_preserve_mapping(self):
        for positions, expected in (
            (None, [-1, 11]),
            ([1, 1], [-1, 12]),
            # A None position (no lazy destination) tracks nowhere.
            ([1, None], [-1, -1]),
        ):
            with self.subTest(positions=positions):
                mapping = torch.tensor([[7, 8], [11, 12]], dtype=torch.int64)
                batch = SimpleNamespace(
                    reqs=[
                        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=p))
                        for p in (None, 0)
                    ],
                    req_pool_indices=torch.arange(2),
                    req_to_token_pool=SimpleNamespace(
                        req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
                    ),
                )
                set_mamba_track_indices_from_reqs(batch, positions)
                self.assertEqual(batch.mamba_track_indices.tolist(), expected)


if __name__ == "__main__":
    unittest.main()

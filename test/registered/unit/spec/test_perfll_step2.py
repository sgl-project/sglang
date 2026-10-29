"""Unit tests for SGLANG_PERFLL_STEP2's counting mode and its three-window decode reserve, and for
step 5a's staged lengths (SGLANG_PERFLL_STEP5A_SEQ_LENS): which pass takes them, the debug check
against the gather they replace, and the forward's wait for the schedule stream only after work
launched there."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import overlap_utils
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.perfll_step2 import (
    DraftExtendInputsCheck,
    NextDraft,
    SeqLensCheck,
    Step2Check,
    Step2Counter,
    make_draft_extend_inputs,
    takes_staging,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(slot: int, allocated: int, committed: int) -> Req:
    sp = SamplingParams(max_new_tokens=16, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=f"r{slot}",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
    )
    req.kv.req_pool_idx = slot
    req.kv.kv_committed_len = committed
    req.kv.kv_allocated_len = allocated
    return req


def _batch(slots, allocated, committed=100):
    """A decode batch of real Req objects (the scheduler's request type)."""
    reqs = [_req(slot, alloc, committed) for slot, alloc in zip(slots, allocated)]
    return SimpleNamespace(reqs=reqs)


class TestPerfllStep2(CustomTestCase):
    def _pass(self, counter, batch):
        counter.count_draft(batch)
        counter.note_extend(batch)

    def test_a_decode_pass_is_staged_only_after_the_same_unpadded_allocated_batch(self):
        """On the scheduler's Req (slot and lengths under req.kv); reading a field Req does not
        have crashed every rank of the first serving run."""
        counter = Step2Counter(width=7, graph_sizes=[1, 2, 4])
        whole = 100 + 21  # committed + three windows
        self._pass(counter, _batch([5, 9], [whole, whole]))
        self._pass(counter, _batch([5, 9], [whole, whole]))
        self._pass(counter, _batch([9, 5], [whole, whole]))
        self._pass(counter, _batch([9, 5, 3], [whole] * 3))
        self._pass(counter, _batch([9, 5, 3], [whole, whole, whole - 1]))
        self._pass(counter, _batch([9, 5], [whole, whole - 1]))
        self._pass(counter, _batch([9, 5], [whole, whole]))
        counter.forget()
        self._pass(counter, _batch([9, 5], [whole, whole]))
        self.assertEqual(
            dict(counter.counts),
            {
                "no_previous_decode": 2,
                "staged": 1,
                "batch_changed": 3,
                "padded": 1,
                "window_not_allocated": 1,
            },
        )
        self.assertEqual(counter.decode_pass_ct, 8)

    def test_mode_two_hands_over_only_inputs_a_post_verify_graph_staged(self):
        counter = Step2Counter(width=7, graph_sizes=[2], needs_next_draft=True)
        whole = 100 + 21
        staged = object()
        counter.note_extend(_batch([5, 9], [whole, whole]), next_draft=staged)
        self.assertIs(counter.count_draft(_batch([5, 9], [whole, whole])), staged)
        counter.note_extend(_batch([5, 9], [whole, whole]), next_draft=None)
        self.assertIsNone(counter.count_draft(_batch([5, 9], [whole, whole])))
        counter.note_extend(_batch([5, 9], [whole, whole]), next_draft=staged)
        self.assertIsNone(counter.count_draft(_batch([9, 5], [whole, whole])))
        self.assertEqual(
            dict(counter.counts),
            {"staged": 1, "not_staged_by_post_verify_graph": 1, "batch_changed": 1},
        )

    def test_the_check_compares_the_draft_slots_with_the_windows_first_columns(self):
        def i64(*v):
            return torch.tensor(v, dtype=torch.int64)

        staged = NextDraft(
            num_requests=2,
            seq_lens=i64(10, 20),
            positions=i64(10, 20),
            mrope=None,
            draft_out_cache_loc=i64(100, 101, 200, 201),
            window=i64(100, 101, 102, 200, 201, 202),
        )
        check = Step2Check(device="cpu")
        seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        window = i64(100, 101, 102, 200, 201, 202)
        check.compare(
            staged,
            seq_lens=seq_lens,
            positions=i64(10, 20),
            mrope=None,
            window=window,
            num_steps=2,
        )
        self.assertEqual(check.mismatch_by_bs[2].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(check.first.tolist()[0], -1)
        window[5] = 7  # past the draft steps' columns: only the window differs
        check.compare(
            staged,
            seq_lens=seq_lens,
            positions=i64(10, 20),
            mrope=None,
            window=window,
            num_steps=2,
        )
        window[3] = 7  # a draft step's slot: both differ
        check.compare(
            staged,
            seq_lens=seq_lens,
            positions=i64(11, 20),
            mrope=None,
            window=window,
            num_steps=2,
        )
        self.assertEqual(check.mismatch_by_bs[2].tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(check.check_ct_by_bs[2], 3)
        # the first difference: the window's last slot in the second comparison
        self.assertEqual(check.first.tolist(), [4, 5, 2, 202, 7])
        self.assertIn(
            "first difference window[5] at B2: staged 202, eager 7", check.summary()
        )

    def test_a_draft_declines_a_staging_of_another_size_plan_or_mrope_choice(self):
        """Mode 3 takes, and mode 2 compares, only a staging the eager path would match: a batch
        with multimodal inputs computes M-RoPE rows with per-request offsets the staging lacks."""
        rows = torch.zeros(2, dtype=torch.int64)
        staged = NextDraft(
            num_requests=2,
            seq_lens=rows,
            positions=rows,
            mrope=torch.zeros(3, 2, dtype=torch.int64),
            draft_out_cache_loc=rows,
            window=rows,
        )
        take = dict(num_requests=2, plans_window=True, with_mrope=True)
        self.assertTrue(takes_staging(staged, **take))
        for field, value in (
            ("num_requests", 3),
            ("plans_window", False),
            ("with_mrope", False),
        ):
            self.assertFalse(takes_staging(staged, **{**take, field: value}), field)

    def test_the_switch_reserves_three_windows(self):
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            window = get_alloc_len_per_decode()
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * window)
            with envs.SGLANG_PERFLL_STEP2.override(1):
                self.assertEqual(get_alloc_reserve_per_decode(), 3 * window)


def _spec_context():
    """Server args of a 7-wide chain draft at page size 1, as in the switch test above."""
    return get_context().override_server_args(
        speculative_algorithm="EAGLE",
        speculative_num_steps=6,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=7,
        page_size=1,
    )


def _decode_batch(reqs, *, penalized=False, sched_idle=True):
    """What eagle_prepare_for_decode reads off a decode batch; the flag starts as an earlier
    pass may have left it."""
    return SimpleNamespace(
        reqs=reqs,
        maybe_evict_swa=lambda: None,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=penalized)
        ),
        cumulate_penalty_output_tokens=lambda: None,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        tree_cache=SimpleNamespace(supports_swa=lambda: False),
        req_to_token_pool=SimpleNamespace(),
        req_pool_indices=torch.tensor([r.kv.req_pool_idx for r in reqs]),
        seq_lens_cpu=torch.tensor([r.kv.kv_committed_len for r in reqs]),
        device="cpu",
        perfll_sched_idle=sched_idle,
    )


def _scheduler(seq_lens_check=None):
    """The fields Scheduler._join_schedule_stream reads."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.forward_stream = "forward"
    scheduler.schedule_stream = "schedule"
    scheduler.future_map = SimpleNamespace(seq_lens_check=seq_lens_check)
    return scheduler


class TestPerfllStep5aSeqLens(CustomTestCase):
    def test_a_staged_pass_takes_the_lengths_its_previous_pass_published(self):
        """Only a pass count_draft calls staged takes them, and taking them consumes nothing:
        the same pass's draft still takes its staged inputs."""
        counter = Step2Counter(width=7, graph_sizes=[2], needs_next_draft=True)
        whole = 100 + 21
        staged, published = object(), torch.tensor([104, 105])
        batch = _batch([5, 9], [whole, whole])
        counter.note_extend(batch, next_draft=staged, new_seq_lens=published)
        self.assertIs(counter.take_seq_lens(batch, sched_idle=True), published)
        self.assertIs(counter.take_seq_lens(batch, sched_idle=False), published)
        self.assertIs(counter.count_draft(batch), staged)
        self.assertEqual(
            (counter.seq_lens_taken_ct, counter.seq_lens_sched_work_ct), (2, 1)
        )
        for name, noted_batch, noted_draft, now in (
            ("batch_changed", batch, staged, _batch([9, 5], [whole, whole])),
            ("not_staged_by_post_verify_graph", batch, None, batch),
            ("window_not_allocated", _batch([5, 9], [whole, whole - 1]), staged, batch),
        ):
            with self.subTest(name):
                counter.note_extend(
                    noted_batch, next_draft=noted_draft, new_seq_lens=published
                )
                self.assertIsNone(counter.take_seq_lens(now, sched_idle=True))
                counter.forget()
        counter.note_extend(batch, next_draft=staged, new_seq_lens=None)
        self.assertIsNone(counter.take_seq_lens(batch, sched_idle=True))
        self.assertIsNone(
            counter.take_seq_lens(_batch([5, 9], [whole, whole]), sched_idle=True)
        )

    def test_the_worker_hands_over_lengths_only_when_its_draft_takes_the_staging(self):
        """SGLANG_PERFLL_STEP5A_SEQ_LENS rides on STEP2=3: below it the worker returns None and
        the scheduler gathers as before; so does a prefill that reuses the last decode pass's
        request slots, which matches that pass's note."""
        whole = 100 + 21
        batch = _batch([5, 9], [whole, whole])
        batch.perfll_sched_idle = True
        published = torch.tensor([104, 105])
        for uses_staged, mode in (
            (False, ForwardMode.DECODE),
            (True, ForwardMode.DECODE),
            (True, ForwardMode.EXTEND),
        ):
            worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
            worker._step2 = Step2Counter(
                width=7, graph_sizes=[2], needs_next_draft=True
            )
            worker._step2_uses_staged = uses_staged
            worker._step2.note_extend(
                batch, next_draft=object(), new_seq_lens=published
            )
            batch.forward_mode = mode
            with self.subTest(uses_staged=uses_staged, mode=mode.name):
                got = worker.take_staged_seq_lens(batch)
                want = uses_staged and mode == ForwardMode.DECODE
                self.assertIs(got, published if want else None)

    def test_the_scheduler_asks_for_staged_lengths_only_with_the_switch(self):
        """Item 2 turns off alone: without SGLANG_PERFLL_STEP5A_SEQ_LENS the scheduler never asks
        the worker, whatever STEP2 / 5A / items 3 and 4 are set to."""
        worker = SimpleNamespace(take_staged_seq_lens=Mock())
        for switch, draft_worker, want in (
            (False, worker, None),
            (True, worker, worker.take_staged_seq_lens),
            (True, None, None),  # a pipeline stage without the draft
        ):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.draft_worker = draft_worker
            with self.subTest(switch=switch, draft=draft_worker is not None):
                with envs.SGLANG_PERFLL_STEP5A_SEQ_LENS.override(switch):
                    scheduler.init_perfll_staged_seq_lens()
                self.assertIs(scheduler._perfll_staged_seq_lens, want)

    def test_the_forward_waits_for_the_schedule_stream_only_after_work_launched_there(
        self,
    ):
        """The WAR barrier sits on the schedule stream behind the previous forward; a staged
        pass's forward reaches it only through a join, which it makes only in a round whose decode
        preparation launched work there (a KV allocation, a penalty update, the debug gather);
        a pass with gathered lengths always joins."""
        whole = 100 + 21  # committed + three windows: nothing to allocate
        staged = torch.tensor([104])
        rounds = (
            # (allocated, penalized, decode alloc fast path, staged lengths, check, joins)
            (whole, False, True, staged, None, False),
            (whole - 1, False, True, staged, None, True),
            (whole, True, True, staged, None, True),
            (whole, False, False, staged, None, True),
            (whole, False, True, None, None, True),
            (whole, False, True, staged, SimpleNamespace(active=True), True),
            (whole, False, True, staged, SimpleNamespace(active=False), False),
        )
        with (
            _spec_context(),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_PERFLL_STEP2.override(3),
            patch.object(eagle_utils, "alloc_for_spec_decode") as alloc,
        ):
            for allocated, penalized, fast, lens, check, joins in rounds:
                batch = _decode_batch([_req(5, allocated, 100)], penalized=penalized)
                with patch.object(eagle_utils, "_DECODE_ALLOC", fast):
                    eagle_utils.eagle_prepare_for_decode(batch)
                scheduler = _scheduler(seq_lens_check=check)
                with (
                    self.subTest(
                        allocated=allocated,
                        penalized=penalized,
                        fast=fast,
                        staged=lens is not None,
                    ),
                    patch.object(scheduler_module.perfll_host, "wait_stream") as wait,
                ):
                    Scheduler._join_schedule_stream(scheduler, batch, lens)
                    calls = [c.args for c in wait.call_args_list]
                    self.assertEqual(calls, [("forward", "schedule")] if joins else [])
                    if lens is not None:
                        # consumed: a later pass cannot skip on this round's preparation
                        self.assertFalse(batch.perfll_sched_idle)
            # only the round short of its reserve allocated
            self.assertEqual(
                [c.kwargs["num_needed_tokens"] for c in alloc.call_args_list],
                [0, 1, 0, 0, 0, 0, 0],
            )

    def test_a_resolve_takes_the_staged_lengths_and_the_check_reads_the_real_buffers(
        self,
    ):
        """The staged lengths become batch.seq_lens as they are; with the debug check the gather
        still runs on the schedule stream, from new_seq_lens_buf, and is compared with them; at
        the next resolve the staged tensor must still hold what the gather read."""
        pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int32))

        def future_map(check):
            with (
                patch.object(overlap_utils, "_is_cuda", False),
                envs.SGLANG_DEBUG_PERFLL_STEP5A_SEQ_LENS_CHECK.override(check),
            ):
                fm = FutureMap(torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool)
            fm.new_seq_lens_buf.copy_(torch.arange(8) * 10)
            return fm

        def batch():
            rows = torch.tensor([3, 1])
            return SimpleNamespace(
                spec_info=SimpleNamespace(future_indices=rows),
                req_pool_indices_cpu=rows,
            )

        fm = future_map(check=False)
        self.assertIsNone(fm.seq_lens_check)
        gathered = batch()
        fm.resolve_seq_lens_cpu(gathered)
        self.assertEqual(gathered.seq_lens.tolist(), [30, 10])
        staged = torch.tensor([30, 10])
        taken = batch()
        with patch.object(
            torch, "index_select", side_effect=AssertionError("gathered")
        ):
            fm.resolve_seq_lens_cpu(taken, staged_seq_lens=staged)
        self.assertIs(taken.seq_lens, staged)
        self.assertEqual(taken.seq_lens_cpu.tolist(), [30, 10])

        fm = future_map(check=True)
        fm.resolve_seq_lens_cpu(batch(), staged_seq_lens=torch.tensor([30, 10]))
        stale = torch.tensor([30, 11])
        fm.resolve_seq_lens_cpu(batch(), staged_seq_lens=stale)
        stale[0] = 31  # rewritten before the next resolve
        fm.resolve_seq_lens_cpu(batch(), staged_seq_lens=torch.tensor([30, 10]))
        fm.resolve_seq_lens_cpu(batch())
        check = fm.seq_lens_check
        self.assertEqual(check.check_ct_by_bs[2], 3)
        self.assertEqual(int(check.mismatch_by_bs[2]), 1)
        self.assertEqual(check.first.tolist(), [1, 2, 11, 10])
        self.assertEqual((int(check.overwritten), check.retire_ct), (1, 3))
        self.assertIn("row 1 at B2: staged 11, gathered 10", check.summary())

    def test_an_int64_bonus_passes_the_deferred_relay_uncast(self):
        """SGLANG_PERFLL_STEP5A_BONUS_I64: the next draft takes the post-verify graph's bonus as
        it is; an int32 bonus is still cast."""
        pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int32))
        with (
            patch.object(overlap_utils, "_is_cuda", False),
            envs.SGLANG_Q35MF_SPEC_RELAY_BYPASS.override(True),
        ):
            fm = FutureMap(torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool)
        for dtype in (torch.int32, torch.int64):
            bonus = torch.tensor([5, 6], dtype=dtype)
            draft_input = SimpleNamespace(
                bonus_tokens=bonus,
                future_indices=torch.tensor([3, 1]),
                future_dsa_topk_indices_available=False,
                dsa_topk_indices=None,
            )
            payload = overlap_utils.RelayPayload(bonus_tokens=bonus)
            fm.defer_relay(draft_input.future_indices, payload, draft_input)
            fm._resolve_spec_extras(SimpleNamespace(spec_info=draft_input))
            with self.subTest(dtype=dtype):
                self.assertEqual(draft_input.bonus_tokens.dtype, torch.int64)
                self.assertEqual(draft_input.bonus_tokens.tolist(), [5, 6])
                self.assertEqual(
                    draft_input.bonus_tokens is bonus, dtype == torch.int64
                )

    def test_a_gated_check_runs_only_while_its_file_exists(self):
        """SGLANG_DEBUG_PERFLL_STEP5A_SEQ_LENS_CHECK_GATE: the check (its gather, comparison and
        the forward's join) follows the gate file, polled while the scheduler is idle; a timed
        workload with the file absent gathers nothing."""
        import os
        import tempfile

        pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int32))
        gate = os.path.join(tempfile.mkdtemp(), "check.on")
        with (
            patch.object(overlap_utils, "_is_cuda", False),
            envs.SGLANG_DEBUG_PERFLL_STEP5A_SEQ_LENS_CHECK_GATE.override(gate),
        ):
            fm = FutureMap(torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool)
        fm.new_seq_lens_buf.copy_(torch.arange(8) * 10)
        check = fm.seq_lens_check
        rows = torch.tensor([3, 1])

        def resolve():
            batch = SimpleNamespace(
                spec_info=SimpleNamespace(future_indices=rows),
                req_pool_indices_cpu=rows,
            )
            fm.resolve_seq_lens_cpu(batch, staged_seq_lens=torch.tensor([30, 10]))

        def poll():
            check._gate_poll = 0.0
            check.poll_gate()

        self.assertFalse(check.active)
        with patch.object(
            torch, "index_select", side_effect=AssertionError("gathered")
        ):
            resolve()
        open(gate, "w").close()
        poll()
        self.assertTrue(check.active)
        resolve()
        resolve()
        os.remove(gate)
        poll()
        self.assertFalse(check.active)
        resolve()
        self.assertEqual(check.check_ct_by_bs[2], 2)
        self.assertEqual((int(check.mismatch_by_bs[2]), check.retire_ct), (0, 1))

    def test_the_check_counts_a_shape_or_dtype_difference_without_comparing(self):
        check = SeqLensCheck(device="cpu")
        check.compare(staged=torch.tensor([1, 2]), gathered=torch.tensor([1]))
        check.compare(
            staged=torch.tensor([1], dtype=torch.int32), gathered=torch.tensor([1])
        )
        self.assertEqual((check.shape_mismatch_ct, check.compare_ct), (2, 0))


def _extend_runner_buffers(max_bs=4, width=7, hidden=8):
    rows = max_bs * width
    return SimpleNamespace(
        input_ids=torch.zeros(rows, dtype=torch.int64),
        seq_lens=torch.zeros(max_bs, dtype=torch.int64),
        positions=torch.zeros(rows, dtype=torch.int64),
        mrope_positions=torch.zeros((3, rows), dtype=torch.int64),
        out_cache_loc=torch.zeros(rows, dtype=torch.int64),
        req_pool_indices=torch.zeros(max_bs, dtype=torch.int64),
        extend_seq_lens=torch.zeros(max_bs, dtype=torch.int32),
        num_correct_drafts=torch.zeros(max_bs, dtype=torch.int32),
        num_accept_tokens=torch.zeros(max_bs, dtype=torch.int32),
        select_index=torch.zeros(max_bs, dtype=torch.int64),
        mamba_track_indices=None,
        hidden_states=torch.zeros(rows, hidden, dtype=torch.bfloat16),
    )


class TestPerfllRsD1(CustomTestCase):
    def test_the_views_are_the_runner_inputs_of_one_batch_size(self):
        rb = _extend_runner_buffers()
        published = torch.zeros(8, dtype=torch.int64)
        views = make_draft_extend_inputs(
            runner_buffers=rb,
            num_requests=2,
            width=7,
            mrope=True,
            published_seq_lens=published,
        )
        self.assertEqual(views.input_ids.data_ptr(), rb.input_ids.data_ptr())
        self.assertEqual(views.input_ids.shape, (14,))
        self.assertEqual(views.seq_lens.shape, (2,))
        self.assertEqual(views.mrope_positions.shape, (3, 14))
        self.assertEqual(views.mrope_positions.stride(0), 28)
        self.assertEqual(views.hidden_states.shape, (14, 8))
        self.assertIs(views.published_seq_lens, published)
        no_mrope = make_draft_extend_inputs(
            runner_buffers=rb,
            num_requests=2,
            width=7,
            mrope=False,
            published_seq_lens=published,
        )
        self.assertIsNone(no_mrope.mrope_positions)

    def test_the_scheduler_binds_the_future_map_only_with_the_switch(self):
        """SGLANG_PERFLL_RS_D1: the scheduler hands the worker its FutureMap's lengths buffer once."""
        for switch in (False, True):
            worker = SimpleNamespace(
                bind_perfll_publish=Mock(), take_staged_seq_lens=Mock()
            )
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.draft_worker = worker
            scheduler.future_map = SimpleNamespace(new_seq_lens_buf=torch.zeros(4))
            with self.subTest(switch=switch):
                with envs.SGLANG_PERFLL_RS_D1.override(switch):
                    scheduler.init_perfll_staged_seq_lens()
                if switch:
                    worker.bind_perfll_publish.assert_called_once_with(
                        new_seq_lens_buf=scheduler.future_map.new_seq_lens_buf
                    )
                else:
                    worker.bind_perfll_publish.assert_not_called()

    def test_a_written_publish_only_records_the_event(self):
        """FutureMap.publish(written=True): the post-verify graph stored the lengths; the call
        keeps the event that gates their copy to the host and stores nothing."""
        pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int32))
        with patch.object(overlap_utils, "_is_cuda", False):
            fm = FutureMap(torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool)
        fm.new_seq_lens_buf.fill_(-1)
        rows = torch.tensor([3, 1])
        recorded = []
        with patch.object(
            torch.get_device_module(torch.device("cpu")),
            "Event",
            return_value=SimpleNamespace(record=lambda: recorded.append(1)),
            create=True,
        ):
            fm.publish(rows, torch.tensor([30, 10]), written=True)
        self.assertTrue(bool((fm.new_seq_lens_buf == -1).all()))
        self.assertEqual(recorded, [1])
        self.assertTrue(fm._publish_fresh)

    def test_a_round_takes_the_written_inputs_except_in_rsoff_blocks(self):
        """The worker takes the views that ride after the layout, unless the same-service A/B
        label of the round is "rsoff" or the post-verify graph did not write them."""
        from sglang.srt.model_executor.runner_backend import ab_variants
        from sglang.srt.speculative import eagle_worker_v2

        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._rs_d1_published = torch.zeros(4, dtype=torch.int64)
        views = object()
        written = SimpleNamespace(prepared_draft_extend_inputs=(1, 2, 3, 4, views))
        unwritten = SimpleNamespace(prepared_draft_extend_inputs=(1, 2, 3, 4))
        self.assertIs(worker._rs_d1_written(written), views)
        self.assertIsNone(worker._rs_d1_written(unwritten))
        for label, want in (("rsoff", None), ("rson", views)):
            with (
                patch.object(ab_variants, "ACTIVE", True),
                patch.object(ab_variants, "current", return_value=label),
            ):
                self.assertIs(worker._rs_d1_written(written), want)
        worker._rs_d1_published = None
        self.assertIsNone(worker._rs_d1_written(written))
        self.assertIs(eagle_worker_v2.ab_variants, ab_variants)

    def test_the_check_counts_each_field_and_keeps_the_first_difference(self):
        import os
        import tempfile

        gate = os.path.join(tempfile.mkdtemp(), "check.on")
        check = DraftExtendInputsCheck(device="cpu", gate=gate)
        self.assertFalse(check.active)
        open(gate, "w").close()
        for _ in range(256):
            check.poll_gate()
        self.assertTrue(check.active)
        check.compare("input_ids", torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
        check.compare(
            "out_cache_loc",
            torch.tensor([5, 6]),
            torch.tensor([5, 9], dtype=torch.int32),
        )
        check.compare("hidden_states", torch.zeros(2, 4), torch.zeros(2, 4))
        check.compare("published", torch.tensor([1]), torch.tensor([1, 2]))
        check.counted()
        counts = dict(zip(check.FIELDS, check.mismatch.tolist()))
        self.assertEqual(
            (
                counts["input_ids"],
                counts["out_cache_loc"],
                counts["hidden_states"],
                counts["published"],
            ),
            (0, 1, 0, 1),
        )
        self.assertIn("first difference out_cache_loc row 1 at 2 rows", check.summary())
        os.remove(gate)
        for _ in range(256):
            check.poll_gate()
        self.assertFalse(check.active)


if __name__ == "__main__":
    unittest.main()

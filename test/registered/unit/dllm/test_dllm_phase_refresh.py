"""Scheduling phase refresh must not advance a diffusion block."""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace

from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Adder:
    def __init__(self):
        self.can_run_list = []
        self.preempt_list = []

    def add_dllm_staging_req(self, req):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE

    def add_one_req(self, req, **kwargs):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _Scheduler(SchedulerDllmMixin):
    """Keep phase selection real; omit GPU allocation and batch construction."""

    def __init__(self, config, waiting, incoming=()):
        self.enable_overlap = False
        self.result_queue = deque()
        self.dllm_config = config
        self.dllm_manager = DllmManager(config)
        self.dllm_manager.add_waiting_reqs(waiting)
        self.waiting_queue = list(incoming)
        self.enable_priority_preemption = False
        self.policy = SimpleNamespace(calc_priority=lambda reqs: None)
        self.tree_cache = None
        self.truncation_align_size = config.block_size

    def get_num_allocatable_reqs(self, running_bs):
        return 16

    def _create_dllm_prefill_adder(self, *args, **kwargs):
        return _Adder()

    def _create_dllm_batch(self, can_run_list, *args, **kwargs):
        return can_run_list

    def schedule(self):
        return self.get_new_batch_dllm(SimpleNamespace(reqs=[], batch_is_full=False))


def _config(separate=False, fdfo=True):
    return SimpleNamespace(
        algorithm="LowConfidence",
        block_size=32,
        mask_id=0,
        max_running_requests=16,
        first_done_first_out_mode=fdfo,
        requires_separate_context_encoding=separate,
    )


def _req(rid, config, prompt_len=128):
    return Req(
        rid=rid,
        origin_input_text="test",
        origin_input_ids=array("q", [1] * prompt_len),
        sampling_params=SamplingParams(max_new_tokens=128),
        dllm_config=config,
    )


class TestDllmPhaseRefresh(CustomTestCase):
    def test_cached_prompt_joins_existing_decode_without_opening_block(self):
        for fdfo in (False, True):
            for result_processed in (False, True):
                with self.subTest(fdfo=fdfo, result_processed=result_processed):
                    config = _config(fdfo=fdfo)
                    cached = _req("cached", config)
                    cached.init_next_round_input()
                    self.assertEqual(cached.dllm_phase, DllmReqPhase.STAGING_PREFILL)
                    # A prefix-cache match occurs after initial phase selection.
                    cached.prefix_len = 128
                    cached.dllm_block_offset = 128
                    cached.dllm_block_id = 1
                    if result_processed:
                        cached.dllm_incomplete_ids = array("q", [0] * 32)

                    peer = _req("decode", config)
                    peer.init_next_round_input()
                    peer.prefix_len = 128
                    peer.dllm_phase = DllmReqPhase.STAGING_DECODE
                    before = (
                        cached.full_untruncated_fill_ids[:],
                        cached.dllm_block_offset,
                        cached.dllm_block_id,
                        cached.dllm_block_done,
                        cached.prefix_len,
                        cached.kv.req_pool_idx,
                    )

                    selected = _Scheduler(config, [cached, peer]).schedule()

                    self.assertEqual(selected, [cached, peer])
                    self.assertEqual(cached.dllm_phase, DllmReqPhase.STAGING_DECODE)
                    self.assertEqual(
                        (
                            cached.full_untruncated_fill_ids,
                            cached.dllm_block_offset,
                            cached.dllm_block_id,
                            cached.dllm_block_done,
                            cached.prefix_len,
                            cached.kv.req_pool_idx,
                        ),
                        before,
                    )

    def test_new_separate_context_request_uses_incoming_initialization(self):
        config = _config(separate=True)
        fresh = _req("fresh", config)
        selected = _Scheduler(config, [], [fresh]).schedule()
        self.assertEqual(selected, [fresh])
        self.assertTrue(fresh.dllm_initialized)
        self.assertEqual(fresh.dllm_phase, DllmReqPhase.STAGING_PREFILL)
        self.assertEqual(list(fresh.full_untruncated_fill_ids), [1] * 128)

    def test_completed_block_joins_decode_before_next_block_is_initialized(self):
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                config = _config(fdfo=fdfo)
                req = _req("completed", config)
                req.init_next_round_input()
                req.output_ids.extend([1] * 32)
                req.full_untruncated_fill_ids = array("q", [1] * 160)
                req.prefix_len = 160
                req.dllm_block_offset = 128
                req.dllm_block_id = 7
                req.dllm_block_done = True
                req.dllm_phase = DllmReqPhase.STAGING_PREFILL
                peer = _req("decode", config)
                peer.init_next_round_input()
                peer.prefix_len = 128
                peer.dllm_phase = DllmReqPhase.STAGING_DECODE

                selected = _Scheduler(config, [req, peer]).schedule()

                self.assertEqual(selected, [req, peer])
                self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)
                self.assertEqual(list(req.full_untruncated_fill_ids), [1] * 160)
                self.assertEqual(list(req.output_ids), [1] * 32)
                self.assertEqual(req.prefix_len, 160)
                self.assertEqual((req.dllm_block_offset, req.dllm_block_id), (128, 7))
                self.assertTrue(req.dllm_block_done)
                self.assertIsNone(req.kv.req_pool_idx)

    def test_phase_at_prompt_boundary(self):
        for prompt_len, prefix_len, expected in (
            (128, 96, DllmReqPhase.STAGING_PREFILL),
            (150, 128, DllmReqPhase.STAGING_DECODE),
            (16, 0, DllmReqPhase.STAGING_DECODE),
            (128, 128, DllmReqPhase.STAGING_DECODE),
        ):
            with self.subTest(prompt_len=prompt_len, prefix_len=prefix_len):
                req = _req("boundary", _config(), prompt_len=prompt_len)
                req.init_next_round_input()
                req.prefix_len = prefix_len

                req.determine_dllm_phase()

                self.assertEqual(req.dllm_phase, expected)

    def test_real_prompt_block_keeps_prefill_priority(self):
        config = _config()
        prompt = _req("prompt", config)
        prompt.init_next_round_input()
        prompt.prefix_len = 32
        decode = _req("decode", config)
        decode.init_next_round_input()
        decode.prefix_len = 128
        decode.dllm_phase = DllmReqPhase.STAGING_DECODE
        self.assertEqual(_Scheduler(config, [prompt, decode]).schedule(), [prompt])


class TestDllmFinalForwardScheduling(CustomTestCase):
    def make_scheduler(self):
        config = _config()
        req = _req("finishing", config)
        req.init_next_round_input()
        req.prefix_len = 128
        req.dllm_block_id = 3
        req.dllm_incomplete_ids = array("q", [1] * 32)
        scheduler = _Scheduler(config, [req])
        scheduler.enable_overlap = True
        return scheduler, req

    def enqueue(self, scheduler, req):
        scheduler.result_queue.append((SimpleNamespace(reqs=[req]), None))

    def test_wait_for_submitted_final_forward_without_completing_block(self):
        scheduler, req = self.make_scheduler()
        self.enqueue(scheduler, req)

        self.assertIsNone(scheduler.schedule())
        self.assertEqual(scheduler.dllm_manager.waiting_queue, [req])
        self.assertEqual(scheduler.dllm_manager.staging_queue, [])
        self.assertFalse(req.dllm_block_done)
        self.assertEqual(req.dllm_block_id, 3)
        self.assertEqual(list(req.output_ids), [])

    def test_unsubmitted_final_forward_still_runs(self):
        scheduler, req = self.make_scheduler()
        self.assertEqual(scheduler.schedule(), [req])

    def test_other_requests_continue_while_final_forward_is_inflight(self):
        scheduler, req = self.make_scheduler()
        peer = _req("peer", scheduler.dllm_config)
        peer.init_next_round_input()
        peer.prefix_len = 128
        peer.dllm_incomplete_ids = array("q", [1] * 32)
        scheduler.dllm_manager.add_waiting_reqs(peer)
        self.enqueue(scheduler, req)

        self.assertEqual(scheduler.schedule(), [peer])
        self.assertEqual(scheduler.dllm_manager.waiting_queue, [req, peer])

    def test_old_request_with_same_rid_does_not_block_new_request(self):
        scheduler, req = self.make_scheduler()
        old_req = _req(req.rid, scheduler.dllm_config)
        self.enqueue(scheduler, old_req)
        self.assertEqual(scheduler.schedule(), [req])

    def test_masked_or_unprocessed_block_still_runs(self):
        for tokens in ([1] * 31 + [0], []):
            with self.subTest(tokens=tokens):
                scheduler, req = self.make_scheduler()
                req.dllm_incomplete_ids = array("q", tokens)
                self.enqueue(scheduler, req)
                self.assertEqual(scheduler.schedule(), [req])

    def test_skip_requires_overlap_and_low_confidence(self):
        for overlap, algorithm in (
            (False, "LowConfidence"),
            (True, "JointThreshold"),
            (True, "Gemma4Renoise"),
        ):
            with self.subTest(overlap=overlap, algorithm=algorithm):
                scheduler, req = self.make_scheduler()
                scheduler.enable_overlap = overlap
                scheduler.dllm_config.algorithm = algorithm
                self.enqueue(scheduler, req)
                self.assertEqual(scheduler.schedule(), [req])

    def test_filtered_last_batch_does_not_hide_inflight_forward(self):
        scheduler, req = self.make_scheduler()
        scheduler.last_batch = ScheduleBatch(
            reqs=[req], dllm_config=scheduler.dllm_config
        )
        snapshot = scheduler.last_batch.copy()
        scheduler.result_queue.append((snapshot, None))
        scheduler.last_batch.filter_batch(reqs_to_exclude={req})

        self.assertEqual(scheduler.last_batch.reqs, [])
        self.assertEqual(snapshot.reqs, [req])
        self.assertIsNone(scheduler.schedule())


if __name__ == "__main__":
    unittest.main()

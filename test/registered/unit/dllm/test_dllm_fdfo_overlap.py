"""CPU checks for the FDFO + LowConfidence overlap draft.

The algorithm result and the FutureMap block relay must agree whether or not
the scheduler overlaps the next batch with result processing.
"""

import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _low_confidence():
    return LowConfidence(
        SimpleNamespace(
            block_size=4,
            mask_id=0,
            first_done_first_out_mode=True,
            algorithm_config={"threshold": 0.5},
        )
    )


def _logits_for(input_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab), -1e4)
    # Masked positions (token 0) predict token 1 with probability ~1.
    logits[:, :, 1] = torch.where(input_ids == 0, 10.0, logits[:, :, 1])
    return logits.reshape(-1, vocab)


class _Runner:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def forward(self, forward_batch, pp_proxy_tensors=None):
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=self.logits),
            can_run_graph=False,
        )


def _req(fdfo=True):
    return Req(
        rid="block-test",
        origin_input_text="test",
        origin_input_ids=array("q", [2, 3]),
        sampling_params=SamplingParams(max_new_tokens=32),
        dllm_config=SimpleNamespace(
            block_size=4,
            mask_id=0,
            max_running_requests=2,
            first_done_first_out_mode=fdfo,
        ),
    )


def _get_empty_dllm_batch(manager):
    scheduler = SimpleNamespace(
        enable_priority_preemption=False,
        _should_skip_prefill=lambda **kwargs: False,
        policy=SimpleNamespace(calc_priority=lambda reqs: None),
        waiting_queue=[],
        _create_dllm_prefill_adder=lambda *args, **kwargs: SimpleNamespace(
            can_run_list=[]
        ),
        dllm_manager=manager,
        _fetch_waiting_reqs=lambda: None,
        _process_dllm_batches=lambda *args, **kwargs: None,
    )
    return SchedulerDllmMixin.get_new_batch_dllm(scheduler, SimpleNamespace(reqs=[]))


class TestDllmResultSnapshot(unittest.TestCase):
    def test_queue_snapshot_preserves_dllm_dispatch(self):
        config = SimpleNamespace(first_done_first_out_mode=True)
        batch = ScheduleBatch(reqs=[object()], dllm_config=config)
        snapshot = batch.copy()
        self.assertTrue(snapshot.is_dllm())
        self.assertIs(snapshot.dllm_config, config)
        batch.reqs.clear()
        self.assertEqual(snapshot.batch_size(), 1)


class TestFdfoLowConfidenceOverlap(unittest.TestCase):
    def test_run_fdfo_matches_step_and_stays_on_device(self):
        algo = _low_confidence()
        input_ids = torch.tensor([[0, 0, 3, 4], [1, 2, 3, 4]], dtype=torch.int64)
        logits = _logits_for(input_ids, vocab=6)

        direct = SimpleNamespace(
            batch_size=2,
            input_ids=input_ids.clone().reshape(-1),
        )
        algo.step(direct, logits, [None, None])

        batched = SimpleNamespace(
            batch_size=2,
            input_ids=input_ids.clone().reshape(-1),
        )
        result = algo._run_fdfo(_Runner(logits), batched, None)

        self.assertIsInstance(result.block_tokens, torch.Tensor)
        self.assertEqual(result.block_tokens.shape, (2, 4))
        self.assertIsInstance(result.block_done, torch.Tensor)
        self.assertEqual(result.block_done.dtype, torch.bool)
        self.assertEqual(
            result.block_tokens.tolist(), direct.input_ids.view(2, 4).tolist()
        )
        self.assertEqual(result.block_done.tolist(), [False, True])

    def test_result_snapshot_survives_next_step(self):
        algo = _low_confidence()
        inputs = torch.tensor([[2, 0, 0, 0]])
        batch = SimpleNamespace(batch_size=1, input_ids=inputs.flatten())
        result = algo.run(_Runner(_logits_for(inputs, 6)), batch)
        batch.input_ids.fill_(5)
        self.assertEqual(result.block_tokens.tolist(), [[2, 1, 1, 1]])
        self.assertEqual(result.block_done.tolist(), [False])


    def test_fdfo_commits_only_after_forward_on_final_tokens(self):
        algo = _low_confidence()
        ids = torch.tensor([[2, 0, 0, 0]])
        batch = SimpleNamespace(batch_size=1, input_ids=ids.flatten())
        runner = _Runner(_logits_for(ids, 6))
        result = algo.run(runner, batch)
        self.assertEqual(result.block_tokens.tolist(), [[2, 1, 1, 1]])
        self.assertEqual(result.block_done.tolist(), [False])
        result = algo.run(runner, batch, result.algo_states)
        self.assertEqual(result.block_tokens.tolist(), [[2, 1, 1, 1]])
        self.assertEqual(result.block_done.tolist(), [True])

    def test_step_falls_back_to_highest_confidence_position(self):
        algo = _low_confidence()
        input_ids = torch.tensor([[0, 0, 3, 4], [1, 2, 3, 4]], dtype=torch.int64)
        logits = torch.zeros((2, 4, 6))
        # Both masked positions stay under the 0.5 threshold. Position 1 is higher.
        logits[0, 0, 1] = 0.2
        logits[0, 1, 2] = 0.4
        batch = SimpleNamespace(batch_size=2, input_ids=input_ids.clone().reshape(-1))

        done = algo.step(batch, logits.reshape(-1, 6), [None, None])

        self.assertEqual(done.tolist(), [False, True])
        self.assertEqual(
            batch.input_ids.view(2, 4).tolist(), [[0, 2, 3, 4], [1, 2, 3, 4]]
        )

    def test_future_map_overwrites_only_rows_with_a_block(self):
        future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SimpleNamespace(),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((4, 8))),
            needs_cpu_seq_lens=False,
        )
        future_map.stash_dllm_block_tokens(
            torch.tensor([2]),
            torch.tensor([[8, 8, 8, 8]]),
        )
        batch = SimpleNamespace(
            is_dllm=lambda: True,
            dllm_config=SimpleNamespace(first_done_first_out_mode=True),
            req_pool_indices=torch.tensor([1, 2]),
            input_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64),
        )
        future_map.resolve_dllm_block_tokens(batch)
        self.assertEqual(
            batch.input_ids.tolist(),
            [0, 0, 0, 0, 8, 8, 8, 8],
        )

        # Even a populated relay must not overwrite sync inputs.
        batch.dllm_config.first_done_first_out_mode = False
        batch.input_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        future_map.resolve_dllm_block_tokens(batch)
        self.assertEqual(batch.input_ids.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        batch.dllm_config.first_done_first_out_mode = True

        future_map.dllm_block_tokens_buf[2] = -1
        batch.input_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        future_map.resolve_dllm_block_tokens(batch)
        self.assertEqual(batch.input_ids.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_block_identity_changes_at_batch_preparation(self):
        req = _req(True)
        req.init_next_round_input()
        req.set_extend_range(0, 4)
        self.assertEqual(req.dllm_block_id, 0)
        batch = ScheduleBatch(
            reqs=[req], device="cpu", dllm_config=req.dllm_config
        )

        class AllocationReached(Exception):
            pass

        def prepare():
            with patch(
                "sglang.srt.managers.schedule_batch.alloc_for_extend",
                side_effect=AllocationReached,
            ), self.assertRaises(AllocationReached):
                batch.prepare_for_extend()

        prepare()
        self.assertEqual(req.dllm_block_id, 1)
        self.assertFalse(req.dllm_block_done)
        prepare()
        self.assertEqual(req.dllm_block_id, 1)
        req.dllm_block_done = True
        req.init_next_round_input()
        self.assertEqual(req.dllm_block_id, 1)
        self.assertTrue(req.dllm_block_done)
        prepare()
        self.assertEqual(req.dllm_block_id, 2)
        self.assertFalse(req.dllm_block_done)

    def test_pending_result_keeps_block_input(self):
        req = _req(True)
        req.init_next_round_input()
        req.dllm_block_id = 1
        block_offset = req.dllm_block_offset
        fill = list(req.full_untruncated_fill_ids)
        manager = DllmManager(req.dllm_config)
        manager.staging_queue = [req]
        self.assertIsNone(_get_empty_dllm_batch(manager))
        self.assertEqual(req.dllm_block_id, 1)
        self.assertFalse(req.dllm_block_done)
        self.assertEqual(req.dllm_block_offset, block_offset)
        self.assertEqual(list(req.full_untruncated_fill_ids), fill)
        self.assertEqual(manager.staging_queue, [])

    def test_incomplete_round_keeps_block_id_and_geometry(self):
        req = _req()
        req.init_next_round_input()
        block_id = req.dllm_block_id
        fill = list(req.full_untruncated_fill_ids)
        req.dllm_incomplete_ids = array("q", [2, 3, 4, 0])
        manager = DllmManager(req.dllm_config)
        manager.staging_queue = [req]
        self.assertIsNone(_get_empty_dllm_batch(manager))
        self.assertEqual(req.dllm_block_id, block_id)
        self.assertEqual(list(req.full_untruncated_fill_ids), fill)
        req.init_next_round_input()
        self.assertEqual(req.dllm_block_id, block_id)
        self.assertFalse(req.dllm_block_done)

    def test_result_keeps_block_ids_captured_before_forward(self):
        req = _req(fdfo=True)
        req.init_next_round_input()
        first_id = req.dllm_block_id
        inputs = torch.tensor([[2, 0, 0, 0]])
        algo = _low_confidence()
        algo.fdfo = True
        runner = _Runner(_logits_for(inputs, 6))
        original_forward = runner.forward

        def forward(*args, **kwargs):
            req.dllm_block_id += 1
            return original_forward(*args, **kwargs)

        runner.forward = forward
        worker = SimpleNamespace(dllm_algorithm=algo, model_runner=runner)
        result = TpModelWorker._forward_batch_generation_dllm(
            worker,
            SimpleNamespace(batch_size=1, input_ids=inputs.flatten()),
            SimpleNamespace(reqs=[req]),
        )
        self.assertGreater(req.dllm_block_id, first_id)
        self.assertEqual(result.dllm_block_ids, (first_id,))
        self.assertIsInstance(result.dllm_block_ids, tuple)

    def test_forward_without_scheduler_batch_has_no_block_ids(self):
        inputs = torch.tensor([[2, 0, 0, 0]])
        worker = SimpleNamespace(
            dllm_algorithm=_low_confidence(),
            model_runner=_Runner(_logits_for(inputs, 6)),
        )
        result = TpModelWorker._forward_batch_generation_dllm(
            worker, SimpleNamespace(batch_size=1, input_ids=inputs.flatten())
        )
        self.assertIsNone(result.dllm_block_ids)


class TestDllmResultProcessing(unittest.TestCase):
    def _process(
        self,
        *,
        fdfo,
        prompt,
        end,
        fill,
        tokens,
        done=None,
        req=None,
        finished=False,
        future_map=None,
        submitted_block_id=None,
    ):
        if req is None:
            req = SimpleNamespace(
                origin_input_ids=list(prompt),
                full_untruncated_fill_ids=array("q", fill),
                extend_range=SimpleNamespace(end=end),
                output_ids=[],
                dllm_block_done=False,
                dllm_block_id=1,
                dllm_incomplete_ids=array("q"),
                dllm_algo_state=None,
                finished=lambda: finished and bool(req.accepted),
                kv=SimpleNamespace(req_pool_idx=1),
                time_stats=SimpleNamespace(set_completion_time=lambda: None),
            )
            req.accepted = []
            req.update_finish_state = lambda new_accepted_len: req.accepted.append(
                new_accepted_len
            )
        streamed = []
        scheduler = SimpleNamespace(
            future_map=future_map,
            forward_stream_ctx=nullcontext(),
            enable_overlap=True,
            tree_cache=None,
            dllm_config=SimpleNamespace(block_size=4, first_done_first_out_mode=fdfo),
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=lambda: None, free_group_end=lambda: None
            ),
            metrics_reporter=SimpleNamespace(
                num_generated_tokens=0, report_prefill_stats=lambda **kwargs: None
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda *args: streamed.append(True)
            ),
        )
        scheduler._clear_dllm_future = lambda req: Scheduler._clear_dllm_future(
            scheduler, req
        )
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = GenerationBatchResult(
            dllm_block_ids=(
                req.dllm_block_id if submitted_block_id is None else submitted_block_id,
            ),
            next_token_ids=torch.tensor([tokens]),
            dllm_block_done=torch.tensor([done]) if fdfo else None,
            dllm_algo_state=[{"round": 1}] if fdfo else None,
        )
        SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, result)
        return req, streamed, scheduler.metrics_reporter.num_generated_tokens

    def test_fdfo_crops_prompt_and_respect_truncated_extend_range(self):
        req, streamed, count = self._process(
            fdfo=True,
            prompt=[2, 3],
            end=4,
            fill=[2, 3, 0, 0, 0, 0],
            tokens=[2, 3, 4, 5],
            done=True,
        )
        self.assertEqual(req.output_ids, [4, 5])
        self.assertEqual(
            list(req.full_untruncated_fill_ids), [2, 3, 4, 5, 0, 0]
        )
        self.assertEqual(req.accepted, [2])
        self.assertEqual(count, 2)
        self.assertEqual(streamed, [True])

    def test_finished_request_clears_future_before_slot_reuse(self):
        future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SimpleNamespace(),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((4, 8))),
            needs_cpu_seq_lens=False,
        )
        future_map.stash_dllm_block_tokens(
            torch.tensor([1]), torch.tensor([[2, 3, 4, 5]])
        )

        def release(req, tree_cache):
            req.kv.req_pool_idx = None

        with patch(
            "sglang.srt.dllm.mixin.scheduler.release_kv_cache", side_effect=release
        ):
            self._process(
                fdfo=True,
                prompt=[2, 3],
                end=4,
                fill=[2, 3, 0, 0],
                tokens=[2, 3, 4, 5],
                done=True,
                finished=True,
                future_map=future_map,
            )
        batch = SimpleNamespace(
            is_dllm=lambda: True,
            dllm_config=SimpleNamespace(first_done_first_out_mode=True),
            req_pool_indices=torch.tensor([1]),
            input_ids=torch.tensor([6, 0, 0, 0]),
        )
        future_map.resolve_dllm_block_tokens(batch)
        self.assertEqual(batch.input_ids.tolist(), [6, 0, 0, 0])

    def test_fdfo_prefill_commits_no_output(self):
        req, streamed, count = self._process(
            fdfo=True,
            done=True,
            prompt=[2, 3, 4, 5, 6],
            end=4,
            fill=[2, 3, 4, 5, 6, 0],
            tokens=[2, 3, 4, 5],
        )
        self.assertEqual(req.output_ids, [])
        self.assertEqual(req.accepted, [])
        self.assertEqual(streamed, [True])
        self.assertEqual(count, 0)

    def test_fdfo_retains_unfinished_then_commits_only_once(self):
        args = dict(fdfo=True, prompt=[2, 3], end=4, fill=[2, 3, 0, 0])
        req, _, count = self._process(**args, tokens=[2, 3, 4, 0], done=False)
        self.assertEqual(req.output_ids, [])
        self.assertEqual(list(req.dllm_incomplete_ids), [2, 3, 4, 0])
        self.assertEqual(req.dllm_algo_state, {"round": 1})
        self.assertEqual(count, 0)
        req, _, count = self._process(**args, tokens=[2, 3, 4, 5], done=True, req=req)
        self.assertEqual(req.output_ids, [4, 5])
        self.assertTrue(req.dllm_block_done)
        self.assertEqual(list(req.dllm_incomplete_ids), [])
        self.assertIsNone(req.dllm_algo_state)
        self.assertEqual(count, 2)
        req, _, count = self._process(**args, tokens=[2, 3, 4, 5], done=True, req=req)
        self.assertEqual(req.output_ids, [4, 5])
        self.assertTrue(req.dllm_block_done)
        self.assertEqual(count, 0)

    def test_old_block_cannot_overwrite_or_complete_new_block(self):
        for old_done in (False, True):
            with self.subTest(fdfo=True, old_done=old_done):
                args = dict(fdfo=True, prompt=[2, 3], end=4, fill=[2, 3, 0, 0])
                req, _, _ = self._process(**args, tokens=[2, 3, 4, 5], done=True)
                req.dllm_block_id += 1
                req.dllm_block_done = False
                req.extend_range.end = 8
                req.full_untruncated_fill_ids = array("q", [2, 3, 4, 5, 6, 0, 0, 0])
                req.dllm_incomplete_ids = array("q", [6, 0, 0, 0])
                state = {"current_block": 2}
                req.dllm_algo_state = state
                req, _, count = self._process(
                    **args,
                    tokens=[7, 7, 7, 7],
                    done=old_done,
                    req=req,
                    submitted_block_id=1,
                )
                self.assertEqual(req.output_ids, [4, 5])
                self.assertEqual(
                    list(req.full_untruncated_fill_ids), [2, 3, 4, 5, 6, 0, 0, 0]
                )
                self.assertEqual(list(req.dllm_incomplete_ids), [6, 0, 0, 0])
                self.assertIs(req.dllm_algo_state, state)
                self.assertFalse(req.dllm_block_done)
                self.assertEqual(count, 0)
                req, _, count = self._process(
                    **args,
                    tokens=[6, 6, 6, 6],
                    done=True,
                    req=req,
                )
                self.assertEqual(req.output_ids, [4, 5, 6, 6, 6, 6])
                self.assertTrue(req.dllm_block_done)
                self.assertEqual(count, 4)

    def test_duplicate_results_never_reopen_completed_block(self):
        args = dict(fdfo=True, prompt=[2, 3], end=4, fill=[2, 3, 0, 0])
        req, _, _ = self._process(**args, tokens=[2, 3, 4, 5], done=True)
        for done in (False, True, True):
            req, _, count = self._process(
                **args, tokens=[7, 7, 7, 7], done=done, req=req
            )
            self.assertEqual(req.output_ids, [4, 5])
            self.assertTrue(req.dllm_block_done)
            self.assertEqual(list(req.dllm_incomplete_ids), [])
            self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()

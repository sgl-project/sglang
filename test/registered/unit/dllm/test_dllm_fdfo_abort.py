import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFdfoAbort(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(weight_version="default")
        override.install()
        self.addCleanup(override.restore)

    def make_scheduler(self, holds_kv=True):
        config = SimpleNamespace(
            block_size=4,
            mask_id=0,
            max_running_requests=2,
            first_done_first_out_mode=True,
            requires_separate_context_encoding=False,
        )
        req = Req(
            rid="cancel-me",
            origin_input_text="test",
            origin_input_ids=array("q", [2, 3]),
            sampling_params=SamplingParams(max_new_tokens=32),
            dllm_config=config,
        )
        req.init_next_round_input()
        req.dllm_block_id = 1
        req.prefix_len = 0
        req.extend_end = 4
        if holds_kv:
            req.kv = ReqKvInfo(req_pool_idx=1, kv_allocated_len=4, kv_committed_len=4)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_continuous_input_polling = False
        scheduler.chunked_req = None
        scheduler.mm_receiver = None
        scheduler.waiting_queue = []
        scheduler.dllm_config = config
        scheduler.dllm_manager = DllmManager(config)
        scheduler.dllm_manager.waiting_queue = [req]
        scheduler.dllm_manager.staging_queue = [req]
        scheduler.grammar_manager = SimpleNamespace(abort_requests=Mock())
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.last_batch = SimpleNamespace(reqs=[req]) if holds_kv else None
        scheduler.tree_cache = SimpleNamespace(finish=Mock())
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )
        scheduler.forward_stream_ctx = nullcontext()
        scheduler.enable_overlap = True
        scheduler.future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SimpleNamespace(),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((4, 8))),
            needs_cpu_seq_lens=False,
        )
        scheduler.future_map.stash_dllm_block_tokens(
            torch.tensor([1]), torch.tensor([[2, 3, 4, 0]])
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=Mock(), free_group_end=Mock()
        )
        scheduler.metrics_reporter = SimpleNamespace(
            num_generated_tokens=0, report_prefill_stats=Mock()
        )
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        return scheduler, req

    def test_abort_invalidates_future_before_slot_release(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                scheduler, req = self.make_scheduler()
                scheduler.enable_overlap = overlap
                calls = []

                def release(req, tree_cache, checkpoint):
                    self.assertFalse(checkpoint)
                    self.assertTrue(req.finished())
                    self.assertEqual(
                        scheduler.future_map.dllm_block_tokens_buf[1].tolist(), [-1] * 4
                    )
                    calls.append(req.rid)
                    req.kv.req_pool_idx = None
                    req.kv.mark_kv_released()

                with patch(
                    "sglang.srt.managers.scheduler.release_kv_cache",
                    side_effect=release,
                ):
                    scheduler.abort_request(AbortReq(rid=req.rid))
                    scheduler.abort_request(AbortReq(rid=req.rid))
                self.assertEqual(calls, [req.rid])
                self.assertEqual(
                    scheduler.ipc_channels.send_to_tokenizer.send_output.call_count, 1
                )
                self.assertTrue(req.finished_output)
                self.assertFalse(req.dllm_block_done)
                self.assertIsNone(req.to_finish)
                self.assertEqual(scheduler.dllm_manager.waiting_queue, [])
                self.assertEqual(scheduler.dllm_manager.staging_queue, [])
                batch = SimpleNamespace(
                    is_dllm=lambda: True,
                    dllm_config=scheduler.dllm_config,
                    req_pool_indices=torch.tensor([1]),
                    input_ids=torch.tensor([6, 0, 0, 0]),
                )
                scheduler.future_map.resolve_dllm_block_tokens(batch)
                self.assertEqual(batch.input_ids.tolist(), [6, 0, 0, 0])

    def test_aborted_request_rejects_incomplete_and_complete_pending_results(self):
        scheduler, req = self.make_scheduler()
        with patch("sglang.srt.managers.scheduler.release_kv_cache"):
            scheduler.abort_request(AbortReq(rid=req.rid))
        original_fill = list(req.full_untruncated_fill_ids)
        req.dllm_incomplete_ids = array("q", [2, 3, 4, 0])
        for done in (False, True):
            with self.subTest(done=done):
                result = GenerationBatchResult(
                    dllm_block_ids=(1,),
                    next_token_ids=torch.tensor([[2, 3, 7, 7]]),
                    dllm_block_done=torch.tensor([done]),
                    dllm_algo_state=[{"late": True}],
                )
                batch = SimpleNamespace(
                    reqs=[req],
                    return_logprob=False,
                    prefill_stats=None,
                    dp_cooperation_info=None,
                )
                with patch(
                    "sglang.srt.dllm.mixin.scheduler.release_kv_cache"
                ) as release:
                    scheduler.process_batch_result_dllm(batch, result)
                release.assert_not_called()
                self.assertEqual(list(req.output_ids), [])
                self.assertEqual(list(req.full_untruncated_fill_ids), original_fill)
                self.assertEqual(list(req.dllm_incomplete_ids), [2, 3, 4, 0])
                self.assertIsNone(req.dllm_algo_state)
                self.assertEqual(scheduler.metrics_reporter.num_generated_tokens, 0)

    def test_abort_after_block_completion_while_waiting_for_admission(self):
        scheduler, req = self.make_scheduler()
        req.sampling_params.normalize(None)
        scheduler.model_config = SimpleNamespace(context_len=64)
        scheduler.tree_cache = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                page_size=4,
            )
        )
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = GenerationBatchResult(
            dllm_block_ids=(1,),
            next_token_ids=torch.tensor([[2, 3, 4, 5]]),
            dllm_block_done=torch.tensor([True]),
        )
        scheduler.process_batch_result_dllm(batch, result)
        self.assertTrue(req.dllm_block_done)
        self.assertEqual(req.dllm_block_id, 1)
        self.assertEqual(req.prefix_len, 0)
        self.assertEqual(req.kv.req_pool_idx, 1)

        def release(req, tree_cache, checkpoint):
            self.assertFalse(checkpoint)
            self.assertTrue(req.kv.holds_kv)
            self.assertEqual(req.kv.kv_allocated_len, 4)
            self.assertEqual(
                scheduler.future_map.dllm_block_tokens_buf[1].tolist(), [-1] * 4
            )
            req.kv.req_pool_idx = None
            req.kv.mark_kv_released()

        with patch(
            "sglang.srt.managers.scheduler.release_kv_cache", side_effect=release
        ) as release_cache:
            scheduler.abort_request(AbortReq(rid=req.rid))
            scheduler.abort_request(AbortReq(rid=req.rid))
        self.assertEqual(release_cache.call_count, 1)
        scheduler.process_batch_result_dllm(batch, result)
        self.assertTrue(req.finished())
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(req.output_ids.tolist(), [4, 5])
        self.assertEqual(req.dllm_block_offset, 0)

    def test_abort_before_first_forward_does_not_clear_another_slot(self):
        scheduler, req = self.make_scheduler(holds_kv=False)
        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            scheduler.abort_request(AbortReq(rid=req.rid))
        release.assert_not_called()
        self.assertTrue(req.finished())
        self.assertEqual(
            scheduler.future_map.dllm_block_tokens_buf[1].tolist(), [2, 3, 4, 0]
        )


if __name__ == "__main__":
    unittest.main()

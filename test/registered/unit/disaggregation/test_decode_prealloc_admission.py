"""Decode-side preallocation must admit every prompt the decode intake accepts.

The decode server clamps max_new_tokens so the prompt plus its output fits the
KV pool. Preallocation then also reserved num_reserved_decode_tokens on top of
the prompt, so a prompt within that reserve of the pool size was accepted but
never preallocated, even on an empty pool, and every request queued behind it
waited forever.
"""

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

KV_CAPACITY = 4096
NUM_RESERVED_DECODE_TOKENS = 512


class TestDecodePreallocAdmission(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_allocator(self, page_size):
        if page_size == 1:
            return TokenToKVPoolAllocator(
                KV_CAPACITY, torch.float16, "cpu", None, False
            )
        return PagedTokenToKVPoolAllocator(
            KV_CAPACITY, page_size, torch.float16, "cpu", None, False
        )

    def _decode_intake_limits(self, allocator):
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                effective_logical_max_total_num_tokens=KV_CAPACITY,
                logical_max_total_num_tokens=KV_CAPACITY,
                token_to_kv_pool_allocator=allocator,
                max_running_requests=4,
                forward_stream=None,
                req_to_token_pool=SimpleNamespace(size=4, max_context_len=1 << 20),
                token_to_kv_pool=SimpleNamespace(size=KV_CAPACITY),
            ),
            # A context longer than the decode pool, so the pool bounds intake.
            model_config=SimpleNamespace(context_len=1 << 20),
            dllm_algorithm=None,
            random_seed=0,
            device="cpu",
        )
        info = TpModelWorker.get_worker_info(worker)
        return info[4], info[5]

    def _accepted_req(self, rid, prompt_len, scheduler, max_req_input_len):
        req = Req(rid, "", [7] * prompt_len, SamplingParams(max_new_tokens=None))
        req.sampling_params.normalize(None)
        req.output_ids = [8]  # the first token, sampled by the prefill server
        Scheduler.init_req_max_new_tokens(scheduler, req)
        self.assertIsNone(validate_input_length(req, max_req_input_len, False))
        self.assertFalse(req.finished())
        return req

    def _new_queue(self, allocator, reqs):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [
            SimpleNamespace(
                req=req,
                waiting_for_input=True,
                kv_receiver=MagicMock(),
                metadata_buffer_index=-1,
                is_rebootstrap=False,
                host_staged=False,
            )
            for req in reqs
        ]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = NUM_RESERVED_DECODE_TOKENS
        queue.max_total_num_tokens = KV_CAPACITY
        queue._num_published_destinations = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._pre_alloc = MagicMock(return_value=torch.arange(4))
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 100
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 100
        queue.req_to_metadata_buffer_idx_allocator.alloc.side_effect = iter(range(100))
        queue.token_to_kv_pool_allocator = allocator
        queue.token_to_kv_pool = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.tree_cache = MagicMock()
        scheduler = MagicMock()
        scheduler.enable_priority_scheduling = False
        scheduler.running_batch.reqs = []
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.enable_decode_hicache = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        queue.scheduler = scheduler
        return queue

    def _intake_scheduler(self, allocator, page_size, max_req_len):
        allocator.max_new_tokens_for_memory = functools.partial(
            BaseTokenToKVPoolAllocator.max_new_tokens_for_memory, allocator
        )
        return SimpleNamespace(
            max_new_tokens_limit=None,
            max_req_len=max_req_len,
            max_total_num_tokens=KV_CAPACITY,
            kv_shard_widening=1,
            page_size=page_size,
            token_to_kv_pool_allocator=allocator,
            sliding_window_size=None,
            chunked_prefill_size=None,
        )

    def test_longest_accepted_prompt_is_preallocated_on_an_empty_pool(self):
        for page_size in (1, 16):
            with self.subTest(page_size=page_size):
                allocator = self._new_allocator(page_size)
                max_req_len, max_req_input_len = self._decode_intake_limits(allocator)
                scheduler = self._intake_scheduler(allocator, page_size, max_req_len)
                # validate_input_length accepts prompts shorter than the limit.
                longest = self._accepted_req(
                    "longest", max_req_input_len - 1, scheduler, max_req_input_len
                )
                short = self._accepted_req("short", 16, scheduler, max_req_input_len)
                queue = self._new_queue(allocator, [longest, short])
                self.assertFalse(queue._check_if_req_exceed_kv_capacity(longest))

                with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
                    preallocated, failed = queue.pop_preallocated()

                # The short request waits for the pool the longest one now holds.
                self.assertEqual(
                    [decode_req.req.rid for decode_req in preallocated], ["longest"]
                )
                self.assertEqual(failed, [])

    def test_retracted_request_near_its_length_limit_resumes_on_an_empty_pool(self):
        for page_size in (1, 16):
            with self.subTest(page_size=page_size):
                allocator = self._new_allocator(page_size)
                max_req_len, max_req_input_len = self._decode_intake_limits(allocator)
                scheduler = self._intake_scheduler(allocator, page_size, max_req_len)
                retracted = self._accepted_req(
                    "retracted", 64, scheduler, max_req_input_len
                )
                # Retracted one token before its max_new_tokens limit.
                retracted.output_ids = [8] * (
                    retracted.sampling_params.max_new_tokens - 1
                )
                retracted.is_retracted = True
                fresh = self._accepted_req("fresh", 16, scheduler, max_req_input_len)
                queue = self._new_queue(allocator, [fresh])
                queue.retracted_queue = [retracted]

                with patch("sglang.srt.disaggregation.decode.restore_kv_cache"):
                    resumed = queue.resume_retracted_reqs()

                self.assertEqual(resumed, [retracted])
                self.assertEqual(queue.retracted_queue, [])


if __name__ == "__main__":
    unittest.main()

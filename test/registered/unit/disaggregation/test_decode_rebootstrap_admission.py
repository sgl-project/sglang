"""A rebootstrapped decode request must be readmitted once the pool is empty.

A PD true-retraction (pause_generation(mode="retract"), or a retracted DeepSeek-V4
request) sends the request back to the prefill worker, which recomputes the
prompt plus the outputs already emitted. Admission sized that rebootstrap as the
prefill length plus max_new_tokens, counting the emitted outputs twice. A request
far enough into its output then needed more than the whole pool, so it was never
readmitted, and every request queued behind it waited forever.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import Req
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
PROMPT_LEN = 1000
MAX_NEW_TOKENS = 3000


class TestDecodeRebootstrapAdmission(CustomTestCase):
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

    def _rebootstrapped_req(self, rid, num_emitted):
        req = Req(
            rid,
            "",
            [7] * PROMPT_LEN,
            SamplingParams(max_new_tokens=MAX_NEW_TOKENS),
        )
        req.sampling_params.normalize(None)
        # The retract popped the boundary token; it is replayed at commit time.
        req.output_ids = [8] * (num_emitted - 1)
        req.pd_rebootstrap_forced_output_id = 8
        req.pd_rebootstrap_in_progress = True
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
                is_rebootstrap=True,
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
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[]),
            submit_prefill_recompute=MagicMock(),
        )
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

    def test_rebootstrap_late_in_its_output_is_preallocated_on_an_empty_pool(self):
        # Prompt plus max_new_tokens fits the pool, as decode intake guarantees.
        self.assertLessEqual(PROMPT_LEN + MAX_NEW_TOKENS, KV_CAPACITY)
        for page_size in (1, 16):
            with self.subTest(page_size=page_size):
                allocator = self._new_allocator(page_size)
                req = self._rebootstrapped_req("rebootstrap", num_emitted=1200)
                queue = self._new_queue(allocator, [req])
                self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))

                with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
                    preallocated, failed = queue.pop_preallocated()

                self.assertEqual(
                    [decode_req.req.rid for decode_req in preallocated],
                    ["rebootstrap"],
                )
                self.assertEqual(failed, [])

    def test_rebootstrap_reserves_only_the_outputs_left_to_decode(self):
        allocator = self._new_allocator(1)
        req = self._rebootstrapped_req("rebootstrap", num_emitted=1200)
        queue = self._new_queue(allocator, [req])
        fill_len = queue._pre_alloc_fill_len(req)
        self.assertEqual(fill_len, PROMPT_LEN + 1199)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            required = queue._required_admission_tokens(req, fill_len, 0, 0)

        # The finished request holds the prompt and max_new_tokens outputs.
        self.assertEqual(required, PROMPT_LEN + MAX_NEW_TOKENS)


if __name__ == "__main__":
    unittest.main()

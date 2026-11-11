"""Scheduler-side duplicate-rid rejection (sgl-project/sglang#49903).

Each tokenizer worker rejects a duplicate rid only against its own
`rid_to_state`, so with `--tokenizer-worker-num > 1` two concurrent requests
with the same rid can both be accepted on different workers. The scheduler is
the first place that sees both, so it must reject the second one at intake --
before it is queued and run -- or the detokenizer merges the two requests'
states under the one rid and streams one client's tokens into the other's
response.

Covers, mirroring the issue's PoC (two tokenizer workers, distinct `rid`s as
the control case):

- `_live_request_with_rid` finds a live request in every list that can hold
  one (waiting, running, chunked, prefill-launched, PD bootstrap/prealloc),
  including lists that hold wrapper items, and returns None once it leaves.
- A second generate request with a live rid is rejected with the same
  "Duplicate request ID detected" abort the tokenizer-side check raises, is
  never queued, and the rid becomes reusable once the first request leaves.
- Distinct rids on two tokenizer workers are both admitted (control case).
- The same rejection applies to embedding requests.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import empty_request_lists, leaving_retirer
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORKER_A = "ipc://tokenizer-worker-0"
WORKER_B = "ipc://tokenizer-worker-1"


def _generate_req(rid, ipc):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="hi",
        input_ids=array("q", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=8),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        http_worker_ipc=ipc,
    )


def _embedding_req(rid, ipc):
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="hi",
        input_ids=array("q", [1]),
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        http_worker_ipc=ipc,
    )


def _live_req(rid):
    return Req(rid, "", array("q", [1]), SamplingParams(max_new_tokens=8))


class TestSchedulerDuplicateRid(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        empty_request_lists(scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_session_radix_cache = False
        scheduler.model_config = SimpleNamespace(hf_eos_token_id=2, vocab_size=1000)
        scheduler.tokenizer = None
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.metrics_collector = None
        scheduler.tree_cache = MagicMock()
        scheduler.max_req_input_len = 1024
        scheduler.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: False,
            is_uno=lambda: False,
            is_none=lambda: True,
            is_some=lambda: False,
        )
        scheduler.validate_dllm_request = lambda req: None
        scheduler.init_req_max_new_tokens = MagicMock()
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        scheduler._record_kv_event_lora_name = MagicMock()
        scheduler._bind_pp_arrival_prefetch = MagicMock()
        scheduler.invariant_checker = MagicMock()
        scheduler.grammar_manager.process_req_with_grammar = lambda req: False
        scheduler.request_retirer = MagicMock()
        scheduler._add_request_to_queue = MagicMock()
        return scheduler

    def _admit(self, scheduler, rid, ipc):
        """Admit one generate request; return the Req the scheduler built."""
        scheduler.handle_generate_request(_generate_req(rid, ipc))
        scheduler._add_request_to_queue.assert_called()
        return scheduler._add_request_to_queue.call_args[0][0]

    def test_live_request_with_rid_searches_every_list(self):
        scheduler = self._new_scheduler()
        self.assertIsNone(scheduler._live_request_with_rid("req-1"))

        waiting = _live_req("waiting")
        running = _live_req("running")
        chunked = _live_req("chunked")
        launched = _live_req("launched")
        bootstrap = _live_req("bootstrap")
        prealloc = _live_req("prealloc")
        move([waiting], scheduler.waiting_queue)
        move([running], scheduler.running_reqs)
        move([chunked], scheduler.chunked_reqs)
        move([launched], scheduler.prefill_launched_reqs)
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=IndexedReqList(ListKind.PD_BOOTSTRAP)
        )
        move([bootstrap], scheduler.disagg_prefill_bootstrap_queue.queue)
        # PD decode prealloc queues hold wrappers, not bare Reqs.
        prealloc_list = IndexedReqList(
            ListKind.PD_PREALLOC, req_of=lambda item: item.req
        )
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            request_lists=lambda: {ListKind.PD_PREALLOC: prealloc_list}
        )
        move([SimpleNamespace(req=prealloc)], prealloc_list)

        for req in (waiting, running, chunked, launched, bootstrap, prealloc):
            self.assertIs(
                scheduler._live_request_with_rid(req.rid),
                req,
                f"live request {req.rid} not found",
            )
        self.assertIsNone(scheduler._live_request_with_rid("no-such-rid"))

    def test_live_request_with_rid_misses_released_requests(self):
        scheduler = self._new_scheduler()
        req = _live_req("req-1")
        move([req], scheduler.waiting_queue)
        self.assertIs(scheduler._live_request_with_rid("req-1"), req)
        # The request finishes and leaves through the retirer.
        leaving_retirer().release(req)
        self.assertIsNone(scheduler._live_request_with_rid("req-1"))

    def test_same_rid_on_two_tokenizer_workers_is_rejected(self):
        scheduler = self._new_scheduler()

        # Tenant A is admitted on worker 0 and is live in the waiting queue.
        first = self._admit(scheduler, "req-1", WORKER_A)
        move([first], scheduler.waiting_queue)

        # Tenant B sends the same rid to worker 1: the scheduler must reject
        # it before it is queued and run, so the detokenizer never sees two
        # live requests under one rid.
        scheduler.handle_generate_request(_generate_req("req-1", WORKER_B))
        self.assertEqual(scheduler._add_request_to_queue.call_count, 1)
        scheduler.request_retirer.retire_unadmitted.assert_called_once()
        rejected = scheduler.request_retirer.retire_unadmitted.call_args[0][0]
        self.assertIsInstance(rejected.finished_reason, FINISH_ABORT)
        self.assertIn(
            "Duplicate request ID detected: req-1", rejected.finished_reason.message
        )
        self.assertEqual(rejected.finished_reason.status_code, 400)

    def test_same_rid_rejected_while_first_request_is_running(self):
        scheduler = self._new_scheduler()
        first = self._admit(scheduler, "req-1", WORKER_A)
        move([first], scheduler.running_reqs)

        scheduler.handle_generate_request(_generate_req("req-1", WORKER_B))
        self.assertEqual(scheduler._add_request_to_queue.call_count, 1)
        scheduler.request_retirer.retire_unadmitted.assert_called_once()

    def test_distinct_rids_on_two_tokenizer_workers_are_admitted(self):
        scheduler = self._new_scheduler()

        first = self._admit(scheduler, "req-a", WORKER_A)
        move([first], scheduler.waiting_queue)
        second = self._admit(scheduler, "req-b", WORKER_B)

        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        scheduler.request_retirer.retire_unadmitted.assert_not_called()
        self.assertEqual(first.rid, "req-a")
        self.assertEqual(second.rid, "req-b")

    def test_rid_reusable_after_first_request_leaves(self):
        scheduler = self._new_scheduler()
        first = self._admit(scheduler, "req-1", WORKER_A)
        move([first], scheduler.waiting_queue)
        # The first request finishes and leaves through the retirer.
        leaving_retirer().release(first)

        second = self._admit(scheduler, "req-1", WORKER_B)
        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        scheduler.request_retirer.retire_unadmitted.assert_not_called()
        self.assertIsNot(second, first)

    def test_same_rid_embedding_request_is_rejected(self):
        scheduler = self._new_scheduler()
        first = self._admit(scheduler, "req-1", WORKER_A)
        move([first], scheduler.waiting_queue)

        # set_finish_with_abort reads the TP rank; state it for the test.
        with get_parallel().override(tp_rank=0):
            scheduler._handle_embedding_request(
                _embedding_req("req-1", WORKER_B), MagicMock()
            )
        # Embedding intake rejections queue an abort, as the other intake
        # rejections in that handler do.
        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        rejected = scheduler._add_request_to_queue.call_args[0][0]
        # set_finish_with_abort stages the abort in to_finish; the scheduler
        # applies it at the request's next commit.
        self.assertIsInstance(rejected.to_finish, FINISH_ABORT)
        self.assertIn(
            "Duplicate request ID detected: req-1", rejected.to_finish.message
        )

    def test_distinct_rid_embedding_request_is_admitted(self):
        scheduler = self._new_scheduler()
        first = self._admit(scheduler, "req-1", WORKER_A)
        move([first], scheduler.waiting_queue)

        # set_finish_with_abort reads the TP rank; state it for the test.
        with get_parallel().override(tp_rank=0):
            scheduler._handle_embedding_request(
                _embedding_req("req-2", WORKER_B), MagicMock()
            )
        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        admitted = scheduler._add_request_to_queue.call_args[0][0]
        self.assertIsNone(admitted.finished_reason)


if __name__ == "__main__":
    unittest.main()

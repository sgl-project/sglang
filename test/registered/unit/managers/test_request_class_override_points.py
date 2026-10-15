"""Subclass-provided request classes must survive every construction site.

A fork that keeps request-local state on a `Req` or `TokenizedGenerateReqInput`
subclass overrides one hook per class. A bare `Req(...)` or
`TokenizedGenerateReqInput(...)` added beside the hook would silently drop the
subclass on that one path, so these cases walk every path.
"""

import asyncio
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

VOCAB = 128


class _TaggedReq(Req):
    def __init__(self, *args, source_rid: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_rid = source_rid


class _TaggedScheduler(Scheduler):
    def _create_generation_req(self, recv_req, *args, **kwargs) -> _TaggedReq:
        return _TaggedReq(*args, source_rid=recv_req.rid, **kwargs)


class _TaggedTokenized(TokenizedGenerateReqInput):
    pass


class _TaggedBatch(BatchTokenizedGenerateReqInput):
    pass


class _TaggedTokenizerManager(TokenizerManager):
    tokenized_generate_req_type = _TaggedTokenized
    batch_tokenized_generate_req_type = _TaggedBatch


def _scheduler(sessions, queued):
    scheduler = Scheduler.__new__(_TaggedScheduler)
    scheduler.session_controller = sessions
    scheduler.enable_session_radix_cache = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.model_config = SimpleNamespace(hf_eos_token_id=None, vocab_size=VOCAB)
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = lambda req: None
    scheduler.init_req_max_new_tokens = lambda req: None
    scheduler._add_request_to_queue = queued.append
    return scheduler


def _recv(rid, session_params=None):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="hello",
        input_ids=array("q", [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=session_params,
        bootstrap_port=1,
    )


class TestRequestClassOverridePoints(CustomTestCase):
    def test_scheduler_builds_every_generation_request_through_the_factory(self):
        """Plain, session, and unknown-session paths all yield the subclass."""
        live = Session(capacity_of_str_len=0, session_id="live", streaming=True)
        paths = {
            "plain": (None, {}),
            "session": (SessionParams(id="live"), {"live": live}),
            "unknown-session": (SessionParams(id="gone"), {}),
        }
        for name, (session_params, sessions) in paths.items():
            with self.subTest(path=name):
                queued = []
                scheduler = _scheduler(sessions, queued)
                recv = _recv(f"rid-{name}", session_params)
                # An mm_input_error stops every path right after construction.
                with get_parallel().override(tp_rank=0):
                    scheduler.handle_generate_request(recv, mm_input_error="stop")
                (req,) = queued
                self.assertIsInstance(req, _TaggedReq)
                self.assertEqual(req.source_rid, recv.rid)
                self.assertEqual(req.rid, recv.rid)

    def test_tokenizer_manager_emits_the_subclass_wire_types(self):
        """Single and batched dispatch hand the subclass types to the scheduler."""
        manager = TokenizerManager.__new__(_TaggedTokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=VOCAB)
        manager.fake_bootstrap_room_counter = 0
        manager.rid_to_state = {
            "r0": SimpleNamespace(time_stats=APIServerReqTimeStats())
        }
        obj = GenerateReqInput(input_ids=[1, 2], rid="r0", sampling_params={})

        with get_context().override_server_args():
            tokenized = manager._create_tokenized_object(obj, "hello", [1, 2])
        self.assertIsInstance(tokenized, _TaggedTokenized)

        sent = []
        manager.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=lambda items: None,
        )
        manager._dispatch_to_scheduler = sent.append
        manager._mark_state_dispatched = lambda rid: None
        asyncio.run(manager._send_batch_request([tokenized]))
        (batch,) = sent
        self.assertIsInstance(batch, _TaggedBatch)
        self.assertEqual(batch.batch, [tokenized])


if __name__ == "__main__":
    unittest.main()

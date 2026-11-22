"""Filler seeds: drawn per request where it enters the scheduler, shared by every rank."""

import pickle
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedGenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import request_receiver  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.session.session_controller import Session  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 128


def _tokenized(rid, *, sampling_seed=None, session_params=None, filler_seed=None):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="hello",
        input_ids=array("q", [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(top_k=50, sampling_seed=sampling_seed),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=session_params,
        bootstrap_port=1,
        filler_seed=filler_seed,
    )


def _receive(sent):
    """The requests the first rank holds once it has drawn their fillers."""
    request_receiver._draw_filler_seeds(sent)
    return sent


def _build_req(recv_req, sessions):
    """The Req one scheduler rank builds from a received request."""
    queued = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.session_controller = sessions
    scheduler.enable_session_radix_cache = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.model_config = SimpleNamespace(
        hf_eos_token_id=None, vocab_size=VOCAB, token_id_vocab_size=VOCAB
    )
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.kv_events_publisher = SimpleNamespace(enable_kv_cache_events=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = lambda req: None
    scheduler.init_req_max_new_tokens = lambda req: None
    scheduler._add_request_to_queue = queued.append
    # An mm_input_error stops the request right after it is built.
    with get_parallel().override(tp_rank=0):
        scheduler.handle_generate_request(recv_req, mm_input_error="stop")
    (req,) = queued
    return req


class TestFillerSeed(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))

    def test_every_rank_samples_with_the_leader_fillers(self):
        """The receiving rank replaces any filler a sender set and draws one per
        request, batched ones included. A rank given the broadcast copies builds
        the same fillers on every construction path (an unknown session is aborted
        but still prefilled); the seeded request keeps its own seed."""
        received = _receive(
            [
                _tokenized("seeded", sampling_seed=123),
                _tokenized("plain", filler_seed=7),
                _tokenized("turn", session_params=SessionParams(id="live")),
                _tokenized("lost", session_params=SessionParams(id="gone")),
                BatchTokenizedGenerateReqInput(batch=[_tokenized("b")]),
            ]
        )
        fillers = [r.filler_seed for r in received[:4]] + [
            received[4].batch[0].filler_seed
        ]
        self.assertNotIn(7, fillers)
        self.assertEqual(len(set(fillers)), 5)

        # A broadcast hands every other rank a pickled copy.
        copies = pickle.loads(pickle.dumps(received[:4]))
        live = Session(capacity_of_str_len=0, session_id="live", streaming=True)
        reqs = [_build_req(r, {"live": live}) for r in copies]
        info = SamplingBatchInfo.from_schedule_batch(
            SimpleNamespace(reqs=reqs, device="cpu"), VOCAB
        )
        self.assertEqual(info.rows_with_seed, [True, False, False, False])
        self.assertEqual(info.sampling_seed.tolist(), [123] + fillers[1:4])


if __name__ == "__main__":
    unittest.main()

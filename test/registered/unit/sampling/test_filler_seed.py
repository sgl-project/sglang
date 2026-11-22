"""Filler seeds: drawn per request where it enters the scheduler, shared by every rank."""

import importlib
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
from sglang.srt.utils.common import set_random_seed  # noqa: E402

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
    """The requests a server's first rank receives from its frontend."""
    receiver = request_receiver.SchedulerRequestReceiver(
        recv_from_tokenizer=SimpleNamespace(drain=lambda limit: list(sent)),
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        request_retirer=None,
        get_last_batch=lambda: None,
        rust_server_mode=True,
    )
    return receiver.recv_requests()


def _relaunch():
    """Reset the receiving side as a restart with the same --random-seed does."""
    importlib.reload(request_receiver)
    set_random_seed(0)


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

    def test_each_received_request_draws_its_own_filler(self):
        """Two launches given the same traffic draw different fillers, and a filler
        a sender put on the wire is replaced."""

        def traffic():
            return [
                _tokenized("a", filler_seed=7),
                BatchTokenizedGenerateReqInput(
                    batch=[_tokenized("b"), _tokenized("c")]
                ),
            ]

        def launch_and_receive():
            _relaunch()
            single, batch = _receive(traffic())
            return [single.filler_seed] + [r.filler_seed for r in batch]

        first, second = launch_and_receive(), launch_and_receive()
        self.assertNotIn(7, first)
        self.assertEqual(len(set(first + second)), 6)
        self.assertTrue(all(0 <= seed < 2**63 for seed in first + second))

    def test_another_rank_samples_with_the_same_seeds(self):
        """A rank given the broadcast requests builds the leader's fillers on every
        construction path (an unknown session is aborted but still prefilled); the
        seeded request keeps its own seed."""
        received = _receive(
            [
                _tokenized("seeded", sampling_seed=123),
                _tokenized("plain"),
                _tokenized("turn", session_params=SessionParams(id="live")),
                _tokenized("lost", session_params=SessionParams(id="gone")),
            ]
        )
        # A broadcast hands every other rank a pickled copy.
        copies = pickle.loads(pickle.dumps(received))
        live = Session(capacity_of_str_len=0, session_id="live", streaming=True)
        reqs = [_build_req(r, {"live": live}) for r in copies]
        info = SamplingBatchInfo.from_schedule_batch(
            SimpleNamespace(reqs=reqs, device="cpu"), VOCAB
        )
        self.assertEqual(info.rows_with_seed, [True, False, False, False])
        self.assertEqual(
            info.sampling_seed.tolist(),
            [123] + [r.filler_seed for r in received[1:]],
        )


if __name__ == "__main__":
    unittest.main()

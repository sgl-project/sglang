"""Subclass-provided request classes must survive every construction site.

A fork that keeps request-local state on a `Req` or `TokenizedGenerateReqInput`
subclass overrides one hook per class. A bare `Req(...)` or
`TokenizedGenerateReqInput(...)` added beside the hook would silently drop the
subclass on that one path, so these cases walk every path. The scheduler side
is swapped through `Scheduler.component_classes`: the request intake builds
each generation request, and the step planner builds its prefill adder.
"""

import ast
import asyncio
import dataclasses
import inspect
import textwrap
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler_assembly
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_intake import RequestIntake
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.scheduler_request_intake import make_request_intake

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

VOCAB = 128


class _TaggedReq(Req):
    def __init__(self, *args, source_rid: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_rid = source_rid


class _TaggedIntake(RequestIntake):
    def create_generation_req(self, recv_req, *args, **kwargs) -> _TaggedReq:
        return _TaggedReq(*args, source_rid=recv_req.rid, **kwargs)


class _TaggedAdder(PrefillAdder):
    pass


class _TaggedScheduler(Scheduler):
    def component_classes(self):
        return dataclasses.replace(
            super().component_classes(),
            request_intake=_TaggedIntake,
            prefill_adder=_TaggedAdder,
        )


class _TaggedTokenized(TokenizedGenerateReqInput):
    pass


class _TaggedBatch(BatchTokenizedGenerateReqInput):
    pass


class _TaggedTokenizerManager(TokenizerManager):
    tokenized_generate_req_type = _TaggedTokenized
    batch_tokenized_generate_req_type = _TaggedBatch


def _intake(sessions, queued):
    classes = Scheduler.__new__(_TaggedScheduler).component_classes()
    intake = make_request_intake(
        classes.request_intake,
        session_controller=sessions,
        model_config=SimpleNamespace(hf_eos_token_id=None, vocab_size=VOCAB),
    )
    intake._maybe_namespace_elastic_radix_cache = lambda req: None
    intake.init_req_max_new_tokens = lambda req: None
    intake.add_request_to_queue = queued.append
    return intake


def _builder_keywords(func_name: str, callee: str) -> dict:
    """The keywords `scheduler_assembly.<func_name>` passes to `callee`, unparsed."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(getattr(scheduler_assembly, func_name)))
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == callee:
            return {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
    raise AssertionError(f"{func_name} never calls {callee}")


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
    def test_scheduler_subclass_supplies_the_component_classes(self):
        """The assembly builds the intake and the planner's adders from the
        classes the scheduler subclass returns."""
        classes = Scheduler.__new__(_TaggedScheduler).component_classes()
        self.assertIs(classes.request_intake, _TaggedIntake)
        self.assertIs(classes.prefill_adder, _TaggedAdder)
        default = scheduler_assembly.default_component_classes()
        self.assertIs(classes.rust_server, default.rust_server)
        self.assertIs(classes.output_streamer, default.output_streamer)

        # The assembly builds both from the table it is handed.
        _builder_keywords("assemble_scheduler", "component_classes.request_intake")
        self.assertEqual(
            _builder_keywords("assemble_scheduler", "StepPlanner")[
                "prefill_adder_class"
            ],
            "component_classes.prefill_adder",
        )

    def test_intake_builds_every_generation_request_through_the_factory(self):
        """Plain, session, and unknown-session paths all yield the subclass."""
        live = Session(capacity_of_str_len=0, session_id="live", streaming=True)
        paths = {
            "plain": (None, {}),
            "session": (SessionParams(id="live"), {"live": live}),
            "unknown-session": (SessionParams(id="gone"), {}),
        }
        enter_scope(self, published_topology())
        for name, (session_params, sessions) in paths.items():
            with self.subTest(path=name):
                queued = []
                intake = _intake(sessions, queued)
                recv = _recv(f"rid-{name}", session_params)
                # An mm_input_error stops every path right after construction.
                with get_parallel().override(tp_rank=0):
                    intake.handle_generate_request(recv, mm_input_error="stop")
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
        manager.positional_embeds_transport = SimpleNamespace(
            prepare_for_dispatch=lambda tokenized_objs: [],
            cancel_for_dispatch=lambda proxies: None,
        )
        manager._dispatch_to_scheduler = sent.append
        manager._mark_state_dispatched = lambda rid: None
        asyncio.run(manager._send_batch_request([tokenized]))
        (batch,) = sent
        self.assertIsInstance(batch, _TaggedBatch)
        self.assertEqual(batch.batch, [tokenized])


if __name__ == "__main__":
    unittest.main()

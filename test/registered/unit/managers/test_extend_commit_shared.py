"""The normal and the PD prefill extend commits share one row loop and its row
helpers, and abort a grammar rejection with the same error."""

import unittest
from collections import Counter
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BRP_MODULE = "sglang.srt.managers.scheduler_components.batch_result_processor"

# The helpers both commits call, once per concluded row.
_SHARED = (
    "commit_extend",
    "host_prefill_result",
    "sampling_mask_finish_reason",
    "apply_prefill_logprobs",
    "apply_prefill_grammar",
)


class _Grammar:
    def __init__(self, reject: bool):
        self.reject = reject
        self.finished = False

    def accept_token(self, token_id):
        if self.reject:
            raise ValueError("token not allowed")

    def is_terminated(self):
        return False


class _CountingProcessor(SchedulerBatchResultProcessor):
    """The real processor, counting the shared helpers; the host copies of the
    result are a fixed one-row output."""

    calls: Counter

    def host_prefill_result(self, batch, result):
        self.calls["host_prefill_result"] += 1
        logits_output = SimpleNamespace(
            hidden_states=None,
            customized_info=None,
            next_token_sampling_mask_status=[SamplingMaskStatus.OK],
        )
        return logits_output, [7], []

    def add_sampling_mask_return_values(self, i, req, logits_output):
        pass

    def _mamba_lazy_keep_pending_write(self, req):
        return False


for _name in _SHARED:
    if _name == "host_prefill_result":
        continue

    def _counted(self, *args, _name=_name, **kwargs):
        self.calls[_name] += 1
        return getattr(SchedulerBatchResultProcessor, _name)(self, *args, **kwargs)

    setattr(_CountingProcessor, _name, _counted)


def _processor():
    kv = SimpleNamespace(
        token_to_kv_pool_allocator=Mock(), tree_cache=None, req_to_token_pool=None
    )
    processor = _CountingProcessor(
        is_generation=True,
        model_config=SimpleNamespace(
            think_end_ids=None, self_opened_think_start_ids=None
        ),
        kv=kv,
        services=SimpleNamespace(output_streamer=Mock()),
        decode_offload_manager=None,
        metrics_collector=None,
        workers=SimpleNamespace(),
        logprob_result_processor=Mock(
            calculate_num_input_logprobs=Mock(return_value=0)
        ),
        beam_coordinator=SimpleNamespace(),
        pending=PendingSteps(),
        request_retirer=make_request_retirer(kv=kv),
    )
    object.__setattr__(processor, "calls", Counter())
    return processor


def _req(reject: bool) -> Req:
    sp = SamplingParams(max_new_tokens=16, temperature=0)
    sp.normalize(None)
    req = Req(
        rid="r0", origin_input_text="", origin_input_ids=[1, 2, 3], sampling_params=sp
    )
    req.return_logprob = True
    req.return_sampling_mask = True
    req.grammar = _Grammar(reject)
    req.disagg = DisaggReqState()
    return req


def _batch(req):
    return SimpleNamespace(
        reqs=[req],
        chunked_req=None,
        num_decode_rows=0,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        return_logprob=True,
        extend_lens=[3],
        extend_logprob_start_lens=[3],
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        forward_iter=1,
    )


def _result():
    return GenerationBatchResult(next_token_ids=torch.tensor([7]))


class TestSharedExtendCommit(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())
        enter_scope(self, patch(f"{BRP_MODULE}.checkpoint_kv_cache"))

    def _commit_normal(self, req):
        processor = _processor()
        processor.process_batch_result_prefill(_batch(req), _result())
        return processor

    def _commit_pd(self, req):
        processor = _processor()
        owner = prefill_owner_shell(
            batch_result_processor=processor,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
            prefilling=[req],
            checkpoint=Mock(),
            send_kv_chunk=Mock(),
            output_streamer=Mock(),
            request_retirer=processor.request_retirer,
            req_to_metadata_buffer_idx_allocator=Mock(),
        )
        owner.kv_senders[req.rid] = Mock()
        owner.commit_extend(_batch(req), _result())
        return processor, owner

    def test_both_commits_call_the_shared_helpers_once(self):
        normal = self._commit_normal(_req(reject=False))
        pd, owner = self._commit_pd(_req(reject=False))
        expected = Counter({name: 1 for name in _SHARED})
        self.assertEqual(normal.calls, expected)
        self.assertEqual(pd.calls, expected)
        self.assertEqual([r.rid for r in owner.transfer], ["r0"])

    def test_a_grammar_rejection_aborts_with_its_error_in_both(self):
        normal_req = _req(reject=True)
        self._commit_normal(normal_req)
        # A running request finishes at its next commit.
        self.assertIsInstance(normal_req.to_finish, FINISH_ABORT)
        self.assertFalse(normal_req.finished())

        pd_req = _req(reject=True)
        _, owner = self._commit_pd(pd_req)
        # A PD prefill request is retired instead of handed over.
        self.assertEqual(owner.transfer, [])
        self.assertTrue(pd_req.finished())

        for reason in (normal_req.to_finish, pd_req.finished_reason):
            self.assertEqual(reason.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.assertIn("Grammar accept_token failed", reason.message)


if __name__ == "__main__":
    unittest.main()

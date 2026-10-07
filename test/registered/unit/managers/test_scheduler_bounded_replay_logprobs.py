"""Reject prompt logprobs at admission under decoder SWA bounded replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_parallel

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PROMPT_LEN = 16


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler.max_req_input_len = 1024
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler.validate_dllm_request = MagicMock(return_value=None)
    scheduler.init_req_max_new_tokens = MagicMock()
    scheduler.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: False,
        is_uno=lambda: False,
    )
    scheduler.tree_cache = SimpleNamespace(cache_controller=None)
    scheduler.grammar_manager = SimpleNamespace(
        process_req_with_grammar=lambda _: False
    )
    scheduler._add_request_to_queue = MagicMock()
    return scheduler


def _admit(*, bounded_replay, logprob_start_len, token_ids_logprob=None, session=None):
    scheduler = _scheduler()
    recv_req = MagicMock(
        session_params=None if session is None else SimpleNamespace(id="sid"),
        session_id=None,
        input_embeds=None,
        bootstrap_port=1,
        pp_prefetch_ticketed=False,
        mm_inputs=None,
        return_logprob=True,
        logprob_start_len=logprob_start_len,
        token_ids_logprob=token_ids_logprob,
        return_routed_experts=False,
    )
    req = MagicMock(
        origin_input_ids=list(range(PROMPT_LEN)),
        return_logprob=True,
        return_sampling_mask=False,
        is_prefill_only=False,
        session=session,
        finished_reason=None,
    )
    if session is not None:
        # Session requests come from Session.create_req, not the Req constructor.
        session.create_req = MagicMock(return_value=req)
        scheduler.session_controller = MagicMock()
        scheduler.session_controller.__contains__.return_value = True
        scheduler.session_controller.get.return_value = session
    with (
        get_context().override_server_args(
            enable_decoder_swa_bounded_replay=bounded_replay
        ),
        get_parallel().override(pp_rank=0),
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch("sglang.srt.managers.scheduler.validate_input_length", return_value=None),
    ):
        scheduler.handle_generate_request(recv_req)
    scheduler._add_request_to_queue.assert_called_once_with(req)
    return req


class TestBoundedReplayPromptLogprobs(CustomTestCase):
    def _assert_rejected(self, req):
        req.set_finish_with_abort.assert_called_once()
        self.assertIn(
            "--enable-decoder-swa-bounded-replay",
            req.set_finish_with_abort.call_args.args[0],
        )
        self.assertEqual(req.logprob_start_len, -1)

    def test_prompt_logprobs_rejected_at_admission(self):
        # Prompt-token logprobs would raise inside the forward pass. Only -1
        # means "no prompt logprobs"; other negative starts resolve to 0.
        for start in (0, PROMPT_LEN // 2, PROMPT_LEN - 1, -2):
            with self.subTest(logprob_start_len=start):
                self._assert_rejected(
                    _admit(bounded_replay=True, logprob_start_len=start)
                )

    def test_output_logprobs_still_admitted(self):
        for start, token_ids_logprob in ((-1, None), (PROMPT_LEN, None), (-1, [5])):
            with self.subTest(
                logprob_start_len=start, token_ids_logprob=token_ids_logprob
            ):
                req = _admit(
                    bounded_replay=True,
                    logprob_start_len=start,
                    token_ids_logprob=token_ids_logprob,
                )
                req.set_finish_with_abort.assert_not_called()

    def test_streaming_session_keeps_output_logprobs(self):
        # A streaming session drops logprob_start_len when the request is
        # scheduled and returns only output logprobs, so it is not rejected.
        session = SimpleNamespace(streaming=True, close_on_finish=False)
        req = _admit(bounded_replay=True, logprob_start_len=0, session=session)
        session.create_req.assert_called_once()
        req.set_finish_with_abort.assert_not_called()

    def test_non_streaming_session_prompt_logprobs_rejected(self):
        session = SimpleNamespace(streaming=False, close_on_finish=False)
        req = _admit(bounded_replay=True, logprob_start_len=0, session=session)
        session.create_req.assert_called_once()
        self._assert_rejected(req)

    def test_prompt_logprobs_admitted_without_bounded_replay(self):
        req = _admit(bounded_replay=False, logprob_start_len=0)
        req.set_finish_with_abort.assert_not_called()


if __name__ == "__main__":
    unittest.main()

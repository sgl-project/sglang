"""Beam groups are initialized only after request preparation succeeds."""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSchedulerBeamRequestValidation(CustomTestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.enable_session_radix_cache = False
        self.scheduler.model_config = SimpleNamespace(
            hf_eos_token_id={1},
            vocab_size=128,
        )
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.scheduler.max_req_input_len = 32
        self.scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        self.scheduler.tokenizer = None
        self.scheduler.dllm_config = None
        self.scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        self.scheduler.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: False,
            is_uno=lambda: False,
        )
        self.scheduler.session_controller = {}
        self.scheduler.output_streamer = MagicMock()
        self.scheduler.grammar_manager = MagicMock()
        self.scheduler.grammar_manager.process_req_with_grammar.return_value = False
        self.scheduler.init_req_max_new_tokens = MagicMock()
        self.scheduler._add_request_to_queue = MagicMock()
        self.scheduler.beam_coordinator = MagicMock()

        self.recv_req = MagicMock(
            session_params=None,
            session_id=None,
            input_embeds=None,
            bootstrap_port=1,
            mm_inputs=None,
            return_logprob=False,
            logprob_start_len=-1,
            token_ids_logprob=None,
            return_routed_experts=False,
        )
        self.req = MagicMock(
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SimpleNamespace(max_new_tokens=10),
            return_sampling_mask=False,
            return_logprob=False,
            is_prefill_only=False,
            logprob_start_len=-1,
            finished_reason=None,
        )

    def _run(self, validate_input_length):
        with (
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=2,
            ),
            patch("sglang.srt.managers.scheduler.Req", return_value=self.req),
            patch(
                "sglang.srt.managers.scheduler.validate_input_length",
                side_effect=validate_input_length,
            ),
            patch(
                "sglang.srt.managers.scheduler.get_serving",
                return_value=SimpleNamespace(allow_auto_truncate=True),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_device",
                return_value=SimpleNamespace(mlx_enable_sampling=False),
            ),
        ):
            self.scheduler.handle_generate_request(self.recv_req)

    def test_initializes_beam_from_finalized_request(self):
        events = []

        def finalize_budget(req):
            events.append("budget")
            req.sampling_params.max_new_tokens = 6

        def validate_prompt(req, _max_input_len, _allow_auto_truncate):
            events.append("prompt")
            req.origin_input_ids = array("q", [2, 3, 4])
            return None

        def initialize_beam(req, recv_req):
            events.append("beam")
            self.assertEqual(list(req.origin_input_ids), [2, 3, 4])
            self.assertEqual(req.sampling_params.max_new_tokens, 6)
            self.assertIs(recv_req, self.recv_req)
            return None

        self.scheduler.init_req_max_new_tokens.side_effect = finalize_budget
        self.scheduler.beam_coordinator.validate_and_init.side_effect = initialize_beam
        self.scheduler.grammar_manager.process_req_with_grammar.side_effect = (
            lambda _req: events.append("grammar") or False
        )
        self.scheduler._add_request_to_queue.side_effect = lambda _req: events.append(
            "queue"
        )

        self._run(validate_prompt)

        self.assertEqual(events, ["budget", "prompt", "beam", "grammar", "queue"])

    def test_invalid_prompt_does_not_initialize_beam(self):
        self._run(lambda _req, _max_input_len, _allow_auto_truncate: "too long")

        self.scheduler.beam_coordinator.validate_and_init.assert_not_called()
        self.req.set_finish_with_abort.assert_called_once_with("too long")
        self.scheduler._add_request_to_queue.assert_called_once_with(self.req)
        self.scheduler.grammar_manager.process_req_with_grammar.assert_not_called()

    def test_legacy_session_does_not_bypass_beam_validation(self):
        session = MagicMock(close_on_finish=False)
        session.create_req.return_value = self.req
        self.scheduler.session_controller = {"session-id": session}
        self.recv_req.session_params = SimpleNamespace(id="session-id")
        error = "Beam search is not supported for session requests."
        self.scheduler.beam_coordinator.validate_and_init.return_value = error

        with patch("sglang.srt.managers.scheduler.prepare_abort") as prepare_abort:
            self._run(lambda _req, _max_input_len, _allow_auto_truncate: None)

        self.scheduler.beam_coordinator.validate_and_init.assert_called_once_with(
            self.req, self.recv_req
        )
        prepare_abort.assert_called_once_with(
            self.req, error, status_code=HTTPStatus.BAD_REQUEST
        )
        self.scheduler.output_streamer.stream_output.assert_called_once_with(
            [self.req], self.req.return_logprob
        )
        self.scheduler.grammar_manager.process_req_with_grammar.assert_not_called()
        self.scheduler._add_request_to_queue.assert_not_called()


if __name__ == "__main__":
    unittest.main()

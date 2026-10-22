"""Scheduler admission: the speculative algorithm validates each request."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestSchedulerSpecRequestValidation(CustomTestCase):
    def _new_scheduler(self, spec_algorithm) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.model_config = SimpleNamespace(
            hf_eos_token_id={1},
            vocab_size=128,
        )
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        scheduler._record_kv_event_lora_name = MagicMock()
        scheduler.spec_algorithm = spec_algorithm
        scheduler.enable_overlap = False
        scheduler.init_req_max_new_tokens = MagicMock()
        scheduler._add_request_to_queue = MagicMock()
        return scheduler

    def _handle(self, scheduler: Scheduler, mm_inputs=None) -> MagicMock:
        recv_req = MagicMock(
            session_params=None,
            session_id=None,
            input_embeds=None,
            bootstrap_port=1,
            mm_inputs=mm_inputs,
        )
        req = MagicMock()
        with (
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=1,
            ),
            patch("sglang.srt.managers.scheduler.Req", return_value=req),
        ):
            scheduler.handle_generate_request(recv_req)
        return req

    def _assert_aborted(self, scheduler: Scheduler, req: MagicMock, error: str):
        req.set_finish_with_abort.assert_called_once_with(error)
        scheduler.init_req_max_new_tokens.assert_called_once_with(req)
        scheduler._add_request_to_queue.assert_called_once_with(req)

    def test_invalid_request_is_aborted_before_scheduler_admission(self):
        scheduler = self._new_scheduler(SpeculativeAlgorithm.UNO)
        error = "UNO request is unsupported."

        with patch(
            "sglang.srt.speculative.uno_validation.validate_uno_request",
            return_value=error,
        ) as validate_uno_request:
            req = self._handle(scheduler)

        validate_uno_request.assert_called_once_with(req)
        self._assert_aborted(scheduler, req, error)

    def test_plugin_algorithm_rejection_is_aborted_before_admission(self):
        """A plugin algorithm's validate_request reaches the same abort path,
        with the request's multimodal-ness, which ``req`` does not carry yet."""
        error = "Plugin algorithm cannot serve this request."
        seen = []

        class _Algo(CustomSpecAlgo):
            def validate_request(self, req, *, has_mm_inputs, enable_overlap):
                seen.append((has_mm_inputs, enable_overlap))
                return error

        scheduler = self._new_scheduler(_Algo("MY_SPEC", factory=MagicMock()))
        req = self._handle(scheduler, mm_inputs=MagicMock())

        self.assertEqual(seen, [(True, False)])
        self._assert_aborted(scheduler, req, error)


if __name__ == "__main__":
    unittest.main()

"""Scheduler containment for unsupported UNO requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestSchedulerUnoRequestValidation(CustomTestCase):
    def test_invalid_request_is_aborted_before_scheduler_admission(self):
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
        scheduler.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: False,
            is_uno=lambda: True,
        )
        scheduler.init_req_max_new_tokens = MagicMock()
        scheduler._add_request_to_queue = MagicMock()

        recv_req = MagicMock(
            session_params=None,
            session_id=None,
            input_embeds=None,
            bootstrap_port=1,
        )
        req = MagicMock()
        error = "UNO request is unsupported."

        with (
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=1,
            ),
            patch("sglang.srt.managers.scheduler.Req", return_value=req),
            patch(
                "sglang.srt.managers.scheduler.validate_uno_request",
                return_value=error,
            ) as validate_uno_request,
        ):
            scheduler.handle_generate_request(recv_req)

        validate_uno_request.assert_called_once_with(req)
        req.set_finish_with_abort.assert_called_once_with(error)
        scheduler.init_req_max_new_tokens.assert_called_once_with(req)
        scheduler._add_request_to_queue.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()

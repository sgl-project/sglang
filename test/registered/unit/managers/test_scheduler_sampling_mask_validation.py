"""Reject PD sampling-mask requests before entering transfer queues."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerSamplingMaskValidation(CustomTestCase):
    def test_disabled_pd_masks_return_bad_request_before_admission(self):
        """Neither PD role may queue mask requests without metadata buffers."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_session_radix_cache = False
                scheduler.model_config = SimpleNamespace(
                    hf_eos_token_id={1}, vocab_size=128
                )
                scheduler.disaggregation_mode = mode
                scheduler.disagg_metadata_buffers = SimpleNamespace(
                    enable_sampling_mask=False
                )
                scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
                scheduler.tokenizer = None
                scheduler.dllm_config = None
                scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
                scheduler.spec_algorithm = SimpleNamespace(
                    is_dflash_family=lambda: False,
                    is_uno=lambda: False,
                )
                scheduler._add_request_to_queue = MagicMock()
                scheduler.output_streamer = MagicMock()
                recv_req = MagicMock(
                    session_params=None,
                    session_id=None,
                    input_embeds=None,
                    bootstrap_port=1,
                    bootstrap_room=9,
                )
                req = MagicMock(return_sampling_mask=True, return_logprob=False)
                with (
                    patch(
                        "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                        return_value=1,
                    ),
                    patch("sglang.srt.managers.scheduler.Req", return_value=req),
                    patch("sglang.srt.managers.scheduler.prepare_abort") as abort,
                ):
                    scheduler.handle_generate_request(recv_req)
                abort.assert_called_once()
                self.assertIs(abort.call_args.args[0], req)
                self.assertIn(
                    "SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1",
                    abort.call_args.args[1],
                )
                self.assertEqual(
                    abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                )
                scheduler.output_streamer.stream_output.assert_called_once_with(
                    [req], False
                )
                scheduler._add_request_to_queue.assert_not_called()


if __name__ == "__main__":
    unittest.main()

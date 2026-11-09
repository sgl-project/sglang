"""Reject PD sampling-mask requests before entering transfer queues."""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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
                scheduler._record_kv_event_lora_name = MagicMock()
                scheduler.spec_algorithm = SimpleNamespace(
                    is_dflash_family=lambda: False,
                    is_uno=lambda: False,
                )
                scheduler._add_request_to_queue = MagicMock()
                scheduler.output_streamer = MagicMock()
                scheduler.tree_cache = MagicMock()
                scheduler.beam_coordinator = MagicMock()
                scheduler.hisparse_coordinator = None
                scheduler.uocr_runtime = None
                scheduler.send_abort_output = MagicMock()
                scheduler.init_request_retirer()
                recv_req = MagicMock(
                    session_params=None,
                    session_id=None,
                    input_embeds=None,
                    bootstrap_port=1,
                    bootstrap_room=9,
                )
                req = MagicMock(return_sampling_mask=True, return_logprob=False)
                req.exit_via = req.send_via = req.kv.retraction_backup = None
                req.kv.holds_kv = req.kv.holds_mamba = False
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
                self.assertEqual(
                    abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                )
                scheduler._add_request_to_queue.assert_not_called()

    def test_rejected_streaming_turn_leaves_session_idle(self):
        """A turn rejected after `create_req` must not wedge its streaming session."""
        for name in ("sampling_mask", "dllm"):
            with self.subTest(rejection=name):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_session_radix_cache = False
                scheduler.model_config = SimpleNamespace(
                    hf_eos_token_id={1}, vocab_size=128
                )
                scheduler.disaggregation_mode = DisaggregationMode.NULL
                scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
                scheduler.tokenizer = None
                scheduler.dllm_config = None
                scheduler.validate_dllm_request = lambda req: (
                    "dllm rejection" if name == "dllm" else None
                )
                scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
                scheduler._record_kv_event_lora_name = MagicMock()
                scheduler.spec_algorithm = SimpleNamespace(
                    is_dflash_family=lambda: False,
                    is_uno=lambda: False,
                    is_none=lambda: True,
                )
                scheduler.send_abort_output = MagicMock()
                scheduler.output_streamer = MagicMock()
                scheduler.tree_cache = MagicMock()
                scheduler.beam_coordinator = MagicMock()
                scheduler.hisparse_coordinator = None
                scheduler.uocr_runtime = None
                scheduler.init_request_retirer()
                scheduler.server_args = SimpleNamespace(sampling_mask_max_tokens=64)
                scheduler._create_generation_req = lambda recv_req, **kw: Req(**kw)
                scheduler.session_controller = SessionController(MagicMock())
                scheduler.session_controller.open(
                    SimpleNamespace(
                        session_id="s",
                        capacity_of_str_len=0,
                        streaming=True,
                        timeout=None,
                    )
                )
                session = scheduler.session_controller.get("s")
                recv_req = MagicMock(
                    rid="r",
                    input_ids=array("q", [5, 6, 7]),
                    input_text="",
                    mm_inputs=None,
                    session_id=None,
                    session_params=SimpleNamespace(
                        id="s",
                        rid=None,
                        offset=None,
                        replace=False,
                        drop_previous_output=False,
                    ),
                    # The default top_k is invalid together with a sampling mask.
                    sampling_params=SamplingParams(max_new_tokens=8),
                    return_sampling_mask=name == "sampling_mask",
                    return_logprob=False,
                    return_hidden_states=False,
                    pp_prefetch_ticketed=False,
                    bootstrap_port=1,
                )
                with (
                    patch(
                        "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                        return_value=1,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_retirer.make_abort_req"
                    ),
                ):
                    scheduler.handle_generate_request(recv_req)
                self.assertFalse(session._inflight)


if __name__ == "__main__":
    unittest.main()

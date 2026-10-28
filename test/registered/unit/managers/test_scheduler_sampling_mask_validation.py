"""Reject PD sampling-mask requests before entering transfer queues."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.runtime_context import get_context
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.scheduler_request_intake import make_request_intake

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSchedulerSamplingMaskValidation(CustomTestCase):
    def test_disabled_pd_masks_return_bad_request_before_admission(self):
        """Neither PD role may queue mask requests without metadata buffers."""
        enter_scope(self, published_topology())
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                buffers = SimpleNamespace(enable_sampling_mask=False)
                output_streamer = MagicMock()
                intake = make_request_intake(
                    model_config=SimpleNamespace(hf_eos_token_id={1}, vocab_size=128),
                    mode_owners=ModeOwners(
                        disagg_prefill=(
                            prefill_owner_shell(metadata_buffers=buffers)
                            if mode == DisaggregationMode.PREFILL
                            else None
                        ),
                        disagg_decode=(
                            decode_owner_shell(metadata_buffers=buffers)
                            if mode == DisaggregationMode.DECODE
                            else None
                        ),
                    ),
                    spec_algorithm=SimpleNamespace(
                        is_dflash_family=lambda: False,
                        is_uno=lambda: False,
                    ),
                    output_streamer=output_streamer,
                    request_retirer=RequestRetirer(
                        kv=MagicMock(),
                        beam_coordinator=MagicMock(),
                        hisparse_coordinator=None,
                        decode_offload_manager=None,
                        pending=PendingSteps(),
                        output_streamer=output_streamer,
                    ),
                )
                get_context().override("test", disaggregation_mode=mode.value)
                intake._maybe_namespace_elastic_radix_cache = MagicMock()
                intake.add_request_to_queue = MagicMock()
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
                        "sglang.srt.managers.scheduler_components.request_intake."
                        "BeamCoordinator.request_beam_width",
                        return_value=1,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_intake.Req",
                        return_value=req,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_retirer.prepare_abort"
                    ) as abort,
                ):
                    intake.handle_generate_request(recv_req)
                abort.assert_called_once()
                self.assertEqual(
                    abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                )
                output_streamer.stream_output.assert_called_once_with([req], False)
                intake.add_request_to_queue.assert_not_called()


if __name__ == "__main__":
    unittest.main()

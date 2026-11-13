"""routed_experts_start_len=-1 resolves on the expanded multimodal prompt."""

import functools
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT = [11, 12, 13, 14]
EXPANDED = list(range(100, 140))  # one image token expanded to many


def _scheduler():
    scheduler = MagicMock()
    scheduler.enable_session_radix_cache = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.validate_dllm_request.return_value = None
    scheduler.spec_algorithm.is_dflash_family.return_value = False
    scheduler.spec_algorithm.is_uno.return_value = False
    scheduler._try_apply_padded_mm_input_ids.return_value = False
    scheduler.max_req_input_len = 1 << 20
    scheduler._handle_generate_request = functools.partial(
        Scheduler._handle_generate_request, scheduler
    )
    return scheduler


def _request():
    return MagicMock(
        session_params=None,
        session_id=None,
        input_embeds=None,
        bootstrap_port=1,
        pp_prefetch_ticketed=False,
        return_logprob=False,
        logprob_start_len=-1,
        return_routed_experts=True,
        routed_experts_start_len=-1,
    )


class TestRoutedExpertsStartLenMultimodal(CustomTestCase):
    def test_minus_one_counts_the_expanded_multimodal_tokens(self):
        scheduler = _scheduler()
        req = MagicMock(
            origin_input_ids=list(PROMPT),
            origin_input_ids_unpadded=list(PROMPT),
            return_sampling_mask=False,
            is_prefill_only=False,
            return_logprob=False,
        )
        scheduler._create_generation_req.return_value = req
        with (
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=1,
            ),
            patch(
                "sglang.srt.managers.scheduler.duplicate_pad_mm_input_ids",
                return_value=list(EXPANDED),
            ),
            patch(
                "sglang.srt.managers.scheduler.validate_input_length",
                return_value=None,
            ),
            patch("sglang.srt.managers.scheduler.get_serving"),
            patch("sglang.srt.managers.scheduler.get_exec") as get_exec,
            patch("sglang.srt.managers.scheduler.get_device") as get_device,
            patch("sglang.srt.managers.scheduler.get_parallel"),
        ):
            get_exec.return_value.kernel.sampling_backend = "flashinfer"
            get_device.return_value.mlx_enable_sampling = False
            Scheduler.handle_generate_request(scheduler, _request())
        self.assertEqual(req.origin_input_ids, EXPANDED)
        self.assertEqual(req.routed_experts_start_len, len(EXPANDED) - 1)


if __name__ == "__main__":
    unittest.main()

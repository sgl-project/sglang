"""Reject PD requests the metadata buffers cannot carry before entering transfer queues."""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
    UpdateRejected,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.srt.speculative.spec_sampling_mask import (
    validate_spec_sampling_mask_request,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _PastSamplingChecks(Exception):
    """Admission reached its first step after the sampling-output checks."""


def _handle_request(mode, req):
    """Run admission on `req`: the abort mock, the queue mock, and whether the
    request got past the sampling-output checks."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = mode
    scheduler.disagg_metadata_buffers = SimpleNamespace(enable_sampling_mask=False)
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler._record_kv_event_lora_name = MagicMock()
    scheduler.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: False,
        is_uno=lambda: False,
        is_none=lambda: True,
    )
    scheduler.server_args = SimpleNamespace(sampling_mask_max_tokens=64)
    scheduler.init_req_max_new_tokens = MagicMock(side_effect=_PastSamplingChecks)
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
        mm_inputs=None,
        bootstrap_port=1,
        bootstrap_room=9,
    )
    if isinstance(req, MagicMock):
        req.exit_via = req.send_via = req.kv.retraction_backup = None
        req.kv.holds_kv = req.kv.holds_mamba = False
    passed = False
    with (
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch(
            "sglang.srt.managers.scheduler.prepare_abort", wraps=prepare_abort
        ) as abort,
    ):
        try:
            scheduler.handle_generate_request(recv_req)
        except _PastSamplingChecks:
            passed = True
    return abort, scheduler._add_request_to_queue, passed


def _spec_algorithm(name):
    return SpeculativeAlgorithm[name.upper()]


def _sampling_req(sampling_params, *, logprobs=False, mask=False):
    params = SamplingParams(max_new_tokens=8, **sampling_params)
    params.normalize(None)
    return Req(
        "r",
        "",
        array("q", [1, 2]),
        params,
        return_logprob=logprobs,
        return_sampling_logprobs=logprobs,
        return_sampling_mask=mask,
    )


class TestSamplingCapture(CustomTestCase):
    def test_only_filtered_sampling_is_captured(self):
        """An unfiltered request reuses its output logprobs, so neither flag may put
        it on the capture path; greedy, top-k, top-p and min-p each must."""
        for sampling_params, filtered in (
            ({}, False),
            ({"top_k": -1, "top_p": 1.0, "temperature": 0.7}, False),
            ({"top_k": 20}, True),
            ({"top_p": 0.9}, True),
            ({"min_p": 0.05}, True),
            ({"temperature": 0.0}, True),
        ):
            for logprobs, mask in ((True, False), (False, True), (False, False)):
                with self.subTest(params=sampling_params, logprobs=logprobs, mask=mask):
                    req = _sampling_req(sampling_params, logprobs=logprobs, mask=mask)
                    captured = filtered and (logprobs or mask)
                    self.assertEqual(req.captures_sampling, captured)
                    self.assertEqual(req.sampling_mask_rows is not None, captured)

    def test_speculative_decoding_admits_what_stays_exact(self):
        """Output logprobs are exact sampling logprobs under any speculative
        algorithm while acceptance is unrelaxed; a filtered request also needs the
        DFlash capture."""
        eagle = _spec_algorithm("eagle")
        dflash = _spec_algorithm("dflash")
        unfiltered = _sampling_req({}, logprobs=True, mask=True)
        greedy = _sampling_req({"temperature": 0.0}, logprobs=True)

        with get_context().override_server_args(speculative_accept_threshold_acc=1.0):
            self.assertIsNone(validate_spec_sampling_mask_request(unfiltered, eagle))
            self.assertIn(
                "top-k/top-p/min-p", validate_spec_sampling_mask_request(greedy, eagle)
            )
            self.assertIsNone(validate_spec_sampling_mask_request(greedy, dflash))
            with envs.SGLANG_SIMULATE_ACC_LEN.override(2):
                self.assertIsNotNone(
                    validate_spec_sampling_mask_request(unfiltered, eagle)
                )
        with get_context().override_server_args(speculative_accept_threshold_acc=0.9):
            for algorithm in (eagle, dflash):
                self.assertIsNotNone(
                    validate_spec_sampling_mask_request(unfiltered, algorithm)
                )

    def test_speculative_decoding_refuses_platforms_whose_verify_is_greedy(self):
        """On CPU, XPU and ROCm without rejection sampling EAGLE verify commits
        argmax for a sampled row, so an unfiltered request would get a logprob
        for a token that was never drawn from that distribution."""
        eagle = _spec_algorithm("eagle")
        unfiltered = _sampling_req({"temperature": 0.7}, logprobs=True, mask=True)
        for platform, use_rejection_sampling, refused in (
            ("cuda", False, False),
            ("cpu", False, True),
            ("cpu", True, True),
            ("xpu", False, True),
            ("hip", False, True),
            ("hip", True, False),
        ):
            with (
                self.subTest(platform=platform, rejection=use_rejection_sampling),
                patch.multiple(
                    eagle_utils,
                    _is_cpu=platform == "cpu",
                    _is_xpu=platform == "xpu",
                    _is_hip=platform == "hip",
                ),
                get_context().override_server_args(
                    speculative_accept_threshold_single=1.0,
                    speculative_accept_threshold_acc=1.0,
                    speculative_use_rejection_sampling=use_rejection_sampling,
                ),
            ):
                error_msg = validate_spec_sampling_mask_request(unfiltered, eagle)
                if refused:
                    self.assertIn("argmax", error_msg)
                else:
                    self.assertIsNone(error_msg)

    def test_speculative_decoding_refuses_plugin_algorithms(self):
        """A plugin algorithm verifies with its own worker, which may commit argmax
        or relax acceptance. Neither flag may be admitted for it, even for an
        unfiltered request, while every in-tree algorithm that verifies through
        eagle_sample is."""
        unfiltered = _sampling_req({"temperature": 0.7}, logprobs=True, mask=True)
        with (
            patch.multiple(eagle_utils, _is_cpu=False, _is_xpu=False, _is_hip=False),
            patch(
                "sglang.srt.speculative.spec_sampling_mask."
                "is_dflash_sampling_verify_available",
                return_value=True,
            ),
            get_context().override_server_args(
                speculative_accept_threshold_single=1.0,
                speculative_accept_threshold_acc=1.0,
            ),
        ):
            self.assertIn(
                "this speculative decoding algorithm",
                validate_spec_sampling_mask_request(
                    unfiltered, CustomSpecAlgo("PLUGIN", lambda server_args: object)
                ),
            )
            for name in (
                "eagle",
                "eagle3",
                "frozen_kv_mtp",
                "standalone",
                "ngram",
                "uno",
                "dflash",
                "dspark",
            ):
                with self.subTest(algorithm=name):
                    self.assertIsNone(
                        validate_spec_sampling_mask_request(
                            unfiltered, _spec_algorithm(name)
                        )
                    )

    def test_threshold_update_is_refused_where_admission_required_one(self):
        """A request admitted because both acceptance thresholds were 1.0 must not
        see one lowered while it runs, under every algorithm that reads them;
        DSpark and plain decoding do not."""
        running = _sampling_req({}, logprobs=True)
        for name, refused in (
            ("eagle", True),
            ("dflash", True),
            ("dspark", False),
            ("none", False),
        ):
            controller = SchedulerInternalStateController(
                SimpleNamespace(
                    spec_algorithm=_spec_algorithm(name),
                    request_lists=lambda: {"running": [running]},
                ),
                record_step_time=False,
            )
            for key in (
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ):
                with self.subTest(algorithm=name, key=key):
                    controller._validate_accept_threshold(key, 1.0)
                    if refused:
                        with self.assertRaises(UpdateRejected):
                            controller._validate_accept_threshold(key, 0.9)
                    else:
                        controller._validate_accept_threshold(key, 0.9)


class TestSchedulerSamplingMaskValidation(CustomTestCase):
    def test_disabled_pd_masks_return_bad_request_before_admission(self):
        """Neither PD role may queue mask requests without metadata buffers."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                abort, add_to_queue, _ = _handle_request(
                    mode, MagicMock(captures_sampling=True, return_logprob=False)
                )
                abort.assert_called_once()
                self.assertEqual(
                    abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                )
                add_to_queue.assert_not_called()

    def test_pd_top_logprobs_beyond_capacity_return_bad_request_before_admission(
        self,
    ):
        capacity = 8
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            for top_logprobs_num in (capacity, capacity + 1):
                with (
                    self.subTest(mode=mode, top_logprobs_num=top_logprobs_num),
                    get_context().override_server_args(
                        disaggregation_max_top_logprobs_num=capacity
                    ),
                ):
                    req = MagicMock(captures_sampling=True, return_logprob=True)
                    req.logprob.top_logprobs_num = top_logprobs_num
                    abort, add_to_queue, _ = _handle_request(mode, req)
                    abort.assert_called_once()
                    self.assertIn(
                        (
                            "top_logprobs_num"
                            if top_logprobs_num > capacity
                            else "return_sampling_mask"
                        ),
                        abort.call_args.args[1],
                    )
                    self.assertEqual(
                        abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                    )
                    add_to_queue.assert_not_called()

    def test_top_k_cap_applies_to_captured_mask_requests_only(self):
        """The cap bounds the captured support. An unfiltered mask request is not
        captured and returns empty rows, so its whole-vocabulary top_k must pass;
        so must a filtered logprobs-only request, which captures no support."""
        for sampling_params, flags, rejected in (
            ({}, {"mask": True}, False),
            ({"top_k": 20}, {"mask": True}, False),
            ({"top_p": 0.9}, {"logprobs": True}, False),
            ({"top_p": 0.9}, {"mask": True}, True),
            ({"top_k": 65}, {"mask": True}, True),
        ):
            with self.subTest(params=sampling_params, flags=flags):
                abort, _, passed = _handle_request(
                    DisaggregationMode.NULL, _sampling_req(sampling_params, **flags)
                )
                self.assertEqual(passed, not rejected)
                if rejected:
                    self.assertIn("return_sampling_mask", abort.call_args.args[1])
                else:
                    abort.assert_not_called()

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
                    # Top-p without a finite top_k cannot return a sampling mask.
                    sampling_params=SamplingParams(max_new_tokens=8, top_p=0.9),
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

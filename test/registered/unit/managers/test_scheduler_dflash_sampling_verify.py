"""Reject non-greedy requests that DFLASH would verify greedily, at admission."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_KERNEL = "sglang.srt.speculative.dflash_worker_v2.is_dflash_sampling_verify_available"


class _Admitted(Exception):
    pass


def _finish_abort(req, message, status_code=None):
    """Leave the finish reason a real prepare_abort sets; the retirer reads it."""
    req.finished_reason = FINISH_ABORT(message, status_code)


def _dflash_worker(
    selector=False,
    selector_sampling=False,
    lilicorr=False,
    lilicorr_sampling=False,
    domino=False,
):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.selector = object() if selector else None
    worker._selector_sampling_enabled = selector_sampling
    worker.lilicorr = object() if lilicorr else None
    worker._lilicorr_sampling_enabled = lilicorr_sampling
    worker._is_domino = domino
    return worker


def _greedy_verify_only(spec_algorithm, draft_worker, mode, kernel=False):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = spec_algorithm
    scheduler.draft_worker = draft_worker
    with (
        patch(
            "sglang.srt.managers.scheduler.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=mode.value),
        ),
        patch(_KERNEL, return_value=kernel),
    ):
        scheduler.init_dflash_greedy_verify_only()
    return scheduler.dflash_greedy_verify_only


def _handle_request(greedy_verify_only, top_k, temperature=1.0):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dflash_greedy_verify_only = greedy_verify_only
    scheduler.enable_overlap = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler._record_kv_event_lora_name = MagicMock()
    scheduler.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: True,
        is_uno=lambda: False,
    )
    # Admitted requests stop at the multimodal step.
    scheduler._get_multimodal_inputs = MagicMock(side_effect=_Admitted)
    scheduler.output_streamer = MagicMock()
    scheduler.tree_cache = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.hisparse_coordinator = None
    scheduler.init_request_retirer()
    recv_req = MagicMock(
        mm_inputs=object(), session_params=None, session_id=None, input_embeds=None
    )
    req = MagicMock(
        return_sampling_mask=False,
        return_logprob=False,
        return_hidden_states=False,
        sampling_params=MagicMock(temperature=temperature, top_k=top_k),
    )
    req.exit_via = req.send_via = req.kv.retraction_backup = None
    req.session = req.finished_reason = None
    req.kv.holds_kv = req.kv.holds_mamba = False
    with (
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch(
            "sglang.srt.managers.scheduler.prepare_abort", side_effect=_finish_abort
        ) as abort,
    ):
        scheduler.handle_generate_request(recv_req)
    return abort


class TestSchedulerDFlashSamplingVerify(CustomTestCase):
    def test_supports_sampling_verify_matches_the_accept_path(self):
        """Non-greedy rows fall back to argmax without a sampled head or the kernel."""
        cases = [
            # Ascend keeps selector sampling off.
            (dict(selector=True), True, False),
            (dict(selector=True, selector_sampling=True), False, True),
            (dict(lilicorr=True, lilicorr_sampling=True), False, True),
            (dict(lilicorr=True), False, False),
            ({}, False, False),
            ({}, True, True),
            # Domino proposes by greedy rollout and leaves both sampled heads unused.
            (dict(domino=True, selector=True, selector_sampling=True), False, False),
            (dict(domino=True, lilicorr=True, lilicorr_sampling=True), False, False),
            (dict(domino=True, selector=True, selector_sampling=True), True, True),
        ]
        for worker_kwargs, kernel, supported in cases:
            with (
                self.subTest(worker=worker_kwargs, kernel=kernel),
                patch(_KERNEL, return_value=kernel),
            ):
                self.assertEqual(
                    _dflash_worker(**worker_kwargs).supports_sampling_verify(),
                    supported,
                )

    def test_greedy_verify_only_is_set_where_dflash_verifies(self):
        dflash = SpeculativeAlgorithm.DFLASH
        for mode in (DisaggregationMode.NULL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                self.assertTrue(_greedy_verify_only(dflash, _dflash_worker(), mode))
                self.assertFalse(
                    _greedy_verify_only(dflash, _dflash_worker(), mode, kernel=True)
                )
        # A non-last PP stage of a PD prefill server holds no draft to query.
        no_draft = DFlashWorkerV2.__new__(DFlashWorkerV2)
        self.assertFalse(
            _greedy_verify_only(dflash, no_draft, DisaggregationMode.PREFILL)
        )
        self.assertFalse(
            _greedy_verify_only(
                SpeculativeAlgorithm.DSPARK, None, DisaggregationMode.NULL
            )
        )

    def test_non_greedy_request_returns_bad_request(self):
        """A request with default sampling params is non-greedy and gets a 400."""
        abort = _handle_request(True, top_k=TOP_K_ALL)
        abort.assert_called_once()
        self.assertEqual(abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST)
        self.assertIn(
            "(temperature=0 or top_k=1), got temperature=1.0", abort.call_args.args[1]
        )

    def test_greedy_or_supported_request_is_admitted(self):
        for greedy_verify_only, top_k in ((True, 1), (False, TOP_K_ALL)):
            with (
                self.subTest(greedy_verify_only=greedy_verify_only, top_k=top_k),
                self.assertRaises(_Admitted),
            ):
                _handle_request(greedy_verify_only, top_k=top_k)


if __name__ == "__main__":
    unittest.main()

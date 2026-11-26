"""DFLASH refuses to start where it would verify non-greedy requests greedily, unless opted in."""

import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_KERNEL = "sglang.srt.speculative.dflash_worker_v2.is_dflash_sampling_verify_available"
_LOGGER = "sglang.srt.speculative.dflash_worker_v2"
_FLAG = "--speculative-dflash-allow-greedy-verification"


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


def _check(worker, *, mode="null", opt_in=False, kernel=False, tp_size=1, tp_rank=0):
    with (
        get_context().override_server_args(
            disaggregation_mode=mode,
            speculative_dflash_allow_greedy_verification=opt_in,
            tp_size=tp_size,
        ),
        get_parallel().override(tp_rank=tp_rank),
        patch(_KERNEL, return_value=kernel),
    ):
        worker._check_sampling_verify_support()


class TestDFlashSamplingVerify(CustomTestCase):
    def test_supports_sampling_verify_matches_the_accept_path(self):
        """Non-greedy rows fall back to argmax without a sampled head or the kernel."""
        cases = [
            # Ascend: selector sampling off and no kernel.
            (dict(selector=True), False, False),
            # A selector draft without selector sampling verifies through the kernel.
            (dict(selector=True), True, True),
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
                    _dflash_worker(**worker_kwargs)._supports_sampling_verify(),
                    supported,
                )

    def test_refuses_to_start_where_it_would_verify_greedily(self):
        for mode in ("null", "decode"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, _FLAG),
            ):
                _check(_dflash_worker(), mode=mode)

    def test_opt_in_starts_with_one_warning_on_tp_rank_0(self):
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            _check(_dflash_worker(), opt_in=True)
        self.assertEqual(len(logs.records), 1)
        with self.assertNoLogs(_LOGGER):
            _check(_dflash_worker(), opt_in=True, tp_size=2, tp_rank=1)

    def test_server_that_verifies_by_sampling_starts_without_a_log(self):
        # A selector draft verifies by sampling without the kernel.
        cases = [({}, True), (dict(selector=True, selector_sampling=True), False)]
        for opt_in in (False, True):
            for worker_kwargs, kernel in cases:
                with (
                    self.subTest(worker=worker_kwargs, kernel=kernel, opt_in=opt_in),
                    self.assertNoLogs(_LOGGER),
                ):
                    _check(
                        _dflash_worker(**worker_kwargs), kernel=kernel, opt_in=opt_in
                    )

    def test_pd_prefill_server_starts_without_a_log(self):
        # A PD prefill server never verifies.
        for opt_in in (False, True):
            with self.subTest(opt_in=opt_in), self.assertNoLogs(_LOGGER):
                _check(_dflash_worker(), mode="prefill", opt_in=opt_in)


if __name__ == "__main__":
    unittest.main()

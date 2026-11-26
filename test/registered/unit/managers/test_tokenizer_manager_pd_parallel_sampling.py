"""TokenizerManager rejects parallel sampling (n > 1) under PD disaggregation.

The n > 1 fan-out copies the first tokenized request for every sample and only
regenerates the rid, so all n samples (and the cache warm-up) carry the same
bootstrap_room. On a PD prefill or decode that means n senders or receivers
share one handshake room: their statuses bleed into each other and the first
to conclude clears the room under the others. Rejecting the request up front
gives the client a 400 instead of n racing transfers.

A request that opens no handshake is let through: one with no bootstrap host,
which the prefill answers alone, and any request on the fake transfer backend.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PastTheGuard(Exception):
    """Raised by the stubbed state init: the request got past the n > 1 check."""


def _make_tm(mode: DisaggregationMode) -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.disaggregation_mode = mode
    tm.auto_create_handle_loop = MagicMock()
    tm.enable_priority_scheduling = False
    tm._init_req_state = MagicMock(side_effect=_PastTheGuard)
    return tm


def _request(n: int, bootstrap_host=None) -> GenerateReqInput:
    return GenerateReqInput(
        input_ids=[1, 2, 3],
        sampling_params={"n": n, "max_new_tokens": 4},
        bootstrap_host=bootstrap_host,
    )


def _submit(
    tm: TokenizerManager, obj: GenerateReqInput, transfer_backend: str = "mooncake"
) -> None:
    async def drive():
        await tm.generate_request(obj).__anext__()

    disagg = SimpleNamespace(disaggregation_transfer_backend=transfer_backend)
    with patch("sglang.srt.managers.tokenizer_manager.get_disagg", return_value=disagg):
        asyncio.run(drive())


class TestParallelSamplingUnderPD(CustomTestCase):
    def test_n_gt_1_is_rejected_on_prefill_and_decode(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode.value):
                with self.assertRaisesRegex(ValueError, "n > 1"):
                    _submit(_make_tm(mode), _request(n=2, bootstrap_host="10.0.0.1"))

    def test_n_gt_1_passes_without_disaggregation(self):
        with self.assertRaises(_PastTheGuard):
            _submit(_make_tm(DisaggregationMode.NULL), _request(n=2))

    def test_n_eq_1_passes_under_disaggregation(self):
        with self.assertRaises(_PastTheGuard):
            _submit(
                _make_tm(DisaggregationMode.PREFILL),
                _request(n=1, bootstrap_host="10.0.0.1"),
            )

    def test_n_gt_1_hostless_passes_on_prefill(self):
        # A request naming no decode is answered by the prefill alone and opens
        # no handshake (test_disaggregation_basic sends such an n=2 request).
        with self.assertRaises(_PastTheGuard):
            _submit(_make_tm(DisaggregationMode.PREFILL), _request(n=2))

    def test_n_gt_1_passes_on_fake_transfer_backend(self):
        with self.assertRaises(_PastTheGuard):
            _submit(
                _make_tm(DisaggregationMode.DECODE),
                _request(n=2, bootstrap_host="10.0.0.1"),
                transfer_backend="fake",
            )


if __name__ == "__main__":
    unittest.main()

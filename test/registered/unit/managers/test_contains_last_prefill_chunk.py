"""`contains_last_prefill_chunk` must mean "some request here consumes the
sampled token".

The flag gates the PP output-communication skip: a batch of nothing but middle
chunks produces tokens nobody reads, so the send/recv can be dropped. It used
to be derived from the batch size (`len(can_run_list) != 1`), which is only a
proxy for "the chunked request is the sole member" and misreads the parked case
-- `chunked_req` set but absent from the batch, so the request that *is* there
does finish its prefill and does consume the token.
"""

import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import _pp_can_skip_output_comm
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


class _Batch:
    """Minimal stand-in for ScheduleBatch as _pp_can_skip_output_comm reads it."""

    def __init__(self, *, reqs, contains_last_prefill_chunk):
        self.reqs = reqs
        self.contains_last_prefill_chunk = contains_last_prefill_chunk
        self.forward_mode = ForwardMode.EXTEND
        self.return_logprob = False


def _flag(chunked_req, can_run_list) -> bool:
    """Run the scheduler's real derivation."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = chunked_req
    return Scheduler._contains_last_prefill_chunk(scheduler, can_run_list)


class TestContainsLastPrefillChunk(CustomTestCase):
    def test_pure_middle_chunk_batch_is_false(self):
        # The only member is the chunked request: nothing consumes the token,
        # which is the case the PP skip exists for.
        chunked = _Req("chunked")
        self.assertFalse(_flag(chunked, [chunked]))

    def test_chunked_req_alongside_others_is_true(self):
        chunked, other = _Req("chunked"), _Req("other")
        self.assertTrue(_flag(chunked, [chunked, other]))

    def test_no_chunked_req_is_true(self):
        self.assertTrue(_flag(None, [_Req("a")]))
        self.assertTrue(_flag(None, [_Req("a"), _Req("b")]))

    def test_parked_chunked_req_is_true(self):
        # Regression. `chunked_req` is set but was parked -- it is not in the
        # batch, so the single request present finishes its prefill and
        # consumes the sampled token. The old size-based derivation returned
        # False here, so PP would skip output comm and leave that request with
        # placeholder zeros instead of its token.
        chunked, present = _Req("parked"), _Req("present")
        self.assertTrue(_flag(chunked, [present]))

    def test_matches_size_derivation_except_when_parked(self):
        # Equivalence sweep against the previous derivation, showing the only
        # behavior change is the parked case.
        def old(chunked_req, can_run_list):
            return chunked_req is None or len(can_run_list) != 1

        chunked, a, b = _Req("chunked"), _Req("a"), _Req("b")
        for chunked_req, can_run_list in [
            (None, [a]),
            (None, [a, b]),
            (chunked, [chunked]),
            (chunked, [chunked, a]),
            (chunked, [chunked, a, b]),
            (chunked, [a, b]),  # parked, but >1 member: both say True
        ]:
            with self.subTest(chunked=chunked_req, batch=can_run_list):
                self.assertEqual(
                    old(chunked_req, can_run_list), _flag(chunked_req, can_run_list)
                )

        # The single divergence: parked, exactly one other request.
        self.assertFalse(old(chunked, [a]))
        self.assertTrue(_flag(chunked, [a]))


class TestPPSkipOutputComm(CustomTestCase):
    """The consumer: dropping the `len(batch.reqs) == 1` proxy must not let the
    skip fire for a batch that has a finishing request in it."""

    def setUp(self):
        super().setUp()
        patcher = patch(
            "sglang.srt.managers.scheduler_pp_mixin.envs."
            "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_skips_pure_middle_chunk_batch(self):
        batch = _Batch(reqs=[_Req("chunk")], contains_last_prefill_chunk=False)
        self.assertTrue(_pp_can_skip_output_comm(batch))

    def test_skips_multi_middle_chunk_batch(self):
        # Newly reachable: with the size proxy gone, a batch of several middle
        # chunks is also pure and can skip. Under a single chunked slot this
        # cannot arise, so it is a no-op today.
        batch = _Batch(reqs=[_Req("c1"), _Req("c2")], contains_last_prefill_chunk=False)
        self.assertTrue(_pp_can_skip_output_comm(batch))

    def test_does_not_skip_when_a_request_finishes(self):
        batch = _Batch(reqs=[_Req("done")], contains_last_prefill_chunk=True)
        self.assertFalse(_pp_can_skip_output_comm(batch))

    def test_does_not_skip_mixed_chunk_batch(self):
        # Mixed chunk merges decode requests in after the flag is computed, and
        # the scheduler sets the flag True there; those requests consume tokens.
        batch = _Batch(
            reqs=[_Req("chunk"), _Req("decode")], contains_last_prefill_chunk=True
        )
        self.assertFalse(_pp_can_skip_output_comm(batch))

    def test_does_not_skip_when_logprobs_requested(self):
        batch = _Batch(reqs=[_Req("chunk")], contains_last_prefill_chunk=False)
        batch.return_logprob = True
        self.assertFalse(_pp_can_skip_output_comm(batch))

    def test_does_not_skip_decode_batch(self):
        batch = _Batch(reqs=[_Req("d")], contains_last_prefill_chunk=False)
        batch.forward_mode = ForwardMode.DECODE
        self.assertFalse(_pp_can_skip_output_comm(batch))


if __name__ == "__main__":
    unittest.main()

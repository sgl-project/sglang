import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    """Minimal stand-in for Req. Deliberately defines no __eq__/__hash__ so
    membership tests compare by identity, exactly as the real Req does."""

    def __init__(self, rid: str):
        self.rid = rid
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False
        self.finished_output = None

    def finished(self):
        return False


def _make_streamer() -> SchedulerOutputStreamer:
    # The dataclass is slots=True, so instance attributes cannot be added or
    # patched after construction; build it through the real constructor.
    return SchedulerOutputStreamer(
        send_to_detokenizer=MagicMock(),
        tree_cache=MagicMock(),
        ps=MagicMock(dp_rank=0, attn_tp_rank=0),
        server_args=MagicMock(),
        is_generation=True,
        spec_algorithm=MagicMock(),
        disaggregation_mode=MagicMock(),
        enable_hicache_storage=lambda: False,
        rust_server=None,
    )


def _accepted_rids(reqs, skip_reqs) -> list[str]:
    """Run the generation stream path and report which reqs were streamed."""
    streamer = _make_streamer()
    accepted = []
    acc = MagicMock()
    acc.accept.side_effect = lambda req: accepted.append(req.rid)
    acc.to_payload.return_value = None

    with patch(
        "sglang.srt.managers.scheduler_components.output_streamer."
        "_GenerationStreamAccumulator",
        return_value=acc,
    ), patch.object(
        SchedulerOutputStreamer, "_maybe_log_time_stats", lambda self, *, req: None
    ):
        streamer._stream_output_generation(reqs, False, skip_reqs)
    return accepted


class TestOutputStreamerSkipReqs(CustomTestCase):
    def setUp(self):
        super().setUp()
        # _stream_output_generation reads the published serving bag
        # (stream_interval); publish defaults for the whole test.
        self._ctx = get_context().override_server_args()
        self._ctx.__enter__()
        self.addCleanup(lambda: self._ctx.__exit__(None, None, None))

    def test_skips_every_mid_prefill_request(self):
        # Regression: `skip_req` used to be a single Req, so with several
        # mid-prefill requests in one batch only the last was suppressed and the
        # rest were streamed despite having appended no new token. DLLM already
        # stages many chunked requests at once, so this fired in production.
        reqs = [_Req("chunk-a"), _Req("chunk-b"), _Req("done")]
        accepted = _accepted_rids(reqs, skip_reqs=[reqs[0], reqs[1]])
        self.assertEqual(accepted, ["done"])

    def test_no_skips_streams_everything(self):
        reqs = [_Req("a"), _Req("b")]
        self.assertEqual(_accepted_rids(reqs, skip_reqs=[]), ["a", "b"])

    def test_single_skip_matches_previous_behavior(self):
        # The old scalar case must be unchanged.
        reqs = [_Req("a"), _Req("mid"), _Req("b")]
        accepted = _accepted_rids(reqs, skip_reqs=[reqs[1]])
        self.assertEqual(accepted, ["a", "b"])

    def test_skip_is_by_identity_not_equality(self):
        # Two distinct requests could compare equal under a future __eq__;
        # skipping must remove only the exact object handed in.
        keep = _Req("same-rid")
        skip = _Req("same-rid")
        accepted = _accepted_rids([keep, skip], skip_reqs=[skip])
        self.assertEqual(len(accepted), 1)

    def test_flags_ignore_skipped_requests(self):
        # The return_* flags drive payload allocation; a skipped request must
        # not switch them on, or the payload reserves space nothing fills.
        mid = _Req("mid")
        mid.return_hidden_states = True
        plain = _Req("plain")
        streamer = _make_streamer()

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            acc = MagicMock()
            acc.to_payload.return_value = None
            return acc

        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer."
            "_GenerationStreamAccumulator",
            side_effect=_capture,
        ), patch.object(
            SchedulerOutputStreamer, "_maybe_log_time_stats", lambda self, *, req: None
        ):
            streamer._stream_output_generation([mid, plain], False, [mid])

        self.assertFalse(captured["return_hidden_states"])


if __name__ == "__main__":
    unittest.main()

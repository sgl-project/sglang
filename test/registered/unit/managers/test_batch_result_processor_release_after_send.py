"""SchedulerBatchResultProcessor.process_batch_result_decode: a request that
finishes in a decode step is sent first, then released (its prefix-cache insert),
in the same free group and before the step returns to the scheduler. Releases
the send depends on, or that a plugin depends on, keep their place before it."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BRP = "sglang.srt.managers.scheduler_components.batch_result_processor"
RETIRER = "sglang.srt.managers.scheduler_components.request_retirer"


class _DecodeReq(FakeReq):
    """Finishes once it holds ``stop_at`` output tokens, with ``reason``; a
    ``finished_len`` of None is the case the send fills in."""

    def __init__(self, rid, *, stop_at=None, reason=None, known_len=True):
        super().__init__(
            rid=rid,
            output_ids=[1],
            finished_len=None,
            finished_reason=None,
            is_retracted=False,
            return_logprob=False,
            return_sampling_mask=False,
            return_hidden_states=False,
            grammar=None,
            require_reasoning=False,
            beam_group=None,
            multimodal_inputs=None,
            customized_info=None,
            session=None,
            time_stats=Mock(),
            kv=SimpleNamespace(
                holds_kv=True, holds_mamba=False, retraction_backup=None
            ),
        )
        self._stop_at = stop_at
        self._reason = reason
        self._known_len = known_len

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self, new_accept_len):
        if self._stop_at is not None and len(self.output_ids) >= self._stop_at:
            self.finished_reason = self._reason or FINISH_LENGTH(self._stop_at)
            if self._known_len:
                self.finished_len = self._stop_at

    def get_mm_token_counts(self):
        return None

    def release_mm_inputs_on_abort(self):
        pass


def _run(case, reqs, *, auxiliary=False):
    """One decode step over ``reqs``; returns the log of the free group, sends,
    auxiliary consumption and releases."""
    override = get_context().override_server_args()
    override.install()
    case.addCleanup(override.restore)
    log = []
    allocator = Mock()
    allocator.free_group_begin.side_effect = lambda: log.append("free_group_begin")
    allocator.free_group_end.side_effect = lambda: log.append("free_group_end")
    streamer = Mock()
    streamer.stream_output.side_effect = lambda reqs, *_: log.append(
        ("send", [(r.rid, r.finished_len) for r in reqs])
    )
    metrics_reporter = Mock(num_generated_tokens=0, forward_ct_decode=0)
    processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        model_config=SimpleNamespace(
            think_end_ids=None, self_opened_think_start_ids=None
        ),
        token_to_kv_pool_allocator=allocator,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        kv_weight_version_tracker=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=SimpleNamespace(),
        logprob_result_processor=None,
        output_streamer=streamer,
        beam_coordinator=Mock(),
        request_retirer=release_only_retirer(),
        abort_request=lambda *args, **kwargs: None,
        still_in_batch=lambda req: True,
        prefill_launched_reqs=None,
    )
    batch = SimpleNamespace(
        reqs=reqs,
        return_logprob=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        batch_size=lambda: len(reqs),
        forward_iter=1,
        mamba_track_mask_cpu=None,
    )
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=torch.tensor([8] * len(reqs)),
    )
    if auxiliary:
        result.auxiliary_host_output = Mock()
        result.auxiliary_host_output.consume.side_effect = lambda *_: log.append(
            "auxiliary"
        )
    with (
        patch(
            f"{RETIRER}.release_kv_cache",
            lambda req, tree, *, checkpoint: log.append(
                ("release", req.rid, checkpoint)
            ),
        ),
        patch.object(SchedulerBatchResultProcessor, "_mamba_prefix_cache_update"),
        patch.object(SchedulerBatchResultProcessor, "_maybe_collect_routed_experts"),
        patch.object(SchedulerBatchResultProcessor, "_maybe_collect_indexer_topk"),
    ):
        processor.process_batch_result_decode(batch, result)
    return log


class TestReleaseAfterSend(CustomTestCase):
    def test_finished_request_is_sent_before_its_release(self):
        """The final output does not wait for the prefix-cache insert, and the
        insert still runs, with the same arguments, before the free group ends
        and the step returns."""
        log = _run(self, [_DecodeReq("done", stop_at=2), _DecodeReq("running")])
        self.assertEqual(
            log,
            [
                "free_group_begin",
                ("send", [("done", 2), ("running", None)]),
                ("release", "done", True),
                "free_group_end",
            ],
        )

    def test_releases_that_must_precede_the_send_keep_their_order(self):
        cases = {
            # The send fills finished_len, which the release reads.
            "unknown_finished_len": _DecodeReq("r", stop_at=2, known_len=False),
            # An abort's release also detaches the request.
            "abort": _DecodeReq("r", stop_at=2, reason=FINISH_ABORT()),
        }
        for name, req in cases.items():
            with self.subTest(name):
                log = _run(self, [req])
                steps = [e[0] if isinstance(e, tuple) else e for e in log]
                self.assertEqual(
                    steps, ["free_group_begin", "release", "send", "free_group_end"]
                )

    def test_auxiliary_output_sees_the_request_released(self):
        """A consumer of auxiliary output defers a finished response only after
        its release (Scheduler.defer_response), so that step keeps the order."""
        log = _run(self, [_DecodeReq("done", stop_at=2)], auxiliary=True)
        self.assertEqual(
            log,
            [
                "free_group_begin",
                ("release", "done", True),
                "auxiliary",
                ("send", [("done", 2)]),
                "free_group_end",
            ],
        )


if __name__ == "__main__":
    unittest.main()

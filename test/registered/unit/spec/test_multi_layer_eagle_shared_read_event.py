"""Unit tests for the multi-layer EAGLE final shared-read event."""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
    OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
    MultiLayerEagleWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RAW_BS = 2
RAW_NUM_TOKENS = 3
PADDED_BS = 6


class _StubEvent:
    def __init__(self):
        self.record_count = 0

    def record(self):
        self.record_count += 1


class _StubDeviceModule:
    def __init__(self):
        self.events = []

    def Event(self):
        event = _StubEvent()
        self.events.append(event)
        return event


def _make_worker(num_steps):
    draft_worker = MultiLayerEagleDraftWorker.__new__(MultiLayerEagleDraftWorker)
    draft_worker.draft_runner_list = [
        SimpleNamespace(shared_read_done_event=None, device_timer=None)
        for _ in range(num_steps)
    ]
    worker = MultiLayerEagleWorkerV2.__new__(MultiLayerEagleWorkerV2)
    worker._draft_worker = draft_worker
    return worker


def _make_step_runner(draft_worker, step):
    out = SimpleNamespace(
        next_token_logits=list(range(PADDED_BS)),
        hidden_states=list(range(PADDED_BS)),
        topk_p=list(range(PADDED_BS)),
        topk_index=list(range(PADDED_BS)),
    )
    runner = SimpleNamespace(
        step=step,
        raw_bs=None,
        out=out,
        device_module=_StubDeviceModule(),
        model_runner=draft_worker.mtp_model_runner(step),
        deepep_adapter=SimpleNamespace(replay=lambda: None),
    )
    runner.replay = lambda bs, seq_lens_sum, spec_info, seq_lens_cpu: out
    return runner


def _prepare_common(composite, num_steps):
    worker = _make_worker(num_steps)
    composite.runners = [
        _make_step_runner(worker._draft_worker, step) for step in range(num_steps)
    ]
    composite.speculative_num_steps = num_steps
    composite.raw_bs = RAW_BS
    composite.bs = RAW_BS
    composite.raw_num_tokens = RAW_NUM_TOKENS
    composite.seq_lens_sum = RAW_BS
    composite.seq_lens_cpu = None
    composite._replay_spec_info = object()
    return worker


def _make_per_step_composite(num_steps):
    composite = MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.__new__(
        MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
    )
    worker = _prepare_common(composite, num_steps)
    composite.prune_draft_extend_logits = True
    return composite, worker


def _make_one_graph_composite(num_steps):
    composite = OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner.__new__(
        OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner
    )
    worker = _prepare_common(composite, num_steps)
    outs = [runner.out for runner in composite.runners]
    first = composite.runners[0]
    first._make_graph_key = lambda bs: ("shape_key", bs)
    first.backend = SimpleNamespace(replay=lambda shape_key, spec_info: outs)
    return composite, worker


class TestLastSharedReadRunner(CustomTestCase):
    def test_last_shared_read_runner_is_final_draft_runner(self):
        """The barrier must read the runner the final draft-extend step writes;
        pointing it at the first one leaves the event unset forever."""
        worker = _make_worker(num_steps=3)
        draft_runners = worker._draft_worker.draft_runner_list

        self.assertIs(worker.last_shared_read_runner, draft_runners[-1])


class TestSharedReadEventPublish(CustomTestCase):
    def _assert_published(self, composite, worker, step):
        runner = composite.runners[step]
        events = runner.device_module.events
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].record_count, 1)
        self.assertIs(worker.last_shared_read_runner, runner.model_runner)
        self.assertIs(worker.last_shared_read_runner.shared_read_done_event, events[0])

    def _assert_not_published(self, composite, step):
        runner = composite.runners[step]
        self.assertEqual(runner.device_module.events, [])
        self.assertIsNone(runner.model_runner.shared_read_done_event)

    def test_per_step_replay_publishes_only_on_final_step(self):
        composite, worker = _make_per_step_composite(num_steps=2)

        composite.replay(0)
        self._assert_not_published(composite, 0)
        self._assert_not_published(composite, 1)

        composite.replay(1)
        self._assert_not_published(composite, 0)
        self._assert_published(composite, worker, 1)

    def test_one_graph_replay_publishes_only_on_final_step(self):
        composite, worker = _make_one_graph_composite(num_steps=2)

        composite.replay(0)
        self._assert_not_published(composite, 0)
        self._assert_not_published(composite, 1)

        composite.replay(1)
        self._assert_not_published(composite, 0)
        self._assert_published(composite, worker, 1)

    def test_single_step_replay_publishes_on_step_zero(self):
        composite, worker = _make_per_step_composite(num_steps=1)

        composite.replay(0)
        self._assert_published(composite, worker, 0)

    def test_replay_returns_batch_sliced_outputs(self):
        composite, _ = _make_per_step_composite(num_steps=2)

        logits_output, topk_p, topk_index = composite.replay(1)

        self.assertEqual(logits_output.next_token_logits, list(range(RAW_BS)))
        self.assertEqual(logits_output.hidden_states, list(range(RAW_NUM_TOKENS)))
        self.assertEqual(topk_p, list(range(RAW_BS)))
        self.assertEqual(topk_index, list(range(RAW_BS)))


if __name__ == "__main__":
    unittest.main()

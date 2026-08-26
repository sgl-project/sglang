"""Unit tests for SGLANG_OPT_STREAM_FINISH_BUDGET: a completion wave's finished
outputs are spread over multiple stream_output passes, each request emitted
exactly once, streaming (unfinished) requests unaffected."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components import output_streamer as os_mod
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubAccumulator:
    """Stands in for _GenerationStreamAccumulator: records accepted reqs and
    mirrors the finished_output bookkeeping accept() performs."""

    instances = []

    def __init__(self, **kwargs):
        self.accepted = []
        _StubAccumulator.instances.append(self)

    def accept(self, *, req):
        if req.finished():
            req.finished_output = True
        self.accepted.append(req.rid)

    def to_payload(self, *, dp_rank, is_idle_batch):
        return None


def _req(rid, finished):
    return SimpleNamespace(
        rid=rid,
        finished=lambda f=finished: f,
        finished_output=False,
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        return_sampling_mask=False,
    )


def _streamer():
    return SchedulerOutputStreamer(
        send_to_detokenizer=MagicMock(),
        tree_cache=MagicMock(),
        ps=MagicMock(attn_tp_rank=1),
        server_args=MagicMock(),
        is_generation=True,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_hicache_storage=lambda: False,
    )


class TestStreamFinishBudget(CustomTestCase):
    def _run_wave(self, budget, n_finished=10, n_streaming=3):
        finished = [_req(f"f{i}", True) for i in range(n_finished)]
        streaming = [_req(f"s{i}", False) for i in range(n_streaming)]
        streamer = _streamer()
        accepted_per_pass = []
        with (
            patch.object(os_mod, "_GenerationStreamAccumulator", _StubAccumulator),
            patch.object(os_mod, "STREAM_FINISH_BUDGET", budget),
            patch.object(os_mod, "get_serving", MagicMock()),
        ):
            _StubAccumulator.instances = []
            streamer._stream_output_generation(finished + streaming, False)
            accepted_per_pass.append(_StubAccumulator.instances[-1].accepted)
            for _ in range(6):
                if not streamer._deferred_finished:
                    break
                streamer._stream_output_generation(streaming, False)
                accepted_per_pass.append(_StubAccumulator.instances[-1].accepted)
        return finished, streaming, streamer, accepted_per_pass

    def test_budget_disabled_emits_all_at_once(self):
        finished, streaming, streamer, passes = self._run_wave(budget=0)
        self.assertEqual(len(passes), 1)
        self.assertEqual(
            passes[0], [r.rid for r in finished] + [r.rid for r in streaming]
        )
        self.assertEqual(len(streamer._deferred_finished), 0)

    def test_wave_spread_over_passes_each_emitted_once(self):
        finished, streaming, streamer, passes = self._run_wave(budget=4)
        finished_emitted = [
            rid for p in passes for rid in p if rid.startswith("f")
        ]
        self.assertEqual(sorted(finished_emitted), sorted(r.rid for r in finished))
        self.assertEqual(len(finished_emitted), len(set(finished_emitted)))
        for p in passes:
            self.assertLessEqual(sum(rid.startswith("f") for rid in p), 4)
        self.assertEqual(len(passes), 3)
        self.assertEqual(len(streamer._deferred_finished), 0)
        self.assertEqual(len(streamer._deferred_rids), 0)
        self.assertTrue(all(r.finished_output for r in finished))

    def test_streaming_reqs_never_deferred(self):
        _, streaming, _, passes = self._run_wave(budget=2)
        self.assertEqual(
            [rid for rid in passes[0] if rid.startswith("s")],
            [r.rid for r in streaming],
        )

    def test_overlap_duplicate_not_double_deferred(self):
        finished = [_req(f"f{i}", True) for i in range(5)]
        streamer = _streamer()
        with (
            patch.object(os_mod, "_GenerationStreamAccumulator", _StubAccumulator),
            patch.object(os_mod, "STREAM_FINISH_BUDGET", 2),
            patch.object(os_mod, "get_serving", MagicMock()),
        ):
            _StubAccumulator.instances = []
            streamer._stream_output_generation(finished, False)
            streamer._stream_output_generation(finished, False)
            while streamer._deferred_finished:
                streamer._stream_output_generation([], False)
            emitted = [
                rid for acc in _StubAccumulator.instances for rid in acc.accepted
            ]
        self.assertEqual(sorted(emitted), sorted(r.rid for r in finished))
        self.assertEqual(len(emitted), len(set(emitted)))


if __name__ == "__main__":
    unittest.main()

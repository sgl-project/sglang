"""BCG replay failures must fall back to eager, mirroring the capture policy."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    DiffusionBreakableCudaGraphRunner,
    _CaptureEntry,
)


class _ExplodingGraph:
    """Fake BreakableCUDAGraph whose replay always raises."""

    def __init__(self):
        self._break_fns = []
        self._segments = []

    def replay(self):
        raise RuntimeError("injected replay failure")


class _RecordingTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return torch.ones(1)


def _bare_runner(transformer) -> DiffusionBreakableCudaGraphRunner:
    runner = DiffusionBreakableCudaGraphRunner.__new__(
        DiffusionBreakableCudaGraphRunner
    )
    runner.transformer = transformer
    runner.entries = {}
    runner._blocked = set()
    runner._pool = None
    runner._disabled_reason = None
    runner.device_module = SimpleNamespace(empty_cache=lambda: None)
    return runner


class TestReplayFallback(unittest.TestCase):
    def test_replay_failure_disables_runner_and_runs_eager(self):
        transformer = _RecordingTransformer()
        runner = _bare_runner(transformer)

        kwargs = {"hidden_states": torch.randn(1, 4, 8)}
        key = runner._signature(kwargs)
        runner.entries[key] = _CaptureEntry(
            graph=_ExplodingGraph(),
            static_kwargs=dict(kwargs),
            static_leaves=[torch.empty(1, 4, 8)],
            output=torch.zeros(1),
            num_segments=1,
        )

        # Replay raises -> eager result, graphs dropped, runner disabled.
        out = runner(**kwargs)
        self.assertTrue(torch.equal(out, torch.ones(1)))
        self.assertEqual(transformer.calls, 1)
        self.assertEqual(runner.entries, {})
        self.assertIsNotNone(runner._disabled_reason)
        self.assertIn("replay failed", runner._disabled_reason)

        # Disabled runner keeps serving eagerly on later calls.
        out = runner(**kwargs)
        self.assertTrue(torch.equal(out, torch.ones(1)))
        self.assertEqual(transformer.calls, 2)


if __name__ == "__main__":
    unittest.main()

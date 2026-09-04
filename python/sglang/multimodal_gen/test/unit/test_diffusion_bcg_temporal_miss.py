import unittest

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    DiffusionBreakableCudaGraphRunner,
    _signature_kwargs,
)


def _video_kwargs(batch, channels, frames, height, width):
    import torch

    return {
        "hidden_states": torch.zeros(batch, channels, frames, height, width),
        "timestep": torch.zeros(1),
    }


def _runner_with_captured(kwargs_list):
    """Minimal runner carrying pre-captured signatures, no CUDA needed."""
    runner = object.__new__(DiffusionBreakableCudaGraphRunner)
    runner.entries = {_signature_kwargs(k): None for k in kwargs_list}
    return runner


class TestTemporalShapeMiss(unittest.TestCase):
    def test_temporal_only_miss_detected(self):
        captured = _runner_with_captured([_video_kwargs(1, 16, 21, 60, 104)])
        serving = _signature_kwargs(_video_kwargs(1, 16, 5, 60, 104))
        self.assertTrue(captured._has_temporal_shape_miss(serving))

    def test_resolution_miss_is_not_temporal(self):
        captured = _runner_with_captured([_video_kwargs(1, 16, 21, 60, 104)])
        serving = _signature_kwargs(_video_kwargs(1, 16, 21, 30, 52))
        self.assertFalse(captured._has_temporal_shape_miss(serving))

    def test_exact_match_is_not_a_miss(self):
        captured = _runner_with_captured([_video_kwargs(1, 16, 21, 60, 104)])
        serving = _signature_kwargs(_video_kwargs(1, 16, 21, 60, 104))
        self.assertFalse(captured._has_temporal_shape_miss(serving))

    def test_non_video_hidden_states_returns_false(self):
        import torch

        captured = _runner_with_captured(
            [{"hidden_states": torch.zeros(1, 32, 32, 32)}]
        )
        serving = _signature_kwargs({"hidden_states": torch.zeros(1, 16, 32, 32)})
        self.assertFalse(captured._has_temporal_shape_miss(serving))


if __name__ == "__main__":
    unittest.main()

"""Triton attention over several window sub-pools, under captured decode graphs
and with value widths that differ per sub-pool."""

import unittest

import torch
from test_triton_several_windows import SeveralWindowsCase, _Model

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTritonSeveralWindowsGraphsAndWidths(SeveralWindowsCase):
    def test_decode_graph_replays_each_layers_window(self):
        """A decode graph captured on one batch replays the same rows and
        token count with other sequence lengths and write ids, per window."""
        for windows in ((8, 4), (16, 8, 3)):
            with self.subTest(windows=windows):
                model = _Model(windows)
                model.bind()
                self._graph_decode(model)

    def test_value_widths_come_from_the_sub_pools(self):
        """Each layer's decode scratch is sized by its own sub-pool's value width,
        whichever layer comes first, eager and under a captured graph."""
        for full_last in (False, True):
            with self.subTest(full_last=full_last):
                model = _Model(
                    (8, 4),
                    full_dim=128,
                    window_dim=64,
                    full_last=full_last,
                    model_window=None,
                )
                self.assertEqual(
                    dict(model.backend.window_lengths),
                    {"window_8": 8, "window_4": 4},
                )
                self.assertEqual(model.backend.v_head_dim, 128)
                model.bind()
                self._extend_and_decode(model)
                graphed = _Model(
                    (8, 4),
                    full_dim=128,
                    window_dim=64,
                    full_last=full_last,
                    model_window=None,
                )
                graphed.bind()
                self._graph_decode(graphed)

    def test_windows_of_different_value_widths(self):
        """Each window layer's decode scratch takes its own width when the windows'
        value widths differ from each other and from the full layers'."""
        model = _Model((8, 4), full_dim=128, window_dim=(64, 32), model_window=None)
        self.assertEqual(model.backend.v_head_dim, 128)
        self.assertEqual(model.backend.extra_v_head_dims, (32, 64))
        model.bind()
        self._extend_and_decode(model)
        graphed = _Model((8, 4), full_dim=128, window_dim=(64, 32), model_window=None)
        graphed.bind()
        self._graph_decode(graphed)


if __name__ == "__main__":
    unittest.main()

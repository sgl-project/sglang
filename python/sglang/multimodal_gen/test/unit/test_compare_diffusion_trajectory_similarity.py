import unittest
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    compute_tensor_metrics,
    compute_uint8_frame_metrics,
    extract_result_frames,
    parse_component_overrides,
    summarize_output_frame_metrics,
    summarize_trajectory_metrics,
)


class TestCompareDiffusionTrajectorySimilarity(unittest.TestCase):
    def test_parse_component_overrides(self):
        overrides = parse_component_overrides(
            ["transformer=/tmp/a", "transformer-2=/tmp/b"]
        )
        self.assertEqual(
            overrides,
            {"transformer": "/tmp/a", "transformer_2": "/tmp/b"},
        )

    def test_compute_tensor_metrics_identical(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metrics = compute_tensor_metrics(tensor, tensor.clone())
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=6)
        self.assertAlmostEqual(metrics["max_abs"], 0.0, places=6)

    def test_summarize_trajectory_metrics_selects_requested_step(self):
        ref = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        cand = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        ref_t = torch.tensor([999.0, 111.0])
        cand_t = torch.tensor([999.0, 111.0])

        summary = summarize_trajectory_metrics(
            ref,
            cand,
            reference_timesteps=ref_t,
            candidate_timesteps=cand_t,
            step_index=-1,
        )

        self.assertEqual(summary["selected_step_index"], 1)
        self.assertEqual(summary["selected_step_metrics"]["reference_timestep"], 111.0)
        self.assertAlmostEqual(
            summary["selected_step_metrics"]["cosine_similarity"], 0.0, places=6
        )
        self.assertEqual(len(summary["per_step_metrics"]), 2)

    def test_summarize_output_frame_metrics_reports_psnr(self):
        ref_frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 16, dtype=np.uint8),
        ]
        cand_frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 32, dtype=np.uint8),
        ]

        summary = summarize_output_frame_metrics(ref_frames, cand_frames)

        self.assertEqual(summary["num_frames"], 2)
        self.assertTrue(np.isinf(summary["frame0_metrics"]["psnr_db"]))
        self.assertGreater(summary["all_frames_metrics"]["mae"], 0.0)

    def test_compute_uint8_frame_metrics_matches_expected_mae(self):
        lhs = np.zeros((1, 2, 2, 1), dtype=np.uint8)
        rhs = np.full((1, 2, 2, 1), 10, dtype=np.uint8)
        metrics = compute_uint8_frame_metrics(lhs, rhs)
        self.assertAlmostEqual(metrics["mae"], 10.0, places=6)
        self.assertGreater(metrics["psnr_db"], 0.0)

    def test_extract_result_frames_falls_back_to_samples(self):
        class DummyResult:
            frames = None
            samples = torch.tensor(
                [
                    [[0.0, 1.0], [0.5, 0.25]],
                    [[0.0, 1.0], [0.5, 0.25]],
                    [[0.0, 1.0], [0.5, 0.25]],
                ]
            )

        frames = extract_result_frames(DummyResult())
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].dtype, np.uint8)
        self.assertEqual(frames[0].shape, (2, 2, 3))

    def test_extract_result_frames_falls_back_to_output_file_path(self):
        tmp_path = Path(self.id().replace(".", "_") + ".png")
        try:
            image = np.full((2, 2, 3), 42, dtype=np.uint8)
            iio.imwrite(tmp_path, image)

            class DummyResult:
                frames = None
                samples = None
                output_file_path = str(tmp_path)

            frames = extract_result_frames(DummyResult())
            self.assertEqual(len(frames), 1)
            self.assertTrue(np.array_equal(frames[0], image))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

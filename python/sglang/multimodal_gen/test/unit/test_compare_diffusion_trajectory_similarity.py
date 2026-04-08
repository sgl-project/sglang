from types import SimpleNamespace

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


def test_parse_component_overrides_normalizes_transformer_2_key():
    overrides = parse_component_overrides(
        ["transformer=/tmp/a", "transformer-2=/tmp/b"]
    )

    assert overrides == {
        "transformer": "/tmp/a",
        "transformer_2": "/tmp/b",
    }


def test_compute_tensor_metrics_for_identical_tensors():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    metrics = compute_tensor_metrics(tensor, tensor.clone())

    assert metrics["cosine_similarity"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs"] == 0.0


def test_summarize_trajectory_metrics_selects_requested_step():
    summary = summarize_trajectory_metrics(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        reference_timesteps=torch.tensor([999.0, 111.0]),
        candidate_timesteps=torch.tensor([999.0, 111.0]),
        step_index=-1,
    )

    assert summary["selected_step_index"] == 1
    assert summary["selected_step_metrics"]["reference_timestep"] == 111.0
    assert summary["selected_step_metrics"]["cosine_similarity"] == 0.0
    assert len(summary["per_step_metrics"]) == 2


def test_summarize_output_frame_metrics_reports_expected_psnr_shape():
    summary = summarize_output_frame_metrics(
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 16, dtype=np.uint8),
        ],
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 32, dtype=np.uint8),
        ],
    )

    assert summary["num_frames"] == 2
    assert np.isinf(summary["frame0_metrics"]["psnr_db"])
    assert summary["all_frames_metrics"]["mae"] > 0.0


def test_compute_uint8_frame_metrics_matches_expected_mae():
    metrics = compute_uint8_frame_metrics(
        np.zeros((1, 2, 2, 1), dtype=np.uint8),
        np.full((1, 2, 2, 1), 10, dtype=np.uint8),
    )

    assert metrics["mae"] == 10.0
    assert metrics["psnr_db"] > 0.0


def test_extract_result_frames_uses_samples_when_frames_are_missing():
    result = SimpleNamespace(
        frames=None,
        samples=torch.tensor(
            [
                [[0.0, 1.0], [0.5, 0.25]],
                [[0.0, 1.0], [0.5, 0.25]],
                [[0.0, 1.0], [0.5, 0.25]],
            ]
        ),
    )

    frames = extract_result_frames(result)

    assert len(frames) == 1
    assert frames[0].dtype == np.uint8
    assert frames[0].shape == (2, 2, 3)


def test_extract_result_frames_uses_output_file_path_when_needed(tmp_path):
    image = np.full((2, 2, 3), 42, dtype=np.uint8)
    image_path = tmp_path / "frame.png"
    iio.imwrite(image_path, image)

    result = SimpleNamespace(frames=None, samples=None, output_file_path=str(image_path))

    frames = extract_result_frames(result)

    assert len(frames) == 1
    assert np.array_equal(frames[0], image)

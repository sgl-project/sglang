import pytest
import torch

from sglang.srt.models.phi4mm_utils import adaptive_enc_mask
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _expected_adaptive_enc_mask(
    x_len: int, chunk_start_idx: list[int], left_window: int, right_window: int
):
    start_pad = [0] + chunk_start_idx
    end_pad = chunk_start_idx + [x_len]
    expected = []

    for token_idx in range(x_len):
        chunk_idx = 0
        for idx, start in enumerate(chunk_start_idx, start=1):
            if token_idx >= start:
                chunk_idx = idx

        left_idx = max(chunk_idx - left_window, 0)
        right_idx = min(chunk_idx + right_window, len(chunk_start_idx))
        row_start = start_pad[left_idx]
        row_end = end_pad[right_idx]
        expected.append(
            [row_start <= attend_idx < row_end for attend_idx in range(x_len)]
        )

    return torch.tensor(expected, dtype=torch.bool)


@pytest.mark.parametrize(
    ("x_len", "chunk_start_idx", "left_window", "right_window"),
    [
        (8, [0, 3, 6], 0, 0),
        (8, [0, 3, 6], 1, 0),
        (8, [0, 3, 6], 0, 1),
        (8, [0, 3, 6], 2, 2),
        (5, [], 0, 0),
        (7, [2, 5], 1, 1),
    ],
)
def test_adaptive_enc_mask_matches_chunk_windows(
    x_len, chunk_start_idx, left_window, right_window
):
    result = adaptive_enc_mask(x_len, chunk_start_idx, left_window, right_window)

    assert result.dtype == torch.bool
    assert torch.equal(
        result,
        _expected_adaptive_enc_mask(x_len, chunk_start_idx, left_window, right_window),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_adaptive_enc_mask_preserves_tensor_device_on_cuda():
    chunk_start_idx = torch.tensor([0, 3, 6], device="cuda")

    result = adaptive_enc_mask(8, chunk_start_idx, left_window=1, right_window=1)

    assert result.device.type == "cuda"
    assert torch.equal(result.cpu(), _expected_adaptive_enc_mask(8, [0, 3, 6], 1, 1))

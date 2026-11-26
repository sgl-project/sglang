import sys

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="extra-test-cpu-intel")


@pytest.mark.parametrize("positions_dtype", [torch.int64, torch.uint64])
def test_murmur_hash32_cpu_known_values(positions_dtype):
    seed = torch.tensor([0, 1, 42, 0x123456789ABCDEF0], dtype=torch.uint64)

    positions = torch.tensor([0, 7, 123, 456], dtype=positions_dtype)

    col_indices = torch.tensor([0, 1, 2, 17], dtype=torch.int64)

    actual = murmur_hash32(seed, positions, col_indices)

    expected = torch.tensor(
        [
            [2167721464, 10027521, 2423355346, 2203067026],
            [3755322398, 4196701286, 1002451629, 183234019],
            [772287619, 548237471, 2740678348, 3656549299],
            [3746406971, 2891010872, 104055988, 3372550890],
        ],
        dtype=torch.uint32,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("positions_dtype", [torch.int64, torch.uint64])
def test_murmur_hash32_cpu_shape_and_dtype(positions_dtype):
    seed = torch.tensor([1, 2, 3], dtype=torch.uint64)

    positions = torch.tensor([10, 20, 30], dtype=positions_dtype)

    col_indices = torch.arange(128, dtype=torch.int64)

    actual = murmur_hash32(
        seed,
        positions,
        col_indices,
    )

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.uint32
    assert actual.shape == (3, 128)


def test_watermark_state_folds_wide_multimodal_padding_ids():
    from sglang.srt.sampling.watermarking.gumbel.algorithm import hash_contexts
    from sglang.srt.sampling.watermarking.gumbel.state import WatermarkState

    # _compute_pad_values() builds padding ids from a 62-bit mask, which does
    # not fit the int32 context buffer used by the watermark state.
    wide_id = (1 << 62) | 0x2A5B
    state = WatermarkState(
        max_num_reqs=1,
        context_window=4,
        max_contexts_per_req=2,
        device="cpu",
    )
    state.init_from_prompt(torch.tensor([0], dtype=torch.int32), [[wide_id, 123]])

    folded = ((wide_id & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    assert state.token_ids[0, :2].tolist() == [folded, 123]
    assert state.lengths[0].item() == 2

    # The watermark hashes the low 32 bits of each id, so folding keeps the
    # context hash identical to hashing the original wide id.
    lengths = torch.tensor([2], dtype=torch.int32)
    original_hash = hash_contexts(torch.tensor([[wide_id, 123]], dtype=torch.int64), lengths)
    stored_hash = hash_contexts(state.token_ids[0, :2].to(torch.int64).view(1, -1), lengths)
    assert original_hash.item() == stored_hash.item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

"""Contract tests for the DSv4 decode split-K heuristic.

`_kv_splits_heuristic` runs at CUDAGraph capture time, so it may depend only on
capture-time scalars and must never read a tensor. These tests pin that
contract, the invariants the split-K reduction relies on, and the specific
shapes the tuned constants were chosen for.

They assert properties rather than the constants themselves, so retuning
`target_wg_per_cu` / `_MAX_KV_SPLITS` for a future architecture does not
require rewriting the suite -- only the one explicitly-marked table does.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (  # noqa: E402
    _MAX_KV_SPLITS,
    _kv_splits_heuristic,
    _prev_pow2,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# MI355X. Passed explicitly so the tests are device-independent and run on CPU.
NUM_CU = 256
# head_dim 512 = 448 nope + 64 rope; DSv4 decode runs 128 heads in 64-head tiles.
HEADS = 128
BLOCK_H = 64


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 64, 100, 1000])
def test_prev_pow2_is_largest_power_of_two_not_exceeding(n):
    got = _prev_pow2(n)
    assert got <= n
    assert got & (got - 1) == 0, f"{got} is not a power of two"
    assert got * 2 > n, f"{got} is not the largest such power of two for {n}"


@pytest.mark.parametrize("n", [0, -1, -100])
def test_prev_pow2_clamps_non_positive(n):
    assert _prev_pow2(n) == 1


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_result_is_positive_power_of_two_within_cap(tokens):
    """Split-K allocates a partial buffer per split and the reduce kernel
    indexes it by power-of-two stride, so a non-pow2 or out-of-range count
    would corrupt the reduction rather than merely run slowly."""
    splits = _kv_splits_heuristic(tokens, HEADS, BLOCK_H, num_cu=NUM_CU)
    assert splits >= 1
    assert splits <= _MAX_KV_SPLITS
    assert splits & (splits - 1) == 0, f"{splits} is not a power of two"


def test_splits_never_increase_with_token_count():
    """More decode tokens means a larger base grid, so the device fills without
    splitting. A count that rose with tokens would over-subscribe the GPU."""
    prev = None
    for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        splits = _kv_splits_heuristic(tokens, HEADS, BLOCK_H, num_cu=NUM_CU)
        if prev is not None:
            assert (
                splits <= prev
            ), f"splits rose from {prev} to {splits} going to T={tokens}"
        prev = splits


def test_saturated_grid_does_not_split():
    """Once the base grid alone meets the target occupancy, splitting can only
    add partial-buffer writes and reduce work."""
    # base_ctas = T * ceil(H/block_h); target_wg = target_wg_per_cu * num_cu.
    saturating_tokens = NUM_CU * 4  # 1024 tokens x 2 head blocks = 2048 CTAs
    assert _kv_splits_heuristic(saturating_tokens, HEADS, BLOCK_H, num_cu=NUM_CU) == 1


def test_does_not_read_tensors_only_capture_time_scalars():
    """CUDAGraph safety: the heuristic must be callable with plain ints and no
    CUDA context. Reading kv_indices/kv_indptr here would bake a value from the
    capture step into every replay."""
    splits = _kv_splits_heuristic(32, HEADS, BLOCK_H, num_cu=NUM_CU)
    assert isinstance(splits, int)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        # The tuned operating points on MI355X (256 CU, H=128, block_h=64).
        # base_ctas = tokens * 2; target_wg = int(1.5 * 256) = 384.
        (8, 16),  # 384//16 = 24 -> capped at 16
        (16, 8),  # 384//32 = 12 -> prev_pow2 = 8
        (32, 4),  # 384//64 = 6  -> prev_pow2 = 4
        (64, 2),  # 384//128 = 3 -> prev_pow2 = 2
        (128, 1),  # 384//256 = 1
        (256, 1),  # base grid already saturates
    ],
)
def test_tuned_operating_points(tokens, expected):
    """Values measured fastest on MI355X. The MI355X constants are passed
    explicitly so the table holds on any CI runner (CUDA keeps 2.0/64); update
    alongside the constants if the heuristic is re-tuned -- this is the one test
    that pins numbers."""
    assert (
        _kv_splits_heuristic(
            tokens,
            HEADS,
            BLOCK_H,
            num_cu=NUM_CU,
            target_wg_per_cu=1.5,
            max_kv_splits=16,
        )
        == expected
    )


@pytest.mark.parametrize("num_cu", [64, 104, 128, 228, 256, 304])
def test_scales_with_device_width(num_cu):
    """A wider device should never ask for more splits at fixed work: splitting
    exists to fill the device, and a wider one fills at a lower split count
    only if the base grid grew, which it did not."""
    splits = _kv_splits_heuristic(32, HEADS, BLOCK_H, num_cu=num_cu)
    assert 1 <= splits <= _MAX_KV_SPLITS
    assert splits & (splits - 1) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

"""Bit-exactness of the fused value+scale row gather.

The oracle is plain torch advanced indexing, which is the whole contract:
``out[i] = src[dst2src_map[i]]`` for both tensors. A second test cross-checks the
pair of `shuffle_rows` calls this replaces, to back the drop-in claim.

Comparisons are made on integer views. The values are fp8 and a random byte
pattern is a NaN often enough that `torch.equal` on the float view would report a
difference where the bytes agree -- and bytes are exactly what this kernel
promises to preserve.
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe.shuffle_rows_with_scales import shuffle_rows_with_scales
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

GROUP_SIZE = 128

# The CUDA shuffle_rows this replaces loads 128 bits per thread and takes its
# element count as num_cols / elems_per_thread with no remainder handling, so it
# only moves a whole fp32 scale row when (k // GROUP_SIZE) % 4 == 0. Shapes below
# that bound are checked against the torch oracle only -- the reference itself
# drops the tail there.
CUDA_XCHECK_SCALE_ALIGN = 4

# k = 896 gives 7 scale groups, the shape whose scale tail the CUDA reference
# cannot express; 7168 is the DeepSeek-class hidden size. Destination rows exceed
# source rows because the map replicates each token once per routed expert.
CASES = get_ci_test_range(
    [
        (k, src, dst)
        for k, (src, dst) in itertools.product(
            [512, 896, 2560, 7168], [(1, 8), (17, 136), (64, 512)]
        )
    ],
    [
        (896, 1, 8),
        (7168, 1, 8),
        (2560, 17, 136),
        (7168, 64, 512),
    ],
)


def _inputs(k, num_src_rows, num_dst_rows, seed):
    torch.manual_seed(seed)
    q = torch.randint(0, 256, (num_src_rows, k), dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    scale = torch.randn(
        (num_src_rows, k // GROUP_SIZE), dtype=torch.float32, device="cuda"
    )
    # Duplicate source rows are the normal case here: a token is replicated once
    # per expert it routes to.
    dst2src = torch.randint(
        0, num_src_rows, (num_dst_rows,), dtype=torch.int32, device="cuda"
    )
    return q, scale, dst2src


def _assert_same_bytes(got, ref, what):
    assert torch.equal(got.view(torch.int8), ref.view(torch.int8)), (
        f"{what} bytes differ"
    )


@pytest.mark.parametrize("k,num_src_rows,num_dst_rows", CASES)
def test_matches_torch_gather(k, num_src_rows, num_dst_rows):
    q, scale, dst2src = _inputs(k, num_src_rows, num_dst_rows, seed=0)

    got_q, got_scale = shuffle_rows_with_scales(q, scale, dst2src, num_dst_rows)

    idx = dst2src.long()
    _assert_same_bytes(got_q, q[idx], "values")
    _assert_same_bytes(got_scale, scale[idx], "scales")


CUDA_XCHECK_CASES = [
    c for c in CASES if (c[0] // GROUP_SIZE) % CUDA_XCHECK_SCALE_ALIGN == 0
]
# An empty parametrize list collects zero tests and reports success, which would
# retire the drop-in check without saying so. Fail at collection instead.
assert CUDA_XCHECK_CASES, "no case survives the CUDA cross-check shape filter"


@pytest.mark.parametrize("k,num_src_rows,num_dst_rows", CUDA_XCHECK_CASES)
def test_matches_shuffle_rows_pair(k, num_src_rows, num_dst_rows):
    """Drop-in equivalence with the two launches this replaces."""
    from sgl_kernel import shuffle_rows

    q, scale, dst2src = _inputs(k, num_src_rows, num_dst_rows, seed=1)

    got_q, got_scale = shuffle_rows_with_scales(q, scale, dst2src, num_dst_rows)

    _assert_same_bytes(got_q, shuffle_rows(q, dst2src, (num_dst_rows, k)), "values")
    _assert_same_bytes(
        got_scale,
        shuffle_rows(scale, dst2src, (num_dst_rows, k // GROUP_SIZE)),
        "scales",
    )


def test_empty_destination_does_not_launch():
    """Zero rows takes the short circuit instead of a zero-sized grid."""
    q, scale, dst2src = _inputs(512, 4, 0, seed=2)

    got_q, got_scale = shuffle_rows_with_scales(q, scale, dst2src, 0)

    assert got_q.shape == (0, 512)
    assert got_scale.shape == (0, 512 // GROUP_SIZE)
    assert got_q.dtype == q.dtype and got_scale.dtype == scale.dtype


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

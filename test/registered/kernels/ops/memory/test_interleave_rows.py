import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "cp_size,tokens,width",
    [(8, 8192, 7168), (8, 17, 128), (2, 7, 33), (8, 1, 576), (8, 0, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_interleave_rows_roundtrip(cp_size, tokens, width, dtype):
    from sglang.kernels.ops.memory.interleave_rows import interleave_rows

    rows = (tokens + cp_size - 1) // cp_size
    x = (
        torch.arange(tokens * width, device="cuda", dtype=torch.int64)
        .reshape(tokens, width)
        .to(dtype)
    )
    expected = x.new_zeros((rows * cp_size, width))
    expected[:tokens] = x
    expected = expected.reshape(rows, cp_size, width).transpose(0, 1).reshape(-1, width)
    rank_major = interleave_rows(x, cp_size, rows * cp_size, to_rank_major=True)
    torch.testing.assert_close(rank_major, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        interleave_rows(rank_major, cp_size, tokens), x, rtol=0, atol=0
    )
    if tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rank_major = interleave_rows(x, cp_size, rows * cp_size, to_rank_major=True)
            restored = interleave_rows(rank_major, cp_size, tokens)
        for v in [3, 11]:
            x.fill_(v)
            graph.replay()
            torch.testing.assert_close(restored, x, rtol=0, atol=0)
            # Check every pad row is overwritten with zero on every replay.
            torch.testing.assert_close(
                rank_major.reshape(cp_size, rows, width)
                .transpose(0, 1)
                .reshape(-1, width)[tokens:],
                x.new_zeros((rows * cp_size - tokens, width)),
                rtol=0,
                atol=0,
            )


def test_row_counts_stay_runtime_args():
    """Row counts change with every batch, so a Triton kernel per token count would be loaded."""
    from sglang.kernels.ops.memory.interleave_rows import _interleave_rows_kernel

    constexpr = {
        param.name: param.is_constexpr for param in _interleave_rows_kernel.params
    }
    assert constexpr == {
        "X": False,
        "Y": False,
        "input_rows": False,
        "rank_rows": False,
        "WIDTH": True,
        "CP_SIZE": True,
        "TO_RANK_MAJOR": True,
        "BLOCK": True,
    }
    assert set(_interleave_rows_kernel.do_not_specialize) == {"input_rows", "rank_rows"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

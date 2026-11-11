"""interleave_rows must match the torch reshape/transpose reference of the
interleave CP row order, in both directions, including the zero padding of a
partial last token group and row counts whose element offsets overflow int32."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

_DEVICE = "cuda"


def _to_sequence_ref(x: torch.Tensor, cp_size: int, output_rows: int) -> torch.Tensor:
    rows = x.shape[0] // cp_size
    return (
        x.reshape(cp_size, rows, *x.shape[1:])
        .transpose(0, 1)
        .flatten(0, 1)[:output_rows]
        .contiguous()
    )


def _to_rank_major_ref(x: torch.Tensor, cp_size: int, output_rows: int) -> torch.Tensor:
    padded = x.new_zeros((output_rows, *x.shape[1:]))
    padded[: x.shape[0]] = x
    return (
        padded.reshape(-1, cp_size, *x.shape[1:]).transpose(0, 1).flatten(0, 1)
    ).contiguous()


def _padded_rows(tokens: int, cp_size: int) -> int:
    return -(-tokens // cp_size) * cp_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [0, 1, 7, 1023, 4096, 32771])
@pytest.mark.parametrize("row_shape", [(1,), (7168,), (8, 128), (16384,)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32])
def test_round_trip_matches_reference(cp_size, tokens, row_shape, dtype):
    from sglang.kernels.ops.memory.interleave_rows import interleave_rows

    torch.manual_seed(tokens + cp_size)
    gathered_rows = _padded_rows(tokens, cp_size)
    if dtype.is_floating_point:
        seq = torch.randn(tokens, *row_shape, device=_DEVICE, dtype=dtype)
    else:
        seq = torch.randint(-(2**30), 2**30, (tokens, *row_shape), device=_DEVICE)
        seq = seq.to(dtype)

    rank_major = interleave_rows(seq, cp_size, gathered_rows, to_rank_major=True)
    torch.testing.assert_close(
        rank_major, _to_rank_major_ref(seq, cp_size, gathered_rows), rtol=0, atol=0
    )

    restored = interleave_rows(rank_major, cp_size, tokens)
    torch.testing.assert_close(
        restored, _to_sequence_ref(rank_major, cp_size, tokens), rtol=0, atol=0
    )
    torch.testing.assert_close(restored, seq, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("to_rank_major", [False, True])
def test_offsets_past_int32(to_rank_major):
    """rows x width > 2**31 elements: a 32-bit offset would wrap and corrupt
    the rows near the end of the tensor."""
    from sglang.kernels.ops.memory.interleave_rows import interleave_rows

    cp_size, width = 8, 7168
    tokens = (2**31 // width) + 4099  # just past 2**31 elements
    rows = _padded_rows(tokens, cp_size)
    need = 2 * rows * width  # int8 input + output
    free, _ = torch.cuda.mem_get_info()
    if free < need + (1 << 30):
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free GPU memory")

    # Encode each row's index in its first bytes so a misplaced row is visible.
    x_rows = tokens if to_rank_major else rows
    x = torch.zeros(x_rows, width, device=_DEVICE, dtype=torch.int8)
    ids = torch.arange(x_rows, device=_DEVICE, dtype=torch.int64)
    for byte in range(4):
        x[:, byte] = ((ids >> (8 * byte)) & 0xFF).to(torch.int8)
    x[:, -1] = 1

    if to_rank_major:
        out = interleave_rows(x, cp_size, rows, to_rank_major=True)
        local = rows // cp_size
        out_rows = torch.arange(rows, device=_DEVICE, dtype=torch.int64)
        expected = out_rows % local * cp_size + out_rows // local
        live = expected < tokens
    else:
        out = interleave_rows(x, cp_size, tokens)
        local = rows // cp_size
        out_rows = torch.arange(tokens, device=_DEVICE, dtype=torch.int64)
        expected = out_rows % cp_size * local + out_rows // cp_size
        live = torch.ones_like(expected, dtype=torch.bool)

    got = torch.zeros_like(expected)
    for byte in range(4):
        got |= (out[:, byte].to(torch.int64) & 0xFF) << (8 * byte)
    assert torch.equal(got[live], expected[live])
    assert torch.equal(out[live, -1], torch.ones_like(out[live, -1]))
    # Padding rows of the partial last token group are zero-filled.
    assert not out[~live].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

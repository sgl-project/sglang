import pytest
import torch

from sglang.srt.batch_invariant_ops import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from sglang.srt.layers.attention.fla.layernorm_gated import (
    MAX_ROWS_PER_BLOCK,
)
from sglang.srt.layers.attention.fla.layernorm_gated import (
    _layer_norm_fwd as layer_norm_fwd,
)
from sglang.srt.layers.attention.fla.layernorm_gated import (
    calc_rows_per_block,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton layernorm gated kernel")
    return torch.device("cuda:0")


@pytest.fixture
def batch_invariant_mode(cuda_device):
    enable_batch_invariant_mode(enable_bmm=False)
    try:
        yield
    finally:
        disable_batch_invariant_mode()


def test_rows_per_block_ignores_row_count_in_batch_invariant_mode(
    cuda_device, batch_invariant_mode
):
    row_counts = [1, 16, 513]
    rows_per_block = [calc_rows_per_block(M, cuda_device) for M in row_counts]

    assert rows_per_block == [MAX_ROWS_PER_BLOCK] * len(row_counts)


def test_layernorm_gated_row_output_is_batch_invariant(
    cuda_device, batch_invariant_mode
):
    torch.manual_seed(1234)
    dtype = torch.float16
    hidden_size = 256
    eps = 1e-6

    base_x = torch.randn(1, hidden_size, device=cuda_device, dtype=dtype)
    base_z = torch.randn(1, hidden_size, device=cuda_device, dtype=dtype)
    weight = torch.randn(hidden_size, device=cuda_device, dtype=dtype)

    solo_out, _, _ = layer_norm_fwd(
        base_x,
        weight,
        None,
        eps,
        z=base_z,
        is_rms_norm=True,
    )

    for batch_rows in [2, 5, 16, 513]:
        x = torch.randn(batch_rows, hidden_size, device=cuda_device, dtype=dtype)
        z = torch.randn(batch_rows, hidden_size, device=cuda_device, dtype=dtype)
        x[0].copy_(base_x[0])
        z[0].copy_(base_z[0])

        batched_out, _, _ = layer_norm_fwd(
            x,
            weight,
            None,
            eps,
            z=z,
            is_rms_norm=True,
        )

        torch.testing.assert_close(
            batched_out[0],
            solo_out[0],
            atol=0,
            rtol=0,
        )

# Adapted from https://github.com/vllm-project/vllm/blob/2c58742dff8613a3bd7496f2008ce927e18d38d1/tests/kernels/mamba/test_mamba_mixer2.py


from unittest.mock import patch

import pytest
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

NUM_GPUS = 2


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "hidden_size_n_groups",
    [
        (64, 1),  # hidden_size be divisible by num_gpus
        (100, 4),  # and n_groups must divide hidden_size
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mixer2_gated_norm_multi_gpu(
    batch_size: int,
    seq_len: int,
    hidden_size_n_groups: tuple[int, int],
    dtype: torch.dtype,
    device: str = "cuda",
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    assert torch.cuda.device_count() == NUM_GPUS

    hidden_size, n_groups = hidden_size_n_groups
    num_processes = NUM_GPUS

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                batch_size,
                seq_len,
                hidden_size,
                n_groups,
                dtype,
                device,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(mixer2_gated_norm_tensor_parallel, NUM_GPUS)


def mixer2_gated_norm_tensor_parallel(
    local_rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    n_groups: int,
    dtype: torch.dtype,
    device: str,
):
    torch.manual_seed(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )

    # initialize distributed
    init_distributed_environment(
        world_size=world_size, rank=local_rank, local_rank=local_rank
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # create random weights an inputs
    weight = torch.rand((hidden_size,), dtype=dtype, device=device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    gate_states = torch.randn(batch_size, seq_len, hidden_size)

    import sglang.srt.layers.attention.mamba.mixer2_rms_norm_gated as m2
    import sglang.srt.model_loader.weight_utils as wu

    # Convenience: Avoid calling initialize_dp_attention
    with patch.object(wu, "get_attention_tp_rank", return_value=local_rank):
        # create gated-norm with TP
        mixer = m2.Mixer2RMSNormGated(
            full_hidden_size=hidden_size,
            full_n_groups=n_groups,
        )
        mixer.weight.weight_loader(mixer.weight, weight)

    with (
        patch.object(m2, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(m2, "get_tensor_model_parallel_rank", return_value=0),
    ):
        # create gated-norm without TP to compute reference
        mixer_single_gpu = m2.Mixer2RMSNormGated(
            full_hidden_size=hidden_size,
            full_n_groups=n_groups,
        )
        # assign weight to single-gpu mixer
        mixer_single_gpu.weight.data = weight

    # generate and compare
    N = hidden_size // world_size
    output = mixer(
        hidden_states[..., local_rank * N : (local_rank + 1) * N],
        gate_states[..., local_rank * N : (local_rank + 1) * N],
    )
    ref_output = mixer_single_gpu(hidden_states, gate_states)
    torch.testing.assert_close(
        output,
        ref_output[..., local_rank * N : (local_rank + 1) * N],
        atol=5e-3,
        rtol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__])

"""AOT vs. JIT benchmark for the depthwise causal conv1d prefill/decode kernels."""

import torch
from sgl_kernel import causal_conv1d_fwd as aot_causal_conv1d_fwd
from sgl_kernel import causal_conv1d_update as aot_causal_conv1d_update

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE, create_random
from sglang.kernels.ops.mamba.causal_conv1d import (
    causal_conv1d_fwd as jit_causal_conv1d_fwd,
)
from sglang.kernels.ops.mamba.causal_conv1d import (
    causal_conv1d_update as jit_causal_conv1d_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

PAD_SLOT_ID = -1
WIDTH = 4

FWD_FN_MAP = {"jit": jit_causal_conv1d_fwd, "aot": aot_causal_conv1d_fwd}
UPDATE_FN_MAP = {"jit": jit_causal_conv1d_update, "aot": aot_causal_conv1d_update}


@marker.parametrize("seqlen", [128, 512, 2048, 8192], [512])
@marker.parametrize("dim", [2048, 4096, 8192], [4096])
@marker.parametrize("dtype", [torch.float16, torch.bfloat16])
@marker.benchmark("impl", ["jit", "aot"])
def benchmark_fwd(seqlen: int, dim: int, dtype: torch.dtype, impl: str):
    """Prefill: one varlen batch of four sequences, conv state written back."""
    batch = 4
    x = create_random(dim, seqlen, dtype=dtype)
    weight = create_random(dim, WIDTH, dtype=dtype)
    bias = create_random(dim, dtype=dtype)
    conv_states = create_random(batch, dim, WIDTH - 1, dtype=dtype)
    lengths = [seqlen // batch] * batch
    lengths[-1] += seqlen - sum(lengths)
    query_start_loc = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        dtype=torch.int32,
        device=DEFAULT_DEVICE,
    )
    cache_indices = torch.arange(batch, dtype=torch.int32, device=DEFAULT_DEVICE)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=DEFAULT_DEVICE)
    return marker.do_bench(
        FWD_FN_MAP[impl],
        input_args=(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            True,
            PAD_SLOT_ID,
        ),
        # x and conv_states are read-modify-write, so both need cloning.
        graph_clone_args=(0, 1, 2, 3),
        memory_args=(x, weight, bias, conv_states),
        memory_output=(x, conv_states),
    )


@marker.parametrize("batch", [1, 8, 64, 256], [64])
@marker.parametrize("dim", [2048, 4096, 8192], [4096])
@marker.parametrize("dtype", [torch.float16, torch.bfloat16])
@marker.benchmark("impl", ["jit", "aot"])
def benchmark_update(batch: int, dim: int, dtype: torch.dtype, impl: str):
    """Decode: one token per sequence, conv state gathered by slot index."""
    entries = max(batch * 4, 64)
    x = create_random(batch, dim, 1, dtype=dtype)
    conv_state = create_random(entries, dim, WIDTH - 1, dtype=dtype)
    weight = create_random(dim, WIDTH, dtype=dtype)
    bias = create_random(dim, dtype=dtype)
    conv_state_indices = torch.randperm(entries, device=DEFAULT_DEVICE)[:batch].to(
        torch.int32
    )
    return marker.do_bench(
        UPDATE_FN_MAP[impl],
        input_args=(
            x,
            conv_state,
            weight,
            bias,
            True,
            None,
            conv_state_indices,
            PAD_SLOT_ID,
        ),
        # conv_state is the large pool; only the gathered rows are touched, so
        # leave it out of the rotation and count just those rows as traffic.
        graph_clone_args=(0, 2, 3, 6),
        memory_args=(x, weight, bias, conv_state_indices),
        memory_output=(x,),
    )


if __name__ == "__main__":
    benchmark_fwd.run()
    benchmark_update.run()

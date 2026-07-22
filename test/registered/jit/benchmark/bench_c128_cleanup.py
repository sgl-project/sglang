"""Benchmark fused offline C128 speculative-draft state cleanup."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, create_empty
from sglang.jit_kernel.dsv4 import (
    C128DraftCleanup,
    clear_unaccepted_c128_draft_states,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

RING_SIZE = 256
HALF = 512
STATE_WIDTH = HALF * 2

BENCHMARK_CONFIGS = [
    (1, 1, 4),
    (8, 1, 4),
    (31, 1, 4),
    (31, 8, 4),
    (31, 32, 4),
    (31, 1, 8),
    (64, 1, 4),
]
CI_CONFIGS = [(31, 1, 4)]


def _clear_with_loop(
    states: Sequence[torch.Tensor],
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    num_draft_tokens: int,
) -> None:
    for state in states:
        clear_unaccepted_c128_draft_states(
            state,
            req_pool_indices,
            seq_lens,
            accept_lens,
            ring_size=RING_SIZE,
            num_draft_tokens=num_draft_tokens,
        )


@marker.parametrize("dtype", [torch.float32, torch.bfloat16], ci_vals=[torch.float32])
@marker.parametrize(
    "num_states,batch_size,num_draft_tokens",
    BENCHMARK_CONFIGS,
    ci_vals=CI_CONFIGS,
)
@marker.parametrize("launch_mode", ["eager", "cuda_graph"])
@marker.benchmark("impl", ["loop", "fused"])
def benchmark(
    dtype: torch.dtype,
    num_states: int,
    batch_size: int,
    num_draft_tokens: int,
    launch_mode: str,
    impl: str,
):
    states = [
        create_empty(batch_size * RING_SIZE, STATE_WIDTH, dtype=dtype)
        for _ in range(num_states)
    ]
    cleanup = C128DraftCleanup(states, ring_size=RING_SIZE)

    req_pool_indices = torch.arange(
        batch_size, dtype=torch.int64, device=DEFAULT_DEVICE
    )
    seq_lens = torch.full(
        (batch_size,), RING_SIZE - 1, dtype=torch.int64, device=DEFAULT_DEVICE
    )
    accept_lens = torch.ones(batch_size, dtype=torch.int32, device=DEFAULT_DEVICE)

    if impl == "loop":

        def fn() -> None:
            _clear_with_loop(
                states,
                req_pool_indices,
                seq_lens,
                accept_lens,
                num_draft_tokens,
            )

    elif impl == "fused":

        def fn() -> None:
            cleanup.clear(
                req_pool_indices,
                seq_lens,
                accept_lens,
                num_draft_tokens=num_draft_tokens,
            )

    else:
        raise ValueError(f"Unknown implementation: {impl}")

    return marker.do_bench(
        fn,
        use_cuda_graph=launch_mode == "cuda_graph",
        # The state tensors are write-only and intentionally stay fixed. Rotating
        # them would clone up to a gigabyte per iteration in the largest case.
        graph_clone_args=None,
        graph_clone_kwargs=None,
        # Only a few acceptance-dependent rows are touched, so aggregate state
        # size is not a meaningful bandwidth denominator.
        disable_log_bandwidth=True,
        memory_args=None,
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()

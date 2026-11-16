from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.kernels.ops.kv_canary.canary_info import launch_canary_info_kernel
from sglang.kernels.ops.kv_canary.verify import CANARY_SLOT_BYTES
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="weekly", runner_config="1-gpu-large")
register_amd_ci(est_time=300, suite="nightly-amd-kernel-1-gpu", nightly=True)

_NUM_SLOTS = 4096
_VOCAB_SIZE = 32768
_L2_FLUSH_FACTOR = 4
_MAX_COPIES = 4096

_X_NAMES = ["bs", "num_nodes"]
_X_VALS = get_benchmark_range(
    full_range=[(bs, w) for bs in (64, 256, 1024) for w in (1, 4)],
    ci_range=[(256, 1), (256, 4)],
)


def _build_inputs(
    *, bs: int, num_nodes: int, generator: torch.Generator, device: torch.device
) -> dict:
    total = bs * num_nodes
    canary_buf = torch.randint(
        0,
        256,
        (_NUM_SLOTS, CANARY_SLOT_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    accept_lens = torch.randint(
        1, num_nodes + 1, (bs,), dtype=torch.int32, generator=generator
    )
    inputs = dict(
        canary_buf=canary_buf,
        accept_lens=accept_lens,
        out_cache_loc=torch.randint(
            1, _NUM_SLOTS, (total,), dtype=torch.int64, generator=generator
        ).view(bs, num_nodes),
        next_token_ids=torch.randint(
            0, _VOCAB_SIZE, (total,), dtype=torch.int32, generator=generator
        ).view(bs, num_nodes),
        temperatures=torch.rand(bs, generator=generator) * 2.0,
        top_ps=torch.rand(bs, generator=generator),
        top_ks=torch.randint(1, 100, (bs,), dtype=torch.int32, generator=generator),
        min_ps=torch.rand(bs, generator=generator),
        out=torch.empty(bs, num_nodes, dtype=torch.int64),
    )
    return {
        name: None if tensor is None else tensor.to(device)
        for name, tensor in inputs.items()
    }


def _working_set_bytes(inputs: dict) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in inputs.values()
        if tensor is not None
    )


def _num_copies(*, inputs: dict, device: torch.device) -> int:
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    needed = -(-_L2_FLUSH_FACTOR * l2_bytes // _working_set_bytes(inputs))
    return max(1, min(_MAX_COPIES, needed))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES,
        x_vals=_X_VALS,
        line_arg="provider",
        line_vals=["canary"],
        line_names=["canary_info"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="kv-canary-info-perf",
        args={},
    )
)
def benchmark(bs: int, num_nodes: int, provider: str) -> Tuple[float, float, float]:
    device = torch.device(DEFAULT_DEVICE)
    generator = torch.Generator().manual_seed(bs * 31 + num_nodes)
    first = _build_inputs(
        bs=bs, num_nodes=num_nodes, generator=generator, device=device
    )
    copies = [first] + [
        _build_inputs(bs=bs, num_nodes=num_nodes, generator=generator, device=device)
        for _ in range(_num_copies(inputs=first, device=device) - 1)
    ]

    def fn() -> None:
        for inputs in copies:
            launch_canary_info_kernel(**inputs)

    return run_benchmark(fn, scale=len(copies))


if __name__ == "__main__":
    benchmark.run(print_data=True)

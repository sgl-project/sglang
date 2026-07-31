"""Compare Kimi-K3 w_kc BMM layouts for local and replicated-Q heads.

The current local Kimi-K3 layout makes the reduction (K) axis contiguous.
Replicated-Q currently materializes a standard contiguous full-head tensor.
This benchmark tests whether one K-contiguous full-head storage can serve both
decode and rank-local prefill/extend views.

Usage::

    python \
      test/registered/kernels/benchmark/communication/bench_kimi_k3_qrep_wkc_layout.py \
      --num-gpu 4
"""

from __future__ import annotations

import atexit
import functools
import logging
import os

import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed as dist
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import (
    get_benchmark_range,
    multigpu_bench_main,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires four GPUs and self-skips in CI",
)

WORLD_SIZE = 4
GLOBAL_HEADS = 96
LOCAL_HEADS = GLOBAL_HEADS // WORLD_SIZE
QK_NOPE_DIM = 128
KV_LORA_RANK = 512
NUM_TOKENS = get_benchmark_range([1, 17, 128], [1, 17, 128])
WEIGHT_NAMES = [
    "local_standard",
    "local_k_contiguous",
    "full_standard",
    "full_k_contiguous",
]
PROVIDERS = list(WEIGHT_NAMES)
if os.environ.get("SGLANG_QREP_BENCH_REVERSE") == "1":
    PROVIDERS.reverse()


@functools.cache
def _init_world() -> ps.GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    assert ps._WORLD is not None
    return ps._WORLD


def _k_contiguous(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(1, 2).contiguous().transpose(1, 2)


@functools.cache
def _weights() -> tuple[torch.Tensor, ...]:
    world = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    generator = torch.Generator(device=device)
    generator.manual_seed(32541 + world.rank_in_group)
    local_standard = torch.randn(
        LOCAL_HEADS,
        QK_NOPE_DIM,
        KV_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_k_contiguous = _k_contiguous(local_standard)
    full_standard = world.all_gather(local_standard, dim=0)
    # Gather the physical [H, V, K] backing so both the full tensor and its
    # local head slices retain K-contiguous [H, K, V] strides.
    full_k_contiguous = world.all_gather(
        local_k_contiguous.transpose(1, 2).contiguous(), dim=0
    ).transpose(1, 2)

    torch.testing.assert_close(local_standard, local_k_contiguous, atol=0, rtol=0)
    torch.testing.assert_close(full_standard, full_k_contiguous, atol=0, rtol=0)
    assert local_standard.stride() == (65536, 512, 1)
    assert local_k_contiguous.stride() == (65536, 1, 128)
    assert full_standard.stride() == (65536, 512, 1)
    assert full_k_contiguous.stride() == (65536, 1, 128)
    return (
        local_standard,
        local_k_contiguous,
        full_standard,
        full_k_contiguous,
    )


@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, provider: str):
    world = _init_world()
    if world.world_size != WORLD_SIZE:
        marker.skip("The Kimi-K3 TP4/DCP4 comparison requires exactly four ranks.")

    weights = dict(zip(WEIGHT_NAMES, _weights(), strict=True))
    weight = weights[provider]
    num_heads = weight.shape[0]
    device = weight.device
    generator = torch.Generator(device=device)
    generator.manual_seed(50484 + num_tokens)
    q_nope = torch.randn(
        num_heads,
        num_tokens,
        QK_NOPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    return marker.do_bench(
        torch.bmm,
        input_args=(q_nope, weight),
        graph_clone_args=(0,),
        sync_multigpu_fn=lambda: dist.barrier(world.device_group),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=(4,),
        main_fn=benchmark.run,
    )

"""Measure Kimi-K3 local Query production before and after q_b storage sharing.

The shared provider uses a contiguous rank slice of the full replicated q_b
buffer, matching the optimized prefill/extend path. Decode consumes the same
full replicated tensor in both designs and therefore has no runtime difference.

Usage::

    python \
      test/registered/kernels/benchmark/communication/bench_kimi_k3_qrep_local_view.py \
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
import torch.nn.functional as F
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import (
    get_benchmark_range,
    multigpu_bench_main,
)
from sglang.srt.layers.dcp.query_weights import (
    bind_parameter_to_replicated_rank_slice_,
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
Q_LORA_RANK = 1536
QK_NOPE_DIM = 128
ROPE_DIM = 64
KV_LORA_RANK = 512
QK_HEAD_DIM = QK_NOPE_DIM + ROPE_DIM
NUM_TOKENS = get_benchmark_range([1, 17, 128], [1, 17, 128])
PROVIDERS = ["separate_local_q_b", "shared_local_q_b"]
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


@functools.cache
def _weights() -> tuple[torch.Tensor, ...]:
    world = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    generator = torch.Generator(device=device)
    generator.manual_seed(32541 + world.rank_in_group)
    separate_q_b = torch.nn.Parameter(
        torch.randn(
            LOCAL_HEADS * QK_HEAD_DIM,
            Q_LORA_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        requires_grad=False,
    )
    full_q_b = world.all_gather(separate_q_b.data, dim=0)
    shared_q_b = torch.nn.Parameter(separate_q_b.detach().clone(), requires_grad=False)
    bind_parameter_to_replicated_rank_slice_(
        shared_q_b,
        full_q_b,
        rank=world.rank_in_group,
        world_size=world.world_size,
    )
    w_kc = torch.randn(
        LOCAL_HEADS,
        QK_NOPE_DIM,
        KV_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    return separate_q_b, shared_q_b, w_kc


def _produce_local_query(
    q_lora: torch.Tensor,
    q_b_weight: torch.Tensor,
    w_kc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.linear(q_lora, q_b_weight).view(q_lora.shape[0], LOCAL_HEADS, QK_HEAD_DIM)
    q_nope, q_rope = query.split((QK_NOPE_DIM, ROPE_DIM), dim=-1)
    q_nope = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    return q_nope, q_rope


@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, provider: str):
    world = _init_world()
    if world.world_size != WORLD_SIZE:
        marker.skip("The Kimi-K3 TP4/DCP4 comparison requires exactly four ranks.")

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    generator = torch.Generator(device=device)
    generator.manual_seed(50484 + num_tokens)
    q_lora = torch.randn(
        num_tokens,
        Q_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    separate_q_b, shared_q_b, w_kc = _weights()
    q_b_weight = separate_q_b if provider == "separate_local_q_b" else shared_q_b

    return marker.do_bench(
        _produce_local_query,
        input_args=(q_lora, q_b_weight, w_kc),
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

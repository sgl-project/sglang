"""Compare the three Kimi-K3 DCP Query production strategies on four GPUs.

This benchmark covers the BF16 decode chain from normalized q_lora through
q_b projection and the absorbed-q BMM. The sharded variants then use either
the production Query AllGather or direct-final NVLS publication. Replicated-Q
uses startup-gathered full-head weights and performs four times the per-rank
projection/BMM work without a per-layer Query collective.

RoPE and attention are deliberately excluded so this isolates the decision
boundary changed by --dcp-direct-q-gather and --dcp-replicate-q-proj.

Usage::

    python test/registered/kernels/benchmark/communication/bench_kimi_k3_dcp_query_chain.py \
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
from sglang.srt.layers.dcp.query import DCPDirectFinalQueryGatherer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
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
NUM_TOKENS = get_benchmark_range([1, 2, 4, 8, 16, 17, 32, 64, 128], [1, 17, 128])
PROVIDERS = ["allgather", "direct_final", "replicated_q"]
MAX_TOKENS = max(NUM_TOKENS)


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
def _gatherer() -> DCPDirectFinalQueryGatherer:
    world = _init_world()
    return DCPDirectFinalQueryGatherer(
        group=world,
        max_tokens=MAX_TOKENS,
        local_heads=LOCAL_HEADS,
        nope_dim=KV_LORA_RANK,
        rope_dim=ROPE_DIM,
        device=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
    )


@functools.cache
def _weights() -> tuple[torch.Tensor, ...]:
    world = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    generator = torch.Generator(device=device)
    generator.manual_seed(32541 + world.rank_in_group)
    local_q_b = torch.randn(
        LOCAL_HEADS * QK_HEAD_DIM,
        Q_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_w_kc = torch.randn(
        LOCAL_HEADS,
        QK_NOPE_DIM,
        KV_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    full_q_b = world.all_gather(local_q_b, dim=0)
    full_w_kc = world.all_gather(local_w_kc, dim=0)
    return local_q_b, local_w_kc, full_q_b, full_w_kc


def _produce_absorbed_query(
    q_lora: torch.Tensor,
    q_b_weight: torch.Tensor,
    w_kc: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.linear(q_lora, q_b_weight).view(q_lora.shape[0], num_heads, QK_HEAD_DIM)
    q_nope, q_rope = query.split((QK_NOPE_DIM, ROPE_DIM), dim=-1)
    q_nope_absorbed = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    return q_nope_absorbed, q_rope


def _production_allgather(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world = _init_world()
    combined = torch.cat((q_rope.transpose(0, 1), q_nope.transpose(0, 1)), dim=-1)
    gathered = world.all_gather(combined, dim=0)
    final_rope, final_nope = gathered.split((ROPE_DIM, KV_LORA_RANK), dim=-1)
    return final_nope.transpose(0, 1), final_rope.transpose(0, 1)


@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, provider: str):
    world = _init_world()
    if world.world_size != WORLD_SIZE:
        marker.skip("The Kimi-K3 TP4/DCP4 comparison requires exactly four ranks.")

    gatherer = _gatherer()
    if provider == "direct_final" and gatherer.state.symm_mem_hdl.multicast_ptr == 0:
        marker.skip("NVLS multicast is unavailable.")

    device = gatherer.state.device
    generator = torch.Generator(device=device)
    generator.manual_seed(50484 + num_tokens)
    q_lora = torch.randn(
        num_tokens,
        Q_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_q_b, local_w_kc, full_q_b, full_w_kc = _weights()

    if provider == "replicated_q":

        def fn(x: torch.Tensor):
            return _produce_absorbed_query(x, full_q_b, full_w_kc, GLOBAL_HEADS)

    elif provider == "allgather":

        def fn(x: torch.Tensor):
            q_nope, q_rope = _produce_absorbed_query(
                x, local_q_b, local_w_kc, LOCAL_HEADS
            )
            return _production_allgather(q_nope, q_rope)

    else:

        def fn(x: torch.Tensor):
            q_nope, q_rope = _produce_absorbed_query(
                x, local_q_b, local_w_kc, LOCAL_HEADS
            )
            return gatherer(q_nope, q_rope)

    return marker.do_bench(
        fn,
        input_args=(q_lora,),
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

"""PCIe pair all-reduce: exact (r0 + r1) + (r2 + r3) on every call, in both rounding
modes, under back-to-back calls of mixed sizes, block counts and variants, eager and
replayed from one CUDA graph (what exercises the flag and slot protocol); and a
row's bits independent of the other rows in the message.

Needs 4 GPUs where ranks 0-1 and 2-3 each share a PCIe switch (e.g. 4x A100 PCIe):

    python test/registered/kernels/ops/communication/test_pcie_pair_all_reduce.py
"""

import os
import random

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.pcie_pair_ar import (
    PciePairCommunicator,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-h100",
    disabled="needs 4 GPUs on two PCIe switches; CI runners are NVLink",
)

HIDDEN = 5120


@pytest.fixture(scope="module")
def groups():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", device_id=dev)
    cpu = dist.new_group(backend="gloo")
    yield rank, dev, cpu
    dist.barrier()
    dist.destroy_process_group()


def _comm(cpu, dev, wide):
    with (
        envs.SGLANG_PCIE_PAIR_AR_FP32.override(wide),
        envs.SGLANG_PCIE_PAIR_AR_MAX_BYTES.override(1 << 30),
    ):
        comm = PciePairCommunicator(cpu, dev)
    if comm.disabled:
        pytest.skip("ranks 0-1 and 2-3 are not two PCIe switch pairs")
    return comm


def _reference(x, wide):
    parts = [torch.empty_like(x) for _ in range(4)]
    dist.all_gather(parts, x)
    f = [p.float() for p in parts]
    if wide:  # pair sums kept in fp32, one rounding
        return ((f[0] + f[1]) + (f[2] + f[3])).bfloat16()
    pair01, pair23 = (f[0] + f[1]).bfloat16(), (f[2] + f[3]).bfloat16()
    return (pair01.float() + pair23.float()).bfloat16()


@pytest.mark.parametrize("wide", [True, False])
def test_exact_eager_and_graph(groups, wide):
    rank, dev, cpu = groups
    comm = _comm(cpu, dev, wide)
    rng = random.Random(7)  # same sequence on every rank
    gen = torch.Generator(device=dev).manual_seed(100 + rank)

    def case():
        rows = rng.choice([1, 2, 3, 4, 6, 7, 12, 18, 24, 33, 48, 64, 100, 300])
        x = torch.randn(rows, HIDDEN, device=dev, generator=gen).bfloat16() * 4
        blocks = rng.choice([None, 1, 2, 3, 4, 8, 16])
        one_shot = rng.choice([None, True, False])
        return x, blocks, one_shot

    def run(x, blocks, one_shot):
        return comm.pcie_pair_all_reduce(x, blocks=blocks, one_shot=one_shot)

    # Eager: checked after the whole burst, so calls run back to back.
    for burst in range(20):
        cases = [case() for _ in range(100)]
        outs = [run(*c) for c in cases]
        bad = sum(
            not torch.equal(out, _reference(x, wide))
            for (x, _, _), out in zip(cases, outs)
        )
        assert bad == 0, f"wide={wide} eager burst {burst}: {bad} wrong"

    # Graph: one capture of 50 mixed calls, replayed with fresh inputs.
    cases = [case() for _ in range(50)]
    static = [x.clone() for x, _, _ in cases]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for (_, b, o), x in zip(cases, static):
            run(x, b, o)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [run(x, b, o) for (_, b, o), x in zip(cases, static)]
    for rep in range(30):
        for x in static:
            x.copy_(torch.randn(x.shape, device=dev, generator=gen).bfloat16())
        graph.replay()
        bad = sum(
            not torch.equal(out, _reference(x, wide)) for x, out in zip(static, outs)
        )
        assert bad == 0, f"wide={wide} graph replay {rep}: {bad} wrong"
    torch.cuda.synchronize()
    comm.destroy()


def test_row_bits_do_not_depend_on_message(groups):
    rank, dev, cpu = groups
    comm = _comm(cpu, dev, True)
    torch.manual_seed(42 + rank)
    token = torch.randn(1, HIDDEN, device=dev).bfloat16()
    ref = comm.pcie_pair_all_reduce(token.clone())[0].clone()
    for rows in (2, 3, 6, 24, 64):
        x = torch.cat([token, torch.randn(rows - 1, HIDDEN, device=dev).bfloat16()])
        assert torch.equal(comm.pcie_pair_all_reduce(x)[0], ref), rows
    comm.destroy()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

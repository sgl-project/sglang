"""Correctness tests for the K3 column-parallel up_proj + multicast
all-gather + add3 (gemm_ag) kernels.

Covers: the fused ``out = up_proj(x) + b (+ c)`` against a torch fp32
reference for every M in the dispatch table, both phases of the push
workspace double buffer, interleaving with the push all-reduce (the two
share the workspace and its phase counters), a CUDA-graph capture/replay
stress loop (regression for a phase-counter/PDL ordering hang), and a
non-K3 template shape (N halved) to keep the kernel generic.

Usage::

    python test/registered/jit/kimi_k3/test_gemm_ag.py   # relaunches under torchrun (8 GPUs)
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.kimi_k3 import all_reduce, gemm_ag
from sglang.jit_kernel.mp import register_comm_cleanup
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

K, N = gemm_ag.K, gemm_ag.N
MB = 1024 * 1024

BATCH = get_ci_test_range(list(range(1, gemm_ag.MAX_TOKENS + 1)), [1, 8, 12])


def _precompile(num_gpus):
    del num_gpus
    gemm_ag._jit_module()


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return coord.cpu_group


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(
        cpu_group, device, max_pull_size=1 * MB, max_push_size=2 * MB
    )
    if comm.disabled or comm.mc_base_ptr == 0:
        raise RuntimeError("gemm_ag requires CustomAllReduceV2 with multicast")
    all_reduce.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    register_comm_cleanup(comm)
    return comm


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


@cache_once
def _weight() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234)
    return (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16).to(_device())


def _make_inputs(m: int, n: int = N, k: int = K):
    g = torch.Generator(device="cpu").manual_seed(500 + m + n)
    x = (torch.randn(m, k, generator=g) * 0.05).to(torch.bfloat16).to(_device())
    b = torch.randn(m, n, generator=g).to(torch.bfloat16).to(_device())
    c = torch.randn(m, n, generator=g).to(torch.bfloat16).to(_device())
    return x, b, c


def _barrier():
    torch.cuda.synchronize()
    dist.barrier()


def _ref(x, weight, b, c):
    ref = x.float() @ weight.float().t() + b.float()
    if c is not None:
        ref += c.float()
    return ref.to(torch.bfloat16)


@pytest.mark.parametrize("bs", BATCH)
@pytest.mark.parametrize("use_c", [False, True])
@torch.inference_mode()
def test_gemm_ag_correctness(bs: int, use_c: bool):
    """Both phases of the double buffer, plus a push AR interleaved (odd
    counter advance) followed by another call."""
    comm = _init_comm()
    weight = _weight()
    x, b, c = _make_inputs(bs)
    c_arg = c if use_c else None
    out = torch.empty(bs, N, dtype=torch.bfloat16, device=_device())
    ref = _ref(x, weight, b, c_arg)
    for _ in range(2):  # both phases
        out.zero_()
        _barrier()
        gemm_ag.gemm_ag_up_proj(
            comm.world_size, x, weight, b, c_arg, out, ws_mc_base=comm.mc_base_ptr
        )
        _barrier()
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    # interleave another push-workspace user, then run once more
    probe = torch.ones(1024, dtype=torch.bfloat16, device=_device())
    all_reduce.all_reduce_push_res(comm.world_size, probe, ws_mc_base=comm.mc_base_ptr)
    out.zero_()
    _barrier()
    gemm_ag.gemm_ag_up_proj(
        comm.world_size, x, weight, b, c_arg, out, ws_mc_base=comm.mc_base_ptr
    )
    _barrier()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("bs", get_ci_test_range([1, 8, 11, 12], [12]))
@torch.inference_mode()
def test_gemm_ag_graph_stress(bs: int):
    """Rank-skewed eager warmup + a 64-call CUDA graph replayed 20x:
    regression test for the phase-counter hang (back-to-back PDL calls
    reading a counter before the previous consumer's flip)."""
    comm = _init_comm()
    weight = _weight()
    x, b, c = _make_inputs(bs)
    out = torch.empty(bs, N, dtype=torch.bfloat16, device=_device())
    ref = _ref(x, weight, b, c)

    def run_once():
        gemm_ag.gemm_ag_up_proj(
            comm.world_size, x, weight, b, c, out, ws_mc_base=comm.mc_base_ptr
        )

    _barrier()
    for _ in range(50):  # eager, no barriers: rank-skewed like a warmup loop
        run_once()
    _barrier()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(64):
                run_once()
        _barrier()
        for _ in range(20):
            graph.replay()
            stream.synchronize()
        dist.barrier()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_gemm_ag_general_shape():
    """The kernel template is not K3-specific: instantiate it at N halved
    (K=3584, N=3584 -> 448 columns per rank) and check correctness."""
    from sglang.jit_kernel.utils import is_arch_support_pdl, load_jit, make_cpp_args

    comm = _init_comm()
    k, n, max_m = 3584, 3584, 4
    args = make_cpp_args(k, n, max_m, is_arch_support_pdl())
    cls = f"GEMMAGKernel<{args}>"
    module = load_jit(
        "kimi_k3_gemm_ag",
        *args,
        cuda_files=["kimi_k3/comm/gemm_ag.cuh"],
        cuda_wrappers=[("run", f"{cls}::run")],
        extra_cuda_cflags=["-O3"],
    )
    g = torch.Generator(device="cpu").manual_seed(4321)
    weight = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).to(_device())
    for m in (1, max_m):
        x, b, c = _make_inputs(m, n=n, k=k)
        out = torch.empty(m, n, dtype=torch.bfloat16, device=_device())
        ref = _ref(x, weight, b, c)
        out.zero_()
        _barrier()
        module.run(comm.obj, x, weight, b, c, out, comm.mc_base_ptr)
        _barrier()
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(8,),
        pre_launch_fn=_precompile,
    )

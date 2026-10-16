"""The fused [base | lora_a] all-reduce of a row-parallel LoRA layer
(SGLANG_OPT_LORA_ROW_PARALLEL_FUSED_ALL_REDUCE) must give the same bits as the
two custom all-reduces it replaces, eager and under a CUDA graph, at every
decode size it is enabled for.

Usage::

    python test_lora_row_parallel_fused_all_reduce.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.lora import layers as lora_layers
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, RANK = 7168, 16
MAX_TOKENS = 16


@cache_once
def _init_comm_once() -> CustomAllReduceV2:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    comm = CustomAllReduceV2(coord.cpu_group, torch.device(f"cuda:{local_rank}"))
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


def test_fusion_eligibility(monkeypatch):
    base = torch.empty(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16)
    lora_a = torch.empty(MAX_TOKENS, RANK, dtype=torch.bfloat16)
    monkeypatch.setattr(lora_layers, "_ROW_PARALLEL_FUSED_ALL_REDUCE", True)
    monkeypatch.setattr(
        lora_layers, "_ROW_PARALLEL_FUSED_ALL_REDUCE_MAX_TOKENS", MAX_TOKENS
    )
    assert lora_layers._fuse_row_lora_all_reduce(base, lora_a)
    too_many = torch.empty(MAX_TOKENS + 1, HIDDEN, dtype=torch.bfloat16)
    assert not lora_layers._fuse_row_lora_all_reduce(
        too_many, torch.empty(MAX_TOKENS + 1, RANK, dtype=torch.bfloat16)
    )
    assert not lora_layers._fuse_row_lora_all_reduce(base, lora_a.float())
    monkeypatch.setattr(lora_layers, "_ROW_PARALLEL_FUSED_ALL_REDUCE", False)
    assert not lora_layers._fuse_row_lora_all_reduce(base, lora_a)


def _both(all_reduce, base: torch.Tensor, lora_a: torch.Tensor):
    separate = (all_reduce(base), all_reduce(lora_a))
    fused = lora_layers._fused_row_lora_all_reduce(all_reduce, base, lora_a)
    return separate, fused


@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_fused_all_reduce_bit_equal(use_graph: bool) -> None:
    comm = _init_comm_once()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    for m in range(1, MAX_TOKENS + 1):
        g = torch.Generator(device=device).manual_seed(1000 * m + rank)
        base = (torch.randn(m, HIDDEN, generator=g, device=device) * 4).bfloat16()
        lora_a = (torch.randn(m, RANK, generator=g, device=device) * 0.1).bfloat16()
        if use_graph:
            static_base, static_lora_a = base.clone(), lora_a.clone()
            graph = torch.cuda.CUDAGraph()
            with comm.capture():
                with torch.cuda.graph(graph):
                    separate, fused = _both(
                        comm.custom_all_reduce, static_base, static_lora_a
                    )
            graph.replay()
        else:
            separate, fused = _both(comm.custom_all_reduce, base, lora_a)
        torch.cuda.synchronize()
        assert torch.equal(separate[0], fused[0]), f"base differs at m={m}"
        assert torch.equal(separate[1], fused[1]), f"lora_a differs at m={m}"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))

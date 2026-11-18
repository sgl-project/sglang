"""CustomAllreduce (v1) CUDA-graph capture when graph inputs are VMM memory.

With expandable_segments, CUDA-graph inputs come from cuMemCreate memory and
cudaIpcGetMemHandle rejects them when v1 registers graph buffers
(sglang#44291). v1 must stage such inputs through its pre-registered IPC
buffer instead.

Usage::

    python test/registered/kernels/ops/communication/test_custom_all_reduce_v1_vmm.py
"""

import os

# PyTorch reads these at the first CUDA allocation, so they have to be set
# before anything touches the GPU. Newer torch prefers PYTORCH_ALLOC_CONF.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import atexit

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.cuda_vmm_utils import is_vmm_pointer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="extra-b", runner_config="8-gpu-h200")

TEST_SIZES = [16, 1024, 32 * 1024, 1024 * 1024]
TEST_LAYERS = 4


@cache_once
def _init_groups():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    device = torch.device(f"cuda:{local_rank}")
    nccl_group = dist.new_group(backend="nccl", device_id=device)
    return device, coord.cpu_group, nccl_group


@cache_once
def _init_comm() -> CustomAllreduce:
    device, cpu_group, _ = _init_groups()
    comm = CustomAllreduce(cpu_group, device)
    if comm.disabled:
        # A skip would turn a failing run into "skipped" on hosts without P2P.
        raise RuntimeError("CustomAllreduce (v1) is disabled on this system")
    register_comm_cleanup(comm)
    return comm


def test_allocator_is_vmm():
    device, _, _ = _init_groups()
    probe = torch.zeros(16, dtype=torch.bfloat16, device=device)
    assert is_vmm_pointer(probe.data_ptr()), (
        "expandable_segments is not active; check PYTORCH_ALLOC_CONF"
    )


@pytest.mark.parametrize("from_graph_pool", [False, True])
@pytest.mark.parametrize("size", TEST_SIZES)
@torch.inference_mode()
def test_graph_all_reduce_matches_nccl(size: int, from_graph_pool: bool):
    device, _, nccl_group = _init_groups()
    comm = _init_comm()
    graph_inp = torch.zeros((TEST_LAYERS, size), dtype=torch.bfloat16, device=device)
    assert is_vmm_pointer(graph_inp.data_ptr())
    assert not comm.tms_cudagraph
    assert comm.should_custom_ar(graph_inp[0])

    graph = torch.cuda.CUDAGraph()
    outs, ptrs = [], []
    with comm.capture():
        with torch.cuda.graph(graph):
            for i in range(TEST_LAYERS):
                # In a server the input is an activation from the graph's own
                # private pool, so cover that source as well.
                x = graph_inp[i] + 0 if from_graph_pool else graph_inp[i]
                ptrs.append(x.data_ptr())
                outs.append(comm.custom_all_reduce(x))
    torch.cuda.synchronize()
    # Checked after capture so the test makes no driver calls inside it; the
    # graph keeps these blocks mapped.
    assert all(is_vmm_pointer(p) for p in ptrs)

    # Small integers stay exact in bf16 after the sum, so compare exactly.
    inp = torch.randint(0, 16, (TEST_LAYERS, size), dtype=torch.bfloat16, device=device)
    ref = inp.clone()
    dist.all_reduce(ref, group=nccl_group)
    graph_inp.copy_(inp)
    graph.replay()
    torch.testing.assert_close(torch.stack(outs), ref, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2,))

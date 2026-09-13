"""A token must retain its bits when batch size changes its receiving TP rank.

Run with ``python test_deterministic_reduce_scatter.py --num-gpu 4``.
This exercises the collective used before DeepEP MoE without a model or DeepEP.
"""

import os

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=45, stage="base-b", runner_config="4-gpu-b200")


@pytest.fixture(scope="module")
def group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # Match --enable-deterministic-inference's CUDA collective settings.
    os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    ps.initialize_model_parallel(tensor_model_parallel_size=world_size)
    yield ps.get_tp_group()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("use_graph", [False, True])
def test_batch_and_destination_invariance(group, dtype, use_graph):
    torch.manual_seed(42 + group.rank_in_group)
    # Non-integer contributions expose changes in floating-point sum order.
    token = torch.randn(2048, device="cuda", dtype=dtype)
    reference = None
    for local_tokens in (1, 2, 3, 8, 32, 128):
        input_ = token.repeat(local_tokens * group.world_size, 1)
        original = input_.clone()
        output = torch.empty((local_tokens, 2048), device="cuda", dtype=dtype)
        if use_graph:
            with group.graph_capture() as capture:
                for _ in range(3):
                    group.reduce_scatter_tensor(output, input_)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture.stream):
                    group.reduce_scatter_tensor(output, input_)
            graph.replay()
        else:
            group.reduce_scatter_tensor(output, input_)

        # Compare all destinations to catch rank-dependent reduction order.
        all_outputs = torch.empty_like(input_)
        dist.all_gather_into_tensor(all_outputs, output, group=group.device_group)
        if reference is None:
            reference = all_outputs[0].clone()
        torch.testing.assert_close(
            all_outputs, reference.expand_as(all_outputs), atol=0, rtol=0
        )
        torch.testing.assert_close(input_, original, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

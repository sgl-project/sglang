import os

import pytest
import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.mem_cache.engram.peer_embedding import PeerVocabParallelEmbedding
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.fixture(scope="module")
def parallel():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=local_rank
    )
    publish_build_topology(tp_size=world_size, attn_dp_size=2, world_rank=rank)
    ps.initialize_model_parallel()
    yield get_parallel()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_peer_gather_into_output(parallel, dtype):
    rows, dim = 128, 7
    embedding = PeerVocabParallelEmbedding(
        rows,
        dim,
        params_dtype=dtype,
        use_attn_tp_group=True,
    )
    # Distinct tables in each DP replica catch lookups into the wrong group.
    table = torch.arange(rows * dim, device="cuda", dtype=torch.float32).reshape(
        rows, dim
    )
    table = ((table % 41) + 50 * parallel.attn_dp_rank).to(dtype)
    embedding.weight_loader(embedding.weight, table)
    embedding.finish_loading()

    ids = torch.tensor([[0, -1, 63, -1], [64, -1, 127, -1]], device="cuda")[:, ::2]
    out = embedding.allocate_output((*ids.shape, dim), ids.device)
    result = embedding.gather(ids, out=out)
    assert result is out
    torch.testing.assert_close(
        embedding.reduce(result), table.bfloat16()[ids], atol=0, rtol=0
    )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

"""Actual EngramEmbedding peer mapping, sharded lookup and graph reload parity."""

import os

import pytest
import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.engram import FP8_BLOCK_SIZE, EngramEmbedding
from sglang.srt.runtime_context import get_model
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.fixture(scope="module")
def group():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Peer table integration is currently enabled only on SM10x")
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        timeout=180,
    )
    publish_build_topology(tp_size=world, world_rank=rank, device="cuda")
    ps.initialize_model_parallel()
    yield ps.get_tp_group()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _reference(weight, scale, ids):
    owned = (ids >= 0) & (ids < weight.shape[0])
    safe = ids.clamp(0, weight.shape[0] - 1)
    values = weight[safe].float().unflatten(-1, (256 // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE))
    values = (values * scale[safe].float().unsqueeze(-1)).flatten(-2)
    return values.bfloat16().masked_fill(~owned.unsqueeze(-1), 0)


@pytest.mark.parametrize("empty_shards", [False, True])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_lookup_reload_and_graph(group, empty_shards, id_dtype):
    bounds = [0, 1, 2, 3] if empty_shards else [0]
    if not empty_shards:
        for i in range(24):
            bounds.append(bounds[-1] + 101 + i)
    size = bounds[-1]
    torch.manual_seed(42)
    weight = torch.randn(size, 256, device="cuda").to(torch.float8_e4m3fn)
    exponent = torch.randint(
        120, 132, (size, 256 // FP8_BLOCK_SIZE), device="cuda", dtype=torch.uint8
    )
    exponent[0] = 0
    scale = exponent.view(torch.float8_e8m0fnu)
    with torch.device("cuda"):
        peer = EngramEmbedding(size, 256, 1, tuple(bounds))
    assert peer._peer_storage is not None
    with torch.device("cuda"), get_model().override(load_format="dummy"):
        assert EngramEmbedding(size, 256, 1, tuple(bounds))._peer_storage is None
    peer._load_rows(peer.weight, weight)
    peer._load_rows(peer.scale, scale)
    peer.finish_load("test")
    ids = torch.tensor(
        [-1, 0, 1, size - 1, size, *bounds], device="cuda", dtype=id_dtype
    ).view(-1, 1)
    expected = _reference(weight, scale, ids)
    torch.testing.assert_close(peer(ids), expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        peer(ids)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = peer(ids)
    # Each rank issues distinct requests; peer lookup needs no forward collective.
    torch.manual_seed(100 + group.rank_in_group)
    ids.copy_(torch.randint(-1, size + 1, ids.shape, device="cuda", dtype=id_dtype))
    graph.replay()
    torch.testing.assert_close(actual, _reference(weight, scale, ids), rtol=0, atol=0)
    # A zero-token rank must not wait for other ranks' lookups.
    local_ids = ids[: group.rank_in_group]
    torch.testing.assert_close(
        peer(local_ids, cp_all_tokens=True),
        _reference(weight, scale, local_ids),
        rtol=0,
        atol=0,
    )
    # Quiesce readers on every rank before mutating shared table storage.
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group.cpu_group)
    # In-place reload preserves the captured peer addresses while updating values.
    weight = (-weight.float()).to(torch.float8_e4m3fn)
    peer._load_rows(peer.weight, weight)
    peer.finish_load("reload")
    graph.replay()
    torch.testing.assert_close(actual, _reference(weight, scale, ids), rtol=0, atol=0)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group.cpu_group)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

"""Sharded Engram reconstruction preserves BF16 bits through HIP collectives."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.engram import EngramEmbedding
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_amd_ci(est_time=60, suite="stage-c-kernel-test-4-gpu-amd-mi35x")
pytestmark = pytest.mark.skipif(
    not is_hip() or "LOCAL_RANK" not in os.environ,
    reason="run through the four-GPU entry point",
)


@pytest.fixture(scope="module")
def group():
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    # initialize_model_parallel reads the widths from the published topology
    publish_build_topology(tp_size=world, world_rank=rank)
    initialize_model_parallel()
    yield get_tp_group()
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


def table(rows):
    weight = torch.ones(rows, 128, dtype=torch.float32)
    weight[:, 1::4] = -1
    weight[:, 2::4] = 0.5
    weight[:, 3::4] = -0.0
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.tensor([0, 1, 120, 140], dtype=torch.uint8).repeat(rows, 1)
    reference = (
        (
            weight.float().reshape(rows, 4, 32)
            * scale.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        )
        .flatten(1)
        .to(torch.bfloat16)
    )
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(False),
        torch.device("cuda"),
    ):
        embed = EngramEmbedding(rows, 128, layer_id=1)
    embed.weight.weight_loader(embed.weight, weight)
    embed.scale.weight_loader(embed.scale, scale.view(torch.float8_e8m0fnu))
    embed.finish_load(label="test")
    return embed, reference


@pytest.mark.parametrize("rows", [1, 17], ids=["1", "17"])
def test_eager_and_graph_reconstruction(group, rows):
    """Reconstructing a sharded row must retain signed zero and BF16 subnormals."""
    embed, reference = table(rows)
    ids = torch.arange(16, device="cuda", dtype=torch.int64).view(-1, 1) % rows
    # the fixture publishes plain TP (attn_dp_size == 1): the all-reduce path
    eager = embed(ids)
    assert torch.equal(
        eager.cpu().view(torch.int16), reference[ids.cpu()].view(torch.int16)
    )
    graph = torch.cuda.CUDAGraph()
    with group.graph_capture() as capture:
        embed(ids)
        with torch.cuda.graph(graph, stream=capture.stream):
            output = embed(ids)
    for shift in (1, 3):
        ids.add_(shift).remainder_(rows)
        graph.replay()
        assert torch.equal(
            output.cpu().view(torch.int16), reference[ids.cpu()].view(torch.int16)
        )


@pytest.mark.parametrize("scatter", [False, True], ids=["False", "True"])
def test_dp_shard_reconstruction(group, scatter):
    """DP distribution must preserve the owning shard's BF16 payload bits."""
    embed, reference = table(17)
    rank, world = group.rank_in_group, group.world_size
    ids = torch.tensor([[rank]], device="cuda", dtype=torch.int64)
    batch = SimpleNamespace(dp_padding_mode=SimpleNamespace(is_max_len=lambda: scatter))

    # Supply a fixed one-token-per-rank DP schedule; exercise real collectives.
    def gather(dst, src, _batch):
        group.all_gather_into_tensor(dst, src)

    # The fixture built plain TP groups; present them to the layer and the DP
    # helpers as a TP-wide attention-DP layout (one attention rank per GPU).
    real = get_parallel()
    dp_layout = SimpleNamespace(
        tp_size=world,
        tp_rank=rank,
        tp_group=real.tp_group,
        attn_tp_group=real.tp_group,
        attn_dp_size=world,
        attn_dp_rank=rank,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
    )
    with (
        patch("sglang.srt.layers.engram.get_global_dp_buffer_len", return_value=world),
        patch("sglang.srt.layers.engram.get_parallel", return_value=dp_layout),
        patch("sglang.srt.layers.dp_attention.get_parallel", return_value=dp_layout),
        patch("sglang.srt.layers.engram.dp_gather_replicate", side_effect=gather),
        patch(
            "sglang.srt.layers.dp_attention.get_dp_local_info",
            return_value=(
                torch.tensor(rank, device="cuda", dtype=torch.int32),
                torch.tensor(1, device="cuda", dtype=torch.int32),
            ),
        ),
    ):
        output = embed._dp_sharded_lookup(ids, batch)
        assert torch.equal(
            output.cpu().view(torch.int16), reference[ids.cpu()].view(torch.int16)
        )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

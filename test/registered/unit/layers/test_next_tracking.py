import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("bs", [0, 1, 3, 129])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mamba_track_gather_graph(bs, dtype):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        gather_mamba_track_indices,
    )

    mapping = torch.arange(300 * 12, device="cuda", dtype=dtype).reshape(300, 12)[
        ::2, ::3
    ]
    requests = torch.arange(bs * 2, device="cuda", dtype=dtype)[::2] % 150
    requests -= 150
    positions = torch.arange(bs * 2, device="cuda", dtype=dtype)[::2] % 4
    gather_mamba_track_indices(mapping, requests, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_mamba_track_indices(mapping, requests, positions)
    for i in range(3):
        positions.add_(1).remainder_(4)
        mapping.add_(i)
        graph.replay()
        expected = (
            mapping[requests.long()]
            .gather(1, positions.long().unsqueeze(1))
            .squeeze(1)
            .long()
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [0, 1, 3, 129])
@pytest.mark.parametrize("position", [0, 1, 3])
def test_mamba_uniform_track_gather_graph(bs, position):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        gather_mamba_track_indices,
    )

    mapping = torch.arange(300 * 12, device="cuda").reshape(300, 12)[::2, ::3]
    requests = torch.arange(bs * 2, device="cuda")[::2] % 150
    gather_mamba_track_indices(mapping, requests, uniform_position=position)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_mamba_track_indices(
            mapping, requests, uniform_position=position
        )
    for i in range(3):
        requests.add_(1).remainder_(150)
        mapping.add_(i)
        graph.replay()
        torch.testing.assert_close(actual, mapping[requests, position], rtol=0, atol=0)


@pytest.mark.parametrize("positions", [[0], [1], [0, 0, 0], [0, 1, 0]])
@pytest.mark.parametrize("override", [False, True])
def test_mamba_track_indices_from_reqs(positions, override):
    from types import SimpleNamespace

    from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

    mapping = torch.arange(20, device="cuda", dtype=torch.int64).reshape(10, 2)
    requests = torch.arange(len(positions), device="cuda", dtype=torch.int64) + 2
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        req_pool_indices=requests,
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=(p if p else None)))
            for p in positions
        ],
    )
    set_mamba_track_indices_from_reqs(
        batch, track_positions=positions if override else None
    )
    expected = mapping[requests, torch.tensor(positions, device="cuda")]
    torch.testing.assert_close(batch.mamba_track_indices, expected, rtol=0, atol=0)
    assert batch.mamba_track_buffer_indices == positions

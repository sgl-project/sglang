import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("bs", [0, 1, 3, 128])
@pytest.mark.parametrize("overlap", ["none", "same", "cross"])
@pytest.mark.parametrize("prepared", [False, True])
def test_combined_state_commit_tracking(bs, overlap, prepared):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_multi,
        prepare_mamba_state_scatter_multi,
    )

    device = "cuda"
    pool = 2 * bs + 2
    indices = torch.arange(bs * 2, device=device, dtype=torch.int32)[::2] // 2
    steps = torch.full((bs * 2,), 2, device=device, dtype=torch.int64)[::2]
    track = indices + bs
    if overlap == "same":
        track = indices.clone()
    elif overlap == "cross":
        track = indices.roll(1)
    tracking = torch.full((bs * 2,), 1, device=device, dtype=torch.int32)[::2]
    storage = torch.randn((2, bs, 65, 6), device=device, dtype=torch.bfloat16)
    pairs = [
        (
            torch.empty((2, pool, 65, 3), device=device, dtype=torch.bfloat16),
            storage.unfold(-1, 3, 1).permute(0, 1, 3, 2, 4),
        ),
        (
            torch.empty((1, pool, 2), device=device, dtype=torch.int64),
            torch.randint(0, 2**40, (1, bs, 4, 2), device=device),
        ),
    ]
    metadata = prepare_mamba_state_scatter_multi(pairs) if prepared else None
    fused_mamba_state_scatter_multi(
        pairs, indices, steps, track, tracking, _metadata=metadata
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_mamba_state_scatter_multi(
            pairs, indices, steps, track, tracking, _metadata=metadata
        )
    for step in [-1, 0, 3, 4]:
        tracking.fill_(step)
        if bs > 1:
            tracking[1] = -1
        for dst, src in pairs:
            dst.fill_(-7)
        graph.replay()
        for dst, src in pairs:
            expected = torch.full_like(dst, -7)
            rows = torch.arange(bs, device=device)
            expected[:, indices.long()] = src[:, rows, steps]
            valid = (tracking >= 0) & (tracking < 4)
            expected[:, track[valid].long()] = src[
                :, rows[valid], tracking[valid].long()
            ]
            assert torch.equal(dst, expected)


def test_state_scatter_metadata_cache_invalidation():
    from types import SimpleNamespace

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    def tensor(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, device="cuda", dtype=dtype)

    states = SimpleNamespace(
        temporal=tensor((2, 4, 7)),
        intermediate_ssm=tensor((2, 1, 4, 7)),
        conv=[tensor((2, 4, 3, 2))],
        intermediate_conv_window=[tensor((2, 1, 4, 3, 2))],
    )
    pool = SimpleNamespace(
        short_conv_pool=SimpleNamespace(
            conv_state=tensor((1, 4, 5)), intermediate_conv_state=tensor((1, 1, 4, 5))
        ),
        ngram_pool=SimpleNamespace(
            context=tensor((4, 2), torch.int64),
            intermediate_context=tensor((1, 4, 2), torch.int64),
        ),
    )
    backend = object.__new__(HybridLinearAttnBackend)
    backend.linear_attn_backend = SimpleNamespace(req_to_token_pool=pool)
    first = backend._prepare_verify_state_scatter(states)
    assert backend._prepare_verify_state_scatter(states) is first
    pool.ngram_pool.context = tensor((4, 2), torch.int64)
    second = backend._prepare_verify_state_scatter(states)
    assert second is not first
    assert second[0][-1][0].data_ptr() == pool.ngram_pool.context.data_ptr()
    old_storage = pool.ngram_pool.context
    old_storage.data = tensor((4, 2), torch.int64)
    third = backend._prepare_verify_state_scatter(states)
    assert third is not second
    assert third[0][-1][0].data_ptr() == old_storage.data_ptr()

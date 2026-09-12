"""Pair-pooling must match torch bitwise, including graph padding and ring state;
rounding differences can change the downstream indexer's top-k selection.
"""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.pair_pool_decode import pair_pool_decode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="pair_pool_decode requires CUDA",
)


@pytest.mark.parametrize("ring_size", (2, 8))
@pytest.mark.parametrize("n", (1, 3, 64))
def test_request_ring_matches_backend_and_graph_replay(n, ring_size):
    from types import SimpleNamespace

    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

    torch.manual_seed(43)
    dim = 512
    state = CompressStatePool(
        size=n * ring_size,
        ring_size=ring_size,
        overlap=False,
        head_dim=dim,
        dtype=torch.float32,
        device="cuda",
        enable_memory_saver=False,
        ratio=2,
    )
    initial = torch.randn_like(state.kv_score_buffer.kv_score)
    initial[-1, :dim] = 0
    initial[-1, dim:] = -torch.inf
    reference = object.__new__(DeepseekV4AttnBackend)
    reference.token_to_kv_pool = SimpleNamespace(
        get_attention_compress_states=lambda layer_id: state
    )
    fused_state = initial.clone()
    kv, score = (torch.randn(n, dim, device="cuda") for _ in range(2))
    pos = torch.ones(n, device="cuda", dtype=torch.int64)
    req = torch.arange(n, device="cuda", dtype=torch.int64)
    raw_loc = torch.arange(256, 256 + n, device="cuda", dtype=torch.int32)
    if n > 1:
        # A padding row shares a live request index and must not touch its ring.
        req[-1] = 0
        raw_loc[-1] = 0
    out_loc = raw_loc.clone()

    def run_fused():
        return pair_pool_decode(
            kv,
            score,
            pos,
            raw_loc,
            out_loc,
            req,
            fused_state[:, :dim],
            fused_state[:, dim:],
            fused_state.shape[0] - 1,
            ring_size=ring_size,
        )

    run_fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = run_fused()
    fused_state.copy_(initial)
    state.kv_score_buffer.kv_score.copy_(initial)
    for step in range(2 * ring_size + 3):
        kv.normal_()
        score.normal_()
        pos.fill_(step)
        out_loc.copy_(torch.where(pos % 2 == 1, raw_loc // 2, -1))
        partner_kv, partner_score = reference._low_ratio_pair_partners(
            layer_id=0, kv=kv, score=score, req=req, pos=pos, pad=raw_loc == 0
        )
        pairs = torch.stack([partner_kv, kv], dim=1)
        weights = torch.stack([partner_score, score], dim=1).softmax(dim=1)
        expected = (
            (pairs * weights).sum(dim=1),
            torch.where(pos % 2 == 1, pos - 1, pos),
            out_loc.clamp_min(0),
        )
        graph.replay()
        for actual, wanted in zip(got, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        torch.testing.assert_close(
            fused_state, state.kv_score_buffer.kv_score, rtol=0, atol=0
        )


@pytest.mark.parametrize("start_pos", (7, 8))
def test_request_ring_matches_target_verify_rows(start_pos):
    from types import SimpleNamespace

    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

    torch.manual_seed(47)
    bs, verify_rows, ring_size, dim = 2, 6, 8, 512
    n = bs * verify_rows
    state = CompressStatePool(
        size=bs * ring_size,
        ring_size=ring_size,
        overlap=False,
        head_dim=dim,
        dtype=torch.float32,
        device="cuda",
        enable_memory_saver=False,
        ratio=2,
    )
    initial = torch.randn_like(state.kv_score_buffer.kv_score)
    initial[-1, :dim] = 0
    initial[-1, dim:] = -torch.inf
    state.kv_score_buffer.kv_score.copy_(initial)
    fused_state = initial.clone()
    reference = object.__new__(DeepseekV4AttnBackend)
    reference.token_to_kv_pool = SimpleNamespace(
        get_attention_compress_states=lambda layer_id: state
    )

    kv, score = (torch.randn(n, dim, device="cuda") for _ in range(2))
    req = torch.arange(bs, device="cuda", dtype=torch.int64).repeat_interleave(
        verify_rows
    )
    pos = (
        torch.arange(start_pos, start_pos + verify_rows, device="cuda")
        .repeat(bs)
        .to(torch.int64)
    )
    raw_loc = torch.arange(256, 256 + n, device="cuda", dtype=torch.int32)
    out_loc = torch.where(pos % 2 == 1, raw_loc // 2, -1)

    partner_kv, partner_score = reference._low_ratio_pair_partners(
        layer_id=0, kv=kv, score=score, req=req, pos=pos, pad=raw_loc == 0
    )
    pairs = torch.stack([partner_kv, kv], dim=1)
    weights = torch.stack([partner_score, score], dim=1).softmax(dim=1)
    expected = (
        (pairs * weights).sum(dim=1),
        torch.where(pos % 2 == 1, pos - 1, pos),
        out_loc.clamp_min(0),
    )

    actual = pair_pool_decode(
        kv,
        score,
        pos,
        raw_loc,
        out_loc,
        req,
        fused_state[:, :dim],
        fused_state[:, dim:],
        fused_state.shape[0] - 1,
        ring_size=ring_size,
    )

    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    torch.testing.assert_close(
        fused_state, state.kv_score_buffer.kv_score, rtol=0, atol=0
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

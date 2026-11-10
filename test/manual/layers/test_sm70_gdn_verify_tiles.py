"""Verify recurrent-state precision across smaller SM70 verification tiles."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 verification tuning requires V100",
)


@pytest.mark.parametrize("tokens", [2, 4])
@pytest.mark.parametrize("seed", [7, 31])
def test_verify_tiles_preserve_states_and_graph_replay(tokens, seed, monkeypatch):
    import sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent as gdn

    torch.manual_seed(seed)
    qkv = torch.randn(1, tokens, 2560, device="cuda", dtype=torch.float16) * 0.1
    q = qkv[:, :, :512].view(1, tokens, 4, 128)
    k = qkv[:, :, 512:1024].view(1, tokens, 4, 128)
    v = qkv[:, :, 1024:].view(1, tokens, 12, 128)
    a = torch.randn(tokens, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    alog = torch.randn(12, device="cuda")
    bias = torch.randn_like(alog)
    state = torch.randn(3, 12, 128, 128, device="cuda", dtype=torch.float16) * 0.1
    state_index = torch.tensor([1], device="cuda", dtype=torch.int32)
    cache_index = torch.tensor([2], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    cache = torch.empty(3, 4, 12, 128, 128, device="cuda", dtype=torch.float16)

    def run():
        return gdn.fused_sigmoid_gating_delta_rule_update(
            alog,
            a,
            bias,
            1.0,
            20.0,
            q,
            k,
            v,
            b,
            state,
            state_index,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            disable_state_update=True,
            intermediate_states_buffer=cache,
            intermediate_state_indices=cache_index,
        )

    original_kernel = gdn.fused_sigmoid_gating_delta_rule_update_kernel
    tiles = []

    class RecordTiles:
        def __getitem__(self, grid):
            def launch(**kwargs):
                tiles.append(kwargs["BV"])
                return original_kernel[grid](**kwargs)

            return launch

    monkeypatch.setattr(
        gdn, "fused_sigmoid_gating_delta_rule_update_kernel", RecordTiles()
    )
    monkeypatch.setenv("SGLANG_SM70_MTP_GDN", "0")
    reference = run()
    reference_state = cache[2, :tokens].clone()
    monkeypatch.setenv("SGLANG_SM70_MTP_GDN", "1")
    result = run()
    assert tiles == [32, 8]
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(cache[2, :tokens], reference_state, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for factor in [-0.5, 2.0, 0.0]:
        qkv.mul_(factor)
        state.mul_(factor)
        graph.replay()
        captured_state = cache[2, :tokens].clone()
        monkeypatch.setenv("SGLANG_SM70_MTP_GDN", "0")
        reference = run()
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
        torch.testing.assert_close(captured_state, cache[2, :tokens], rtol=0, atol=0)

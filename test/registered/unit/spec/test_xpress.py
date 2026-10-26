import pytest
import torch

from sglang.srt.models.xpress import XPressRefinerHead


def _head(device, topc=0):
    torch.manual_seed(17)
    head = XPressRefinerHead(61, 32, 8, rank=32, mlp_hidden=64, topc=topc)
    return head.to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )


def test_xpress_mixer_folds_training_residual():
    head = _head("cpu")
    raw = torch.randn_like(head.mix_L)
    h = torch.randn(2, 8, 32)
    prev = torch.randint(0, 61, (2, 8))
    cache = head.hidden_cache(h)
    x = head.in_proj(torch.cat([cache, head.w1(prev)], dim=-1))
    causal = raw.tril()
    mixed = x + torch.bmm(causal, x.permute(2, 1, 0)).permute(2, 1, 0)
    expected = head.w2(
        mixed
        + head.mlp_down(
            torch.nn.functional.silu(head.mlp_gate(mixed)) * head.mlp_up(mixed)
        )
    )
    head.fold_from_raw_(raw)
    torch.testing.assert_close(head.refine_bias(prev, cache), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("topc", [0, 8])
def test_xpress_fused_jacobi_matches_reference(topc):
    head = _head("cuda", topc=topc)
    head.fold_from_raw_(torch.randn_like(head.mix_L) * 0.1)
    head.build_fused_buffers()
    base = torch.randn(2, 8, 61, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16)
    anchor = torch.randint(0, 61, (2,), device="cuda")
    block = torch.cat([anchor[:, None], base[:, 1:].argmax(dim=-1)], dim=1)
    candidates = base[:, 1:].topk(topc, dim=-1).indices if topc else None
    cache = head.hidden_cache(hidden)
    for _ in range(4):
        prev = torch.cat([anchor[:, None], block[:, :-1]], dim=1)
        refined = base + head.refine_bias(prev, cache)
        if candidates is None:
            block[:, 1:] = refined[:, 1:].argmax(dim=-1)
        else:
            scores = refined[:, 1:].gather(-1, candidates)
            block[:, 1:] = candidates.gather(
                -1, scores.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
    actual = head.jacobi_refine_greedy(base, hidden, anchor, 4)
    torch.testing.assert_close(actual, block[:, 1:])

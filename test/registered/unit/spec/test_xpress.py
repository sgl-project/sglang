from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel, _radix_topk
from sglang.srt.models.xpress import Qwen3XPressModel, XPressRefinerHead
from sglang.srt.speculative.dflash_worker_v2 import _validate_xpress_lm_head


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


def test_xpress_mixer_is_causal():
    head = _head("cpu")
    head.fold_from_raw_(torch.randn_like(head.mix_L) * 0.3)
    hidden = torch.randn(2, 8, 32)
    prev = torch.randint(0, 61, (2, 8))
    hcache = head.hidden_cache(hidden)
    bias = head.refine_bias(prev, hcache)
    changed = prev.clone()
    changed[:, 5] = (changed[:, 5] + 1) % 61
    changed_bias = head.refine_bias(changed, hcache)
    torch.testing.assert_close(changed_bias[:, :5], bias[:, :5])
    assert not torch.allclose(changed_bias[:, 5:], bias[:, 5:])


def test_xpress_target_head_validation():
    head = _head("cpu")
    target_head = nn.Linear(32, 64, bias=False)  # A padded vocabulary is valid.
    config = SimpleNamespace(hidden_size=32)
    _validate_xpress_lm_head(target_head, head, config)

    class QuantizedMethod:
        def apply(self, layer, x, bias):
            return x

    target_head.quant_method = QuantizedMethod()
    with pytest.raises(ValueError, match="unquantized"):
        _validate_xpress_lm_head(target_head, head, config)

    target_head.quant_method = None
    with pytest.raises(ValueError, match="shape"):
        _validate_xpress_lm_head(nn.Linear(31, 64, bias=False), head, config)


def test_xpress_checkpoint_load_folds_mixer(monkeypatch):
    model = Qwen3XPressModel.__new__(Qwen3XPressModel)
    nn.Module.__init__(model)
    model.xpress_head = _head("cpu")
    head = model.xpress_head
    raw = torch.randn_like(head.mix_L)
    weights = [
        ("xpress_head.w1.weight", torch.randn_like(head.w1.weight)),
        ("xpress_head.down_h.weight", torch.randn_like(head.down_h.weight)),
        ("xpress_head.down_g.weight", torch.randn_like(head.down_g.weight)),
        ("xpress_head.in_proj.weight", torch.randn_like(head.in_proj.weight)),
        ("xpress_head.mlp.gate_proj.weight", torch.randn_like(head.mlp_gate.weight)),
        ("xpress_head.mlp.up_proj.weight", torch.randn_like(head.mlp_up.weight)),
        ("xpress_head.mlp.down_proj.weight", torch.randn_like(head.mlp_down.weight)),
        ("xpress_head.w2.weight", torch.randn_like(head.w2.weight)),
        ("xpress_head.mix.L", raw),
    ]

    def load_base_weights(self, mapped):
        params = dict(self.named_parameters())
        for name, weight in mapped:
            params[name].data.copy_(weight)

    monkeypatch.setattr(DFlashDraftModel, "load_weights", load_base_weights)
    model.load_weights(weights)
    torch.testing.assert_close(head.mix_L, raw.tril() + torch.eye(head.block_size))
    torch.testing.assert_close(head._fused_buf["mix_kjc"], head.mix_L.permute(1, 2, 0))


def test_xpress_post_load_refreshes_fused_mixer():
    model = Qwen3XPressModel.__new__(Qwen3XPressModel)
    nn.Module.__init__(model)
    model.xpress_head = _head("cpu")
    model.post_load_weights()
    old_mixer = model.xpress_head._fused_buf["mix_kjc"].clone()
    model.xpress_head.fold_from_raw_(torch.randn_like(model.xpress_head.mix_L))
    model.post_load_weights()
    assert not torch.equal(model.xpress_head._fused_buf["mix_kjc"], old_mixer)


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
    candidates = (
        _radix_topk(base[:, 1:].reshape(-1, 61), topc)[1].view(2, 7, topc)
        if topc
        else None
    )
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

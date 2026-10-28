import sys

import pytest
import torch
from sgl_kernel.flash_mla_mega import (
    flash_mla_mega_decode,
    flash_mla_mega_prefill,
    is_flash_mla_mega_available,
    permute_mega_q_weight,
    permute_mega_wo_weight,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FlashMLA MegaAttention requires SM100/SM103",
)


def _pack_q(q):
    tokens, heads, dim = q.shape
    return q.view(tokens, heads, dim // 16, 16).transpose(1, 2).contiguous().view_as(q)


def _unpack_output(data, scale):
    tokens, groups, hidden = data.shape
    sf = scale.contiguous().view(torch.uint8).float()
    values = data.float().view(tokens, groups, hidden // 32, 32)
    values = values * torch.exp2(sf - 127).unsqueeze(-1)
    return (
        values.view(tokens, groups, 16, 8, 32)
        .transpose(2, 3)
        .reshape(tokens, groups * 8, 512)
    )


def _rope(x, positions, table, inverse=False):
    out = x.float().clone()
    cos, sin = table[positions.long()].chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    if inverse:
        sin = -sin
    even, odd = out[..., -64::2].clone(), out[..., -63::2].clone()
    out[..., -64::2] = even * cos - odd * sin
    out[..., -63::2] = even * sin + odd * cos
    return out


def _reference(q, kv, indices, lens, positions, table, sink):
    q = _rope(q, positions, table).bfloat16().float()
    selected = kv[indices.clamp_min(0), 0].float()
    logits = torch.einsum("thd,tkd->thk", q, selected) / 512**0.5
    valid = (indices >= 0) & (
        torch.arange(indices.shape[-1], device=q.device)[None] < lens[:, None]
    )
    logits.masked_fill_(~valid[:, None], -torch.inf)
    logits = torch.cat((logits, sink[None, :, None].expand(q.shape[0], -1, -1)), dim=-1)
    probabilities = logits.softmax(-1)[..., :-1]
    out = torch.einsum("thk,tkd->thd", probabilities, selected)
    return _rope(out, positions, table, inverse=True)


def _inputs(heads):
    torch.manual_seed(123)
    device = "cuda"
    q = torch.randn(3, heads, 512, device=device, dtype=torch.bfloat16)
    kv = torch.randn(128, 1, 512, device=device, dtype=torch.bfloat16)
    indices = torch.arange(128, device=device, dtype=torch.int32).repeat(3, 1)
    indices[1, 48:64] = -1
    lens = torch.tensor([128, 96, 64], device=device, dtype=torch.int32)
    positions = torch.tensor([0, 17, 255], device=device, dtype=torch.int32)
    angles = (
        torch.arange(256, device=device)[:, None]
        / torch.arange(1, 33, device=device)[None]
    )
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    return q, kv, indices, lens, positions, table, sink


@pytest.mark.parametrize("heads", [64, 128])
def test_mega_prefill_matches_unfused_reference(heads):
    assert is_flash_mla_mega_available(), "The SM100 wheel must expose MegaAttention"
    q, kv, indices, lens, positions, table, sink = _inputs(heads)
    data, scale, _, _ = flash_mla_mega_prefill(
        _pack_q(q),
        kv,
        indices[:, None],
        positions,
        table,
        512**-0.5,
        attn_sink=sink,
        topk_length=lens,
    )
    expected = _reference(q, kv, indices, lens, positions, table, sink)
    actual = _unpack_output(data, scale)
    relative_rmse = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rmse < 0.04
    assert scale.dtype == torch.int32 and scale.stride(0) == 1
    assert scale.stride(-1) % 4 == 0


def test_mega_decode_matches_gathered_prefill():
    q, kv, indices, lens, positions, table, sink = _inputs(64)
    groups = kv.float().view(128, 16, 32)
    sf = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)))
    codes = (groups / sf[..., None]).to(torch.float8_e4m3fn)
    dequant = (codes.float() * sf[..., None]).view_as(kv).bfloat16()
    raw = torch.empty(128 * 528, device="cuda", dtype=torch.uint8)
    raw[: 128 * 512] = codes.view(torch.uint8).reshape(-1)
    raw[128 * 512 :] = (sf.log2() + 127).to(torch.uint8).reshape(-1)
    cache = raw.view(1, 128, 1, 528)
    packed_q = _pack_q(q)
    decoded = flash_mla_mega_decode(
        packed_q,
        cache,
        indices,
        positions,
        table,
        512**-0.5,
        attn_sink=sink,
        topk_length=lens,
    )
    prefilled = flash_mla_mega_prefill(
        packed_q,
        dequant,
        indices[:, None],
        positions,
        table,
        512**-0.5,
        attn_sink=sink,
        topk_length=lens,
    )
    torch.testing.assert_close(
        _unpack_output(*decoded[:2]),
        _unpack_output(*prefilled[:2]),
        rtol=0.05,
        atol=0.01,
    )


def test_mega_weight_permutations_preserve_scale_association():
    heads, rows, width = 64, 64, 128
    q_weight = (
        torch.randint(0, 112, (heads * 512 * width,), device="cuda")
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
        .view(heads * 512, width)
    )
    q_scale = torch.arange(heads * 512, device="cuda", dtype=torch.int32)[:, None]
    pq, ps = permute_mega_q_weight(q_weight, q_scale, heads)
    q_perm = (
        torch.arange(heads * 512, device="cuda")
        .view(heads, 32, 16)
        .transpose(0, 1)
        .reshape(-1)
    )
    torch.testing.assert_close(pq.view(torch.uint8), q_weight.view(torch.uint8)[q_perm])
    torch.testing.assert_close(ps, q_scale[q_perm])

    wo = (
        torch.randint(0, 112, (rows * 4096,), device="cuda")
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
        .view(1, rows, 4096)
    )
    sf_bytes = (
        torch.arange(128, device="cuda", dtype=torch.uint8)[None, None]
        .expand(1, rows, -1)
        .contiguous()
    )
    sf = sf_bytes.view(torch.int32).transpose(1, 2).contiguous().transpose(1, 2)
    pw, pws = permute_mega_wo_weight(wo, sf)
    wo_perm = (
        torch.arange(4096, device="cuda").view(8, 16, 32).transpose(0, 1).reshape(-1)
    )
    torch.testing.assert_close(pw.view(torch.uint8), wo.view(torch.uint8)[..., wo_perm])
    expected_sf = sf_bytes[..., wo_perm[::32] // 32]
    torch.testing.assert_close(pws.contiguous().view(torch.uint8), expected_sf)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

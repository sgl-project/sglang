"""Hopper expert GEMMs and decode attention, and the MTP verify commit."""

import sys
import types

import pytest
import torch

from sglang.srt.models.inkling2_mtp import Inkling2ForConditionalGenerationMTP
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=65, stage="base-b-kernel-unit", runner_config="1-gpu-large")

hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires a Hopper GPU",
)


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


EXPERTS, LOCAL, TOP_K = 64, 8, 8


def _routing(tokens, seed=0):
    from sglang.kernels.ops.inkling2 import marlin_sm90
    from sglang.kernels.ops.inkling2.route import route_metadata

    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(EXPERTS, device="cuda", generator=g)[:TOP_K]
            for _ in range(tokens)
        ]
    ).int()
    flat = ids.flatten()
    weights = torch.rand(flat.numel(), device="cuda", generator=g)
    _, dst2src, _, counts, offsets, *_ = route_metadata(
        flat, weights, 0, LOCAL, EXPERTS
    )
    routing = marlin_sm90.route(counts, offsets, flat.numel(), tokens, dst2src)
    return flat, routing, (counts, offsets, dst2src)


def _check_rows(out, flat, xq, top_k, weight):
    for row in range(flat.numel()):
        expert = int(flat[row])
        if expert >= LOCAL:
            continue
        ref = xq[row // top_k].double() @ weight[expert].T
        err = (out[row].double() - ref).abs().max() / ref.abs().max()
        assert err < 1e-2, (row, err)


@hopper
@pytest.mark.parametrize("tokens", [1, 9, 64])
def test_marlin_nvfp4(tokens):
    from sglang.kernels.ops.inkling2 import experts_sm90, marlin_sm90

    n, k = 512, 1024
    packed = torch.randint(0, 256, (LOCAL, n, k // 2), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(LOCAL, n, k // 16, device="cuda") * 4 + 0.25).to(
        torch.float8_e4m3fn
    )
    codes = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2).long()
    grid = torch.tensor(E2M1, dtype=torch.float64, device="cuda")
    values = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0).double()
    weight = (
        values.view(LOCAL, n, k // 16, 16) * scales.double().unsqueeze(-1)
    ).flatten(-2)
    flat, routing, metadata = _routing(tokens)
    xq, _ = experts_sm90.fake_quantize(
        torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    )
    marlin = marlin_sm90.repack(packed, scales)
    _check_rows(
        marlin_sm90.gemm(xq, marlin, n, routing, TOP_K), flat, xq, TOP_K, weight
    )
    out = marlin_sm90.gemv(xq, marlin, n, metadata, TOP_K, tokens)
    _check_rows(out, flat, xq, TOP_K, weight)


@hopper
@pytest.mark.parametrize("tokens", [1, 64])
def test_marlin_mxfp8(tokens):
    from sglang.kernels.ops.inkling2 import experts_sm90, marlin_sm90

    n, k = 512, 1024
    weight8 = (torch.randn(LOCAL, n, k, device="cuda") * 50).to(torch.float8_e4m3fn)
    exponents = torch.randint(
        118, 128, (LOCAL, n, k // 32), dtype=torch.uint8, device="cuda"
    )
    weight = (
        weight8.double().view(LOCAL, n, k // 32, 32)
        * torch.exp2(exponents.double() - 127).unsqueeze(-1)
    ).flatten(-2)
    flat, routing, metadata = _routing(tokens, seed=1)
    xq = experts_sm90.fake_quantize_mxfp8(
        torch.randn(tokens * TOP_K, k, device="cuda", dtype=torch.bfloat16)
    )
    marlin = marlin_sm90.repack_mxfp8(weight8, exponents)
    _check_rows(marlin_sm90.gemm(xq, marlin, n, routing, 1), flat, xq, 1, weight)
    _check_rows(
        marlin_sm90.gemv(xq, marlin, n, metadata, 1, tokens), flat, xq, 1, weight
    )


@hopper
@pytest.mark.parametrize("qlen", [1, 8])
@pytest.mark.parametrize("window", [None, 1023])
def test_decode_attention(window, qlen):
    from sglang.kernels.ops.inkling2.decode_attention_sm90 import decode_partials

    d, group, page, extent = 128, 6, 128, 1024
    seqlens = [qlen, 77, 2049, 0 if qlen == 1 else 2111]
    batch, max_pages = len(seqlens), 24
    k_cache = torch.randn(
        batch * max_pages, page, 1, d, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    page_table = (
        torch.randperm(batch * max_pages, device="cuda").view(batch, max_pages).int()
    )
    q = torch.randn(batch * qlen, group, d, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(batch * qlen, group, extent, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor(seqlens, device="cuda", dtype=torch.int32)
    scale = d**-0.5
    o, m, l = decode_partials(
        q, k_cache, v_cache, page_table, lens, bias, scale, window_size_left=window
    )
    m = m.reshape(m.shape[0], -1, m.shape[-1])
    o = o.reshape(o.shape[0], m.shape[1], *o.shape[-2:])
    l = torch.ones_like(m) if l is None else l.reshape(m.shape)
    weights = torch.where(
        m == float("-inf"), torch.zeros_like(m), torch.exp(m - m.amax(0))
    )
    total = (weights * l).sum(0)
    out = (weights[..., None] * o).sum(0) / total[..., None]
    for t in range(batch * qlen):
        b, n = t // qlen, seqlens[t // qlen]
        if n == 0:
            continue
        pos = torch.arange(n, device="cuda")
        pages, slots = page_table[b, pos // page].long(), pos % page
        keys, vals = k_cache[pages, slots, 0].float(), v_cache[pages, slots, 0].float()
        dist = n - qlen + t % qlen - pos
        rel = torch.where(
            (dist >= 0) & (dist < extent),
            bias[t][:, dist.clamp(0, extent - 1)],
            torch.zeros(1, device="cuda"),
        )
        s = scale * (q[t].float() @ keys.T) + rel
        s = s.masked_fill((dist < 0)[None, :], float("-inf"))
        if window is not None:
            s = s.masked_fill((dist > window)[None, :], float("-inf"))
        ref = torch.softmax(s, -1) @ vals
        assert (out[t] - ref).abs().max() < 2e-2


def _fake_model(req_pool_size, hidden):
    model = types.SimpleNamespace(
        _f_rnn_buf=torch.full((req_pool_size + 1, hidden), -1.0, device="cuda")
    )

    def store_f_rnn(req_pool_indices, rows):
        model._f_rnn_buf.index_copy_(0, req_pool_indices.view(-1).long(), rows)

    model.store_f_rnn = store_f_rnn
    return model


@pytest.mark.parametrize("valid_mask", [None, "gated"])
def test_store_verify_commit_routes_invalid_rows_to_sink(valid_mask):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    req_pool_size, hidden, stride, bs, real_bs = 16, 8, 4, 6, 3
    model = _fake_model(req_pool_size, hidden)
    verify_hidden = torch.arange(
        bs * stride * hidden, dtype=torch.float32, device="cuda"
    ).view(bs * stride, hidden)
    req_pool_indices = torch.tensor([5, 2, 9, 0, 0, 0], device="cuda")
    commit_lens = torch.tensor([1, 4, 2, 0, 1, 3], dtype=torch.int32, device="cuda")
    valid = torch.arange(bs, device="cuda") < real_bs if valid_mask == "gated" else None
    num_rows = bs if valid is not None else real_bs

    Inkling2ForConditionalGenerationMTP.store_verify_commit(
        model,
        req_pool_indices=req_pool_indices[:num_rows],
        verify_hidden=verify_hidden[: num_rows * stride],
        commit_lens=commit_lens[:num_rows],
        stride=stride,
        valid=valid,
    )

    expected = torch.full((req_pool_size + 1, hidden), -1.0, device="cuda")
    rows = verify_hidden.view(bs, stride, hidden)
    for i in range(real_bs):
        expected[req_pool_indices[i]] = rows[i, commit_lens[i] - 1]
    buf = model._f_rnn_buf
    assert torch.equal(buf[:req_pool_size], expected[:req_pool_size])
    if valid is None:
        assert torch.equal(buf[req_pool_size], expected[req_pool_size])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Execution test for the CUDA raw-fp8 tilelang DSA sparse path.

Ported from the probe validated on sm_121 (DGX Spark GB10); requires SM89+
for fp8 tensor-core MMA. Two properties, each of which has been observed to
fail when the path is broken:
  1. one-hot with GLM-5.3-Flash's NoPE geometry (d_tail=0): a single valid
     index per token makes softmax exactly 1.0 and
     the fp8 prob-quant exact, so the kernel must return the gathered V row
     nearly bit-exactly (decisive for gather/mask/normalize plumbing);
  2. spread case vs an fp32 reference computed from the SAME quantized
     inputs must sit inside the analytic fp8-prob-GEMM budget (0.04), and a
     scrambled-index negative control must blow past it (proves the
     comparator discriminates).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

tilelang_kernel = pytest.importorskip(
    "sglang.kernels.ops.attention.dsa.tilelang_kernel"
)

requires_fp8_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs CUDA SM89+ for fp8 tensor-core MMA",
)

S, H, DV, TOPK, POOL = 4, 32, 512, 2112, 32768  # GLM-5.3-Flash NoPE geometry
SM_SCALE = DV**-0.5


def _rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _make_inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, H, DV, device="cuda", dtype=torch.float32, generator=g) * 0.5
    kv = torch.randn(POOL, 1, DV, device="cuda", dtype=torch.float32, generator=g) * 0.5
    idx = torch.randint(1, POOL, (S, 1, TOPK), device="cuda", generator=g).to(
        torch.int32
    )
    idx[:, :, -37:] = -1  # exercise the padded-tail mask path
    return q.to(torch.bfloat16), kv.to(torch.bfloat16), idx


def _ref_attn_fp32(q32, kv32, idx):
    out = torch.empty(S, H, DV, device="cuda", dtype=torch.float32)
    kvf = kv32.squeeze(1)
    for i in range(S):
        ids = idx[i, 0].long()
        m = ids >= 0
        k = kvf[ids.clamp(min=0)]
        logits = (q32[i] @ k.T) * SM_SCALE
        logits[:, ~m] = float("-inf")
        p = torch.softmax(logits, dim=-1)
        out[i] = p @ k
    return out


@requires_fp8_cuda
def test_fp8_one_hot_exact():
    q, kv, _ = _make_inputs(0)
    kv_fp8 = kv.to(torch.float8_e4m3fn)
    idx_hot = torch.full((S, 1, TOPK), -1, device="cuda", dtype=torch.int32)
    hot = torch.randint(1, POOL, (S,), device="cuda")
    idx_hot[:, 0, 0] = hot.to(torch.int32)
    out = tilelang_kernel.tilelang_sparse_fwd(q, kv_fp8, idx_hot, SM_SCALE, d_v=DV)
    torch.cuda.synchronize()
    expect = kv_fp8.float().squeeze(1)[hot.long()].unsqueeze(1).expand(S, H, DV)
    rel = _rel_err(out.reshape(S, H, DV), expect)
    assert rel < 1e-3, f"one-hot rel err {rel:.6f} exceeds 1e-3"


@requires_fp8_cuda
def test_fp8_one_hot_with_rope_tail_exact():
    tail = 64
    g = torch.Generator(device="cuda").manual_seed(2)
    q = torch.randn(
        S, H, DV + tail, device="cuda", dtype=torch.float32, generator=g
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(
        POOL, 1, DV + tail, device="cuda", dtype=torch.float32, generator=g
    ).to(torch.float8_e4m3fn)
    idx_hot = torch.full((S, 1, TOPK), -1, device="cuda", dtype=torch.int32)
    hot = torch.randint(1, POOL, (S,), device="cuda", generator=g)
    idx_hot[:, 0, 0] = hot.to(torch.int32)
    out = tilelang_kernel.tilelang_sparse_fwd(q, kv, idx_hot, SM_SCALE, d_v=DV)
    torch.cuda.synchronize()
    expect = kv.float().squeeze(1)[hot.long(), :DV].unsqueeze(1).expand(S, H, DV)
    rel = _rel_err(out.reshape(S, H, DV), expect)
    assert rel < 1e-3, f"one-hot tail rel err {rel:.6f} exceeds 1e-3"


@requires_fp8_cuda
def test_fp8_spread_within_budget_and_negative_control_fails():
    q, kv, idx = _make_inputs(1)
    kv_fp8 = kv.to(torch.float8_e4m3fn)
    out = tilelang_kernel.tilelang_sparse_fwd(q, kv_fp8, idx, SM_SCALE, d_v=DV)
    torch.cuda.synchronize()
    ref = _ref_attn_fp32(q.to(torch.float8_e4m3fn).float(), kv_fp8.float(), idx)
    rel = _rel_err(out.reshape(S, H, DV), ref)
    assert rel < 0.04, f"spread-case rel err {rel:.5f} exceeds fp8-prob budget"

    idx_bad = idx.clone()
    idx_bad[:, :, : TOPK // 2] = idx[:, :, TOPK // 2 : TOPK].flip(-1)[:, :, : TOPK // 2]
    out_bad = tilelang_kernel.tilelang_sparse_fwd(q, kv_fp8, idx_bad, SM_SCALE, d_v=DV)
    torch.cuda.synchronize()
    rel_bad = _rel_err(out_bad.reshape(S, H, DV), ref)
    assert rel_bad > 0.04, (
        "scrambled indices still matched the reference — the comparator is "
        f"not discriminating (rel err {rel_bad:.5f})"
    )


@requires_fp8_cuda
@pytest.mark.parametrize("tokens", [7999, 8000, 8192])
def test_sm90_large_fp8_tile_preserves_runner_contract(tokens, monkeypatch):
    from sglang.srt.environ import envs

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the experimental prefill tile is SM90-only")
    heads = 16
    generator = torch.Generator(device="cuda").manual_seed(1729)
    q = (torch.randn(tokens, heads, DV, device="cuda", generator=generator) * 0.5).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(POOL, 1, DV, device="cuda", generator=generator) * 0.5).to(
        torch.float8_e4m3fn
    )
    indices = torch.randint(
        1,
        POOL,
        (tokens, 1, TOPK),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    )
    indices[:, :, -37:] = -1
    indices[0] = -1
    indices[1] = -1
    indices[1, 0, 0] = 7
    blocks = []
    factory = tilelang_kernel.sparse_mla_fwd_decode_partial_fp8

    def record_tile(*args, **kwargs):
        blocks.append(kwargs["block_I"])
        return factory(*args, **kwargs)

    monkeypatch.setattr(
        tilelang_kernel, "sparse_mla_fwd_decode_partial_fp8", record_tile
    )
    with envs.SGLANG_OPT_DSA_SM90_LARGE_FP8_TILE.override(False):
        baseline = tilelang_kernel.tilelang_sparse_fwd(q, kv, indices, SM_SCALE)
    with envs.SGLANG_OPT_DSA_SM90_LARGE_FP8_TILE.override(True):
        actual = tilelang_kernel.tilelang_sparse_fwd(q, kv, indices, SM_SCALE)
    assert blocks == [32, 64 if tokens >= 8000 else 32]
    actual = actual.reshape(tokens, heads, DV)
    baseline = baseline.reshape_as(actual)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0]) == 0
    assert _rel_err(actual[1], kv[7, 0].float().expand(heads, DV)) < 1e-3
    assert _rel_err(actual, baseline) < 0.04
    # A float32 reference from the same quantized inputs checks attention,
    # masking, and normalization independently of either tile choice.
    expected = []
    for row in range(2, 6):
        ids = indices[row, 0].long()
        values = kv.float().squeeze(1)[ids.clamp(min=0)]
        logits = q[row].float() @ values.T * SM_SCALE
        logits[:, ids < 0] = float("-inf")
        expected.append(logits.softmax(-1) @ values)
    expected = torch.stack(expected)
    assert _rel_err(actual[2:6], expected) < 0.04
    bad = indices.clone()
    bad[2:6, :, :1024] = indices[2:6, :, 1024:2048].flip(-1)
    with envs.SGLANG_OPT_DSA_SM90_LARGE_FP8_TILE.override(True):
        wrong = tilelang_kernel.tilelang_sparse_fwd(q, kv, bad, SM_SCALE)
    assert _rel_err(wrong.reshape_as(actual)[2:6], expected) > 0.04

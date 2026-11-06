"""Correctness of the SM100 Q8KV8 sparse MLA kernel.

References are exact fp32 attention over the dequantized e4m3 inputs, so the remaining error is the
e4m3 rounding of P and the BF16 output.
"""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode,
)
from sglang.kernels.ops.attention.sparse_mla_q8kv8_sm100 import (
    D_QK,
    D_V,
    sparse_mla_q8kv8_sm100_fwd,
    sparse_mla_q8kv8_sm100_trtllm,
)
from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="sparse_mla_q8kv8_sm100 needs an SM100 GPU",
)

# The e4m3 rounding of P dominates the output error; max_logits and lse stay in fp32.
OUT_TOL = 3e-2
STAT_TOL = 1e-3


def _quant(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = (x.float().abs().amax() / 448.0).clamp(min=1e-8)
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale.item()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: float):
    actual, expected = actual.float(), expected.float()
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite), (
        f"{name}: non-finite entries differ"
    )
    assert torch.equal(actual[~finite], expected[~finite]), f"{name}: inf values differ"
    if finite.any():
        diff = (actual[finite] - expected[finite]).norm()
        err = (diff / expected[finite].norm().clamp(min=1e-12)).item()
        assert err < tol, f"{name}: rel_l2 {err:.3e} >= {tol}"


def _make_case(s_q, s_kv, h_q, topk, invalid_frac, seed, row_bytes=D_QK):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(s_q, h_q, D_QK, generator=g, device="cuda") * 0.3).bfloat16()
    kv = (torch.randn(s_kv, D_QK, generator=g, device="cuda") * 0.3).bfloat16()
    kv[:, D_V:] *= (
        2.0  # RoPE dims with a different range than NoPE, as in real activations
    )
    q8, q_scale = _quant(q)
    k8, k_scale = _quant(kv)
    if row_bytes != D_QK:
        rows = torch.zeros(s_kv, row_bytes, dtype=torch.uint8, device="cuda")
        rows[:, :D_QK] = k8.view(torch.uint8)
        k8 = rows[:, :D_QK].view(torch.float8_e4m3fn)
    keys = torch.rand(s_q, s_kv, generator=g, device="cuda")
    indices = keys.topk(topk, dim=-1).indices.to(torch.int32)
    if invalid_frac > 0:
        bad = torch.rand(s_q, topk, generator=g, device="cuda") < invalid_frac
        coin = torch.rand(s_q, topk, generator=g, device="cuda") < 0.5
        junk = torch.where(coin, -1, s_kv + 7).to(torch.int32)
        indices = torch.where(bad, junk, indices)
    sm_scale = 1.0 / math.sqrt(192 + 64)
    return q8, k8, indices, q_scale * k_scale * sm_scale, k_scale


def _reference(q8, k8, indices, bmm1, bmm2, attn_sink=None, topk_length=None):
    s_q, s_kv = q8.shape[0], k8.shape[0]
    qf, kf = q8.float(), k8.float()
    outs, max_logits, lses = [], [], []
    for i in range(0, s_q, 16):
        rows = indices[i : i + 16]
        valid = (rows >= 0) & (rows < s_kv)
        if topk_length is not None:
            pos = torch.arange(rows.shape[1], device=rows.device)
            valid &= pos[None, :] < topk_length[i : i + 16, None]
        k_rows = kf[rows.clamp(0, s_kv - 1).long()]
        # q: query token, h: head, k: top-k slot, d: dim.
        logits = torch.einsum("qhd,qkd->qhk", qf[i : i + 16], k_rows) * bmm1
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        lse = torch.logsumexp(logits, -1)
        denom = lse if attn_sink is None else torch.logaddexp(lse, attn_sink[None, :])
        p = torch.exp(logits - denom[..., None]).nan_to_num(0.0)
        outs.append(torch.einsum("qhk,qkd->qhd", p, k_rows[..., :D_V]) * bmm2)
        max_logits.append(logits.amax(-1))
        lses.append(lse.masked_fill(lse == float("-inf"), float("inf")))
    return torch.cat(outs), torch.cat(max_logits), torch.cat(lses)


# name -> shape and features; `splits` lists the num_splits values to run (None: chosen by the op).
CASES = {
    "topk2048": dict(s_q=64, s_kv=4096, h_q=64, topk=2048, splits=(None, 1, 2, 4)),
    "decode_split8": dict(s_q=4, s_kv=16384, h_q=64, topk=2048, splits=(None, 8)),
    "decode_split_sink_topk_length": dict(
        s_q=9,
        s_kv=8192,
        h_q=64,
        topk=2048,
        sink=True,
        topk_len=True,
        invalid_frac=0.2,
        splits=(8, 4, 2),
    ),
    "topk512": dict(s_q=64, s_kv=20480, h_q=64, topk=512, splits=(None, 1, 2)),
    "topk64": dict(s_q=64, s_kv=4096, h_q=64, topk=64),
    "topk2176": dict(s_q=48, s_kv=8192, h_q=64, topk=2176, splits=(None, 2)),
    "invalid_indices": dict(
        s_q=64, s_kv=4096, h_q=64, topk=1024, invalid_frac=0.3, splits=(1, 2)
    ),
    "mostly_invalid": dict(
        s_q=32, s_kv=4096, h_q=64, topk=1024, invalid_frac=0.97, splits=(1, 2, 4)
    ),
    "multi_wave_split": dict(s_q=200, s_kv=4096, h_q=64, topk=1024, splits=(2,)),
    "sink_topk_length": dict(
        s_q=64,
        s_kv=4096,
        h_q=64,
        topk=1024,
        invalid_frac=0.1,
        sink=True,
        topk_len=True,
        splits=(1, 2, 4),
    ),
    "h128": dict(s_q=40, s_kv=4096, h_q=128, topk=512, splits=(1, 2)),
    "h32_padded": dict(s_q=40, s_kv=4096, h_q=32, topk=512, sink=True),
    "paged_kv": dict(s_q=64, s_kv=8192, h_q=64, topk=2048, paged=True),
    "strided_kv_rows": dict(s_q=64, s_kv=4096, h_q=64, topk=512, row_bytes=640),
    "s_kv_equals_topk": dict(s_q=64, s_kv=2048, h_q=64, topk=2048),
    "persistent_items": dict(
        s_q=600, s_kv=4096, h_q=64, topk=512, topk_len=True, sink=True
    ),
    "persistent_topk2048": dict(
        s_q=400, s_kv=8192, h_q=64, topk=2048, invalid_frac=0.05
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_reference(name):
    case = dict(CASES[name])
    splits = case.pop("splits", (None,))
    sink, topk_len = case.pop("sink", False), case.pop("topk_len", False)
    paged = case.pop("paged", False)
    invalid_frac = case.pop("invalid_frac", 0.0)
    row_bytes = case.pop("row_bytes", D_QK)
    s_q, h_q, topk = case["s_q"], case["h_q"], case["topk"]
    q8, k8, indices, bmm1, bmm2 = _make_case(
        **case, invalid_frac=invalid_frac, seed=0, row_bytes=row_bytes
    )
    attn_sink = topk_length = None
    if sink:
        g = torch.Generator(device="cuda").manual_seed(1)
        attn_sink = torch.randn(h_q, generator=g, device="cuda")
        attn_sink[:3] = float("inf")  # zeroes the row
        attn_sink[3:6] = float("-inf")  # no-op
    if topk_len:
        g = torch.Generator(device="cuda").manual_seed(2)
        topk_length = torch.randint(
            0, topk + 1, (s_q,), generator=g, device="cuda", dtype=torch.int32
        )
        topk_length[:2] = 0
        topk_length[2:4] = topk
    expected = _reference(q8, k8, indices, bmm1, bmm2, attn_sink, topk_length)
    kv = k8.view(-1, 1, 64, D_QK) if paged else k8
    for num_splits in splits:
        actual = sparse_mla_q8kv8_sm100_fwd(
            q8, kv, indices, bmm1, bmm2, attn_sink, topk_length, num_splits
        )
        for label, a, e, tol in zip(
            ("out", "max_logits", "lse"),
            actual,
            expected,
            (OUT_TOL, STAT_TOL, STAT_TOL),
        ):
            _assert_close(f"{name} splits={num_splits} {label}", a, e, tol)


@pytest.mark.parametrize("h_q", [32, 64])
def test_caller_owned_outputs(h_q):
    q8, k8, indices, bmm1, bmm2 = _make_case(40, 4096, h_q, 512, 0.0, seed=3)
    expected = sparse_mla_q8kv8_sm100_fwd(q8, k8, indices, bmm1, bmm2)
    buffers = (
        torch.empty(40, h_q, D_V, dtype=torch.bfloat16, device="cuda"),
        torch.empty(40, h_q, dtype=torch.float32, device="cuda"),
        torch.empty(40, h_q, dtype=torch.float32, device="cuda"),
    )
    out, max_logits, lse = sparse_mla_q8kv8_sm100_fwd(
        q8,
        k8,
        indices,
        bmm1,
        bmm2,
        out=buffers[0],
        max_logits=buffers[1],
        lse=buffers[2],
    )
    for result, buffer, ref in zip((out, max_logits, lse), buffers, expected):
        assert result.data_ptr() == buffer.data_ptr()
        assert torch.equal(buffer, ref)


def test_rejects_misaligned_indices():
    q8, k8, indices, bmm1, bmm2 = _make_case(8, 4096, 64, 512, 0.0, seed=4)
    storage = torch.empty(8 * 512 + 1, dtype=torch.int32, device="cuda")
    shifted = storage[1:].view(8, 512)
    shifted.copy_(indices)
    with pytest.raises(Exception, match="16B-aligned"):
        sparse_mla_q8kv8_sm100_fwd(q8, k8, shifted, bmm1, bmm2)


# The trtllm-layout cases build their inputs the way DSA's `trtllm` backend does: Q and K roped and
# quantized by mla_quantize_and_rope_for_fp8 (scale 1.0), K in a 576 B pool at scattered slots, and
# page_table_1 from transform_index_page_table_decode (-1 past each request's valid top-k).
H, KV_LORA, ROPE, TOPK, PAGE = 64, 512, 64, 2048, 64
SCALING = (192 + 64) ** -0.5  # GLM-5: qk_nope_head_dim 192 + qk_rope_head_dim 64
MIXED_LENS = [
    37,
    500,
    2047,
    2048,
    2049,
    3000,
    9000,
    1,
    64,
    65,
    1500,
    4096,
    777,
    2100,
    128,
    5000,
]


def _sglang_inputs(seq_lens_list, seed=0):
    if not seq_lens_list:
        raise ValueError("seq_lens_list must be non-empty")
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch, max_len = len(seq_lens_list), max(seq_lens_list)
    num_slots = (batch * max_len * 9 // 8 + PAGE - 1) // PAGE * PAGE
    perm = torch.randperm(num_slots, generator=g, device="cuda").to(torch.int32)
    req_to_token = perm[: batch * max_len].view(batch, max_len).contiguous()
    inv_freq = 1.0 / (8e6 ** (torch.arange(0, ROPE, 2, device="cuda").float() / ROPE))
    angles = torch.arange(max_len, device="cuda")[:, None].float() * inv_freq[None]
    cos_sin_cache = torch.cat([angles.cos(), angles.sin()], -1).float()

    # mla_quantize_and_rope_for_fp8 takes Q and K of the same tokens; the unused side is zeros.
    pool = torch.empty(num_slots, D_QK, dtype=torch.float8_e4m3fn, device="cuda")
    for start in range(0, num_slots, 65536):
        n = min(65536, num_slots - start)
        k_nope = (torch.randn(n, KV_LORA, generator=g, device="cuda") * 0.3).bfloat16()
        k_rope = (torch.randn(n, ROPE, generator=g, device="cuda") * 0.3).bfloat16()
        k_pos = torch.randint(0, max_len, (n,), generator=g, device="cuda")
        _, k8, k_rope8 = mla_quantize_and_rope_for_fp8(
            k_nope.new_zeros(n, 1, KV_LORA),
            k_rope.new_zeros(n, 1, ROPE),
            k_nope,
            k_rope,
            k_pos,
            cos_sin_cache,
            False,
            KV_LORA,
            ROPE,
        )
        pool[start : start + n, :KV_LORA] = k8.view(n, KV_LORA)
        pool[start : start + n, KV_LORA:] = k_rope8.view(n, ROPE)
    q_nope = (
        torch.randn(batch, H, KV_LORA, generator=g, device="cuda") * 0.3
    ).bfloat16()
    q_rope = (torch.randn(batch, H, ROPE, generator=g, device="cuda") * 0.3).bfloat16()
    q_pos = torch.tensor([s - 1 for s in seq_lens_list], device="cuda")
    q8, _, _ = mla_quantize_and_rope_for_fp8(
        q_nope,
        q_rope,
        q_nope.new_zeros(batch, KV_LORA),
        q_rope.new_zeros(batch, ROPE),
        q_pos,
        cos_sin_cache,
        False,
        KV_LORA,
        ROPE,
    )
    topk_indices = torch.full((batch, TOPK), -1, dtype=torch.int32, device="cuda")
    for i, s in enumerate(seq_lens_list):
        n = min(s, TOPK)
        topk_indices[i, :n] = torch.randperm(s, generator=g, device="cuda")[:n].to(
            torch.int32
        )
    page_table_1 = transform_index_page_table_decode(
        page_table=req_to_token, topk_indices=topk_indices, page_size=1
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    return q8, pool, page_table_1, seq_lens


def _exact_decode(q8, pool, page_table_1, seq_lens):
    outs = []
    for i in range(q8.shape[0]):
        slots = page_table_1[i, : min(int(seq_lens[i]), TOPK)]
        k = pool[slots[slots >= 0].long()].float()
        p = torch.softmax((q8[i].float() @ k.T) * SCALING, -1)
        outs.append(p @ k[:, :KV_LORA])
    return torch.stack(outs)


def _trtllm_layout(q8, pool, page_table_1):
    batch = q8.shape[0]
    return (
        q8.view(batch, 1, H, D_QK),
        pool.view(-1, 1, PAGE, D_QK),
        page_table_1.unsqueeze(1),
    )


SGLANG_LENS = {
    "b16_mixed": MIXED_LENS,
    "b1_long": [6000],
    "b64": [
        int(x)
        for x in torch.randint(
            1, 6000, (64,), generator=torch.Generator().manual_seed(0)
        )
    ],
    "b200": [
        int(x)
        for x in torch.randint(
            1, 3000, (200,), generator=torch.Generator().manual_seed(1)
        )
    ],
}


@pytest.mark.parametrize("name", list(SGLANG_LENS))
def test_trtllm_layout_from_sglang_metadata(name):
    q8, pool, page_table_1, seq_lens = _sglang_inputs(SGLANG_LENS[name])
    batch = q8.shape[0]
    query, kv_cache, block_tables = _trtllm_layout(q8, pool, page_table_1)
    out = sparse_mla_q8kv8_sm100_trtllm(
        query, kv_cache, block_tables, SCALING, seq_lens=seq_lens
    )
    assert out.shape == (batch, 1, H, D_V) and out.dtype == torch.bfloat16
    _assert_close(
        name,
        out.view(batch, H, D_V),
        _exact_decode(q8, pool, page_table_1, seq_lens),
        OUT_TOL,
    )
    # sglang writes -1 past min(seq_len, topk), so seq_lens only lets the kernel skip blocks.
    no_seq_lens = sparse_mla_q8kv8_sm100_trtllm(query, kv_cache, block_tables, SCALING)
    assert torch.equal(no_seq_lens, out)
    flat, _, _ = sparse_mla_q8kv8_sm100_fwd(
        q8, kv_cache, page_table_1, SCALING, topk_length=seq_lens
    )
    assert torch.equal(flat, out.view(batch, H, D_V))


def test_trtllm_layout_matches_trtllm_gen():
    decode = pytest.importorskip("flashinfer.decode")
    q8, pool, page_table_1, seq_lens = _sglang_inputs(MIXED_LENS)
    query, kv_cache, block_tables = _trtllm_layout(q8, pool, page_table_1)
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    expected = decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=192,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max(MIXED_LENS),
        sparse_mla_top_k=TOPK,
        bmm1_scale=SCALING,
        backend="trtllm-gen",
    )
    actual = sparse_mla_q8kv8_sm100_trtllm(
        query, kv_cache, block_tables, SCALING, seq_lens=seq_lens
    )
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # Both kernels round P to e4m3, each within OUT_TOL of the exact result.
    _assert_close("vs trtllm-gen", actual, expected, 2 * OUT_TOL)


def test_trtllm_layout_cuda_graph_replay_matches_eager():
    q8, pool, page_table_1, seq_lens = _sglang_inputs(MIXED_LENS)
    query, kv_cache, block_tables = _trtllm_layout(q8, pool, page_table_1)
    eager = sparse_mla_q8kv8_sm100_trtllm(
        query, kv_cache, block_tables, SCALING, seq_lens=seq_lens
    )
    replayed = torch.empty_like(eager)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed.copy_(
            sparse_mla_q8kv8_sm100_trtllm(
                query, kv_cache, block_tables, SCALING, seq_lens=seq_lens
            )
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

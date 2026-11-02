"""Owned DSA DCP route on ROCm: the aiter FP8 asm sparse MLA, the decode
indexer row split, and the row-chunked prefill a2a.

Covers:
1. ``build_dcp_csr`` packs exactly the owned slots of the localized page table.
2. ``asm_sparse_mla_dcp`` matches the Triton owned-count sparse MLA (output, and
   its natural-log LSE against the Triton base-2 one), a row that owns no slot
   drops out of the merge, and CSR row slices match.
3. The AITER fused RoPE + fp8 query row + KV write at this rank's owned locs
   writes what the pool's DCP-widened write does.
4. A row slice of the packed top-k with ``row_to_batch`` gives the same rows as
   the whole batch, which is what the per-rank decode indexer split relies on.
5. ``DcpA2aChunkPipeline`` over row chunks equals one ``dcp_a2a_lse_reduce``.
6. ``DcpA2aChunkPlanner`` measures the stage costs on one layer, then plans the
   chunk count its schedule model finishes first with.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)

FP8 = torch.float8_e4m3fn
H, KV_LORA_RANK, ROPE_DIM, TOPK, DCP = 64, 512, 64, 2048, 4
# The RoPE rounds its own way before the fp8 cast: one e4m3 step (3 mantissa
# bits), absolute below the smallest normal.
FP8_RTOL, FP8_ATOL = 2**-3, torch.finfo(torch.float8_e4m3fn).smallest_normal


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _owned_table(rows: int, ctx: int, rank: int):
    from sglang.kernels.ops.attention.dsa.transform_index import (
        transform_index_page_table_dcp,
    )

    table = torch.stack(
        [torch.randperm(ctx, device="cuda")[:TOPK] for _ in range(rows)]
    ).int()
    # Row 0 owns nothing on rank 0; row 1 has a short top-k.
    table[0, :] = -1
    table[0, 0] = 1
    table[1, 100:] = -1
    return transform_index_page_table_dcp(table, DCP, rank)


def test_build_dcp_csr_packs_owned_slots():
    from sglang.kernels.ops.attention.dsa.asm_sparse_mla_dcp import build_dcp_csr

    torch.manual_seed(0)
    local, counts = _owned_table(rows=37, ctx=40000, rank=0)
    indptr, indices = build_dcp_csr(local, counts)

    assert indptr[0].item() == 0
    torch.testing.assert_close(indptr[1:] - indptr[:-1], counts.int())
    for r in range(local.shape[0]):
        n = counts[r].item()
        got = indices[indptr[r] : indptr[r] + n]
        torch.testing.assert_close(got, local[r, :n].int())


@pytest.mark.parametrize("T", [24, 144, 2304])
def test_asm_sparse_mla_matches_triton(T):
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.kernels.ops.attention.dsa.asm_sparse_mla_dcp import (
        asm_sparse_mla_dcp,
        asm_sparse_mla_dcp_available,
        build_dcp_csr,
        is_asm_q8,
    )
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_splitk,
    )

    if not asm_sparse_mla_dcp_available():
        pytest.skip("aiter has no non-persistent fp8 gqa64 MLA kernel")
    torch.manual_seed(1)
    ctx, sm_scale = 60000, 0.1
    kv = (
        torch.randn(ctx // DCP + 1000, 1, KV_LORA_RANK + ROPE_DIM, device="cuda") * 0.5
    ).to(FP8)
    q_nope = torch.randn(T, H, 192, device="cuda").bfloat16()
    q_pe = torch.randn(T, H, ROPE_DIM, device="cuda").bfloat16()
    w_kc = (torch.randn(H, 192, KV_LORA_RANK, device="cuda") * 0.08).bfloat16()
    local, counts = _owned_table(T, ctx, rank=0)
    indptr, indices = build_dcp_csr(local, counts)
    one = torch.ones(1, dtype=torch.float32, device="cuda")

    # The query prepare packs: one fp8 [q_nope | q_pe] row.
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    q8 = torch.cat([q_nope_out, q_pe], -1).to(FP8)
    assert is_asm_q8(*q8.split([KV_LORA_RANK, ROPE_DIM], -1))
    out, lse = asm_sparse_mla_dcp(q8, kv, indptr, indices, sm_scale, KV_LORA_RANK, one)

    # The Triton kernel casts q to the fp8 KV dtype on load: feed it the same q.
    ref_out, ref_lse = triton_sparse_mla_decode_splitk(
        q_nope=q8[..., :KV_LORA_RANK].to(torch.bfloat16).contiguous(),
        q_rope=q8[..., KV_LORA_RANK:].to(torch.bfloat16).contiguous(),
        kv=kv,
        indices=local.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=KV_LORA_RANK,
        workspace=[],
        return_lse=True,
        lengths=counts,
    )
    triton_out = ref_out.view(T, H, KV_LORA_RANK)
    triton_lse = ref_lse.view(T, H)
    owned = counts > 0

    # Both kernels round on their own way to the same fp32 result: the asm
    # one must be as close to it as the Triton one.
    qf, kvf = q8.float(), kv.float()[:, 0]
    exact_out = torch.zeros(T, H, KV_LORA_RANK, device="cuda")
    exact_lse = torch.full((T, H), float("-inf"), device="cuda")
    for t in owned.nonzero().flatten().tolist():
        sel = kvf[local[t, : counts[t]].long()]
        s = qf[t] @ sel.T * sm_scale
        exact_lse[t] = torch.logsumexp(s, -1)
        exact_out[t] = torch.softmax(s, -1) @ sel[:, :KV_LORA_RANK]
    asm_err = _rel_err(out[owned], exact_out[owned])
    triton_err = _rel_err(triton_out[owned], exact_out[owned])
    assert asm_err < 1.1 * triton_err + 1e-3, (asm_err, triton_err)
    torch.testing.assert_close(lse[owned], exact_lse[owned], rtol=0, atol=1e-2)
    torch.testing.assert_close(
        triton_lse[owned], exact_lse[owned] / math.log(2), rtol=0, atol=1e-2
    )
    # A row that owns no slot drops out of the merge with another rank's row.
    assert counts[0] == 0
    assert out[0].eq(0).all() and lse[0].eq(float("-inf")).all()
    other_out = torch.randn(1, H, KV_LORA_RANK, device="cuda").bfloat16()
    other_lse = torch.randn(1, H, device="cuda")
    merged, _ = dcp_lse_combine_triton(
        torch.stack([out[:1], other_out]),
        torch.stack([lse[:1], other_lse]),
        is_lse_base_on_e=True,
    )
    torch.testing.assert_close(merged, other_out, rtol=0, atol=0)

    # Row chunks of the batch read their slice of the same CSR.
    lo, hi = T // 3, T
    out_c, lse_c = asm_sparse_mla_dcp(
        q8[lo:hi].contiguous(),
        kv,
        indptr[lo : hi + 1],
        indices,
        sm_scale,
        KV_LORA_RANK,
        one,
    )
    torch.testing.assert_close(out_c, out[lo:hi], rtol=0, atol=0)
    torch.testing.assert_close(lse_c, lse[lo:hi], rtol=0, atol=0)


def _rope_interleaved(x, cos, sin):
    x0, x1 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[:, None].float(), sin[:, None].float()
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], -1).flatten(-2)


@pytest.mark.parametrize("T,rank", [(24, 0), (144, 1), (2304, 3)])
def test_fused_rope_cat_and_cache_matches_dcp_write(T, rank, monkeypatch):
    from types import SimpleNamespace

    from aiter.ops.triton.fusions.fused_kv_cache import (
        fused_qk_rope_cat_and_cache_mla,
    )

    from sglang.kernels.ops.kvcache.mla_buffer import (
        set_mla_kv_buffer_triton_fp8_quant,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        forward_mla_rocm,
    )

    torch.manual_seed(4)
    rows, max_pos = 4096, 65536
    monkeypatch.setattr(
        forward_mla_rocm,
        "get_parallel",
        lambda: SimpleNamespace(attn_dcp_size=DCP, attn_dcp_rank=rank),
    )
    # Distinct widened slots, slot 0 (the reserved padding slot) among them.
    loc = torch.randperm(rows * DCP, device="cuda")[:T]
    loc[T // 2] = 0
    batch = SimpleNamespace(out_cache_loc=loc)
    owned_loc = forward_mla_rocm._dcp_owned_cache_loc(batch)
    assert forward_mla_rocm._dcp_owned_cache_loc(batch) is owned_loc

    positions = torch.randint(0, max_pos, (T,), device="cuda")
    freqs = torch.rand(max_pos, ROPE_DIM // 2, device="cuda") * 6.3
    cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
    q_nope = torch.randn(T, H, KV_LORA_RANK, device="cuda").bfloat16()
    q_pe = torch.randn(T, H, ROPE_DIM, device="cuda").bfloat16()
    k_nope = torch.randn(T, 1, KV_LORA_RANK, device="cuda").bfloat16()
    k_pe = torch.randn(T, 1, ROPE_DIM, device="cuda").bfloat16()

    kv = torch.zeros(rows, 1, KV_LORA_RANK + ROPE_DIM, device="cuda", dtype=FP8)
    q8, *_ = fused_qk_rope_cat_and_cache_mla(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv,
        owned_loc,
        positions,
        cos,
        sin,
        None,
        False,
        apply_scale=False,
        q_out_dtype=FP8,
    )

    c, s = cos[positions], sin[positions]
    ref_q = torch.cat([q_nope.float(), _rope_interleaved(q_pe, c, s)], -1)
    torch.testing.assert_close(
        q8.float(), ref_q.to(FP8).float(), rtol=FP8_RTOL, atol=FP8_ATOL
    )
    ref_kv = torch.zeros_like(kv)
    set_mla_kv_buffer_triton_fp8_quant(
        ref_kv,
        loc,
        k_nope,
        _rope_interleaved(k_pe, c, s).bfloat16(),
        FP8,
        dcp_world_size=DCP,
        dcp_rank=rank,
    )
    torch.testing.assert_close(
        kv[..., :KV_LORA_RANK].float(),
        ref_kv[..., :KV_LORA_RANK].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(kv.float(), ref_kv.float(), rtol=FP8_RTOL, atol=FP8_ATOL)


@pytest.mark.parametrize("rows", [24, 144])
def test_packed_topk_row_slice_matches_whole_batch(rows):
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_packed_v2

    torch.manual_seed(2)
    page_size, max_len = 64, 16384
    lengths = torch.randint(
        TOPK // 2, max_len, (rows,), device="cuda", dtype=torch.int32
    )
    pages = max_len // page_size
    page_table = torch.stack(
        [torch.randperm(4 * pages, device="cuda")[:pages] for _ in range(rows)]
    ).int()
    scores = torch.randn(rows, max_len, device="cuda")

    whole = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    all_rows = torch.arange(rows, dtype=torch.int32, device="cuda")
    topk_transform_packed_v2(
        scores.clone(),
        lengths,
        page_table,
        whole,
        page_size,
        row_starts=torch.zeros_like(all_rows),
        row_to_batch=all_rows,
    )

    per = -(-rows // DCP)
    split = torch.empty(per * DCP, TOPK, dtype=torch.int32, device="cuda")
    for rank in range(DCP):
        lo, hi = min(rank * per, rows), min(rank * per + per, rows)
        if hi == lo:
            continue
        row_ids = torch.arange(lo, hi, dtype=torch.int32, device="cuda")
        topk_transform_packed_v2(
            scores[lo:hi].clone(),
            lengths[lo:hi],
            page_table,
            split[rank * per : rank * per + hi - lo],
            page_size,
            row_starts=torch.zeros_like(row_ids),
            row_to_batch=row_ids,
        )

    # The top-k order within a row is unspecified; the selected set is not.
    torch.testing.assert_close(
        split[:rows].sort(-1).values, whole.sort(-1).values, rtol=0, atol=0
    )
    # Each selected slot is the page-table translation of one of the row's top-k.
    for r in (0, rows - 1):
        n = lengths[r].item()
        pos = torch.topk(scores[r, :n], TOPK).indices
        expect = page_table[r, pos // page_size] * page_size + pos % page_size
        assert set(whole[r].tolist()) == set(expect.int().tolist())


def _mock_group(world_size: int):
    group = MagicMock()
    group.world_size = world_size
    group.all_to_all_single = MagicMock(side_effect=lambda out, inp: out.copy_(inp))
    return group


def _partials(rows: int, D: int):
    partial = torch.randn(rows, H, KV_LORA_RANK, device="cuda").bfloat16()
    lse = torch.randn(rows, H, device="cuda") * 3
    w_vc = (torch.randn(H, KV_LORA_RANK, D, device="cuda") * 0.04).bfloat16()

    def project(x):
        return torch.bmm(x.transpose(0, 1), w_vc).transpose(0, 1).contiguous()

    def attend(lo, hi):
        return partial[lo:hi], lse[lo:hi].contiguous()

    return partial, lse, project, attend


@pytest.mark.parametrize("rows,chunk", [(4096, 1024), (5000, 1250), (300, 128)])
@pytest.mark.parametrize("timed", [False, True])
def test_a2a_chunk_pipeline_matches_reduce(rows, chunk, timed):
    from sglang.srt.layers.dcp import DcpA2aChunkPipeline, dcp_a2a_lse_reduce

    torch.manual_seed(3)
    N, D = 2, 256
    group = _mock_group(N)
    partial, lse, project, attend = _partials(rows, D)
    chunks = [(lo, min(rows, lo + chunk)) for lo in range(0, rows, chunk)]

    # Project per chunk for the reference too: bmm rounding depends on the batch.
    projected = torch.cat([project(partial[lo:hi]) for lo, hi in chunks])
    ref = dcp_a2a_lse_reduce(projected, lse, group, is_lse_base_on_e=False)

    pipe = DcpA2aChunkPipeline(chunks, group, project, False, timed=timed)
    out = pipe.run(attend)

    assert pipe.consumed and pipe.rows == rows
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    sizes = [hi - lo for lo, hi in chunks]
    if timed and sizes[0] != sizes[-1]:
        from sglang.srt.layers.dcp.comm import _median_costs

        costs = _median_costs([pipe.timing], chunks)
        assert len(costs) == 7 and min(costs) >= 0 and costs[-1] > 0


# Stage costs (us): compute, a2a and combine as (fixed, per row), then the host
# time per chunk, in the range measured at GLM-5.2 DCP4 shapes on MI355X.
_COSTS = [20.0, 0.11, 33.0, 0.152, 10.0, 0.038, 240.0]


def test_plan_pipeline_chunks_follows_costs():
    from sglang.srt.layers.dcp.comm import modeled_pipeline_us, plan_pipeline_chunks

    plans = {rows: plan_pipeline_chunks(_COSTS, rows) for rows in (512, 4096, 32768)}
    for rows, chunks in plans.items():
        assert chunks[0][0] == 0 and chunks[-1][1] == rows
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
        # No even chunking with one chunk more or less models faster.
        n = len(chunks)
        for m in (n - 1, n + 1):
            if 1 <= m <= rows:
                per = -(-rows // m)
                other = [(lo, min(rows, lo + per)) for lo in range(0, rows, per)]
                assert modeled_pipeline_us(_COSTS, other) >= modeled_pipeline_us(
                    _COSTS, chunks
                )
    # Issuing a chunk costs more than a small batch's exchange hides: no split.
    assert len(plans[512]) == 1
    # The hidden compute grows with the rows, the per-chunk cost does not.
    assert 1 < len(plans[4096]) < len(plans[32768])
    # Free per-chunk costs leave nothing to stop the split but the host.
    cheap = _COSTS[:-1] + [_COSTS[-1] / 8]
    assert len(plan_pipeline_chunks(cheap, 4096)) > len(plans[4096])


def test_planner_measures_then_plans():
    from sglang.srt.layers.dcp import DcpA2aChunkPlanner
    from sglang.srt.layers.dcp.comm import plan_pipeline_chunks

    torch.manual_seed(5)
    rows, D = 3000, 256
    group = _mock_group(1)
    _, _, project, attend = _partials(2 * rows, D)
    planner = DcpA2aChunkPlanner(group)

    # The first prefill measures on every layer it reaches with those rows.
    for layer in range(3):
        p = planner.pipeline(layer, rows, project, False)
        assert p is not None and p.timing is not None
        assert len(p.chunks) == 2 and p.chunks[-1][1] == rows
        p.run(attend)
    assert planner.pipeline(3, rows // 2, project, False) is None
    torch.cuda.synchronize()

    # A later forward reaching the first of them takes the costs.
    planner.settle(1)
    assert planner.costs is None
    planner.settle(0)
    assert planner.measured_rows == rows and min(planner.costs) >= 0
    want = plan_pipeline_chunks(planner.costs, rows)
    got = planner.pipeline(0, rows, project, False)
    assert (got.chunks if got is not None else [(0, rows)]) == want
    assert got is None or got.timing is None

    # A larger prefill is measured again.
    again = planner.pipeline(0, 2 * rows, project, False)
    assert again.timing is not None
    # One that never reached a second layer keeps the costs, and is not retried.
    costs = list(planner.costs)
    planner.settle(0)
    assert planner.costs == costs and planner.measured_rows == 2 * rows
    retry = planner.pipeline(0, 2 * rows, project, False)
    assert retry is None or retry.timing is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""The single-launch replacements of the DeepSeek-V4.1 decode bookkeeping on HIP are
bitwise the torch chains (or the two-step launches) they replace."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
    expand_same_length_rows,
    gather_i32,
    sparse_buffers,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildCausalSwaPageIndices,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase  # noqa: F401

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and torch.cuda.is_available()),
    reason="HIP decode glue for the DeepSeek-V4.1 path.",
)

DEVICE = "cuda"
SWA_WINDOW = 128
PAGE_INDEX_ALIGNED_SIZE = 64


def _seed(s: int) -> torch.Generator:
    torch.manual_seed(s)
    return torch.Generator(device=DEVICE).manual_seed(s)


def _expand_reference(seq_lens, req_pool_indices, qo_len, offset, extend_seq_lens):
    """The torch chain of make_forward_metadata_from_raw_verify /
    init_forward_metadata_dspark_draft_block + make_core_attn_metadata's head."""
    bs = len(seq_lens)
    seq = seq_lens + offset
    ext = (
        torch.full_like(seq, qo_len)
        if extend_seq_lens is None
        else extend_seq_lens.to(device=seq.device, dtype=seq.dtype)
    )
    ext = torch.minimum(ext, seq).clamp_min_(1)
    casual = seq[:, None] + torch.arange(
        -qo_len + 1, 1, dtype=torch.int32, device=seq.device
    )
    casual = casual.flatten()
    idx = torch.arange(bs, dtype=torch.int32, device=seq.device).repeat_interleave(
        qo_len
    )
    req_rep = req_pool_indices[idx]
    casual32 = casual.to(torch.int32)
    return seq, ext, casual32, casual32 - 1, req_rep


@pytest.mark.parametrize("seq_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("with_ext", [False, True])
def test_expand_same_length_rows(seq_dtype, with_ext):
    _seed(1)
    for bs, qo, offset in ((1, 5, 5), (2, 5, 5), (7, 5, 0), (33, 3, 3), (0, 5, 5)):
        seq_lens = torch.randint(0, 5000, (bs,), device=DEVICE).to(seq_dtype)
        req = torch.randint(0, 64, (bs,), device=DEVICE, dtype=torch.int64)
        ext = (
            torch.randint(-2, 9, (bs,), device=DEVICE, dtype=torch.int32)
            if with_ext
            else None
        )
        ref = _expand_reference(seq_lens, req, qo, offset, ext)
        got = expand_same_length_rows(
            seq_lens,
            req,
            qo_len=qo,
            seq_lens_offset=offset,
            extend_seq_lens=None if ext is None else ext.to(seq_dtype),
        )
        for r, g in zip(ref, got):
            assert r.dtype == g.dtype and r.shape == g.shape, (r, g)
            torch.testing.assert_close(r, g, rtol=0, atol=0)


def test_gather_i32():
    _seed(2)
    table = torch.randint(0, 1 << 20, (4097,), device=DEVICE, dtype=torch.int64)
    table[-1] = -1
    for n in (0, 1, 10, 3000):
        idx = torch.randint(-1, 4097, (n,), device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(
            gather_i32(table, idx), table[idx].to(torch.int32), rtol=0, atol=0
        )


@pytest.mark.parametrize("ratios", [(1,), (2,), (1, 2), ()])
def test_sparse_buffers_low_ratio_raw_indices(ratios):
    _seed(3)
    for rows in (1, 10, 513):
        lens = {
            r: torch.randint(1, 4000, (rows,), device=DEVICE, dtype=torch.int32)
            for r in (4, 0, 1, 2)
        }
        kw = dict(
            c4_topk_lengths_clamp1=lens[4],
            c4_topk_lengths_raw=lens[0],
            c1_topk_lengths_clamp1=lens[1] if 1 in ratios else None,
            c2_topk_lengths_clamp1=lens[2] if 2 in ratios else None,
            index_topk=512,
            page_index_align=PAGE_INDEX_ALIGNED_SIZE,
        )
        ref = sparse_buffers(**kw)
        for r in ratios:
            ref[f"c{r}_sparse_raw_indices"] = torch.full_like(
                ref[f"c{r}_sparse_page_indices"], -1
            )
        got = sparse_buffers(**kw, low_ratio_raw_indices=True)
        assert ref.keys() == got.keys()
        for k in ref:
            torch.testing.assert_close(ref[k], got[k], rtol=0, atol=0)


def _hip_get_swa_page_indices(req_to_token, mapping, req_rep, seq_lens_casual):
    """DeepseekV4HipRadixBackend.get_swa_page_indices + _pad_last_dim."""
    pos_causal = seq_lens_casual - 1
    offsets = pos_causal.unsqueeze(1) - torch.arange(
        SWA_WINDOW, dtype=torch.int32, device=DEVICE
    ).unsqueeze(0)
    invalid = offsets < 0
    offsets.masked_fill_(invalid, 0)
    raw = req_to_token[req_rep[:, None], offsets]
    raw.masked_fill_(invalid, -1)
    swa = mapping[raw].to(torch.int32)
    width = -(-SWA_WINDOW // PAGE_INDEX_ALIGNED_SIZE) * PAGE_INDEX_ALIGNED_SIZE
    return F.pad(swa, (0, width - SWA_WINDOW), value=-1)


def test_causal_swa_page_indices_matches_hip_torch_chain():
    _seed(4)
    pool, max_ctx, n_req = 20000, 4096, 16
    mapping = torch.cat(
        [
            torch.randint(0, 5000, (pool,), device=DEVICE, dtype=torch.int64),
            torch.tensor([-1], device=DEVICE, dtype=torch.int64),
        ]
    )
    req_to_token = torch.randint(
        0, pool, (n_req, max_ctx), device=DEVICE, dtype=torch.int32
    )
    # unset slots inside a row's reach read the mapping's -1 sentinel through the wrap
    req_to_token[3, 100:110] = -1
    for rows in (1, 5, 10, 160):
        req_rep = torch.randint(0, n_req, (rows,), device=DEVICE, dtype=torch.int64)
        req_rep[0] = 3
        seq = torch.randint(1, max_ctx, (rows,), device=DEVICE, dtype=torch.int32)
        seq[0] = 105
        ref = _hip_get_swa_page_indices(req_to_token, mapping, req_rep, seq)
        got = BuildCausalSwaPageIndices.triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=mapping,
            req_pool_indices_repeated=req_rep,
            seq_lens_casual=seq,
            swa_window=SWA_WINDOW,
            page_index_aligned_size=PAGE_INDEX_ALIGNED_SIZE,
        )
        torch.testing.assert_close(ref, got, rtol=0, atol=0)


def test_init_compression_metadata_padded_c128_page_indices():
    _seed(5)
    page_size = 256
    for rows, max_pages in ((5, 37), (10, 64), (1, 3)):
        seq = torch.randint(1, max_pages * page_size, (rows,), device=DEVICE).to(
            torch.int32
        )
        pos = seq - 1
        out_loc = torch.randint(0, 1 << 20, (rows,), device=DEVICE, dtype=torch.int64)
        page_table = torch.randint(
            0, 1000, (rows, max_pages), device=DEVICE, dtype=torch.int32
        )
        ref = init_compression_metadata(seq, pos, out_loc, page_table, page_size)
        got = init_compression_metadata(
            seq,
            pos,
            out_loc,
            page_table,
            page_size,
            page_indices_align=PAGE_INDEX_ALIGNED_SIZE,
        )
        for i, (r, g) in enumerate(zip(ref, got)):
            if i == 8:
                width = r.shape[-1]
                padded = -(-width // PAGE_INDEX_ALIGNED_SIZE) * PAGE_INDEX_ALIGNED_SIZE
                r = F.pad(r, (0, padded - width), value=-1)
            torch.testing.assert_close(r, g, rtol=0, atol=0)


def test_engram_gather_strided_ids():
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    _seed(6)
    rows, dim, blk = 3000, 256, 128
    weight = (torch.randn(rows, dim, device=DEVICE) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(100, 140, (rows, dim // blk), device=DEVICE).to(torch.uint8)
    for t, layers, cols in ((1, 3, 8), (10, 2, 8), (37, 4, 6)):
        table = torch.randint(0, rows, (t, layers, cols), device=DEVICE)
        ids = table[:, 1]
        assert t == 1 or not ids.is_contiguous()
        ref = torch.empty(t * cols, dim, dtype=torch.bfloat16, device=DEVICE)
        got = torch.empty_like(ref)
        for out, i in ((ref, ids.reshape(-1)), (got, ids)):
            engram_gather(
                weight.data_ptr(),
                scale.data_ptr(),
                i,
                out,
                dim,
                blk,
                row_lo=500,
                row_hi=2500,
            )
        torch.testing.assert_close(ref, got, rtol=0, atol=0)


def test_rope_copy_into_matches_rope_then_copy():
    from sglang.kernels.ops.attention import deepseek_v4_rope as rope

    _seed(7)
    prev = rope._USE_BATCHED_ROPE
    rope.set_batched_rope(True)
    try:
        head_dim, rd, max_pos = 512, 64, 8192
        freqs = torch.polar(
            torch.ones(max_pos, rd // 2, device=DEVICE),
            torch.rand(max_pos, rd // 2, device=DEVICE) * 6.28,
        )
        for tokens, heads, pad_heads in ((5, 16, 64), (10, 32, 64), (1, 8, 64)):
            q = torch.randn(tokens, heads, head_dim, device=DEVICE).to(torch.bfloat16)
            pos = torch.randint(0, max_pos, (tokens,), device=DEVICE)
            ref_pad = torch.zeros(tokens, pad_heads, head_dim, device=DEVICE).to(
                torch.bfloat16
            )
            got_pad = ref_pad.clone()
            q_ref = q.clone()
            rope.apply_rotary_emb_triton(q_ref[..., -rd:], freqs, positions=pos)
            ref_pad[:, :heads].copy_(q_ref)
            q_in = q.clone()
            assert (
                rope.apply_rotary_emb_copy_into(
                    q_in, got_pad[:, :heads], freqs, pos, rd
                )
                is not None
            )
            torch.testing.assert_close(q_in, q, rtol=0, atol=0)
            assert torch.equal(ref_pad.view(torch.int16), got_pad.view(torch.int16))
    finally:
        rope.set_batched_rope(prev)


def test_finalize_and_out_tokens_into_buffers():
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        FinalizeAcceptLensResult,
        finalize_accept_lens_triton,
    )
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        build_out_tokens_triton,
    )

    _seed(8)
    stride, gamma, max_bs = 5, 4, 16
    for bs in (1, 3, 16):
        correct = torch.randint(0, stride, (bs,), device=DEVICE, dtype=torch.int32)
        trim = torch.randint(0, 3, (bs,), device=DEVICE, dtype=torch.int32)
        prefix = torch.randint(0, 9000, (bs,), device=DEVICE, dtype=torch.int64)
        bonus = torch.randint(0, 1 << 17, (bs,), device=DEVICE, dtype=torch.int64)
        draft = torch.randint(0, 1 << 17, (bs, gamma), device=DEVICE, dtype=torch.int64)
        ref = finalize_accept_lens_triton(
            correct_len=correct, cap_trim_lens=trim, prefix_lens=prefix
        )
        bufs = FinalizeAcceptLensResult(
            commit_lens=torch.full((max_bs,), 7, dtype=torch.int32, device=DEVICE)[:bs],
            new_seq_lens=torch.full((max_bs,), 7, dtype=torch.int64, device=DEVICE)[
                :bs
            ],
            cap_trim_lens=torch.full((max_bs,), 7, dtype=torch.int32, device=DEVICE)[
                :bs
            ],
        )
        got = finalize_accept_lens_triton(
            correct_len=correct, cap_trim_lens=trim, prefix_lens=prefix, out=bufs
        )
        for name in ("commit_lens", "new_seq_lens", "cap_trim_lens"):
            torch.testing.assert_close(
                getattr(ref, name), getattr(got, name), rtol=0, atol=0
            )
            assert getattr(got, name).data_ptr() == getattr(bufs, name).data_ptr()
        kw = dict(
            draft_tokens=draft,
            correct_len=correct,
            bonus=bonus,
            verify_num_draft_tokens=stride,
            gamma=gamma,
        )
        ref_tokens = build_out_tokens_triton(
            **{**kw, "correct_len": correct.to(torch.int64)}
        )
        out_buf = torch.full((max_bs, stride), 7, dtype=torch.int64, device=DEVICE)
        build_out_tokens_triton(**kw, out=out_buf[:bs])
        torch.testing.assert_close(ref_tokens, out_buf[:bs], rtol=0, atol=0)


def test_k_norm_rope_flashmla_reads_strided_kv():
    """The DSpark commit KV projection now hands the writer column slices of the
    stacked output on ROCm; the paged cache must come out byte-identical."""
    from sglang.kernels.ops.attention.dsv4.elementwise import (
        fused_k_norm_rope_flashmla,
    )

    _seed(9)
    head_dim, rd, page_size, stages, max_pos = 512, 64, 128, 3, 8192
    freqs = torch.polar(
        torch.ones(max_pos, rd // 2, device=DEVICE),
        torch.rand(max_pos, rd // 2, device=DEVICE) * 6.28,
    )
    weight = torch.randn(head_dim, device=DEVICE).to(torch.bfloat16)
    page_bytes = -(-584 * page_size // 576) * 576
    for tokens in (1, 5, 10):
        kv_all = torch.randn(tokens, stages * head_dim, device=DEVICE).to(
            torch.bfloat16
        )
        pos = torch.randint(0, max_pos, (tokens,), device=DEVICE)
        loc = torch.randperm(4 * page_size, device=DEVICE)[:tokens].to(torch.int32)
        caches = []
        for strided in (False, True):
            cache = torch.zeros(4, page_bytes, dtype=torch.uint8, device=DEVICE)
            for kv in kv_all.split(head_dim, dim=-1):
                fused_k_norm_rope_flashmla(
                    kv=kv if strided else kv.contiguous(),
                    kv_weight=weight,
                    eps=1e-6,
                    freqs_cis=freqs,
                    positions=pos,
                    out_loc=loc,
                    kvcache=cache,
                    page_size=page_size,
                )
            caches.append(cache)
        assert torch.equal(caches[0], caches[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

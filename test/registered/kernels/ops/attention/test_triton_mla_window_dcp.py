"""DCP8 draft partials over the draft KV ring must merge to dense fp32 attention
over each row's window."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.triton_draft_hip import MlaDcpWindowVerify
from sglang.srt.layers.dcp.layout import is_mla_dcp_lse_base_on_e
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

PAGE = 64
Q_LEN = 8  # the draft block
# window_left = HF sliding_window - 1 (get_dflash_attention_sliding_window_size),
# MLA5's 1024; the window only sets each row's key count, so k4s's 256 takes no other path.
WINDOW_LEFT = 1023
# All 96 heads: under DCP the layer attends with the group's gathered q.
HEADS, KV_LORA, ROPE = 96, 512, 64
SCALE = (128 + 64) ** -0.5
DCP = 8
# 48 rows x 6 head blocks exceed MI355X's 256 CUs (fused sparse kernel).
CONTEXTS = [0, 5, 250, 256, 300, 3000]


def _layout(ctx, dcp, seed):
    """req_to_token of requests at shuffled page-aligned loc ranges, so
    loc % dcp == position % dcp as the DCP allocator keeps it."""
    g = torch.Generator().manual_seed(seed)
    seq_lens = [c + Q_LEN for c in ctx]
    cycle = PAGE * dcp
    span = -(-max(seq_lens) // cycle) * cycle
    order = torch.randperm(len(ctx) + 2, generator=g)[: len(ctx)]
    req_to_token = torch.full((len(ctx) + 3, max(seq_lens) + 16), -1, dtype=torch.int32)
    pool_idx = torch.randperm(req_to_token.shape[0], generator=g)[: len(ctx)]
    for i, s in enumerate(seq_lens):
        base = int(order[i]) * span
        req_to_token[pool_idx[i], :s] = torch.arange(base, base + s, dtype=torch.int32)
    return req_to_token, pool_idx, (len(ctx) + 2) * span


def _layer():
    return SimpleNamespace(
        layer_id=2,
        sliding_window_size=WINDOW_LEFT,
        tp_q_head_num=HEADS,
        qk_head_dim=KV_LORA + ROPE,
        v_head_dim=KV_LORA,
        scaling=SCALE,
    )


def _backend(rank, req_to_token, kv_shard, ring):
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.forward_metadata = SimpleNamespace(custom_mask=None)
    backend.mla_dcp_window_verify = MlaDcpWindowVerify(
        SimpleNamespace(
            req_to_token=req_to_token,
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _: kv_shard),
            kv_index_translator=SimpleNamespace(full_v2p_table=None),
            page_size=PAGE,
            dcp_size=DCP,
            dcp_rank=rank,
        ),
        SimpleNamespace(draft_kv_ring=ring),
    )
    return backend


def _forward_batch(pool_idx, positions):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=pool_idx.cuda(),
        # each request's keys before its window: its first row's position
        seq_lens=positions[::Q_LEN].cuda(),
        positions=positions.cuda(),
    )


def _inputs(ctx):
    req_to_token, pool_idx, num_locs = _layout(ctx, DCP, seed=DCP + len(ctx))
    positions = torch.tensor(
        [c + i for c in ctx for i in range(Q_LEN)], dtype=torch.int64
    )
    g = torch.Generator(device="cuda").manual_seed(7)
    keys = torch.randn(
        num_locs,
        1,
        KV_LORA + ROPE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    keys = (keys * 0.5).to(fp8_dtype)  # the draft KV follows --kv-cache-dtype
    q = torch.randn(
        positions.numel(),
        HEADS,
        KV_LORA + ROPE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    return req_to_token, pool_idx, positions, keys, q


def _merged_rank_partials(q, keys, req_to_token, pool_idx, positions, ring, floors):
    """Every simulated rank's (out, lse) over its own ring shard, merged:
    (out [rows, H, KV_LORA], natural-log lse [rows, H])."""
    rows = positions.numel()
    forward_batch = _forward_batch(pool_idx, positions)
    outs, lses = [], []
    for rank in range(DCP):
        shard = _ring_shard(ring, keys, req_to_token, pool_idx, positions, rank, floors)
        backend = _backend(rank, req_to_token.cuda(), shard, ring)
        out, lse = backend._forward_extend_dcp(
            q, None, None, _layer(), forward_batch, True, 0.0, None
        )
        assert out.shape == (rows, HEADS * KV_LORA) and lse.shape == (rows, HEADS)
        outs.append(out.view(rows, HEADS, KV_LORA))
        lses.append(lse)
    # The log base forward_absorb_rocm_core hands the merge for this backend.
    base_e = is_mla_dcp_lse_base_on_e("triton")
    out, lse = dcp_lse_combine_triton(
        torch.stack(outs), torch.stack(lses), is_lse_base_on_e=base_e, return_lse=True
    )
    return out, lse if base_e else lse * math.log(2.0)


def _ring_shard(ring, keys, req_to_token, pool_idx, positions, rank, floors):
    """One rank's draft pool after the worker wrote each request's last ring span
    of keys at DFlashDraftKVRing.locs, kept under the pool's DCP owner rule; rows
    below a request's floor hold another request's stale keys."""
    rows = draft_kv_ring_pool_rows(
        req_pool_rows=req_to_token.shape[0],
        rows_per_req=ring.rows_per_req,
        page_size=PAGE,
    )
    shard = torch.zeros(rows, *keys.shape[1:], device="cuda", dtype=keys.dtype)
    for i, req in enumerate(pool_idx.tolist()):
        end = int(positions[i * Q_LEN + Q_LEN - 1]) + 1
        pos = torch.arange(max(0, end - ring.span), end)
        pos = pos[pos % ring.dcp_size == rank]
        locs = ring.locs(torch.full_like(pos, req), pos)
        rows = keys[req_to_token[req, pos].long().cuda()]
        stale = (pos < floors[i]).cuda()
        rows[stale] = (rows[stale].float() * -4).to(rows.dtype)
        shard[(locs // ring.dcp_size).cuda()] = rows
    return shard


def _reference(q, keys, req_to_token, pool_idx, positions, floors):
    """fp32 attention of each row over the keys of [p - window_left, p], from the
    request's floor on: (out [rows, H, KV_LORA], natural-log lse [rows, H])."""
    positions = positions.tolist()
    outs, lses = [], []
    for r, p in enumerate(positions):
        req = int(pool_idx[r // Q_LEN])
        lo = max(0, p - WINDOW_LEFT, floors[r // Q_LEN])
        locs = req_to_token[req, lo : p + 1].long().cuda()
        k = keys[locs, 0].float()  # exact for bf16 and fp8
        s = (q[r].float() @ k.T) * SCALE
        lses.append(torch.logsumexp(s, -1))
        outs.append(torch.softmax(s, -1) @ k[:, :KV_LORA])
    return torch.stack(outs), torch.stack(lses)


def _assert_matches(out, lse, ref_out, ref_lse):
    # Rounding the partials, probabilities and output to BF16 costs a few 1e-3;
    # FP8 q and probabilities, or a wrong window or log base, cost far more.
    err = ((out.float() - ref_out).norm() / ref_out.norm()).item()
    assert err < 1e-2, err
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=2e-3)


def test_ring_rows_read_as_the_request_window():
    req_to_token, pool_idx, positions, keys, q = _inputs(CONTEXTS)
    ring = DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=WINDOW_LEFT, block_size=Q_LEN, dcp_size=DCP
        ),
        dcp_size=DCP,
        req_pool_rows=req_to_token.shape[0],
        device="cuda",
        window_left=WINDOW_LEFT,
    )
    # Prefix-cache hits must not read stale keys below the first extend.
    floors = [max(0, c - 37) for c in CONTEXTS]
    ring.min_pos[pool_idx.cuda()] = torch.tensor(
        floors, dtype=torch.int32, device="cuda"
    )
    out, lse = _merged_rank_partials(
        q, keys, req_to_token, pool_idx, positions, ring, floors
    )
    _assert_matches(
        out, lse, *_reference(q, keys, req_to_token, pool_idx, positions, floors)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

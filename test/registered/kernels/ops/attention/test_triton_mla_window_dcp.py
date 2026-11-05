"""DCP draft partials must merge to dense fp32 attention over each row's window."""

import math
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.triton_draft_hip import (
    MlaDcpWindowVerify,
    init_draft_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_mla_dcp_lse_base_on_e,
)
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
    draft_kv_ring_unsupported_reason,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

PAGE = 64
Q_LEN = 8  # the draft block
# window_left = HF sliding_window - 1 (get_dflash_attention_sliding_window_size):
# the k4s MLA layer's 256 and MLA5's 1024.
WINDOWS = [255, 1023]
# All 96 heads: under DCP the layer attends with the group's gathered q.
HEADS, KV_LORA, ROPE = 96, 512, 64
SCALE = (128 + 64) ** -0.5
# 48 rows x 6 head blocks exceed MI355X's 256 CUs (fused sparse kernel); one
# request takes the split-K path at DCP1. In "edge" the 1024 window's start
# crosses position 0 inside the block.
CONTEXTS = {
    "batch": [0, 5, 250, 256, 300, 3000],
    "single": [3000],
    "edge": [1020, 1021],
}
# DCP8 edge and single requests use the same split count; include the boundary
# positions in its batch instead of compiling a separate edge case.
DCP8_BATCH = CONTEXTS["batch"] + CONTEXTS["edge"]


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


def _layer(window_left, layer_id=2):
    return SimpleNamespace(
        layer_id=layer_id,
        sliding_window_size=window_left,
        tp_q_head_num=HEADS,
        qk_head_dim=KV_LORA + ROPE,
        v_head_dim=KV_LORA,
        scaling=SCALE,
    )


def _backend(dcp, rank, req_to_token, kv_shard, ring=None):
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.forward_metadata = SimpleNamespace(custom_mask=None)
    backend.mla_dcp_window_verify = MlaDcpWindowVerify(
        SimpleNamespace(
            req_to_token=req_to_token,
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _: kv_shard),
            kv_index_translator=SimpleNamespace(full_v2p_table=None),
            page_size=PAGE,
            dcp_size=dcp,
            dcp_rank=rank,
        ),
        SimpleNamespace(draft_kv_ring=ring),
    )
    return backend


def _forward_batch(pool_idx, positions, mode=ForwardMode.TARGET_VERIFY):
    return SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=pool_idx.cuda(),
        # each request's keys before its window: its first row's position
        seq_lens=positions[::Q_LEN].cuda(),
        positions=positions.cuda(),
    )


def _inputs(ctx, dcp, kv_dtype):
    req_to_token, pool_idx, num_locs = _layout(ctx, dcp, seed=dcp + len(ctx))
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
    keys = (keys * 0.5).to(kv_dtype)
    q = torch.randn(
        positions.numel(),
        HEADS,
        KV_LORA + ROPE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    return req_to_token, pool_idx, positions, keys, q


def _merged_rank_partials(
    q, keys, req_to_token, pool_idx, positions, layer, dcp, ring=None, floors=None
):
    """Every simulated rank's (out, lse) over its own cache shard, merged:
    (out [rows, H, KV_LORA], natural-log lse [rows, H])."""
    rows = positions.numel()
    forward_batch = _forward_batch(pool_idx, positions)
    outs, lses = [], []
    for rank in range(dcp):
        if ring is None:
            # This rank caches global row loc (loc % dcp == rank) at loc // dcp.
            shard = keys[rank::dcp].contiguous()
        else:
            shard = _ring_shard(
                ring, keys, req_to_token, pool_idx, positions, rank, floors
            )
        backend = _backend(dcp, rank, req_to_token.cuda(), shard, ring)
        out, lse = backend._forward_extend_dcp(
            q, None, None, layer, forward_batch, True, 0.0, None
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


def _ring_shard(ring, keys, req_to_token, pool_idx, positions, rank, floors=None):
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
        if floors is not None:
            stale = (pos < floors[i]).cuda()
            rows[stale] = (rows[stale].float() * -4).to(rows.dtype)
        shard[(locs // ring.dcp_size).cuda()] = rows
    return shard


def _reference(q, keys, req_to_token, pool_idx, positions, window_left, floors=None):
    """fp32 attention of each row over the keys of [p - window_left, p], from the
    request's floor on: (out [rows, H, KV_LORA], natural-log lse [rows, H])."""
    positions = positions.tolist()
    outs, lses = [], []
    for r, p in enumerate(positions):
        req = int(pool_idx[r // Q_LEN])
        lo = max(0, p - window_left, floors[r // Q_LEN] if floors else 0)
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


@pytest.mark.parametrize(
    "dcp,ctx",
    [(1, CONTEXTS[name]) for name in CONTEXTS]
    + [(8, DCP8_BATCH), (8, CONTEXTS["single"])],
    ids=["dcp1-batch", "dcp1-single", "dcp1-edge", "dcp8-batch", "dcp8-single"],
)
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, fp8_dtype])
@pytest.mark.parametrize("window_left", WINDOWS)
def test_rank_partials_merge_to_dense_window(dcp, kv_dtype, window_left, ctx):
    req_to_token, pool_idx, positions, keys, q = _inputs(ctx, dcp, kv_dtype)
    out, lse = _merged_rank_partials(
        q, keys, req_to_token, pool_idx, positions, _layer(window_left), dcp
    )
    _assert_matches(
        out, lse, *_reference(q, keys, req_to_token, pool_idx, positions, window_left)
    )


@pytest.mark.parametrize("window_left", WINDOWS)
@pytest.mark.parametrize(
    "ctx,with_floor",
    [
        (CONTEXTS["batch"], False),
        (CONTEXTS["single"], False),
        (CONTEXTS["batch"], True),
    ],
    ids=["batch", "single", "prefix-cache-floor"],
)
def test_ring_rows_read_as_the_request_window(window_left, ctx, with_floor):
    dcp = 8
    req_to_token, pool_idx, positions, keys, q = _inputs(ctx, dcp, fp8_dtype)
    ring = DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=window_left, block_size=Q_LEN, dcp_size=dcp
        ),
        dcp_size=dcp,
        req_pool_rows=req_to_token.shape[0],
        device="cuda",
        window_left=window_left,
    )
    floors = [max(0, c - 37) for c in ctx] if with_floor else None
    if floors is not None:
        # Prefix-cache hits must not read stale keys below the first extend.
        ring.min_pos[pool_idx.cuda()] = torch.tensor(
            floors, dtype=torch.int32, device="cuda"
        )
    out, lse = _merged_rank_partials(
        q,
        keys,
        req_to_token,
        pool_idx,
        positions,
        _layer(window_left),
        dcp,
        ring=ring,
        floors=floors,
    )
    _assert_matches(
        out,
        lse,
        *_reference(q, keys, req_to_token, pool_idx, positions, window_left, floors),
    )


def test_ring_admits_triton_only_under_dcp_on_rocm():
    """Only ROCm's triton draft backend reads a ring window by index, and only under
    DCP; elsewhere it keeps dev's refusal."""

    config = {
        "num_hidden_layers": 1,
        "dflash_config": {"attention_mode": "mla"},
        "layer_types": ["sliding_attention"],
    }

    def reason(dcp, hip):
        with mock.patch("sglang.srt.utils.is_hip", return_value=hip):
            return draft_kv_ring_unsupported_reason(
                config, draft_backend="triton", draft_window_size=None, dcp_size=dcp
            )

    assert reason(dcp=8, hip=True) is None
    assert reason(dcp=1, hip=True) is not None
    assert reason(dcp=8, hip=False) is not None


def test_window_indices_are_built_once_per_forward():
    req_to_token, pool_idx, _ = _layout(CONTEXTS["batch"], 8, seed=0)
    positions = torch.tensor(
        [c + i for c in CONTEXTS["batch"] for i in range(Q_LEN)], dtype=torch.int64
    )
    verify = _backend(8, 3, req_to_token.cuda(), None).mla_dcp_window_verify
    forward_batch = _forward_batch(pool_idx, positions)
    window_left = WINDOWS[0]
    first = verify.window_sparse_indices(_layer(window_left, 2), forward_batch)
    # The next sliding layer of the same forward reuses them.
    again = verify.window_sparse_indices(_layer(window_left, 3), forward_batch)
    assert again[0] is first[0] and again[1] is first[1]
    # A new forward rebuilds them, here to the same values.
    rebuilt = verify.window_sparse_indices(
        _layer(window_left, 4), _forward_batch(pool_idx, positions)
    )
    assert rebuilt[0] is not first[0]
    torch.testing.assert_close(rebuilt[0], first[0], rtol=0, atol=0)
    torch.testing.assert_close(rebuilt[1], first[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,causal", [(ForwardMode.EXTEND, True), (ForwardMode.TARGET_VERIFY, False)]
)
def test_other_sliding_windows_still_refuse(mode, causal):
    req_to_token, pool_idx, _ = _layout([0], 8, seed=0)
    backend = _backend(8, 0, req_to_token.cuda(), None)
    q = torch.zeros(Q_LEN, HEADS, KV_LORA + ROPE, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="sliding window"):
        backend._forward_extend_dcp(
            q,
            None,
            None,
            _layer(WINDOWS[0]),
            _forward_batch(pool_idx, torch.arange(Q_LEN), mode),
            causal,
            0.0,
            None,
        )


def test_verify_metadata_builds_no_rail():
    """Under DCP every verify layer of a ROCm MLA draft reads its owner rows
    itself, so the target-verify metadata packs no global prefix or window rail,
    eager or for a graph: the translator here cannot pack one, and the eager
    window rail would size itself with a host sync."""
    bs = 2
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.use_mla = True
    backend.topk = 0
    backend.dcp_size, backend.dcp_rank = 8, 0
    backend.sliding_window_size = WINDOWS[1]
    backend.req_to_token = backend.token_to_kv_pool = None
    backend.kv_index_translator = SimpleNamespace(full_v2p_table=None)
    backend.page_size = PAGE
    init_draft_backend(
        backend, SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DFLASH)
    )
    backend.dllm_attention = None
    backend.device = "cuda"
    backend.use_sliding_window_kv_pool = False
    backend.target_verify_num_tokens_per_req = Q_LEN
    backend._verify_mask = None
    # As __init__ leaves them once the draft's window moved to the layers.
    backend.window_kv_indptr = None
    for name in ("kv_indptr", "qo_indptr", "mask_indptr"):
        setattr(backend, name, torch.zeros(bs + 1, dtype=torch.int64, device="cuda"))
    # Never written: the fill helper returns before reading it.
    backend.cuda_graph_kv_indices = torch.empty(0, dtype=torch.int64, device="cuda")
    spec_info = SimpleNamespace(num_tokens_per_req=Q_LEN, custom_mask=None)
    pool_idx = torch.tensor([1, 0], device="cuda")
    seq_lens = torch.tensor([300, 3000], device="cuda")
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        spec_info=spec_info,
        req_pool_indices=pool_idx,
        seq_lens=seq_lens,
        seq_lens_sum=3300,
        kv_loc_plan=None,
        out_cache_loc=None,
    )

    backend.init_forward_metadata(forward_batch)
    metadata = backend.forward_metadata
    assert metadata.kv_indptr is None
    assert metadata.window_kv_indices is None

    # The graph path packs none either (no plan to pack one from).
    backend._update_target_verify_buffers(None, bs, seq_lens, spec_info, pool_idx)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

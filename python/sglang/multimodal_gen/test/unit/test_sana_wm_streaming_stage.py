# SPDX-License-Identifier: Apache-2.0
"""S1c tests — streaming denoising stage: KV accumulator + chunk-loop composition.

The accumulator is the load-bearing new logic: GDN/STATE blocks copy-forward the
previous chunk's recurrent state; softmax/CONCAT blocks concatenate the rolling
+ sink K/V along **dim=1** (our (B,N,H,D) softmax cache layout — the reference's
dim=2 would concat the head axis and silently corrupt every softmax block).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.sana_wm import (
    SanaWMArchConfig,
    SanaWMConfig,
)
from sglang.multimodal_gen.runtime import server_args as _sa_mod
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    SanaWMTransformer3DModel,
    _CACHE_TYPE_CONCAT,
    _CACHE_TYPE_STATE,
    _NUM_STREAM_CACHE_SLOTS,
    _SLOT_CAM_K,
    _SLOT_CAM_V,
    _SLOT_FFN_TCONV,
    _SLOT_K,
    _SLOT_SHORTCONV,
    _SLOT_TYPE_FLAG,
    _SLOT_V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    SanaWMStreamingDenoisingStage as Stage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

B, Hh, D = 1, 2, 4


# --------------------------------------------------------------------------- #
# 1. Chunk schedule
# --------------------------------------------------------------------------- #


def test_autoregressive_segments_first_chunk_absorbs_remainder():
    assert Stage._autoregressive_segments(13, 3) == [0, 4, 7, 10, 13]
    seg = Stage._autoregressive_segments(121, 3)
    assert seg[0] == 0 and seg[1] == 4 and seg[-1] == 121
    assert len(seg) - 1 == 40  # num chunks
    assert all(seg[i + 1] - seg[i] == 3 for i in range(1, len(seg) - 1))


# --------------------------------------------------------------------------- #
# 2. KV accumulator
# --------------------------------------------------------------------------- #


def _state_block(seed):
    g = torch.Generator().manual_seed(seed)
    blk = [None] * _NUM_STREAM_CACHE_SLOTS
    blk[_SLOT_K] = torch.randn(B, Hh, D, D, generator=g)  # recurrent state_kv
    blk[_SLOT_V] = torch.randn(B, Hh, D, 1, generator=g)  # state_z
    blk[_SLOT_CAM_K] = torch.randn(B, Hh, D, D, generator=g)  # cam state
    blk[_SLOT_SHORTCONV] = torch.randn(B * 6, 3, Hh * D, generator=g)
    blk[_SLOT_TYPE_FLAG] = torch.tensor([_CACHE_TYPE_STATE])
    blk[_SLOT_FFN_TCONV] = torch.randn(B, Hh * D, 1, 6, generator=g)
    return blk


def _concat_block(n_tok, seed):
    g = torch.Generator().manual_seed(seed)
    blk = [None] * _NUM_STREAM_CACHE_SLOTS
    for slot in (_SLOT_K, _SLOT_V, _SLOT_CAM_K, _SLOT_CAM_V):
        blk[slot] = torch.randn(B, n_tok, Hh, D, generator=g)  # (B, N, H, D)
    blk[_SLOT_TYPE_FLAG] = torch.tensor([_CACHE_TYPE_CONCAT])
    blk[_SLOT_FFN_TCONV] = torch.randn(B, Hh * D, 1, 6, generator=g)
    return blk


def test_accumulate_state_block_copies_previous_chunk():
    # 3 chunks, 1 STATE block. chunk2 must copy-forward chunk1's state.
    kv = [[_state_block(c)] for c in range(3)]
    cur, sink_num = Stage._accumulate_kv_cache(
        kv, chunk_idx=2, chunk_indices=[0, 4, 7, 10], num_cached_blocks=2,
        sink_token=True, num_blocks=1,
    )
    assert cur[0][_SLOT_K] is kv[1][0][_SLOT_K]  # carried from chunk 1
    assert cur[0][_SLOT_V] is kv[1][0][_SLOT_V]
    assert cur[0][_SLOT_CAM_K] is kv[1][0][_SLOT_CAM_K]
    assert float(cur[0][_SLOT_TYPE_FLAG].item()) == _CACHE_TYPE_STATE


def test_accumulate_concat_block_concats_on_token_axis_dim1():
    # chunk0 K has 4 tokens, chunk1 K has 3 -> chunk2 prefix = 4+3=7 on dim=1.
    n0, n1 = 4, 3
    kv = [[_concat_block(n0, 0)], [_concat_block(n1, 1)], [[None] * _NUM_STREAM_CACHE_SLOTS]]
    cur, _ = Stage._accumulate_kv_cache(
        kv, chunk_idx=2, chunk_indices=[0, n0, n0 + n1, n0 + n1 + 3],
        num_cached_blocks=5, sink_token=False, num_blocks=1,
    )
    acc_k = cur[0][_SLOT_K]
    assert acc_k.shape == (B, n0 + n1, Hh, D)  # dim=1 grew; head axis (dim 2) intact
    # exact content: chunk0 then chunk1 along dim=1
    torch.testing.assert_close(acc_k[:, :n0], kv[0][0][_SLOT_K], atol=0, rtol=0)
    torch.testing.assert_close(acc_k[:, n0:], kv[1][0][_SLOT_K], atol=0, rtol=0)
    assert float(cur[0][_SLOT_TYPE_FLAG].item()) == _CACHE_TYPE_CONCAT


def test_accumulate_sink_includes_chunk_zero():
    # num_cached_blocks=2 at chunk 3 -> sink_start=2>0 -> valid = [0, 2]
    n = 3
    kv = [[_concat_block(n, c)] for c in range(3)] + [[[None] * _NUM_STREAM_CACHE_SLOTS]]
    cur, sink_num = Stage._accumulate_kv_cache(
        kv, chunk_idx=3, chunk_indices=[0, n, 2 * n, 3 * n, 4 * n],
        num_cached_blocks=2, sink_token=True, num_blocks=1,
    )
    # valid = [0, 2] -> 2 chunks of n tokens each
    assert cur[0][_SLOT_K].shape == (B, 2 * n, Hh, D)
    assert sink_num == n  # chunk_indices[1] - chunk_indices[0]


def test_accumulate_chunk_zero_is_empty():
    kv = [[[None] * _NUM_STREAM_CACHE_SLOTS]]
    cur, sink_num = Stage._accumulate_kv_cache(
        kv, 0, [0, 3], num_cached_blocks=2, sink_token=True, num_blocks=1
    )
    assert sink_num == 0 and cur[0][_SLOT_K] is None


# --------------------------------------------------------------------------- #
# 3. Model-level chunk-loop composition (depth >= 4 -> a softmax/CONCAT block)
# --------------------------------------------------------------------------- #

MC, MT, MHt, MWt = 8, 9, 2, 2  # 9 latent frames, block=3 -> chunks [0,3),[3,6),[6,9)


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


@pytest.fixture
def _global_args():
    prev = _sa_mod._global_server_args
    set_global_server_args(
        SimpleNamespace(
            comfyui_mode=False,
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            attention_backend=None,
        )
    )
    try:
        yield
    finally:
        set_global_server_args(prev)


def _depth4_model():
    arch = SanaWMArchConfig(
        in_channels=MC, out_channels=MC, num_layers=4,  # block 3 -> softmax/CONCAT
        num_attention_heads=2, attention_head_dim=16, linear_head_dim=16,
        num_cross_attention_heads=2, cross_attention_head_dim=16, cross_attention_dim=32,
        caption_channels=32, model_max_length=8, softmax_every_n=4,
        update_rule="torch_recurrent", cam_update_rule="torch_recurrent", chunk_size=None,
    )
    m = SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch)).double().eval()
    for b in m.blocks:
        b.cross_attn = _ZeroCross()
    return m


def test_streaming_loop_runs_and_accumulates_concat_block(_global_args):
    """Drive forward_long chunk-by-chunk (the stage's core loop) on a depth-4
    model: accumulate -> denoise(save=False) -> clean(save=True). Verify finite
    output, threaded GDN state, and that the softmax block (idx 3) accumulates a
    growing K window across chunks (the CONCAT path the depth-2 fixture can't)."""
    m = _depth4_model()
    assert [b.softmax_main for b in m.blocks] == [False, False, False, True]
    torch.manual_seed(0)
    latents = torch.randn(B, MC, MT, MHt, MWt, dtype=torch.float64)
    y = torch.randn(B, 4, 32, dtype=torch.float64)
    cam = torch.randn(B, MT, 20, dtype=torch.float64)
    plk = torch.randn(B, 48, MT, MHt, MWt, dtype=torch.float64)

    seg = Stage._autoregressive_segments(MT, 3)  # [0,3,6,9]
    num_chunks = len(seg) - 1
    kv = [[[None] * _NUM_STREAM_CACHE_SLOTS for _ in range(4)] for _ in range(num_chunks)]
    concat_k_lens = []

    for ci in range(num_chunks):
        chunk_kv, sink_num = Stage._accumulate_kv_cache(
            kv, ci, seg, num_cached_blocks=2, sink_token=True, num_blocks=4
        )
        s, e = seg[ci], seg[ci + 1]
        # softmax block 3 prefix K length entering this chunk
        pk = chunk_kv[3][_SLOT_K]
        concat_k_lens.append(0 if pk is None else pk.shape[1])
        fidx = torch.arange(s, e) if sink_num > 0 else None
        lat = latents[:, :, s:e]
        ts = torch.full((B, e - s), 500.0, dtype=torch.float64)
        # denoise step (save=False reads the accumulated prefix)
        out, _ = m.forward_long(
            hidden_states=lat, encoder_hidden_states=y, timestep=ts,
            camera_conditions=cam, chunk_plucker=plk,
            kv_cache=chunk_kv, save_kv_cache=False, start_f=s, end_f=e, frame_index=fidx,
        )
        assert out.shape == (B, MC, e - s, MHt, MWt)
        assert torch.isfinite(out).all()
        # clean pass writes this chunk's KV
        ts0 = torch.zeros(B, 1, e - s, dtype=torch.float64)
        _, updated = m.forward_long(
            hidden_states=lat, encoder_hidden_states=y, timestep=ts0,
            camera_conditions=cam, chunk_plucker=plk,
            kv_cache=chunk_kv, save_kv_cache=True, start_f=s, end_f=e, frame_index=fidx,
        )
        kv[ci] = updated
        # block 3 (softmax) stores current-chunk K as (B, N_tok, H, D) where
        # N_tok = frames * H * W; blocks 0-2 (GDN) store recurrent state (B,heads,hd,hd).
        n_tok = (e - s) * MHt * MWt
        assert kv[ci][3][_SLOT_K].shape == (B, n_tok, 2, 16)  # heads=2, head_dim=16
        assert kv[ci][0][_SLOT_K].shape == (B, 2, 16, 16)  # GDN state_kv
        assert float(kv[ci][3][_SLOT_TYPE_FLAG].item()) == _CACHE_TYPE_CONCAT
        assert float(kv[ci][0][_SLOT_TYPE_FLAG].item()) == _CACHE_TYPE_STATE

    # softmax prefix grows (in TOKENS = frames*H*W): chunk0 sees 0, chunk1 sees
    # chunk0's 3*4=12, chunk2 sees the rolling+sink window.
    S_tok = MHt * MWt
    assert concat_k_lens[0] == 0
    assert concat_k_lens[1] == 3 * S_tok
    assert concat_k_lens[2] >= 3 * S_tok

# SPDX-License-Identifier: Apache-2.0
"""S1a-2c tests — streaming `forward_long` assembly on the #26153 SANA-WM DiT.

Built up in stages, each pinned at fp64 / CPU / atol=1e-9:
  Stage 0 — RoPE windowing: `WanRotaryPosEmbed` must produce freqs at GLOBAL
    frame positions for a chunk `[start, end)`, equal to the corresponding slice
    of the full table. This is the prerequisite that keeps a chunk's queries
    aligned with the carried K and the softmax concat-window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime import server_args as _sa_mod
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    BidirectionalGDNUCPESinglePathLiteLA,
    SanaWMBlock,
    WanRotaryPosEmbed,
    _SLOT_CAM_K,
    _SLOT_FFN_TCONV,
    _SLOT_K,
    _SLOT_TYPE_FLAG,
    _SLOT_V,
    _build_ucpe_apply_fns,
    _CACHE_TYPE_CONCAT,
    _CACHE_TYPE_STATE,
    _slice_rope_to_current_chunk,
    process_camera_conditions_ucpe,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

HEAD_DIM = 112
H, W = 2, 3
S = H * W
T = 6


def _rope():
    return WanRotaryPosEmbed(attention_head_dim=HEAD_DIM, patch_size=(1, 1, 1))


def test_rope_full_equals_range_from_zero():
    rope = _rope()
    dense = rope((T, H, W), torch.device("cpu"))
    ranged = rope(((0, T), H, W), torch.device("cpu"))
    torch.testing.assert_close(ranged, dense, atol=1e-9, rtol=0)


def test_rope_window_equals_slice_of_full():
    rope = _rope()
    start, end = 2, 5
    full = rope((end, H, W), torch.device("cpu"))  # frames 0..end-1
    window = rope(((start, end), H, W), torch.device("cpu"))  # frames start..end-1
    # full is frame-major (1, 1, end*S, D/2); slice frames [start:end].
    torch.testing.assert_close(
        window, full[:, :, start * S : end * S, :], atol=1e-9, rtol=0
    )


def test_rope_frame_index_overrides_to_global_positions():
    rope = _rope()
    start, end = 2, 5
    fidx = torch.arange(start, end)
    by_index = rope((end - start, H, W), torch.device("cpu"), frame_index=fidx)
    by_range = rope(((start, end), H, W), torch.device("cpu"))
    torch.testing.assert_close(by_index, by_range, atol=1e-9, rtol=0)


def test_slice_rope_to_current_chunk_is_noop_when_sized():
    rope = _rope()
    freqs = rope(((1, 4), H, W), torch.device("cpu"))  # 3 frames -> 3*S tokens
    assert _slice_rope_to_current_chunk(freqs, 3 * S) is freqs
    assert _slice_rope_to_current_chunk(None, 3 * S) is None
    # wider table trimmed to the trailing chunk
    wide = rope(((0, 4), H, W), torch.device("cpu"))  # 4 frames
    trimmed = _slice_rope_to_current_chunk(wide, 3 * S)
    torch.testing.assert_close(trimmed, wide[:, :, -3 * S :, :], atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- #
# Stage 1 — attention-level forward_long: cached branch methods
# --------------------------------------------------------------------------- #

AB, AHEADS, ADIM = 1, 2, 16  # head_dim/2 divisible by 4 (UCPE homog. 4-vectors), like 112
AC = AHEADS * ADIM  # in_dim == heads*head_dim
AHW = (T, H, W)
AN = T * S


def _arope():
    # RoPE table sized to the attention's head_dim (NOT the Stage-0 HEAD_DIM).
    return WanRotaryPosEmbed(attention_head_dim=ADIM, patch_size=(1, 1, 1))


def _attn(softmax_main=False):
    m = BidirectionalGDNUCPESinglePathLiteLA(
        in_dim=AC,
        heads=AHEADS,
        head_dim=ADIM,
        update_rule="torch_recurrent",  # match the cached recurrent scan exactly
        cam_update_rule="torch_recurrent",
        softmax_main=softmax_main,
        use_chunked_softmax_attention=softmax_main,
    ).double().eval()
    return m


def _softmax_attn():
    # Build a GDN module (no LocalAttention -> no server-args/backend needed),
    # then flip to softmax mode: softmax blocks have no short conv, and the
    # cached softmax path only reads softmax_attn.softmax_scale (a scalar).
    m = _attn(softmax_main=False)
    m.softmax_main = True
    m.conv_k = None
    m.conv_k_cam = None
    m.softmax_attn = SimpleNamespace(softmax_scale=ADIM**-0.5)
    return m


def _x():
    torch.manual_seed(5)
    return torch.randn(AB, AN, AC, dtype=torch.float64)


def _prope(start=0, end=T):
    # Build UCPE apply fns co-windowed with freqs for frames [start, end).
    torch.manual_seed(7)
    cam = torch.randn(AB, T, 20, dtype=torch.float64)[:, start:end]
    raymats = process_camera_conditions_ucpe(cam, HW=(end - start, H, W), patch_size=(1, 1, 1))
    raymats_flat = raymats.reshape(AB, -1, 4, 4)
    freqs = _arope()(((start, end), H, W), torch.device("cpu"))
    return _build_ucpe_apply_fns(ADIM, raymats_flat, freqs), freqs


def _empty_cache():
    return [None] * 10


def test_main_gdn_cached_single_chunk_reduces_to_dense():
    attn = _attn(softmax_main=False)
    x = _x()
    rope_emb = _arope()((T, H, W), torch.device("cpu"))
    dense, _, _ = attn._main_branch_gdn(x, AHW, rope_emb)
    cache = _empty_cache()
    cached, _, _ = attn._main_branch_gdn_cached(x, AHW, rope_emb, cache, True)
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert cache[_SLOT_K].shape == (AB, AHEADS, ADIM, ADIM)
    assert cache[_SLOT_V].shape == (AB, AHEADS, ADIM, 1)
    assert float(cache[_SLOT_TYPE_FLAG].item()) == _CACHE_TYPE_STATE


def test_cam_gdn_cached_single_chunk_reduces_to_dense():
    attn = _attn(softmax_main=False)
    x = _x()
    (apply_q, apply_kv, apply_o), _ = _prope()
    _, beta, decay = attn._main_branch_gdn(x, AHW, None)
    dense = attn._cam_branch(x, AHW, apply_q, apply_kv, apply_o, beta, decay)
    cache = _empty_cache()
    cached = attn._cam_branch_cached(
        x, AHW, apply_q, apply_kv, apply_o, beta, decay, cache, True
    )
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert cache[_SLOT_CAM_K].shape == (AB, AHEADS, ADIM, ADIM)


def test_softmax_main_cached_last_chunk_matches_whole():
    attn = _softmax_attn()
    x = _x()
    rope_full = _arope()(((0, T), H, W), torch.device("cpu"))
    whole, _, _ = attn._main_branch_softmax_cached(x, AHW, rope_full, _empty_cache(), True)

    split = 2
    rope0 = _arope()(((0, split), H, W), torch.device("cpu"))
    rope1 = _arope()(((split, T), H, W), torch.device("cpu"))
    cache = _empty_cache()
    _o0, _, _ = attn._main_branch_softmax_cached(
        x[:, : split * S, :], (split, H, W), rope0, cache, True
    )
    o1, _, _ = attn._main_branch_softmax_cached(
        x[:, split * S :, :], (T - split, H, W), rope1, cache, True
    )
    # Last chunk's queries attend to [chunk0_K || chunk1_K] == full K.
    torch.testing.assert_close(o1, whole[:, split * S :, :], atol=1e-9, rtol=0)


def test_forward_long_gdn_reduces_to_dense_no_camera():
    attn = _attn(softmax_main=False)
    x = _x()
    rope_emb = _arope()((T, H, W), torch.device("cpu"))
    dense = attn(x, AHW, rope_emb, None)
    cache = _empty_cache()
    cached, ret = attn.forward_long(x, AHW, rope_emb, None, kv_cache=cache, save_kv_cache=True)
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert ret is cache and cache[_SLOT_K] is not None


def test_forward_long_gdn_reduces_to_dense_with_camera():
    attn = _attn(softmax_main=False)
    x = _x()
    (apply_q, apply_kv, apply_o), freqs = _prope()
    prope_fns = (apply_q, apply_kv, apply_o)
    dense = attn(x, AHW, freqs, prope_fns)
    cache = _empty_cache()
    cached, _ = attn.forward_long(x, AHW, freqs, prope_fns, kv_cache=cache, save_kv_cache=True)
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert cache[_SLOT_CAM_K] is not None


# --------------------------------------------------------------------------- #
# Stage 2 — block-level forward_long (cross-attn stubbed; needs server args
# only to construct the cross-attn's LocalAttention)
# --------------------------------------------------------------------------- #


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


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


def _block():
    b = SanaWMBlock(
        hidden_size=AC,
        num_heads=AHEADS,
        head_dim=ADIM,
        mlp_ratio=2.0,
        t_kernel_size=3,
        qk_norm=True,
        cross_norm=True,
        conv_kernel_size=4,
        k_conv_only=True,
        softmax_main=False,
        use_chunk_plucker_post_attn=False,
        update_rule="torch_recurrent",
        cam_update_rule="torch_recurrent",
    ).double().eval()
    # Stub cross-attn (unchanged/uncached in forward_long) to avoid the CUDA
    # attention backend; both forward and forward_long add the same zero.
    b.cross_attn = _ZeroCross()
    return b


def test_block_forward_long_reduces_to_dense(_global_args):
    block = _block()
    x = _x()
    y = torch.randn(AB, 4, AC, dtype=torch.float64)
    t6 = torch.randn(AB, 1, T, 6 * AC, dtype=torch.float64)
    (apply_q, apply_kv, apply_o), freqs = _prope()
    prope_fns = (apply_q, apply_kv, apply_o)

    dense = block(x, y, t6, AHW, freqs, prope_fns, None, None)
    cache = _empty_cache()
    out, ret = block.forward_long(
        x, y, t6, AHW, freqs, prope_fns, None, None, kv_cache=cache, save_kv_cache=True
    )
    torch.testing.assert_close(out, dense, atol=1e-9, rtol=0)
    assert ret is cache
    assert cache[_SLOT_K] is not None  # main GDN state
    assert cache[_SLOT_CAM_K] is not None  # cam state
    assert cache[_SLOT_FFN_TCONV] is not None  # FFN temporal tail

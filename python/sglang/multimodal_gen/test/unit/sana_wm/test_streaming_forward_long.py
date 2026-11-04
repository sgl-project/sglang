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
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.sana_wm import (
    SanaWMArchConfig,
    SanaWMConfig,
)
from sglang.multimodal_gen.runtime.models.dits import sana_wm
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _CACHE_TYPE_STATE,
    _SLOT_CAM_K,
    _SLOT_FFN_TCONV,
    _SLOT_K,
    _SLOT_TYPE_FLAG,
    _SLOT_V,
    BidirectionalGDNUCPESinglePathLiteLA,
    SanaWMBlock,
    SanaWMTransformer3DModel,
    WanRotaryPosEmbed,
    _build_ucpe_apply_fns,
    _slice_rope_to_current_chunk,
    process_camera_conditions_ucpe,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    compute_chunk_plucker,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    streaming_refiner,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    self_forcing_denoise_chunk,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead

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

AB, AHEADS, ADIM = (
    1,
    2,
    16,
)  # head_dim/2 divisible by 4 (UCPE homog. 4-vectors), like 112
AC = AHEADS * ADIM  # in_dim == heads*head_dim
AHW = (T, H, W)
AN = T * S


def _arope():
    # RoPE table sized to the attention's head_dim (NOT the Stage-0 HEAD_DIM).
    return WanRotaryPosEmbed(attention_head_dim=ADIM, patch_size=(1, 1, 1))


def _attn(softmax_main=False):
    m = (
        BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=AC,
            heads=AHEADS,
            head_dim=ADIM,
            update_rule="torch_recurrent",  # match the cached recurrent scan exactly
            cam_update_rule="torch_recurrent",
            softmax_main=softmax_main,
            use_chunked_softmax_attention=softmax_main,
        )
        .double()
        .eval()
    )
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
    raymats = process_camera_conditions_ucpe(
        cam, HW=(end - start, H, W), patch_size=(1, 1, 1)
    )
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
    whole, _, _ = attn._main_branch_softmax_cached(
        x, AHW, rope_full, _empty_cache(), True
    )

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
    cached, ret = attn.forward_long(
        x, AHW, rope_emb, None, kv_cache=cache, save_kv_cache=True
    )
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert ret is cache and cache[_SLOT_K] is not None


def test_forward_long_gdn_reduces_to_dense_with_camera():
    attn = _attn(softmax_main=False)
    x = _x()
    (apply_q, apply_kv, apply_o), freqs = _prope()
    prope_fns = (apply_q, apply_kv, apply_o)
    dense = attn(x, AHW, freqs, prope_fns)
    cache = _empty_cache()
    cached, _ = attn.forward_long(
        x, AHW, freqs, prope_fns, kv_cache=cache, save_kv_cache=True
    )
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert cache[_SLOT_CAM_K] is not None


# --------------------------------------------------------------------------- #
# Stage 2 — block-level forward_long (cross-attn stubbed; needs server args
# only to construct the cross-attn's LocalAttention)
# --------------------------------------------------------------------------- #


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


def _block():
    b = (
        SanaWMBlock(
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
        )
        .double()
        .eval()
    )
    # Stub cross-attn (unchanged/uncached in forward_long) to avoid the CUDA
    # attention backend; both forward and forward_long add the same zero.
    b.cross_attn = _ZeroCross()
    return b


def test_block_forward_long_reduces_to_dense():
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


# --------------------------------------------------------------------------- #
# Stage 3 — model-level forward_long (tiny CPU model; depth 2 => all-GDN blocks,
# no softmax/LocalAttention in the main path; cross-attn stubbed)
# --------------------------------------------------------------------------- #

MB, MC, MT, MH, MW = 1, 8, 4, 2, 2


def _tiny_model(num_layers=2):
    arch = SanaWMArchConfig(
        in_channels=MC,
        out_channels=MC,
        # 2 layers: (i+1)%4 != 0 -> no softmax blocks; 4: block 3 is a softmax
        # (concat-window) block
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        linear_head_dim=16,
        num_cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=32,
        model_max_length=8,
        softmax_every_n=4,
        update_rule="torch_recurrent",
        cam_update_rule="torch_recurrent",
        chunk_size=None,
    )
    m = SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch)).double().eval()
    for b in m.blocks:
        b.cross_attn = _ZeroCross()
    return m


def _model_inputs():
    torch.manual_seed(0)
    return dict(
        hidden_states=torch.randn(MB, MC, MT, MH, MW, dtype=torch.float64),
        encoder_hidden_states=torch.randn(MB, 4, 32, dtype=torch.float64),
        timestep=torch.randint(
            1, 1000, (MB, 1, MT)
        ).double(),  # framewise for both paths
        camera_conditions=torch.randn(MB, MT, 20, dtype=torch.float64),
        chunk_plucker=torch.randn(MB, 48, MT, MH, MW, dtype=torch.float64),
    )


def test_model_forward_long_single_chunk_reduces_to_dense():
    m = _tiny_model()
    inp = _model_inputs()
    with torch.no_grad():
        dense = m(**inp)
        out, cache = m.forward_long(
            **inp, kv_cache=None, save_kv_cache=True, start_f=0, end_f=MT
        )
    torch.testing.assert_close(out, dense, atol=1e-9, rtol=0)
    assert len(cache) == len(m.blocks)
    assert cache[0][_SLOT_K] is not None
    assert cache[0][_SLOT_CAM_K] is not None
    assert cache[0][_SLOT_FFN_TCONV] is not None


def test_model_forward_long_two_chunks_runs_and_windows():
    m = _tiny_model()
    inp = _model_inputs()
    split = 2
    with torch.no_grad():
        o0, c0 = m.forward_long(
            hidden_states=inp["hidden_states"][:, :, :split],
            encoder_hidden_states=inp["encoder_hidden_states"],
            timestep=inp["timestep"][:, :, :split],
            camera_conditions=inp["camera_conditions"],  # full; sliced internally
            chunk_plucker=inp["chunk_plucker"],
            kv_cache=None,
            save_kv_cache=True,
            start_f=0,
            end_f=split,
        )
        o1, c1 = m.forward_long(
            hidden_states=inp["hidden_states"][:, :, split:],
            encoder_hidden_states=inp["encoder_hidden_states"],
            timestep=inp["timestep"][:, :, split:],
            camera_conditions=inp["camera_conditions"],
            chunk_plucker=inp["chunk_plucker"],
            kv_cache=c0,
            save_kv_cache=True,
            start_f=split,
            end_f=MT,
        )
    assert o0.shape == (MB, MC, split, MH, MW)
    assert o1.shape == (MB, MC, MT - split, MH, MW)
    assert torch.isfinite(o0).all() and torch.isfinite(o1).all()
    assert c1 is not None and len(c1) == len(m.blocks)


# --------------------------------------------------------------------------- #
# Host overhead — streaming paths must not sync the host per step, chunk or
# block, and must match the per-call formulations they replaced
# --------------------------------------------------------------------------- #


def test_camera_helpers_run_without_host_reads():
    # meta stands in for CUDA: reading any value back raises there
    plucker = compute_chunk_plucker(torch.randn(1, 9, 20, device="meta"), (2, 3, 4))
    assert plucker.shape == (1, 48, 2, 3, 4)
    raymats = process_camera_conditions_ucpe(
        torch.randn(1, 2, 20, device="meta"), HW=(2, 3, 4), patch_size=(1, 1, 1)
    )
    assert raymats.shape == (1, 2, 3, 4, 4, 4)
    # rays of a frame with an undefined camera still map to the identity
    camera = torch.randn(1, 2, 20, dtype=torch.float64)
    camera[0, 1, :16] = float("nan")
    raymats = process_camera_conditions_ucpe(camera, HW=(2, 3, 4), patch_size=(1, 1, 1))
    torch.testing.assert_close(
        raymats[0, 1],
        torch.eye(4, dtype=torch.float64).expand(3, 4, 4, 4),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(raymats[0, 0]).all()


@torch.no_grad()
def test_forward_long_windows_chunk_inputs_once():
    model = _tiny_model(num_layers=4)
    torch.manual_seed(0)
    inputs = dict(
        hidden_states=torch.randn(1, 8, 2, 2, 2, dtype=torch.float64),
        encoder_hidden_states=torch.randn(1, 4, 32, dtype=torch.float64),
        timestep=torch.full((1, 1, 2), 500.0, dtype=torch.float64),
        camera_conditions=torch.randn(1, 4, 20, dtype=torch.float64),
        chunk_plucker=torch.randn(1, 48, 4, 2, 2, dtype=torch.float64),
        frame_index=torch.arange(2, 4),
        start_f=2,
        end_f=4,
    )

    def run_steps(reuse):
        outputs, caches = [], []
        for step in range(3):
            if step == 0 or not reuse:
                model._chunk_inputs_cache = None
                model._ucpe_apply_fns_cache = None
                model._plucker_emb_cache = None
            with patch.object(torch, "tensor", wraps=torch.tensor) as make_tensor:
                out, cache = model.forward_long(**inputs, save_kv_cache=True)
            # the cache-type flags are host tensors: uploading them blocks per block
            assert all("device" not in call.kwargs for call in make_tensor.mock_calls)
            outputs.append(out)
            caches.append(cache)
        return outputs, caches

    expected, _ = run_steps(reuse=False)
    with patch.object(
        sana_wm,
        "process_camera_conditions_ucpe",
        wraps=sana_wm.process_camera_conditions_ucpe,
    ) as ucpe:
        actual, caches = run_steps(reuse=True)
    # the chunk's steps share one UCPE build (fresh slices missed the cache)
    assert ucpe.call_count == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert {block[_SLOT_TYPE_FLAG].device.type for block in caches[0]} == {"cpu"}


def test_self_forcing_chunk_never_reads_device_timesteps():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    set_timesteps = scheduler.set_timesteps

    def set_device_timesteps(*args, **kwargs):
        set_timesteps(*args, **kwargs)
        scheduler.timesteps = scheduler.timesteps.as_subclass(SyncOnRead)

    scheduler.set_timesteps = set_device_timesteps
    contexts = []

    def forward_long(**kwargs):
        latents = kwargs["hidden_states"]
        return latents * 0.5, ["cache"]

    chunk = {"lat": torch.randn(1, 3, 2, 2, 2)}

    def run(forward_ctx):
        chunk["lat"] = torch.ones(1, 3, 2, 2, 2)
        return self_forcing_denoise_chunk(
            transformer=SimpleNamespace(forward_long=forward_long),
            scheduler=scheduler,
            sigmas=[1.0, 0.6, 0.3],
            get_chunk=lambda: chunk["lat"],
            set_chunk=lambda denoised: chunk.update(lat=denoised),
            cond_mask=torch.zeros(1, 3, 2, 2, 2),
            embeds=None,
            mask=None,
            camera_conditions=None,
            chunk_plucker=None,
            chunk_kv=None,
            start_f=0,
            end_f=2,
            frame_index=None,
            do_cfg=False,
            cfg_scale=1.0,
            target_dtype=torch.float32,
            device=torch.device("cpu"),
            forward_ctx=forward_ctx,
        )

    assert run(None) == ["cache"]
    realtime = chunk["lat"].as_subclass(torch.Tensor).clone()

    def forward_ctx(timestep):
        contexts.append(timestep)
        return patch.object(torch, "is_grad_enabled", torch.is_grad_enabled)

    run(forward_ctx)
    # the batch path reads the schedule once per chunk for its forward contexts
    assert contexts == [1000, 600, 300, 0]
    torch.testing.assert_close(
        chunk["lat"].as_subclass(torch.Tensor), realtime, rtol=0, atol=0
    )


def test_refiner_rope_builds_without_uploads_and_once_per_block():
    rope_calls = []

    def rope(pixel_coords, device):
        rope_calls.append(pixel_coords)
        return pixel_coords.sum(-1)

    rope.patch_size_t, rope.patch_size = 1, 2
    rope.scale_factors, rope.causal_offset = (8, 32, 32), 1
    transformer = SimpleNamespace(rope=rope)
    with patch.object(torch, "tensor", wraps=torch.tensor) as upload:
        coords = streaming_refiner.build_rotary_emb_for_absolute_positions(
            transformer=transformer,
            batch_size=2,
            frame_positions=range(3, 6),
            height=4,
            width=6,
            device=torch.device("cpu"),
            fps=24.0,
        )
    assert upload.call_count == 0
    # the per-call upload formulation it replaced
    grid = torch.stack(
        torch.meshgrid(
            torch.tensor([3.0, 4.0, 5.0]),
            torch.arange(0, 4, 2, dtype=torch.float32),
            torch.arange(0, 6, 2, dtype=torch.float32),
            indexing="ij",
        )
    )
    ends = grid + torch.tensor((1, 2, 2), dtype=torch.float32).view(3, 1, 1, 1)
    latent = torch.stack([grid, ends], -1).flatten(1, 3).unsqueeze(0).repeat(2, 1, 1, 1)
    pixel = latent * torch.tensor((8, 32, 32)).view(1, -1, 1, 1)
    pixel[:, 0] = (pixel[:, 0] + 1 - 8).clamp(min=0)
    pixel[:, 0] = pixel[:, 0] / 24.0
    torch.testing.assert_close(
        (rope_calls[0], coords), (pixel, pixel.sum(-1)), rtol=0, atol=0
    )

    core = streaming_refiner._RefinerCore(transformer, torch.device("cpu"), None)
    block = dict(batch_size=2, height=4, width=6, fps=24.0)
    first = core._block_rotary_emb(frame_positions=range(3, 6), **block)
    # a block's denoise levels and capture pass share the table
    assert core._block_rotary_emb(frame_positions=range(3, 6), **block) is first
    core._block_rotary_emb(frame_positions=range(6, 9), **block)
    assert len(rope_calls) == 3


def test_refiner_block_never_reads_device_sigmas():
    transformer = SimpleNamespace(
        config=SimpleNamespace(patch_size=1, patch_size_t=1),
        transformer_blocks=[None],
    )
    refiner = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        _predict_x0_active_block=lambda *, active, sigma_cur, **_: (
            active * (1.0 - sigma_cur)
        ),
        _capture_block_kv=lambda **_: [(torch.zeros(1, 1), torch.zeros(1, 1))],
    )
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    outputs = []
    for schedule in (sigmas, sigmas.as_subclass(SyncOnRead)):
        runner = streaming_refiner.RefinerChunkRunner(
            refiner,
            prompt_embeds=None,
            prompt_attention_mask=None,
            fps=24.0,
            sigmas=schedule,
            source_sink_frames=0,
            block_size=2,
            kv_max_frames=4,
            seed=0,
            spatial_shape=(2, 2),
        )
        outputs.append(
            runner.refine_block(
                block_idx=0,
                clean_block=torch.ones(1, 3, 2, 2, 2),
                block_start=0,
                block_end=2,
            ).as_subclass(torch.Tensor)
        )
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)

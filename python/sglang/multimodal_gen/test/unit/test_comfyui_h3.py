# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 ComfyUI contracts: layout, adapter, step kwargs, profile, QKV."""
from __future__ import annotations

import math

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
    comfyui_layout_to_packed,
    serialize_comfyui_layout,
)

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (
        torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0
    ) * 32.0


def _frame_grid(h, w):
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(
        _axis_from_sqrt_area(h, 2, area),
        _axis_from_sqrt_area(w, 2, area),
        indexing="ij",
    )
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(
        w, 2, area
    )


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _ref_t_span(blk):
    kind = blk["kind"]
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(blk["ref_audio_t"])
    if kind in ("video", "video_audio"):
        return max(float(blk["ref_audio_t"]), sum(_video_t_spans(blk["latent_t"])))
    return 0.0


def _audio_grid(cursor, t, w_low, w_high):
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


class ComfyUIPackedLayout:
    """Standalone replica of ComfyUI ``PackedLayout`` (no comfy import)."""

    def __init__(
        self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None
    ):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        row = text_len

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        cursor = float(text_len)
        for blk in refs or ():
            cursor += _ref_t_span(blk)

        if keyframes:
            for kf in keyframes:
                cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]
                video_latent = kf.get("latent")
                if video_latent is not None:
                    vt = video_latent.shape[2]
                    n = vt * frame_rows
                    segments.append(("cond", n))
                    pos.append(_video_grid(vt, frame, cond_t))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                audio_latent = kf.get("audio_latent")
                if audio_latent is not None:
                    rt = audio_latent.shape[-1]
                    segments.append(("cond_audio", rt * 2))
                    pos.append(_audio_grid(cond_t, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2

        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(
                            _audio_grid(
                                cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])
                            )
                        )
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


def _sgld_used(packed) -> int:
    return int(packed["cu_seqlens"][1])


def _sgld_segment_kinds(packed, text_len: int) -> list[str]:
    used = _sgld_used(packed)
    img_set = {int(i) for i in packed["img_pos"].tolist()}
    audio_set = {int(i) for i in packed["audio_pos"].tolist()}
    update = {
        int(pos): bool(flag)
        for pos, flag in zip(packed["img_pos"].tolist(), packed["update_mask"].tolist())
    }
    audio_update_mask = packed.get("audio_update_mask")
    if audio_update_mask is None:
        audio_update = {int(i): True for i in packed["audio_pos"].tolist()}
    else:
        audio_update = {
            int(pos): bool(flag)
            for pos, flag in zip(
                packed["audio_pos"].tolist(), audio_update_mask.tolist()
            )
        }

    kinds: list[str] = []
    for row in range(used):
        if row < text_len:
            kind = "text"
        elif row in img_set:
            kind = "video" if update[row] else "cond_or_ref_img"
        elif row in audio_set:
            kind = "audio" if audio_update[row] else "cond_or_ref_audio"
        else:
            kind = "unknown"
        if not kinds or kinds[-1] != kind:
            kinds.append(kind)
    return kinds


def _comfy_coarse_kinds(layout: ComfyUIPackedLayout) -> list[str]:
    mapping = {
        "text": "text",
        "cond": "cond_or_ref_img",
        "ref_img": "cond_or_ref_img",
        "cond_audio": "cond_or_ref_audio",
        "ref_audio": "cond_or_ref_audio",
        "audio": "audio",
        "video": "video",
    }
    kinds: list[str] = []
    for _, _, kind in layout.segments:
        mapped = mapping[kind]
        if not kinds or kinds[-1] != mapped:
            kinds.append(mapped)
    return kinds


def _assert_used_prefix_aligned(comfy: ComfyUIPackedLayout, packed: dict) -> None:
    used = _sgld_used(packed)
    assert used == comfy.seq_len
    assert _sgld_segment_kinds(packed, comfy.signature[0]) == _comfy_coarse_kinds(comfy)
    assert torch.equal(packed["img_pos"], comfy.img_pos)
    assert torch.equal(packed["audio_pos"], comfy.audio_pos)
    assert torch.equal(packed["update_mask"], comfy.img_update)
    sgld_audio_update = packed.get("audio_update_mask")
    if sgld_audio_update is None:
        assert bool(comfy.audio_update.all())
    else:
        assert torch.equal(sgld_audio_update, comfy.audio_update)
    sgld_pos = packed["img_position_ids"][:used].to(torch.float32)
    comfy_pos = comfy.position_ids.to(torch.float32)
    assert torch.equal(sgld_pos, comfy_pos)


def test_t2va_used_prefix_matches_comfyui():
    kwargs = dict(text_len=8, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    comfy = ComfyUIPackedLayout(**kwargs)
    packed = minimax_h3_packed_sequence(**kwargs, include_keyframe_cond=False)
    _assert_used_prefix_aligned(comfy, packed)
    assert [k for _, _, k in comfy.segments] == ["text", "audio", "video"]


def test_fl2va_first_last_used_prefix_matches_comfyui():
    text_len, latent_t, latent_h, latent_w, audio_t = 11, 37, 48, 76, 203
    frame_count = 124
    comfy = ComfyUIPackedLayout(
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=[
            {
                "resolved_frame_index": 0,
                "latent": torch.zeros(1, 24, 1, latent_h, latent_w),
            },
            {
                "resolved_frame_index": frame_count - 1,
                "latent": torch.zeros(1, 24, 1, latent_h, latent_w),
            },
        ],
    )
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0, -1],
        frame_count=frame_count,
    )
    _assert_used_prefix_aligned(comfy, packed)
    assert [k for _, _, k in comfy.segments] == ["text", "cond", "cond", "audio", "video"]


def test_ref2va_image_plus_video_audio_used_prefix_matches_comfyui():
    kwargs = dict(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=5)
    refs = [
        {"kind": "image", "latent_h": 4, "latent_w": 4},
        {
            "kind": "video_audio",
            "ref_audio_t": 3,
            "latent_t": 2,
            "latent_h": 4,
            "latent_w": 4,
        },
    ]
    comfy = ComfyUIPackedLayout(**kwargs, refs=refs)
    packed = minimax_h3_packed_sequence_ref2va_blocks(**kwargs, ref_blocks=refs)
    _assert_used_prefix_aligned(comfy, packed)
    assert [k for _, _, k in comfy.segments] == [
        "text",
        "ref_img",
        "ref_audio",
        "ref_img",
        "audio",
        "video",
    ]


@pytest.mark.xfail(strict=True, reason="D3: SGLD packed_sequence has no cond_audio")
def test_gap_keyframe_audio_latent_cond_audio():
    latent_h, latent_w = 4, 4
    comfy = ComfyUIPackedLayout(
        4,
        2,
        latent_h,
        latent_w,
        3,
        keyframes=[
            {
                "resolved_frame_index": 0,
                "latent": torch.zeros(1, 24, 1, latent_h, latent_w),
                "audio_latent": torch.zeros(1, 32, 2, 2),
            }
        ],
    )
    assert any(kind == "cond_audio" for _, _, kind in comfy.segments)
    packed = minimax_h3_packed_sequence(
        text_len=4,
        latent_t=2,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=3,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0],
        frame_count=5,
    )
    comfy_cond_audio = sum(
        stop - start for start, stop, kind in comfy.segments if kind == "cond_audio"
    )
    sgld_audio_cond = int((~packed.get("audio_update_mask", torch.ones(1))).sum())
    assert sgld_audio_cond == comfy_cond_audio


@pytest.mark.xfail(
    strict=True, reason="D3: SGLD only accepts keyframe signatures (0,), (-1,), (0, -1)"
)
def test_gap_arbitrary_frame_idx():
    packed = minimax_h3_packed_sequence(
        text_len=4,
        latent_t=37,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=True,
        keyframe_frame_indices=[51],
        frame_count=124,
    )
    comfy = ComfyUIPackedLayout(
        4,
        37,
        4,
        4,
        3,
        keyframes=[
            {"resolved_frame_index": 51, "latent": torch.zeros(1, 24, 1, 4, 4)},
        ],
    )
    _assert_used_prefix_aligned(comfy, packed)


def test_translated_layout_covers_keyframe_audio_and_arbitrary_idx():
    latent_h, latent_w = 4, 4
    comfy = ComfyUIPackedLayout(
        4,
        2,
        latent_h,
        latent_w,
        3,
        keyframes=[
            {
                "resolved_frame_index": 51,
                "latent": torch.zeros(1, 24, 1, latent_h, latent_w),
                "audio_latent": torch.zeros(1, 32, 2, 2),
            }
        ],
    )
    packed = comfyui_layout_to_packed(serialize_comfyui_layout(comfy))
    _assert_used_prefix_aligned(comfy, packed)
    assert packed["seq_len"] % 64 == 0
    assert packed["seq_len"] >= comfy.seq_len
    assert any(kind == "cond_audio" for _, _, kind in comfy.segments)
    assert int((~packed["audio_update_mask"]).sum()) > 0


@pytest.mark.xfail(
    strict=True, reason="D3: ComfyUI does not pad; SGLD pads seq_len to a multiple of 64"
)
def test_gap_sgld_pads_seq_len_to_64():
    kwargs = dict(text_len=8, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    comfy = ComfyUIPackedLayout(**kwargs)
    packed = minimax_h3_packed_sequence(**kwargs, include_keyframe_cond=False)
    assert packed["seq_len"] == comfy.seq_len
    assert packed["seq_len"] % 64 == 0
    assert int((packed["token_tags"] < 0).sum()) > 0


# --- adapter ---

import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.minimax_h3 import (
    MiniMaxH3Adapter,
    drop_h3_pinned_sampling_fields,
    worker_transformer_options,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams


def test_begin_sampler_run_bumps_id_and_clears_sent_conds() -> None:
    class _State:
        session_id = "abc"
        _run_id = 0
        _sent_conds = {("old",)}

        begin_sampler_run = SGLDiffusionExecutor.begin_sampler_run
        comfyui_session_id = SGLDiffusionExecutor.comfyui_session_id

    state = _State()
    SGLDiffusionExecutor.begin_sampler_run(state)
    assert state._run_id == 1
    assert state._sent_conds == set()
    assert SGLDiffusionExecutor.comfyui_session_id(state) == "abc:1"
    SGLDiffusionExecutor.begin_sampler_run(state)
    assert state._run_id == 2
    assert SGLDiffusionExecutor.comfyui_session_id(state) == "abc:2"


def test_h3_sampling_params_kwargs_omit_pinned_fields() -> None:
    kwargs = drop_h3_pinned_sampling_fields(
        {
            "prompt": " ",
            "guidance_scale": 4.5,
            "height": 48,
            "width": 28,
            "num_frames": 1,
            "num_inference_steps": 1,
            "save_output": False,
            "suppress_logs": True,
        }
    )
    MiniMaxH3SamplingParams(**kwargs)
    assert "guidance_scale" not in kwargs
    assert "num_frames" not in kwargs


def test_worker_transformer_options_drop_comfy_objects() -> None:
    class ModelSampling:
        pass

    opts = worker_transformer_options(
        {
            "minimax_h3_sigma_shift_video": 12.0,
            "sample_sigmas": torch.tensor([1.0, 0.0]),
            "model_sampling": ModelSampling(),
            "patches": {"x": 1},
        }
    )
    assert opts["minimax_h3_sigma_shift_video"] == 12.0
    assert "model_sampling" not in opts
    assert "patches" not in opts


def test_minimax_h3_is_registered() -> None:
    types = registered_model_types()
    assert "minimax_h3" in types
    assert get_adapter_class("minimax_h3") is MiniMaxH3Adapter
    assert get_adapter_class("minimax_h3").pipeline_class_name == "MiniMaxH3Pipeline"


def test_h3_pack_fills_extra_req_and_unpack_restores_shapes() -> None:
    adapter = MiniMaxH3Adapter()
    video = torch.ones(1, 24, 2, 4, 4)
    audio = torch.ones(1, 32, 2, 3) * 2
    x = [video, audio]
    timestep = torch.tensor([500.0])
    context = torch.ones(1, 8, 16)
    payload = {"audio_scale": 1.0, "layout": object()}
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0])
    packed = adapter.pack(
        x,
        timestep,
        context,
        minimax_payload=payload,
        transformer_options={"sample_sigmas": sample_sigmas},
    )
    assert packed.latents.shape == video.shape
    assert packed.extra_req["audio_latents"].shape == audio.shape
    assert packed.extra_req["h3_payload"]["audio_scale"] == 1.0
    assert "layout" not in packed.extra_req["h3_payload"]
    assert "h3_layout" not in packed.extra_req
    assert torch.equal(packed.extra_req["h3_sample_sigmas"], sample_sigmas)
    assert packed.extra_req["comfyui_cache_fp"]["spatial"] == (2, 4, 4)
    assert packed.extra_req["comfyui_cache_fp"]["sigmas"] == (1.0, 0.0, 3)
    assert packed.prompt_embeds[0].shape == (8, 16)
    assert packed.timesteps is timestep

    pred = [torch.zeros_like(video), torch.zeros_like(audio)]
    out = adapter.unpack(pred, packed, x)
    assert isinstance(out, list)
    assert out[0].shape == video.shape
    assert out[1].shape == audio.shape


def test_h3_audio_scale_is_undone_on_pack_and_restored_on_unpack() -> None:
    adapter = MiniMaxH3Adapter()
    video = torch.ones(1, 24, 2, 4, 4)
    audio = torch.ones(1, 32, 2, 3)
    x = [video, audio]
    timestep = torch.tensor([200.0])
    context = torch.ones(1, 4, 8)
    packed = adapter.pack(
        x,
        timestep,
        context,
        minimax_payload={"audio_scale": 0.5},
        transformer_options={
            "minimax_h3_sigma_shift_video": 12.0,
            "minimax_h3_sigma_shift_audio": 3.0,
        },
    )
    assert not torch.equal(packed.extra_req["audio_latents"], audio)
    pred = [torch.ones_like(video), torch.ones_like(audio)]
    out = adapter.unpack(pred, packed, x)
    assert out[0].shape == video.shape
    assert out[1].shape == audio.shape


# --- step kwargs ---

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _FORWARD_SUPPORTED_KWARGS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
    build_step_forward_kwargs,
    comfyui_payload_to_branch_inputs,
    pack_comfy_output,
    pack_audio,
    patchify_video,
    serialize_comfyui_layout,
    unpatchify_video,
    unpack_audio,
)


def _t2va_tensors():
    video = torch.randn(1, 24, 2, 4, 4)
    audio = torch.randn(1, 32, 2, 3)
    context = torch.randn(1, 8, 16)
    timestep = torch.tensor([500.0])
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0])
    return video, audio, context, timestep, sample_sigmas


def test_t2va_forward_kwargs_are_supported_and_complete():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload={},
        sample_sigmas=sample_sigmas,
        timestep=timestep,
    )
    assert inputs["used"] == 8 + 3 * 2 + 2 * 2 * 2
    assert inputs["packed"]["seq_len"] % 64 == 0
    assert inputs["packed"]["seq_len"] >= inputs["used"]
    assert torch.equal(inputs["text_embeddings"], context[0])
    assert inputs["video_rows"].shape[0] == 2 * 2 * 2
    assert inputs["audio_rows"].shape[0] == 6

    fk, branch = build_step_forward_kwargs(inputs, device=torch.device("cpu"))
    assert set(fk) <= _FORWARD_SUPPORTED_KWARGS
    for required in (
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "block_token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "block_combined_indices",
    ):
        assert required in fk, required
    assert torch.equal(fk["prompt_embeds"], context[0].to(fk["prompt_embeds"].device))
    assert fk["x"].shape[1] == inputs["packed"]["seq_len"]
    assert branch.seq_len == inputs["packed"]["seq_len"]


def test_context_is_used_as_refined_prompt_embeds():
    video, audio, _context, timestep, sample_sigmas = _t2va_tensors()
    refined = torch.randn(8, 32)
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        refined,
        minimax_payload={},
        sample_sigmas=sample_sigmas,
        timestep=timestep,
    )
    fk, _branch = build_step_forward_kwargs(inputs, device=torch.device("cpu"))
    assert torch.equal(fk["prompt_embeds"], refined)


def test_pack_comfy_output_crops_to_unpadded_shape():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload={},
        sample_sigmas=sample_sigmas,
        timestep=timestep,
    )
    _fk, branch = build_step_forward_kwargs(inputs, device=torch.device("cpu"))
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
        MiniMaxH3ComfyUIRunState,
    )

    state = MiniMaxH3ComfyUIRunState(
        branch=branch,
        used=inputs["used"],
        orig_video_shape=inputs["orig_video_shape"],
        padded_video_shape=inputs["padded_video_shape"],
    )
    v_video = torch.zeros(inputs["video_rows"].shape)
    v_audio = torch.zeros(inputs["audio_rows"].shape)
    out_video, out_audio = pack_comfy_output(v_video, v_audio, state)
    assert out_video.shape == video.shape
    assert out_audio.shape == audio.shape


def test_patchify_roundtrip():
    video = torch.arange(1 * 24 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 24, 2, 4, 4)
    rows = patchify_video(video)
    restored = unpatchify_video(rows, 2, 2, 2)
    assert torch.equal(restored, video)
    audio = torch.arange(1 * 32 * 2 * 3, dtype=torch.float32).reshape(1, 32, 2, 3)
    packed = pack_audio(audio)
    assert torch.equal(unpack_audio(packed), audio)


def test_keyframes_without_layout_are_still_rejected():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    with pytest.raises(NotImplementedError, match="PackedLayout"):
        comfyui_payload_to_branch_inputs(
            video,
            audio,
            context,
            minimax_payload={"keyframes": [{"resolved_frame_index": 0}]},
            sample_sigmas=sample_sigmas,
            timestep=timestep,
        )


def test_stale_layout_is_rebuilt_from_current_video_shape():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    stale = ComfyUIPackedLayout(8, 2, 2, 2, 3)
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload={},
        sample_sigmas=sample_sigmas,
        timestep=timestep,
        layout=serialize_comfyui_layout(stale),
    )
    assert stale.signature == (8, 2, 2, 2, 3)
    assert tuple(video.shape[2:]) == (2, 4, 4)
    assert inputs["video_rows"].shape[0] == 2 * 2 * 2
    assert inputs["used"] == 8 + 3 * 2 + 2 * 2 * 2


def test_keyframes_with_serialized_layout_build_kwargs():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    kf_latent = torch.randn(1, 24, 1, 4, 4)
    layout = ComfyUIPackedLayout(
        8,
        2,
        4,
        4,
        3,
        keyframes=[{"resolved_frame_index": 0, "latent": kf_latent}],
    )
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload={
            "keyframes": [{"resolved_frame_index": 0, "latent": kf_latent}],
            "cond_video_latents": [kf_latent],
            "visual_cond_noise_aug": 1.0,
        },
        sample_sigmas=sample_sigmas,
        timestep=timestep,
        layout=serialize_comfyui_layout(layout),
    )
    assert inputs["video_rows"].shape[0] == 4 + 2 * 2 * 2
    fk, branch = build_step_forward_kwargs(inputs, device=torch.device("cpu"))
    assert set(fk) <= _FORWARD_SUPPORTED_KWARGS
    assert branch.video_target_start == 4


def test_uniform_denoise_mask_scales_video_timestep():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload={},
        sample_sigmas=sample_sigmas,
        timestep=timestep,
        denoise_mask=torch.full((1, 1, 2, 4, 4), 0.5),
    )
    sigma_v = float(timestep.reshape(-1)[0] / 1000.0)
    # Text / pad keep the unmasked video timestep; only target rows remapped.
    assert inputs["t_video"] == pytest.approx(1.0 - sigma_v)
    assert inputs["video_rows_t"] is not None
    assert float(inputs["video_rows_t"][0]) == pytest.approx(1.0 - 0.5 * sigma_v)


def test_per_frame_denoise_mask_overlays_only_video_targets():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    mask = torch.ones(1, 1, 2, 4, 4)
    mask[:, :, 0] = 1.0
    mask[:, :, 1] = 0.25
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload={},
        sample_sigmas=sample_sigmas,
        timestep=timestep,
        denoise_mask=mask,
    )
    fk, branch = build_step_forward_kwargs(inputs, device=torch.device("cpu"))
    sigma_v = float(timestep.reshape(-1)[0] / 1000.0)
    unique = fk["unique_timesteps"]
    inverse = fk["inverse_indices"]
    target_ts = unique[inverse[branch.img_target_seq_idx]]
    frame0 = target_ts[:4]
    frame1 = target_ts[4:]
    assert torch.allclose(frame0, torch.full_like(frame0, 1.0 - 1.0 * sigma_v), atol=1e-5)
    assert torch.allclose(frame1, torch.full_like(frame1, 1.0 - 0.25 * sigma_v), atol=1e-5)


def test_ref2va_image_forward_kwargs_are_supported():
    video, audio, context, timestep, sample_sigmas = _t2va_tensors()
    ref_latent = torch.randn(1, 24, 1, 4, 4)
    payload = {
        "refs": [
            {
                "kind": "image",
                "latent_h": 4,
                "latent_w": 4,
                "latent": ref_latent,
            }
        ],
        "text_token_tags": torch.ones(8, dtype=torch.long),
        "visual_cond_noise_aug": 1.0,
    }
    inputs = comfyui_payload_to_branch_inputs(
        video,
        audio,
        context,
        minimax_payload=payload,
        sample_sigmas=sample_sigmas,
        timestep=timestep,
    )
    assert inputs["video_rows"].shape[0] == 4 + 2 * 2 * 2
    assert inputs["audio_rows"].shape[0] == 6
    fk, branch = build_step_forward_kwargs(inputs, device=torch.device("cpu"))
    assert set(fk) <= _FORWARD_SUPPORTED_KWARGS
    assert branch.video_target_start == 4
    assert branch.audio_target_start == 0

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
        MiniMaxH3ComfyUIRunState,
    )

    state = MiniMaxH3ComfyUIRunState(
        branch=branch,
        used=inputs["used"],
        orig_video_shape=inputs["orig_video_shape"],
        padded_video_shape=inputs["padded_video_shape"],
    )
    v_video = torch.zeros(2 * 2 * 2, inputs["video_rows"].shape[1])
    v_audio = torch.zeros(inputs["audio_rows"].shape)
    out_video, out_audio = pack_comfy_output(v_video, v_audio, state)
    assert out_video.shape == video.shape
    assert out_audio.shape == audio.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optional GPU smoke")
def test_optional_gpu_step_output_shapes():
    pytest.skip("no tiny H3 checkpoint on this machine; do not download weights")


# --- profile ---

from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import MiniMaxH3Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    create_comfyui_pipeline_stages,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
    MiniMaxH3ComfyUIStepStage,
)


class _StubPipeline:
    def __init__(self):
        self._stage_name_mapping = {}
        self._stages = []
        self.modules = {"transformer": object(), "scheduler": object()}

    def get_module(self, name):
        return self.modules[name]

    def add_stage(self, stage, stage_name=None):
        name = stage_name or stage.__class__.__name__
        self._stage_name_mapping[name] = stage
        self._stages.append(stage)
        return self


def test_create_comfyui_pipeline_stages_dispatches_to_pipeline_hook():
    called = {}

    class _Hooked(_StubPipeline):
        def create_comfyui_stages(self, server_args):
            called["server_args"] = server_args

    pipeline = _Hooked()
    create_comfyui_pipeline_stages(pipeline, server_args="marker")
    assert called["server_args"] == "marker"
    assert pipeline._stage_name_mapping == {}


def test_h3_comfyui_stages_are_step_stage_only(monkeypatch):
    created = {}

    def _fake_init(self, transformer):
        self.transformer = transformer
        created["transformer"] = transformer

    monkeypatch.setattr(MiniMaxH3ComfyUIStepStage, "__init__", _fake_init)
    pipeline = _StubPipeline()
    MiniMaxH3Pipeline.create_comfyui_stages(pipeline, server_args=None)
    assert list(pipeline._stage_name_mapping) == ["MiniMaxH3ComfyUIStepStage"]
    assert "DenoisingStage" not in pipeline._stage_name_mapping
    assert "ComfyUILatentPreparationStage" not in pipeline._stage_name_mapping
    assert created["transformer"] is pipeline.modules["transformer"]


def test_h3_profile_hook_installs_step_stage(monkeypatch):
    def _fake_init(self, transformer):
        self.transformer = transformer

    monkeypatch.setattr(MiniMaxH3ComfyUIStepStage, "__init__", _fake_init)

    class _H3(_StubPipeline):
        create_comfyui_stages = MiniMaxH3Pipeline.create_comfyui_stages

    pipeline = _H3()
    create_comfyui_pipeline_stages(pipeline, server_args=None)
    assert list(pipeline._stage_name_mapping) == ["MiniMaxH3ComfyUIStepStage"]
    assert "DenoisingStage" not in pipeline._stage_name_mapping


def test_comfyui_transformer_load_uses_gguf_override(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec as comfyui_checkpoints_spec

    called = {}

    class _Spec:
        dit_cls_name = "MiniMaxH3DiTModel"
        param_names_mapping = {}
        inherit_config_mapping = False
        convert_weights = None
        strict = True

        def build_dit_config(self, server_args):
            arch = SimpleNamespace(param_names_mapping={})
            return SimpleNamespace(arch_config=arch)

    class _ModelCls:
        param_names_mapping = {}

    class _Pipeline:
        pipeline_name = "MiniMaxH3Pipeline"
        model_path = "/tmp/minimax_h3_fl2va_bf16.safetensors"
        modules = {"scheduler": "sched"}

        def get_module(self, name):
            return self.modules[name]

    monkeypatch.setattr(
        comfyui_checkpoints_spec, "get_comfyui_checkpoint_spec", lambda name: _Spec()
    )
    monkeypatch.setattr(
        comfyui_checkpoints_spec.ModelRegistry,
        "resolve_model_cls",
        lambda name: (_ModelCls, None),
    )
    monkeypatch.setattr(
        comfyui_checkpoints_spec,
        "resolve_precision",
        lambda *args, **kwargs: "bf16",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.resolve_transformer_gguf_to_load",
        lambda server_args, component: "/tmp/dit.gguf",
    )

    loaded = SimpleNamespace(parameters=lambda: [])

    def _fake_gguf(**kwargs):
        called.update(kwargs)
        return loaded

    monkeypatch.setattr(
        comfyui_checkpoints_spec, "_load_comfyui_gguf_transformer", _fake_gguf
    )

    server_args = SimpleNamespace(
        transformer_weights_path="/tmp/dit.gguf",
        model_paths={},
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        pin_cpu_memory=True,
        should_start_component_on_cpu=lambda name: False,
        should_use_fsdp_for_component=lambda name: False,
    )
    modules = comfyui_checkpoints_spec.load_comfyui_transformer(
        _Pipeline(), server_args
    )
    assert modules["transformer"] is loaded
    assert called["gguf_file"] == "/tmp/dit.gguf"


# --- qkv load ---

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (
    get_comfyui_checkpoint_spec,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    _reorder_grouped_qkv_to_qkv,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _concat_qkv_rows() -> torch.Tensor:
    # Runtime / ComfyUI layout: [Q_all, K_all, V_all], 2 heads × head_dim 2.
    return torch.tensor(
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        dtype=torch.float32,
    ).reshape(12, 1)


def test_grouped_reorder_is_not_identity_on_comfyui_concat_qkv() -> None:
    concat = _concat_qkv_rows()
    reordered = _reorder_grouped_qkv_to_qkv(
        concat,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=2,
    )
    assert not torch.equal(reordered, concat)


def test_comfyui_h3_spec_disables_grouped_qkv_reorder() -> None:
    spec = get_comfyui_checkpoint_spec("MiniMaxH3Pipeline")
    assert spec is not None
    pipeline_config = SimpleNamespace(dit_config=None)
    server_args = SimpleNamespace(pipeline_config=pipeline_config)
    dit_config = spec.build_dit_config(server_args)
    assert dit_config.arch_config.qkv_checkpoint_grouped is False
    assert MiniMaxH3DiTArchConfig().qkv_checkpoint_grouped is True


def test_concat_qkv_attention_skips_grouped_weight_loader() -> None:
    _ensure_single_process_parallel_runtime()
    concat_arch = MiniMaxH3DiTArchConfig(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_dim=16,
        qkv_checkpoint_grouped=False,
    )
    concat_attn = MiniMaxH3Attention(concat_arch, None, prefix="blocks.0.attn")
    concat_weight = concat_attn.qkv_proj.weight
    assert getattr(concat_weight, "checkpoint_mapping_unsafe", False) is False
    assert getattr(concat_weight, "rank_local_weight_transform", None) is None

    grouped_arch = MiniMaxH3DiTArchConfig(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_dim=16,
        qkv_checkpoint_grouped=True,
    )
    grouped_attn = MiniMaxH3Attention(grouped_arch, None, prefix="blocks.1.attn")
    grouped_weight = grouped_attn.qkv_proj.weight
    assert grouped_weight.checkpoint_mapping_unsafe is True
    assert grouped_weight.rank_local_weight_transform is not None

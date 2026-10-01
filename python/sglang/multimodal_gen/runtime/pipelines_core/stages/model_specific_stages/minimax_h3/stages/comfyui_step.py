# SPDX-License-Identifier: Apache-2.0
"""ComfyUI per-step DiT stage for MiniMax H3.

ComfyUI owns the sampler loop. This stage runs one ``MiniMaxH3DiTModel.forward``
via ``MiniMaxH3DenoiseBranch.forward_kwargs`` and writes unpacked
``[video, audio]`` velocities onto ``batch.noise_pred``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

_logger = logging.getLogger(__name__)

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _FORWARD_SUPPORTED_KWARGS,
)
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_or_create_run_state,
    get_run_state,
    pop_run_state,
    release_run_state,
    session_id_from_req,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

DEFAULT_SIGMA_SHIFT_VIDEO = 12.0
DEFAULT_SIGMA_SHIFT_AUDIO = 3.0
_PATCH_SIZE = (1, 2, 2)
_SEG_TOKEN_TAG = {
    "text": 1,
    "video": 0,
    "cond": 0,
    "ref_img": 0,
    "audio": 2,
    "cond_audio": 2,
    "ref_audio": 2,
}


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma, from_shift, to_shift):
    """d(sigma_to)/d(sigma_from) at the same base-grid point."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


def patchify_video(latent, patch_size=_PATCH_SIZE):
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=_PATCH_SIZE):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    _b, _c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, latent.shape[1])


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def pad_to_patch_size(video, patch_size=_PATCH_SIZE):
    pt, ph, pw = patch_size
    t, h, w = video.shape[2], video.shape[3], video.shape[4]
    pad_t = (pt - t % pt) % pt
    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw
    if pad_t == pad_h == pad_w == 0:
        return video
    return torch.nn.functional.pad(video, (0, pad_w, 0, pad_h, 0, pad_t))


def _as_text_embeddings(context: torch.Tensor) -> torch.Tensor:
    if context.ndim == 3:
        return context[0]
    return context


def _payload_has_keyframes(payload: dict[str, Any] | None) -> bool:
    return bool((payload or {}).get("keyframes"))


def _payload_refs(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    refs = (payload or {}).get("refs") or []
    if not isinstance(refs, (list, tuple)):
        raise TypeError(f"minimax_payload.refs must be a sequence, got {type(refs)!r}")
    return list(refs)


def _apply_cond_noise(rows: torch.Tensor, aug: float, seed: int) -> torch.Tensor:
    if aug >= 1.0:
        return rows
    gen = torch.Generator("cpu").manual_seed(int(seed))
    noise = torch.randn(rows.shape, generator=gen, dtype=torch.float32)
    return aug * rows + (1.0 - aug) * noise.to(rows.device)


def _ref_blocks_for_packed(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for ref in refs:
        block = {
            key: value
            for key, value in ref.items()
            if key not in {"latent", "audio_latent"}
        }
        if "kind" not in block and "type" in block:
            block["kind"] = block["type"]
        blocks.append(block)
    return blocks


def serialize_comfyui_layout(layout: Any) -> dict[str, Any] | None:
    """Turn a ComfyUI ``PackedLayout`` into pickle/IPC-safe tensors."""
    required = (
        "seq_len",
        "signature",
        "segments",
        "position_ids",
        "img_pos",
        "img_update",
        "audio_pos",
        "audio_update",
    )
    if layout is None or any(not hasattr(layout, name) for name in required):
        return None
    return {
        "seq_len": int(layout.seq_len),
        "signature": tuple(int(value) for value in layout.signature),
        "segments": [
            (int(start), int(stop), str(kind)) for start, stop, kind in layout.segments
        ],
        "position_ids": layout.position_ids.detach().cpu().to(torch.float64),
        "img_pos": layout.img_pos.detach().cpu().to(torch.long),
        "img_update": layout.img_update.detach().cpu().to(torch.bool),
        "audio_pos": layout.audio_pos.detach().cpu().to(torch.long),
        "audio_update": layout.audio_update.detach().cpu().to(torch.bool),
    }


def _layout_signature_matches(
    layout: dict[str, Any],
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    text: torch.Tensor,
) -> bool:
    """ComfyUI ``PackedLayout.signature`` is ``(text, T, H, W, audio_t)``.

    A cached layout from an earlier sampler pass (e.g. pre-upscale 14×8
    tokens) must not be applied to a different spatial grid.
    """
    sig = layout.get("signature")
    if sig is None:
        return False
    values = tuple(int(v) for v in sig[:5])
    if len(values) < 5:
        return False
    text_len, latent_t, latent_h, latent_w, audio_t = values
    return (
        text_len == int(text.shape[0])
        and latent_t == int(video_x.shape[2])
        and latent_h == int(video_x.shape[3])
        and latent_w == int(video_x.shape[4])
        and audio_t == int(audio_x.shape[-1])
    )


def comfyui_layout_to_packed(layout: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the SGLD packed dict from a serialized ComfyUI layout.

    Only pads ``seq_len`` to 64. Row order, RoPE ids, and update masks stay
    exactly as ComfyUI computed them.
    """
    used = int(layout["seq_len"])
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )
    position_ids = layout["position_ids"].to(dtype=torch.float64)
    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[:used] = position_ids[:used]
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    text_pos = torch.empty(0, dtype=torch.long)
    for start, stop, kind in layout["segments"]:
        token_tags[start:stop] = _SEG_TOKEN_TAG.get(kind, -1)
        if kind == "text":
            text_pos = torch.arange(start, stop, dtype=torch.long)
    return {
        "seq_len": seq_len,
        "img_pos": layout["img_pos"].view(-1).to(torch.long),
        "audio_pos": layout["audio_pos"].view(-1).to(torch.long),
        "text_pos": text_pos,
        "update_mask": layout["img_update"].view(-1).to(torch.bool),
        "audio_update_mask": layout["audio_update"].view(-1).to(torch.bool),
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def _latent_items(payload: dict[str, Any], key: str) -> list[Any]:
    items = payload.get(key) or []
    if not isinstance(items, (list, tuple)):
        return [items]
    return [item for item in items if item is not None]


def _assemble_media_rows(
    video: torch.Tensor,
    audio: torch.Tensor,
    payload: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    video_target = patchify_video(video, _PATCH_SIZE)
    audio_target = pack_audio(audio)
    vis_aug = float(
        payload.get("visual_cond_noise_aug", MINIMAX_H3_IMGVID_COND_TIMESTEP)
    )
    aud_aug = float(
        payload.get("audio_cond_noise_aug", MINIMAX_H3_AUDIO_REF_COND_TIMESTEP)
    )
    seed = int(payload.get("seed", 0))

    cond_video_src = list(_latent_items(payload, "cond_video_latents"))
    cond_audio_src = list(_latent_items(payload, "cond_audio_latents"))
    if not cond_video_src and not cond_audio_src:
        for block in list(payload.get("keyframes") or []) + _payload_refs(payload):
            if block.get("latent") is not None:
                cond_video_src.append(block["latent"])
            if block.get("audio_latent") is not None:
                cond_audio_src.append(block["audio_latent"])

    cond_video: list[torch.Tensor] = []
    for latent in cond_video_src:
        rows = patchify_video(
            pad_to_patch_size(
                latent.to(device=video.device, dtype=torch.float32),
                _PATCH_SIZE,
            ),
            _PATCH_SIZE,
        )
        cond_video.append(_apply_cond_noise(rows, vis_aug, seed))
    cond_audio: list[torch.Tensor] = []
    for latent in cond_audio_src:
        rows = pack_audio(latent.to(device=audio.device, dtype=torch.float32))
        cond_audio.append(_apply_cond_noise(rows, aud_aug, seed + 1))

    video_rows = (
        torch.cat(cond_video + [video_target], dim=0) if cond_video else video_target
    )
    audio_rows = (
        torch.cat(cond_audio + [audio_target], dim=0) if cond_audio else audio_target
    )
    return video_rows, audio_rows


def _token_tags_for_payload(
    packed: dict[str, torch.Tensor],
    payload: dict[str, Any],
) -> torch.Tensor:
    tags = packed["token_tags"]
    text_tags = payload.get("text_token_tags")
    if text_tags is None:
        return tags
    text_pos = packed["text_pos"].view(-1)
    text_tags = text_tags.reshape(-1).to(dtype=torch.long)
    if int(text_tags.numel()) != int(text_pos.numel()):
        raise ValueError(
            "text_token_tags length "
            f"{int(text_tags.numel())} != packed text_len {int(text_pos.numel())}"
        )
    tags = tags.clone()
    tags[text_pos] = text_tags
    return tags


def _sigma_list(sample_sigmas: Any) -> list[float] | None:
    n = _sample_sigmas_len(sample_sigmas)
    if n is None:
        return None
    if torch.is_tensor(sample_sigmas):
        return [float(v) for v in sample_sigmas.reshape(-1).tolist()]
    return [float(v) for v in sample_sigmas]


def _as_spatial_mask(mask: Any) -> torch.Tensor | None:
    if mask is None:
        return None
    values = mask
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    if int(values.numel()) == 0:
        return None
    if values.ndim == 5:
        values = values[0, 0]
    elif values.ndim == 4:
        values = values[0]
    elif values.ndim == 2:
        values = values.unsqueeze(0)
    return values.to(dtype=torch.float32)


def mask_row_values(mask: Any, latent_t: int, lat_h: int, lat_w: int):
    """Match ComfyUI ``mask_row_values``: [T,H,W] → one float per 2×2 patch row."""
    values = _as_spatial_mask(mask)
    if values is None:
        return None
    if values.ndim != 3:
        raise ValueError(
            f"denoise_mask must be [T,H,W] after squeeze, got {tuple(values.shape)}"
        )
    if values.shape[0] == 1 and latent_t > 1:
        values = values.expand(latent_t, *values.shape[1:])
    elif values.shape[0] != latent_t:
        raise ValueError(
            f"denoise_mask T={int(values.shape[0])} != padded latent T={latent_t}"
        )
    pad_w = lat_w - int(values.shape[-1])
    pad_h = lat_h - int(values.shape[-2])
    if pad_w > 0 or pad_h > 0:
        values = torch.nn.functional.pad(
            values, (0, max(pad_w, 0), 0, max(pad_h, 0)), mode="replicate"
        )
    values = values[:latent_t, :lat_h, :lat_w]
    values = values.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
    rows = values.reshape(-1)
    if bool((rows >= 1.0 - 1e-3).all()):
        return None
    return rows


def _audio_mask_values(mask: Any):
    if mask is None:
        return None
    values = mask.reshape(-1).to(dtype=torch.float32)
    if int(values.numel()) == 0:
        return None
    if bool((values >= 1.0 - 1e-3).all()):
        return None
    return values


def _overlay_per_row_timesteps(
    fk: dict[str, Any],
    branch: MiniMaxH3DenoiseBranch,
    video_rows_t: torch.Tensor | None,
    audio_rows_t: torch.Tensor | None,
) -> dict[str, Any]:
    """Apply ComfyUI per-row denoise_mask only on target video/audio rows."""
    if video_rows_t is None and audio_rows_t is None:
        return fk
    unique = fk["unique_timesteps"].to(torch.float32)
    inverse = fk["inverse_indices"].clone()
    device = inverse.device
    extras = [unique.reshape(-1).detach().cpu()]
    if video_rows_t is not None:
        extras.append(video_rows_t.reshape(-1).to(torch.float32).cpu())
    if audio_rows_t is not None:
        extras.append(audio_rows_t.reshape(-1).to(torch.float32).cpu())
    new_unique = torch.unique(torch.cat(extras), sorted=True)
    inverse = torch.searchsorted(new_unique, unique.cpu()).to(device)[inverse]
    if video_rows_t is not None:
        mapped = torch.searchsorted(
            new_unique, video_rows_t.reshape(-1).to(torch.float32).cpu()
        )
        tgt = branch.img_target_seq_idx
        if int(mapped.numel()) != int(tgt.numel()):
            raise ValueError(
                "denoise_mask rows "
                f"{int(mapped.numel())} != video target rows {int(tgt.numel())}"
            )
        inverse[tgt] = mapped.to(device)
    if audio_rows_t is not None:
        mapped = torch.searchsorted(
            new_unique, audio_rows_t.reshape(-1).to(torch.float32).cpu()
        )
        tgt = branch.audio_target_seq_idx
        if int(mapped.numel()) != int(tgt.numel()):
            raise ValueError(
                "audio_denoise_mask rows "
                f"{int(mapped.numel())} != audio target rows {int(tgt.numel())}"
            )
        inverse[tgt] = mapped.to(device)
    fk["unique_timesteps"] = new_unique.to(device=device, dtype=unique.dtype)
    fk["inverse_indices"] = inverse
    fk["block_combined_indices"] = (
        branch.block_token_tags
        + inverse[branch.local_row_slice] * MINIMAX_H3_ADALN_MODALITY_NUM
    )
    return fk


def comfyui_payload_to_branch_inputs(
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    context: torch.Tensor,
    minimax_payload: dict[str, Any] | None,
    sample_sigmas: Any,
    timestep: torch.Tensor,
    transformer_options: dict[str, Any] | None = None,
    layout: dict[str, Any] | None = None,
    denoise_mask: Any = None,
    audio_denoise_mask: Any = None,
) -> dict[str, Any]:
    """Translate one ComfyUI H3 step into Branch + ``forward_kwargs`` inputs.

    ``context`` is raw Qwen states (last dim 5120). Token refinement is
    precomputed once on the worker after the Branch is created.
    """
    payload = minimax_payload or {}
    orig_shape = tuple(video_x.shape)
    video = pad_to_patch_size(video_x.to(torch.float32), _PATCH_SIZE)
    audio = audio_x.to(torch.float32)
    text = _as_text_embeddings(context)
    if layout is not None and not _layout_signature_matches(
        layout, video_x, audio, text
    ):
        _logger.warning(
            "Ignoring stale ComfyUI PackedLayout signature %s for video %s "
            "audio %s text_len %s",
            layout.get("signature"),
            tuple(video_x.shape),
            tuple(audio.shape),
            int(text.shape[0]),
        )
        layout = None
    if layout is None and _payload_has_keyframes(payload):
        raise NotImplementedError(
            "MiniMaxH3 ComfyUI step needs a serialized PackedLayout for fl2va keyframes"
        )
    text_len = int(text.shape[0])
    latent_t, latent_h, latent_w = (
        int(video.shape[2]),
        int(video.shape[3]),
        int(video.shape[4]),
    )
    audio_t = int(audio.shape[-1])
    refs = _payload_refs(payload)
    if layout is not None:
        packed = comfyui_layout_to_packed(layout)
    elif refs:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            ref_blocks=_ref_blocks_for_packed(refs),
        )
    else:
        packed = minimax_h3_packed_sequence(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            include_keyframe_cond=False,
        )
    used = int(packed["cu_seqlens"][1])
    video_rows, audio_rows = _assemble_media_rows(video, audio, payload)

    opts = transformer_options or {}
    shift_v = float(opts.get("minimax_h3_sigma_shift_video", DEFAULT_SIGMA_SHIFT_VIDEO))
    shift_a = float(opts.get("minimax_h3_sigma_shift_audio", DEFAULT_SIGMA_SHIFT_AUDIO))
    sigma_v = (timestep.reshape(-1)[0] / 1000.0).to(torch.float32).clamp(min=1e-6)
    sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
    t_video = float(1.0 - sigma_v)
    t_audio = float(1.0 - sigma_a)
    t_pin_v = max(t_video, MINIMAX_H3_IMGVID_COND_TIMESTEP)
    t_pin_a = max(t_audio, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP)
    # ComfyUI only remaps the video/audio *target* segments. Text / pad keep t_video.
    video_rows_t = mask_row_values(denoise_mask, latent_t, latent_h, latent_w)
    if video_rows_t is not None:
        video_rows_t = (1.0 - video_rows_t * float(sigma_v)).clamp(max=t_pin_v)
    audio_rows_t = _audio_mask_values(audio_denoise_mask)
    if audio_rows_t is not None:
        audio_rows_t = (1.0 - audio_rows_t * float(sigma_a)).clamp(max=t_pin_a)

    return {
        "packed": packed,
        "text_embeddings": text,
        "token_tags": _token_tags_for_payload(packed, payload),
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "t_video": t_video,
        "t_audio": t_audio,
        "video_rows_t": video_rows_t,
        "audio_rows_t": audio_rows_t,
        "used": used,
        "orig_video_shape": orig_shape,
        "padded_video_shape": tuple(video.shape),
        "sample_sigmas": sample_sigmas,
        "sigma_v": float(sigma_v),
        "shift_v": shift_v,
        "shift_a": shift_a,
        "imgvid_cond_noise_aug": float(
            payload.get("visual_cond_noise_aug", MINIMAX_H3_IMGVID_COND_TIMESTEP)
        ),
        "audio_ref_cond_noise_aug": float(
            payload.get("audio_cond_noise_aug", MINIMAX_H3_AUDIO_REF_COND_TIMESTEP)
        ),
    }


def build_step_forward_kwargs(
    inputs: dict[str, Any],
    *,
    branch: MiniMaxH3DenoiseBranch | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], MiniMaxH3DenoiseBranch]:
    """Assemble DiT ``forward`` kwargs from ``comfyui_payload_to_branch_inputs``."""
    if device is None:
        device = inputs["video_rows"].device
    if branch is None:
        branch = MiniMaxH3DenoiseBranch(
            packed=inputs["packed"],
            text_embeddings=inputs["text_embeddings"],
            token_tags=inputs["token_tags"],
            device=device,
        )
    plan = branch.prepare_timestep_plan(
        video_timesteps=[inputs["t_video"]],
        audio_timesteps=[inputs["t_audio"]],
        imgvid_cond_noise_aug=inputs["imgvid_cond_noise_aug"],
        audio_ref_cond_noise_aug=inputs["audio_ref_cond_noise_aug"],
    )
    fk = branch.forward_kwargs(
        video_rows=inputs["video_rows"].to(device=device, dtype=torch.float32),
        audio_rows=inputs["audio_rows"].to(device=device, dtype=torch.float32),
        step_timesteps=plan[0],
    )
    fk = _overlay_per_row_timesteps(
        fk, branch, inputs.get("video_rows_t"), inputs.get("audio_rows_t")
    )
    extra = set(fk) - _FORWARD_SUPPORTED_KWARGS
    if extra:
        raise TypeError(
            f"MiniMaxH3 ComfyUI step produced unsupported forward kwargs: {sorted(extra)}"
        )
    return fk, branch


def pack_comfy_output(
    v_video: torch.Tensor,
    v_audio: torch.Tensor,
    state: MiniMaxH3ComfyUIRunState,
) -> list[torch.Tensor]:
    _b, _c, orig_t, orig_h, orig_w = state.orig_video_shape
    _pb, _pc, padded_t, padded_h, padded_w = state.padded_video_shape
    video = unpatchify_video(
        v_video,
        padded_t,
        padded_h // 2,
        padded_w // 2,
        c=int(state.orig_video_shape[1]),
        patch_size=_PATCH_SIZE,
    )
    video = video[:, :, :orig_t, :orig_h, :orig_w]
    # DiT returns target video rows, but audio_pos includes reference rows.
    audio = unpack_audio(v_audio[state.branch.audio_target_slice])
    return [video, audio]


def _sample_sigmas_len(sample_sigmas: Any) -> int | None:
    if sample_sigmas is None:
        return None
    if torch.is_tensor(sample_sigmas):
        return int(sample_sigmas.reshape(-1).numel())
    if isinstance(sample_sigmas, (list, tuple)):
        return len(sample_sigmas)
    return None


@dataclass
class MiniMaxH3ComfyUIRunState:
    branch: MiniMaxH3DenoiseBranch
    used: int
    orig_video_shape: tuple[int, ...]
    padded_video_shape: tuple[int, ...]
    sample_sigmas: Any = None
    cache_mounted: bool = False
    refined: bool = False
    steps_done: int = 0
    signature: tuple[Any, ...] | None = None


class MiniMaxH3ComfyUIStepStage(PipelineStage):
    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer
        self._cache_stage = None

    def verify_input(self, batch: Req, server_args: ServerArgs):
        bind_comfyui_session(batch)
        return super().verify_input(batch, server_args)

    def _cache_dit_stage(self):
        if self._cache_stage is None:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
                MiniMaxH3DenoisingStage,
            )

            self._cache_stage = MiniMaxH3DenoisingStage(transformer=self.transformer)
        return self._cache_stage

    def _maybe_mount_cache_dit(
        self, batch: Req, state: MiniMaxH3ComfyUIRunState
    ) -> None:
        if state.cache_mounted:
            return
        n_sigmas = _sample_sigmas_len(state.sample_sigmas)
        if n_sigmas is None or n_sigmas < 2:
            return
        self._cache_dit_stage()._maybe_enable_cache_dit(n_sigmas - 1, batch)
        state.cache_mounted = True

    def _release_run_state(self, state: MiniMaxH3ComfyUIRunState | None) -> None:
        if state is None:
            return
        if state.cache_mounted and self._cache_stage is not None:
            self._cache_stage._unmount_cache_dit()
            state.cache_mounted = False
        state.branch = None

    def _read_step_tensors(
        self, batch: Req
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extra = getattr(batch, "extra", None) or {}
        video = batch.latents
        audio = batch.audio_latents
        if audio is None:
            audio = extra.get("audio_latents")
        context = extra.get("h3_context")
        if context is None:
            embeds = batch.prompt_embeds
            if isinstance(embeds, list) and embeds:
                context = embeds[0]
            else:
                context = embeds
        if video is None or audio is None or context is None:
            raise ValueError(
                "MiniMaxH3 ComfyUI step requires latents, audio_latents, and prompt_embeds"
            )
        return video, audio, context

    def _bind_run_state(
        self,
        batch: Req,
        inputs: dict[str, Any],
        sample_sigmas: Any,
        device: torch.device,
    ) -> MiniMaxH3ComfyUIRunState:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
            _precompute_refined_prompt_embeds,
        )

        signature = (
            inputs["orig_video_shape"],
            inputs["used"],
            int(inputs["text_embeddings"].shape[0]),
        )
        existing = get_run_state(batch)
        sigmas = _sigma_list(sample_sigmas)
        is_first_sigma = False
        if sigmas:
            is_first_sigma = abs(float(inputs["sigma_v"]) - sigmas[0]) < 1e-4
        if existing is not None and (
            existing.signature != signature
            or (is_first_sigma and existing.steps_done > 0)
        ):
            self._release_run_state(existing)
            pop_run_state(batch)
            existing = None

        def _factory() -> MiniMaxH3ComfyUIRunState:
            branch = MiniMaxH3DenoiseBranch(
                packed=inputs["packed"],
                text_embeddings=inputs["text_embeddings"],
                token_tags=inputs["token_tags"],
                device=device,
            )
            return MiniMaxH3ComfyUIRunState(
                branch=branch,
                used=inputs["used"],
                orig_video_shape=inputs["orig_video_shape"],
                padded_video_shape=inputs["padded_video_shape"],
                sample_sigmas=sample_sigmas,
                signature=signature,
            )

        state = (
            existing
            if existing is not None
            else get_or_create_run_state(batch, _factory)
        )
        if not state.refined:
            _precompute_refined_prompt_embeds(
                self.transformer, state.branch, device=device
            )
            state.refined = True
        return state

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        bind_comfyui_session(batch)
        extra = getattr(batch, "extra", None) or {}
        video_x, audio_x, context = self._read_step_tensors(batch)
        payload = extra.get("h3_payload") or {}
        transformer_options = extra.get("h3_transformer_options") or {}
        sample_sigmas = extra.get("h3_sample_sigmas")
        if sample_sigmas is None:
            sample_sigmas = batch.sigmas
        timestep = batch.timesteps
        if timestep is None:
            timestep = batch.timestep
        if timestep is None:
            raise ValueError("MiniMaxH3 ComfyUI step requires timesteps")

        inputs = comfyui_payload_to_branch_inputs(
            video_x,
            audio_x,
            context,
            payload,
            sample_sigmas,
            timestep,
            transformer_options=transformer_options,
            layout=extra.get("h3_layout"),
            denoise_mask=extra.get("h3_denoise_mask"),
            audio_denoise_mask=extra.get("h3_audio_denoise_mask"),
        )
        state = self._bind_run_state(batch, inputs, sample_sigmas, video_x.device)
        self._maybe_mount_cache_dit(batch, state)
        fk, _branch = build_step_forward_kwargs(
            inputs, branch=state.branch, device=video_x.device
        )
        v_video, v_audio = self.transformer(**fk)
        batch.noise_pred = pack_comfy_output(v_video, v_audio, state)
        state.steps_done += 1
        n_sigmas = _sample_sigmas_len(sample_sigmas)
        if n_sigmas is not None and n_sigmas >= 2 and state.steps_done >= n_sigmas - 1:
            self._release_run_state(state)
            release_run_state(session_id_from_req(batch))
        return batch


__all__ = [
    "DEFAULT_SIGMA_SHIFT_AUDIO",
    "DEFAULT_SIGMA_SHIFT_VIDEO",
    "MiniMaxH3ComfyUIRunState",
    "MiniMaxH3ComfyUIStepStage",
    "build_step_forward_kwargs",
    "comfyui_layout_to_packed",
    "comfyui_payload_to_branch_inputs",
    "mask_row_values",
    "pack_audio",
    "pack_comfy_output",
    "patchify_video",
    "serialize_comfyui_layout",
    "time_shift_sigma",
    "time_shift_slope",
    "unpack_audio",
    "unpatchify_video",
]

# SPDX-License-Identifier: Apache-2.0

import math
import os
import time
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.sana_wm_camera import (
    SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
    SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    SANA_WM_DEFAULT_TRANSLATION_SPEED,
    coerce_sana_wm_action_camera_to_world,
    coerce_sana_wm_camera_to_world,
    coerce_sana_wm_intrinsics_vec4,
    default_sana_wm_static_camera,
    flatten_sana_wm_camera_conditions,
    latent_frame_sana_wm_camera_conditions,
    pad_or_trim_sana_wm_frames,
    relative_sana_wm_camera_poses,
    sana_wm_action_num_frames,
    sana_wm_default_horizontal_fov_deg,
    scale_sana_wm_intrinsics_to_latent,
    validate_sana_wm_motion_params,
)
from sglang.multimodal_gen.runtime.utils.sana_wm_runtime_cache import (
    clear_sana_wm_request_runtime_cache,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

_SANA_WM_DIAGNOSTICS_ENVS = (
    "SGLANG_SANA_WM_DIAGNOSTICS",
    "SGLANG_SANA_WM_LOG_TENSOR_STATS",
)
_SANA_WM_DIAGNOSTIC_MAX_EXACT_ELEMENTS = 4_194_304
_SANA_WM_DIAGNOSTIC_MAX_SAMPLE_ELEMENTS = 65_536

_SANA_WM_DEFAULT_VAE_TILE_MIN_FRAMES = 96
_SANA_WM_DEFAULT_VAE_TILE_STRIDE_FRAMES = 64
_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY = "sana_wm_condition_image_preprocess"
_SANA_WM_PRECOMPUTED_PROPE_FNS_KEY = "precomputed_prope_fns"
_SANA_WM_PRECOMPUTED_PLUCKER_EMB_KEY = "precomputed_plucker_emb"


def _clear_sana_wm_precomputed_static_conditioning(batch: Req) -> None:
    extra = getattr(batch, "extra", None)
    if extra is None:
        return
    extra.pop(_SANA_WM_PRECOMPUTED_PROPE_FNS_KEY, None)
    extra.pop(_SANA_WM_PRECOMPUTED_PLUCKER_EMB_KEY, None)


def _sana_wm_effective_guidance_scale(batch: Req) -> float:
    cfg_scale = getattr(batch, "true_cfg_scale", None)
    if cfg_scale is None:
        cfg_scale = getattr(batch, "guidance_scale", 1.0)
    if cfg_scale is None:
        return 1.0
    return float(cfg_scale)


def _sana_wm_has_negative_condition(batch: Req) -> bool:
    """Return True if a negative/unconditional CFG condition is present.

    Upstream SANA-WM uses an empty negative prompt as the unconditional branch,
    so ``negative_prompt=""`` is still a valid CFG condition for this model.
    """
    neg_prompt = getattr(batch, "negative_prompt", None)
    if neg_prompt is not None:
        if isinstance(neg_prompt, (list, tuple)):
            return len(neg_prompt) > 0
        return True

    neg_embeds = getattr(batch, "negative_prompt_embeds", None)
    if neg_embeds is None:
        return False
    if isinstance(neg_embeds, torch.Tensor):
        return True
    try:
        return len(neg_embeds) > 0
    except TypeError:
        return bool(neg_embeds)


def _sana_wm_should_do_cfg(batch: Req) -> bool:
    return bool(getattr(batch, "do_classifier_free_guidance", False)) or (
        _sana_wm_effective_guidance_scale(batch) > 1.0
        and _sana_wm_has_negative_condition(batch)
    )


def sana_wm_stage_tp_world_size() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    try:
        return get_tp_world_size()
    except AssertionError:
        return 1


def sana_wm_stage_tp_rank() -> int:
    if sana_wm_stage_tp_world_size() <= 1:
        return 0
    try:
        return get_tp_rank()
    except AssertionError:
        return 0


def sana_wm_is_tp_rank0() -> bool:
    return sana_wm_stage_tp_rank() == 0


def sana_wm_broadcast_tensor_dict_from_tp_rank0(
    tensor_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    if sana_wm_stage_tp_world_size() <= 1:
        if tensor_dict is None:
            raise RuntimeError("SANA-WM TP broadcast payload is missing on rank 0.")
        return tensor_dict
    broadcasted = get_tp_group().broadcast_tensor_dict(tensor_dict, src=0)
    if broadcasted is None:
        raise RuntimeError("SANA-WM TP broadcast returned no payload.")
    return broadcasted


def _pack_sana_wm_text_outputs(
    outputs: tuple[
        list[torch.Tensor],
        list[torch.Tensor | None],
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[int]],
    ],
) -> dict[str, Any]:
    embeds, masks, pooled, embeds_masks, seq_lens = outputs
    return {
        "embeds_count": len(embeds),
        "embeds": {str(index): tensor for index, tensor in enumerate(embeds)},
        "masks_count": len(masks),
        "masks": {str(index): tensor for index, tensor in enumerate(masks)},
        "pooled_count": len(pooled),
        "pooled": {str(index): tensor for index, tensor in enumerate(pooled)},
        "embeds_masks_count": len(embeds_masks),
        "embeds_masks": {
            str(index): tensor for index, tensor in enumerate(embeds_masks)
        },
        "seq_lens": seq_lens,
    }


def _unpack_sana_wm_text_outputs(
    payload: dict[str, Any],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
]:
    def ordered_tensors(name: str) -> list[Any]:
        values = payload.get(name, {})
        return [
            values[str(index)] for index in range(int(payload.get(f"{name}_count", 0)))
        ]

    return (
        ordered_tensors("embeds"),
        ordered_tensors("masks"),
        ordered_tensors("pooled"),
        ordered_tensors("embeds_masks"),
        payload.get("seq_lens", []),
    )


def _resolve_sana_wm_vae_frame_tile_value(
    pipeline_config: SanaWMPipelineConfig,
    direct_attr: str,
    vae_config_attr: str,
    default: int,
) -> int:
    direct_value = getattr(pipeline_config, direct_attr, default)
    if direct_value != default:
        return int(direct_value)

    vae_config = getattr(pipeline_config, "vae_config", None)
    nested_value = getattr(vae_config, vae_config_attr, None)
    return int(nested_value or default)


def sana_wm_diagnostics_enabled() -> bool:
    """Whether to emit detailed SANA-WM tensor-quality diagnostics."""
    return any(
        os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on", "debug"}
        for name in _SANA_WM_DIAGNOSTICS_ENVS
    )


def log_sana_wm_tensor_stats(label: str, tensor: torch.Tensor | None) -> None:
    if not sana_wm_diagnostics_enabled():
        return
    if tensor is None:
        logger.info("[SANA-WM diagnostics] %s: None", label)
        return
    if not isinstance(tensor, torch.Tensor):
        logger.info(
            "[SANA-WM diagnostics] %s: non-tensor type=%s",
            label,
            type(tensor).__name__,
        )
        return

    with torch.no_grad():
        data = tensor.detach()
        if data.numel() == 0:
            logger.info(
                "[SANA-WM diagnostics] %s: shape=%s dtype=%s device=%s empty",
                label,
                tuple(data.shape),
                data.dtype,
                data.device,
            )
            return

        numel = data.numel()
        sampled = numel > _SANA_WM_DIAGNOSTIC_MAX_EXACT_ELEMENTS
        sample_stride = 1
        if sampled:
            sample_stride = max(
                1, math.ceil(numel / _SANA_WM_DIAGNOSTIC_MAX_SAMPLE_ELEMENTS)
            )
            indices = torch.arange(0, numel, sample_stride, device=data.device)
            stats = torch.take(data, indices).float()
        else:
            stats = data.float().reshape(-1)
        finite = torch.isfinite(stats)
        finite_ratio = float(finite.float().mean().item())
        finite_stats = stats[finite] if bool(finite.any().item()) else stats
        flat = finite_stats.reshape(-1)
        stride = max(1, flat.numel() // 4096)
        fingerprint = float(flat[::stride].sum().item())
        std = float(finite_stats.std(unbiased=False).item())
        logger.info(
            "[SANA-WM diagnostics] %s: shape=%s dtype=%s device=%s "
            "finite=%.6f min=%.6g max=%.6g mean=%.6g std=%.6g "
            "l2=%.6g fingerprint=%.6g sampled=%s sample_stride=%d sample_size=%d",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            finite_ratio,
            float(finite_stats.min().item()),
            float(finite_stats.max().item()),
            float(finite_stats.mean().item()),
            std,
            float(torch.linalg.vector_norm(finite_stats).item()),
            fingerprint,
            sampled,
            sample_stride,
            stats.numel(),
        )


def configure_sana_wm_ltx2_vae_for_long_video(
    vae: Any,
    pipeline_config: SanaWMPipelineConfig,
    *,
    log_info: Any | None = None,
) -> None:
    min_frames = _resolve_sana_wm_vae_frame_tile_value(
        pipeline_config,
        "vae_tile_sample_min_num_frames",
        "tile_sample_min_num_frames",
        _SANA_WM_DEFAULT_VAE_TILE_MIN_FRAMES,
    )
    stride_frames = _resolve_sana_wm_vae_frame_tile_value(
        pipeline_config,
        "vae_tile_sample_stride_num_frames",
        "tile_sample_stride_num_frames",
        _SANA_WM_DEFAULT_VAE_TILE_STRIDE_FRAMES,
    )

    use_tiling = bool(getattr(pipeline_config, "vae_tiling", True))
    if use_tiling and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling(
                tile_sample_min_num_frames=min_frames,
                tile_sample_stride_num_frames=stride_frames,
            )
        except TypeError:
            vae.enable_tiling()

    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = bool(
            getattr(pipeline_config, "vae_framewise_encoding", True)
        )
    if hasattr(vae, "use_framewise_decoding"):
        vae.use_framewise_decoding = bool(
            getattr(pipeline_config, "vae_framewise_decoding", True)
        )

    if hasattr(vae, "tile_sample_min_num_frames"):
        vae.tile_sample_min_num_frames = min_frames
    if hasattr(vae, "tile_sample_stride_num_frames"):
        vae.tile_sample_stride_num_frames = stride_frames

    if log_info is not None:
        log_info(
            "SANA-WM VAE tiling configured: spatial=%s, framewise_encode=%s, "
            "framewise_decode=%s, tile_frames_min=%d, tile_frames_stride=%d",
            getattr(vae, "use_tiling", use_tiling),
            getattr(vae, "use_framewise_encoding", None),
            getattr(vae, "use_framewise_decoding", None),
            getattr(vae, "tile_sample_min_num_frames", min_frames),
            getattr(vae, "tile_sample_stride_num_frames", stride_frames),
        )


class SanaWMDecodingStage(DecodingStage):
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM decoding stage inputs."""
        result = super().verify_input(batch, server_args)
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        configure_sana_wm_ltx2_vae_for_long_video(
            self.vae,
            server_args.pipeline_config,
            log_info=self.log_info,
        )
        frames = super().decode(latents, server_args, vae_dtype=vae_dtype)
        log_sana_wm_tensor_stats("decode.frames", frames)
        return frames


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value if isinstance(value, torch.Tensor) else None


def _sana_wm_tensor_or_tensor_list(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return V.is_tensor(value)
    return V.list_of_tensors(value)


def _sana_wm_optional_tensor_or_tensor_list_allow_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return V.is_tensor(value)
    if isinstance(value, (list, tuple)):
        return all(
            isinstance(item, torch.Tensor) and V.is_tensor(item) for item in value
        )
    return False


def _sana_wm_none_or_positive_int(value: Any) -> bool:
    return value is None or V.positive_int(value)


def _sana_wm_condition_inputs_dict(value: Any) -> bool:
    return value is None or isinstance(value, dict)


def _sana_wm_condition_inputs_mutually_compatible(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    if value.get("action") is None:
        return True
    return (
        value.get("camera_to_world") is None
        and value.get("camera_conditions") is None
        and value.get("chunk_plucker") is None
    )


def _sana_wm_condition_inputs_motion_params_valid(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    motion_keys = ("translation_speed", "rotation_speed_deg", "pitch_limit_deg")
    if not any(key in value for key in motion_keys):
        return True
    try:
        validate_sana_wm_motion_params(
            translation_speed=value.get(
                "translation_speed", SANA_WM_DEFAULT_TRANSLATION_SPEED
            ),
            rotation_speed_deg=value.get(
                "rotation_speed_deg", SANA_WM_DEFAULT_ROTATION_SPEED_DEG
            ),
            pitch_limit_deg=value.get(
                "pitch_limit_deg", SANA_WM_DEFAULT_PITCH_LIMIT_DEG
            ),
        )
    except ValueError:
        return False
    return True


def _sana_wm_condition_image_not_empty(value: Any) -> bool:
    return not isinstance(value, list) or len(value) > 0


def _sana_wm_camera_conditions_ready(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, torch.Tensor)
        and V.is_tensor(value)
        and value.dim() == 3
        and value.shape[-1] == 20
    )


def _sana_wm_chunk_plucker_ready(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, torch.Tensor)
        and V.is_tensor(value)
        and value.dim() == 5
        and value.shape[1] == 48
    )


def _to_device_dtype(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def _cat_optional_tensors(
    neg: torch.Tensor | None,
    pos: torch.Tensor | None,
) -> torch.Tensor | None:
    if neg is None and pos is None:
        return None
    if neg is None:
        return pos
    if pos is None:
        return neg
    return torch.cat([neg, pos], dim=0)


def _text_sequence_dim(tensor: torch.Tensor) -> int:
    return -2 if tensor.ndim >= 3 else -1


def _pad_text_sequence(
    tensor: torch.Tensor | None,
    target_length: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    seq_dim = _text_sequence_dim(tensor)
    current_length = tensor.shape[seq_dim]
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        index = [slice(None)] * tensor.ndim
        index[seq_dim] = slice(0, target_length)
        return tensor[tuple(index)]

    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = target_length - current_length
    padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=seq_dim)


def _default_attention_mask_for_embeds(embeds: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (embeds.shape[0], embeds.shape[-2]),
        device=embeds.device,
        dtype=torch.long,
    )


def _align_sana_wm_cfg_text_conditions(
    pos_embeds: torch.Tensor,
    neg_embeds: torch.Tensor | None,
    pos_mask: torch.Tensor | None,
    neg_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if neg_embeds is None:
        return pos_embeds, neg_embeds, pos_mask, neg_mask

    target_length = max(pos_embeds.shape[-2], neg_embeds.shape[-2])
    pos_embeds = _pad_text_sequence(pos_embeds, target_length)
    neg_embeds = _pad_text_sequence(neg_embeds, target_length)

    if pos_mask is None:
        pos_mask = _default_attention_mask_for_embeds(pos_embeds)
    if neg_mask is None:
        neg_mask = _default_attention_mask_for_embeds(neg_embeds)
    pos_mask = _pad_text_sequence(pos_mask, target_length)
    neg_mask = _pad_text_sequence(neg_mask, target_length)
    return pos_embeds, neg_embeds, pos_mask, neg_mask


class SanaWMTextEncodingStage(TextEncodingStage):
    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if sana_wm_stage_tp_world_size() > 1 and not sana_wm_is_tp_rank0():
            return []
        return super().component_uses(server_args, stage_name)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM text encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check("text_encoders", self.text_encoders, lambda x: len(x) == 1)
        result.add_check("tokenizers", self.tokenizers, lambda x: len(x) == 1)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            _sana_wm_optional_tensor_or_tensor_list_allow_empty,
        )
        result.add_check(
            "negative_prompt",
            batch.negative_prompt,
            lambda x: (
                not batch.do_classifier_free_guidance
                or _first_tensor(batch.negative_prompt_embeds) is not None
                or V.string_or_list_strings(x)
            ),
        )
        return result

    @staticmethod
    def _text_encoder_max_length(server_args: ServerArgs) -> int:
        encoder_cfg = server_args.pipeline_config.text_encoder_configs[0]
        arch_config = getattr(encoder_cfg, "arch_config", None)
        return int(getattr(arch_config, "text_len", 300) or 300)

    @staticmethod
    def _chi_prompt(server_args: ServerArgs) -> str:
        parts = getattr(server_args.pipeline_config, "chi_prompt", ()) or ()
        return "\n".join(parts)

    @staticmethod
    def _select_official_prompt_window(
        tensor: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        seq_dim = _text_sequence_dim(tensor)
        if tensor.shape[seq_dim] <= max_length:
            return tensor
        index = [slice(None)] * tensor.ndim
        tail_start = tensor.shape[seq_dim] - max_length + 1
        select = torch.cat(
            [
                torch.zeros(1, device=tensor.device, dtype=torch.long),
                torch.arange(
                    tail_start,
                    tensor.shape[seq_dim],
                    device=tensor.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        index[seq_dim] = select
        return tensor[tuple(index)]

    @staticmethod
    def _seq_lens_from_masks(masks: list[torch.Tensor | None]) -> list[list[int]]:
        seq_lens = []
        for mask in masks:
            if mask is None:
                seq_lens.append([])
            else:
                seq_lens.append([int(x) for x in mask.long().sum(dim=-1).tolist()])
        return seq_lens

    def _encode_text_on_tp_rank0(
        self,
        *args,
        **kwargs,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor | None],
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[int]],
    ]:
        if sana_wm_stage_tp_world_size() <= 1:
            return self.encode_text(*args, **kwargs)

        payload = None
        if sana_wm_is_tp_rank0():
            payload = _pack_sana_wm_text_outputs(self.encode_text(*args, **kwargs))
        payload = sana_wm_broadcast_tensor_dict_from_tp_rank0(payload)
        return _unpack_sana_wm_text_outputs(payload)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if len(self.text_encoders) != 1:
            raise ValueError(
                "SANA-WM stage-1 expects exactly one Gemma-2 text encoder."
            )
        assert batch.prompt is not None

        max_length = self._text_encoder_max_length(server_args)
        chi_prompt = self._chi_prompt(server_args)
        prompt_text = batch.prompt
        if isinstance(prompt_text, str):
            prompt_text = [prompt_text]
        else:
            prompt_text = list(prompt_text)

        tokenizer = self.tokenizers[0]
        if chi_prompt:
            prompt_text = [chi_prompt + text for text in prompt_text]
            max_length_all = len(tokenizer.encode(chi_prompt)) + max_length - 2
        else:
            max_length_all = max_length

        (
            prompt_embeds_list,
            prompt_masks_list,
            pooler_embeds_list,
            prompt_embeds_masks_list,
            _prompt_seq_lens_list,
        ) = self._encode_text_on_tp_rank0(
            prompt_text,
            server_args,
            encoder_index=[0],
            return_attention_mask=True,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
        )

        prompt_embeds_list = [
            self._select_official_prompt_window(tensor, max_length)
            for tensor in prompt_embeds_list
        ]
        prompt_masks_list = [
            self._select_official_prompt_window(tensor, max_length)
            for tensor in prompt_masks_list
        ]
        prompt_embeds_masks_list = [
            self._select_official_prompt_window(tensor, max_length)
            for tensor in prompt_embeds_masks_list
        ]
        prompt_seq_lens_list = self._seq_lens_from_masks(prompt_masks_list)

        has_preencoded_negative = (
            _first_tensor(getattr(batch, "negative_prompt_embeds", None)) is not None
        )
        # Pre-initialise so references below are always bound regardless of which branch runs.
        neg_embeds_list: list[torch.Tensor] = []
        neg_masks_list: list[torch.Tensor] = []
        neg_pooler_embeds_list: list[torch.Tensor] = []
        neg_embeds_masks_list: list[torch.Tensor] = []
        neg_seq_lens_list: list[torch.Tensor] = []

        if batch.do_classifier_free_guidance and not has_preencoded_negative:
            negative_prompt = batch.negative_prompt
            if not isinstance(negative_prompt, (str, list)) or (
                isinstance(negative_prompt, list)
                and not all(isinstance(text, str) for text in negative_prompt)
            ):
                raise TypeError(
                    "SANA-WM CFG negative_prompt must be a string or a list of "
                    f"strings, got {type(negative_prompt).__name__}."
                )
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                _neg_seq_lens_list,
            ) = self._encode_text_on_tp_rank0(
                negative_prompt,
                server_args,
                encoder_index=[0],
                return_attention_mask=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            neg_seq_lens_list = self._seq_lens_from_masks(neg_masks_list)

        self._append_positive_text_outputs(
            batch,
            prompt_embeds_list,
            prompt_masks_list,
            pooler_embeds_list,
            prompt_embeds_masks_list,
            prompt_seq_lens_list,
        )

        if batch.do_classifier_free_guidance:
            if has_preencoded_negative:
                self._align_preencoded_negative_text_outputs(batch, prompt_embeds_list)
            else:
                self._append_negative_text_outputs(
                    batch,
                    prompt_embeds_list,
                    neg_embeds_list,
                    neg_masks_list,
                    neg_pooler_embeds_list,
                    neg_embeds_masks_list,
                    neg_seq_lens_list,
                )

        self.log_info(
            "SANA-WM text encoded with chi_prompt=%s, prompt_window=%d, "
            "positive_raw_window=%d",
            "yes" if chi_prompt else "no",
            max_length,
            max_length_all,
        )
        log_sana_wm_tensor_stats("text.prompt_embeds", prompt_embeds_list[0])
        if batch.do_classifier_free_guidance:
            neg_for_log = (
                _first_tensor(batch.negative_prompt_embeds)
                if has_preencoded_negative
                else neg_embeds_list[0]
            )
            log_sana_wm_tensor_stats("text.negative_prompt_embeds", neg_for_log)

        return batch


class SanaWMDenoisingStage(DenoisingStage):
    @property
    def parallelism_type(self) -> StageParallelismType:
        if self.server_args.enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the SANA-WM denoising loop.

        Wraps the base-class ``forward`` with a try/finally so that the
        pre-computed static conditioning tensors (Plücker emb, prope_fns) that
        were persisted in ``batch.extra`` by ``SanaWMBeforeDenoisingStage``
        are always freed — even when a denoising step raises an exception.
        Without this guard those tensors would remain allocated in GPU memory
        until the batch object itself is garbage-collected.
        """
        try:
            return super().forward(batch, server_args)
        except BaseException:
            # Ensure large pre-computed tensors are released on the error path.
            # _finalize_denoising_loop (happy path) handles this for successful runs.
            _clear_sana_wm_precomputed_static_conditioning(batch)
            clear_sana_wm_request_runtime_cache(batch)
            raise

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM denoising stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            _sana_wm_tensor_or_tensor_list,
        )
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check(
            "guidance_scale",
            _sana_wm_effective_guidance_scale(batch),
            V.non_negative_float,
        )
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            (
                _sana_wm_tensor_or_tensor_list
                if _sana_wm_should_do_cfg(batch)
                else _sana_wm_optional_tensor_or_tensor_list_allow_empty
            ),
        )
        extra = getattr(batch, "extra", None)
        result.add_check("extra", extra, lambda x: x is None or isinstance(x, dict))
        extra = extra or {}
        result.add_check(
            "camera_conditions",
            extra.get("camera_conditions"),
            _sana_wm_camera_conditions_ready,
        )
        result.add_check(
            "chunk_plucker",
            extra.get("chunk_plucker"),
            _sana_wm_chunk_plucker_ready,
        )
        return result

    @staticmethod
    def _write_serial_cfg_latent_model_input(
        buffer: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        expected_shape = (batch_size * 2, *latents.shape[1:])
        if tuple(buffer.shape) != expected_shape:
            raise ValueError(
                "SANA-WM serial CFG latent buffer shape mismatch: expected "
                f"{expected_shape}, got {tuple(buffer.shape)}."
            )
        buffer[:batch_size].copy_(latents)
        buffer[batch_size:].copy_(latents)
        return buffer

    @staticmethod
    def _prepare_step_timesteps(
        step_timestep: torch.Tensor,
        frame_condition_limit: torch.Tensor,
        token_condition_limit: torch.Tensor,
        *,
        do_cfg: bool,
        cfg_parallel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = step_timestep.float()
        model_timestep = torch.minimum(
            timestep.expand(frame_condition_limit.shape),
            frame_condition_limit,
        )
        if do_cfg and not cfg_parallel:
            model_timestep = torch.cat([model_timestep, model_timestep], dim=0)
        per_token_timesteps = torch.minimum(
            timestep.expand(token_condition_limit.shape),
            token_condition_limit,
        )
        return model_timestep, per_token_timesteps

    @staticmethod
    def _combine_serial_cfg_noise_in_place(
        noise_pred: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return noise_pred_text.sub_(noise_pred_uncond).mul_(guidance_scale).add_(
            noise_pred_uncond
        )

    @staticmethod
    def _combine_cfg_parallel_noise(
        noise_pred: torch.Tensor,
        guidance_scale: float,
        cfg_rank: int,
    ) -> torch.Tensor:
        """Combine CFG branches across exactly 2 CFG-parallel ranks.

        Rank 0 holds the positive-branch prediction, rank 1 holds the
        negative-branch prediction.  The all-reduce sums the two scaled
        contributions so every rank ends up with the full CFG output.

        ``cfg_world_size != 2`` is rejected at the start of the denoising loop,
        so ``cfg_rank`` is always 0 or 1 here.
        """
        if cfg_rank not in (0, 1):
            raise ValueError(
                "SANA-WM CFG parallel combine expects cfg_rank 0 or 1, "
                f"got cfg_rank={cfg_rank}."
            )
        if cfg_rank == 0:
            partial = guidance_scale * noise_pred
        else:
            partial = (1.0 - guidance_scale) * noise_pred
        return cfg_model_parallel_all_reduce(partial)

    def _prepare_denoising_loop(
        self, batch: Req, server_args: ServerArgs
    ) -> DenoisingContext:
        if batch.latents is None:
            raise ValueError("SANA-WM denoising requires initialized latents.")
        if batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM denoising expects 5D latents shaped (B, C, T, H, W), "
                f"got {tuple(batch.latents.shape)}."
            )

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE.get(
            getattr(server_args.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        scheduler = getattr(
            batch, "scheduler", None
        ) or get_or_create_request_scheduler(batch, self.scheduler)
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("SANA-WM denoising requires prepared timesteps.")

        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        clear_sana_wm_request_runtime_cache(batch)
        self._maybe_enable_cache_dit_and_torch_compile(num_inference_steps, batch)

        latents = batch.latents.to(device=device, dtype=target_dtype)
        init_condition_latents = latents[:, :, :1].clone()
        condition_mask = torch.zeros(
            (
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )
        condition_mask[:, :, :1] = 1

        pos_embeds = _to_device_dtype(
            _first_tensor(server_args.pipeline_config.get_pos_prompt_embeds(batch)),
            device=device,
            dtype=target_dtype,
        )
        pos_mask = _to_device_dtype(
            _first_tensor(batch.prompt_attention_mask), device=device
        )
        if pos_embeds is None:
            raise ValueError("SANA-WM denoising requires positive prompt embeds.")

        do_cfg = _sana_wm_should_do_cfg(batch)
        guidance_scale = _sana_wm_effective_guidance_scale(batch)
        neg_embeds = None
        neg_mask = None
        if do_cfg:
            neg_embeds = _to_device_dtype(
                _first_tensor(server_args.pipeline_config.get_neg_prompt_embeds(batch)),
                device=device,
                dtype=target_dtype,
            )
            neg_mask = _to_device_dtype(
                _first_tensor(batch.negative_attention_mask), device=device
            )
            if neg_embeds is None:
                raise ValueError("SANA-WM CFG requires negative prompt embeds.")

            pos_embeds, neg_embeds, pos_mask, neg_mask = (
                _align_sana_wm_cfg_text_conditions(
                    pos_embeds, neg_embeds, pos_mask, neg_mask
                )
            )

        extra = batch.extra or {}
        chunk_kwargs = {}
        for key in ("chunk_index", "chunk_size", "chunk_split_strategy"):
            value = extra.get(key)
            if value is not None:
                chunk_kwargs[key] = value
        camera_conditions = _to_device_dtype(
            extra.get("camera_conditions"), device=device, dtype=target_dtype
        )
        chunk_plucker = _to_device_dtype(
            extra.get("chunk_plucker"), device=device, dtype=target_dtype
        )
        precomputed_prope_fns = extra.get(_SANA_WM_PRECOMPUTED_PROPE_FNS_KEY)
        precomputed_plucker_emb = _to_device_dtype(
            extra.get(_SANA_WM_PRECOMPUTED_PLUCKER_EMB_KEY),
            device=device,
            dtype=target_dtype,
        )

        cfg_parallel = bool(server_args.enable_cfg_parallel and do_cfg)
        cfg_rank = get_classifier_free_guidance_rank() if cfg_parallel else 0
        if cfg_parallel:
            cfg_world_size = get_classifier_free_guidance_world_size()
            if cfg_world_size != 2:
                raise ValueError(
                    f"SANA-WM CFG parallel requires exactly 2 CFG ranks (one for "
                    f"the positive branch, one for the negative branch), but "
                    f"cfg_world_size={cfg_world_size}. "
                    "Set --cfg-parallel-size 2 (or disable with --no-cfg-parallel)."
                )

        if cfg_parallel:
            if cfg_rank == 1:
                branch_embeds = neg_embeds
                branch_mask = neg_mask
            else:
                branch_embeds = pos_embeds
                branch_mask = pos_mask
            model_kwargs = {
                "encoder_hidden_states": branch_embeds,
                "encoder_attention_mask": branch_mask,
                "camera_conditions": (
                    None if precomputed_prope_fns is not None else camera_conditions
                ),
                "chunk_plucker": (
                    None if precomputed_plucker_emb is not None else chunk_plucker
                ),
                _SANA_WM_PRECOMPUTED_PROPE_FNS_KEY: precomputed_prope_fns,
                _SANA_WM_PRECOMPUTED_PLUCKER_EMB_KEY: precomputed_plucker_emb,
            }
        else:
            model_kwargs = {
                "encoder_hidden_states": (
                    torch.cat([neg_embeds, pos_embeds], dim=0)
                    if do_cfg
                    else pos_embeds
                ),
                "encoder_attention_mask": (
                    _cat_optional_tensors(neg_mask, pos_mask) if do_cfg else pos_mask
                ),
                "camera_conditions": (
                    None
                    if precomputed_prope_fns is not None
                    else (
                        torch.cat([camera_conditions, camera_conditions], dim=0)
                        if do_cfg and camera_conditions is not None
                        else camera_conditions
                    )
                ),
                "chunk_plucker": (
                    None
                    if precomputed_plucker_emb is not None
                    else (
                        torch.cat([chunk_plucker, chunk_plucker], dim=0)
                        if do_cfg and chunk_plucker is not None
                        else chunk_plucker
                    )
                ),
                _SANA_WM_PRECOMPUTED_PROPE_FNS_KEY: precomputed_prope_fns,
                _SANA_WM_PRECOMPUTED_PLUCKER_EMB_KEY: precomputed_plucker_emb,
            }
        model_kwargs.update(chunk_kwargs)
        model_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            model_kwargs,
        )

        serial_cfg_latent_model_input = (
            torch.empty(
                (latents.shape[0] * 2, *latents.shape[1:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            if do_cfg and not cfg_parallel
            else None
        )
        timestep_frame_condition_limit = (
            1.0 - condition_mask[:, :, :, 0, 0].float()
        ) * 1000.0
        timestep_token_condition_limit = (
            1.0 - condition_mask.flatten(2).squeeze(1).float()
        ) * 1000.0

        self.log_info(
            "SANA-WM flow_euler_ltx denoising: latent=%s, steps=%d, cfg=%s, "
            "cfg_parallel=%s, guidance_scale=%.4f, first_frame_locked=yes",
            tuple(latents.shape),
            len(timesteps),
            do_cfg,
            cfg_parallel,
            guidance_scale,
        )
        log_sana_wm_tensor_stats("denoise.input_latents", latents)

        return DenoisingContext(
            scheduler=scheduler,
            extra_step_kwargs={},
            target_dtype=target_dtype,
            autocast_enabled=(
                bool(getattr(server_args.pipeline_config, "enable_autocast", False))
                and target_dtype != torch.float32
                and not getattr(server_args, "disable_autocast", False)
            ),
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            image_kwargs={},
            pos_cond_kwargs={},
            neg_cond_kwargs={},
            latents=latents,
            boundary_timestep=None,
            z=None,
            reserved_frames_mask=None,
            seq_len=None,
            guidance=torch.empty(0, device=device, dtype=target_dtype),
            is_warmup=batch.is_warmup,
            cfg_policy=None,
            extra={
                "cfg_parallel": cfg_parallel,
                "cfg_rank": cfg_rank,
                "condition_mask": condition_mask,
                "do_cfg": do_cfg,
                "guidance_scale": guidance_scale,
                "init_condition_latents": init_condition_latents,
                "model_kwargs": model_kwargs,
                "serial_cfg_latent_model_input": serial_cfg_latent_model_input,
                "start_time": time.perf_counter(),
                "timestep_frame_condition_limit": timestep_frame_condition_limit,
                "timestep_token_condition_limit": timestep_token_condition_limit,
            },
        )

    def _prepare_step_attn_metadata(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ) -> Any | None:
        return None

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        cfg_parallel = bool(ctx.extra["cfg_parallel"])
        cfg_rank = int(ctx.extra["cfg_rank"])
        condition_mask = ctx.extra["condition_mask"]
        do_cfg = bool(ctx.extra["do_cfg"])
        model_kwargs = ctx.extra["model_kwargs"]

        if cfg_parallel:
            latent_model_input = ctx.latents
        elif do_cfg:
            latent_model_input = self._write_serial_cfg_latent_model_input(
                ctx.extra["serial_cfg_latent_model_input"],
                ctx.latents,
            )
        else:
            latent_model_input = ctx.latents

        model_timestep, per_token_timesteps = self._prepare_step_timesteps(
            step.t_device,
            ctx.extra["timestep_frame_condition_limit"],
            ctx.extra["timestep_token_condition_limit"],
            do_cfg=do_cfg,
            cfg_parallel=cfg_parallel,
        )

        with set_forward_context(
            current_timestep=step.step_index,
            attn_metadata=None,
            forward_batch=batch,
        ):
            noise_pred = step.current_model(
                hidden_states=latent_model_input.to(ctx.target_dtype),
                timestep=model_timestep,
                **model_kwargs,
            )

        if do_cfg:
            guidance_scale = float(ctx.extra["guidance_scale"])
            if cfg_parallel:
                noise_pred = self._combine_cfg_parallel_noise(
                    noise_pred, guidance_scale, cfg_rank
                )
            else:
                noise_pred = self._combine_serial_cfg_noise_in_place(
                    noise_pred,
                    guidance_scale,
                )

        latents_dtype = ctx.latents.dtype
        latents_shape = ctx.latents.shape
        batch_size, channels, _, _, _ = latents_shape
        scheduler_output = ctx.scheduler.step(
            -noise_pred.reshape(batch_size, channels, -1).transpose(1, 2),
            step.t_device,
            ctx.latents.reshape(batch_size, channels, -1).transpose(1, 2),
            per_token_timesteps=per_token_timesteps,
            return_dict=False,
        )[0]
        denoised_latents = scheduler_output.transpose(1, 2).reshape(latents_shape)

        tokens_to_denoise = (
            step.t_device.float() / 1000.0 - 1e-6 < (1.0 - condition_mask)
        )
        ctx.latents = torch.where(tokens_to_denoise, denoised_latents, ctx.latents)
        if ctx.latents.dtype != latents_dtype:
            ctx.latents = ctx.latents.to(latents_dtype)

        if sana_wm_diagnostics_enabled() and (
            step.step_index == 0 or step.step_index == len(ctx.timesteps) - 1
        ):
            log_sana_wm_tensor_stats(
                f"denoise.step_{step.step_index}.noise_pred", noise_pred
            )
            log_sana_wm_tensor_stats(
                f"denoise.step_{step.step_index}.latents", ctx.latents
            )

    def _finalize_denoising_loop(
        self, ctx: DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        clear_sana_wm_request_runtime_cache(batch)
        _clear_sana_wm_precomputed_static_conditioning(batch)

        log_sana_wm_tensor_stats("denoise.output_latents", ctx.latents)
        unchanged = (
            ctx.latents[:, :, :1] - ctx.extra["init_condition_latents"]
        ).abs().max().item()
        self.log_info(
            "SANA-WM flow_euler_ltx denoising finished in %.4f seconds; "
            "first_frame_max_delta=%.6g",
            time.perf_counter() - ctx.extra["start_time"],
            float(unchanged),
        )
        super()._finalize_denoising_loop(ctx, batch, server_args)


class SanaWMBeforeDenoisingStage(PipelineStage):

    def __init__(
        self,
        vae,
        transformer,
        scheduler,
        pipeline_config: SanaWMPipelineConfig,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.pipeline_config = pipeline_config

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses: list[ComponentUse] = []
        pipeline_config = getattr(server_args, "pipeline_config", self.pipeline_config)
        if self.vae is not None:
            vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
            uses.append(
                ComponentUse(
                    stage_name=stage_name,
                    component_name="vae",
                    target_dtype=vae_dtype,
                )
            )
        if self.transformer is not None:
            uses.append(
                ComponentUse(
                    stage_name=stage_name,
                    component_name="transformer",
                    phase="transformer",
                    preferred_ready_after_request=True,
                    memory_intensive=True,
                )
            )
        return uses

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM before-denoising stage inputs."""
        result = VerificationResult()
        pipeline_config = getattr(server_args, "pipeline_config", self.pipeline_config)
        vae_stride = getattr(pipeline_config, "vae_stride", (8, 32, 32))
        height_stride = int(vae_stride[1])
        width_stride = int(vae_stride[2] if len(vae_stride) > 2 else height_stride)

        result.add_check(
            "condition_image",
            getattr(batch, "condition_image", None),
            [V.not_none, _sana_wm_condition_image_not_empty],
        )
        result.add_check(
            "height",
            batch.height,
            [V.positive_int, V.divisible(height_stride)],
        )
        result.add_check(
            "width",
            batch.width,
            [V.positive_int, V.divisible(width_stride)],
        )
        result.add_check(
            "num_frames",
            getattr(batch, "num_frames", None),
            _sana_wm_none_or_positive_int,
        )
        result.add_check(
            "num_inference_steps",
            batch.num_inference_steps,
            V.positive_int,
        )
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            _sana_wm_tensor_or_tensor_list,
        )
        result.add_check(
            "guidance_scale",
            _sana_wm_effective_guidance_scale(batch),
            V.non_negative_float,
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            (
                _sana_wm_tensor_or_tensor_list
                if _sana_wm_should_do_cfg(batch)
                else _sana_wm_optional_tensor_or_tensor_list_allow_empty
            ),
        )
        result.add_check(
            "condition_inputs",
            getattr(batch, "condition_inputs", None),
            [
                _sana_wm_condition_inputs_dict,
                _sana_wm_condition_inputs_mutually_compatible,
                _sana_wm_condition_inputs_motion_params_valid,
            ],
        )
        return result

    @torch.no_grad()
    def _vae_encode_image(
        self,
        image: torch.Tensor,  # (1, C, H, W) or (1, C, 1, H, W) in [0, 1] float
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a single image frame through the VAE encoder."""
        vae = self.vae
        configure_sana_wm_ltx2_vae_for_long_video(vae, self.pipeline_config)
        vae_dtype = PRECISION_TO_TYPE.get(
            self.pipeline_config.vae_precision, torch.bfloat16
        )

        # Normalize to [0, 1] then shift to [-1, 1] as expected by the VAE.
        # uint8 tensors are always in [0, 255]; for float tensors the caller
        # should supply [0, 1] values — we accept both for robustness.
        if not image.is_floating_point():
            image = image.float() / 255.0
        elif image.max() > 1.5:
            image = image / 255.0
        image = (image * 2.0 - 1.0).to(device=device, dtype=vae_dtype)

        # Add temporal dim if not present: (B, C, H, W) → (B, C, 1, H, W)
        if image.dim() == 4:
            image = image.unsqueeze(2)

        log_sana_wm_tensor_stats("first_frame.pixel_input_normalized", image)
        with self.use_declared_component(
            component_name="vae",
            module=vae,
            target_dtype=vae_dtype,
        ) as active_vae:
            vae = active_vae if active_vae is not None else vae
            z = self._extract_vae_latents(vae.encode(image)).float()

        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)
        scaling_factor = self._get_vae_scaling_factor(vae)
        if sana_wm_diagnostics_enabled():
            logger.info(
                "[SANA-WM diagnostics] VAE encode normalization: "
                "has_latents_mean_std=%s scaling_factor=%.6g",
                isinstance(latents_mean, torch.Tensor)
                and isinstance(latents_std, torch.Tensor),
                scaling_factor,
            )
        if isinstance(latents_mean, torch.Tensor) and isinstance(
            latents_std, torch.Tensor
        ):
            latents_mean = latents_mean.to(device=z.device, dtype=z.dtype).view(
                1, -1, 1, 1, 1
            )
            latents_std = latents_std.to(device=z.device, dtype=z.dtype).view(
                1, -1, 1, 1, 1
            )
            z = (z - latents_mean) * scaling_factor / latents_std
        else:
            # Legacy VAE convention: encode applies shift before scaling.
            shift_factor = getattr(vae, "shift_factor", None)
            if shift_factor is not None:
                z = z - (
                    shift_factor.to(z.device, z.dtype)
                    if isinstance(shift_factor, torch.Tensor)
                    else shift_factor
                )
            z = z * scaling_factor

        log_sana_wm_tensor_stats("first_frame.latent_normalized", z)
        return z.to(dtype=dtype)  # (1, 128, 1, H_sp, W_sp)

    @staticmethod
    def _extract_vae_latents(encoded: Any) -> torch.Tensor:
        """Return deterministic VAE latents from common Diffusers encode() outputs.

        Handles DiagonalGaussianDistribution (latent_dist attribute), plain
        tensors, and tuple wrappers — up to 8 nesting levels to guard against
        infinite loops from unexpected encoder return types.
        """
        for _ in range(8):
            if isinstance(encoded, torch.Tensor):
                return encoded

            latent_dist = getattr(encoded, "latent_dist", None)
            if latent_dist is not None:
                if hasattr(latent_dist, "mode"):
                    return latent_dist.mode()
                mean = getattr(latent_dist, "mean", None)
                if isinstance(mean, torch.Tensor):
                    return mean
                if callable(mean):
                    return mean()
                if hasattr(latent_dist, "sample"):
                    return latent_dist.sample()

            if isinstance(encoded, tuple) and encoded:
                encoded = encoded[0]
                continue

            break

        raise TypeError(
            "Unsupported VAE encode output for SANA-WM first-frame conditioning: "
            f"{type(encoded).__name__}"
        )

    def _get_vae_scaling_factor(self, vae) -> float:
        scaling_factor = (
            getattr(getattr(vae, "config", None), "scaling_factor", None)
            or getattr(vae, "scaling_factor", None)
            or getattr(
                self.pipeline_config.vae_config.arch_config,
                "scaling_factor",
                None,
            )
            or 1.0
        )
        if isinstance(scaling_factor, torch.Tensor):
            return float(scaling_factor.item())
        scaling_factor = float(scaling_factor)
        return 1.0 if scaling_factor == 0.0 else scaling_factor

    def _prepare_noise_latents(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        generator: (
            torch.Generator | list[torch.Generator] | tuple[torch.Generator, ...]
        ),
    ) -> torch.Tensor:
        if isinstance(generator, (list, tuple)):
            if not generator:
                raise ValueError("SANA-WM generator list must not be empty.")
            if len(generator) == 1:
                return randn_tensor(
                    shape, generator=generator[0], device=device, dtype=dtype
                )
            if len(generator) != shape[0]:
                raise ValueError(
                    "SANA-WM generator list length must match latent batch size; "
                    f"got {len(generator)} generators for batch {shape[0]}."
                )
            sample_shape = (1, *shape[1:])
            return torch.cat(
                [
                    randn_tensor(
                        sample_shape,
                        generator=sample_generator,
                        device=device,
                        dtype=dtype,
                    )
                    for sample_generator in generator
                ],
                dim=0,
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def _generator_from_seed(
        seed: int | list[int] | tuple[int, ...] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Generator | list[torch.Generator]:
        if seed is None:
            seed = 0
        if isinstance(seed, (list, tuple)):
            if not seed:
                raise ValueError("SANA-WM seed list must not be empty.")
            if len(seed) == 1:
                seed = seed[0]
            elif len(seed) == batch_size:
                return [
                    torch.Generator(device=device).manual_seed(int(sample_seed))
                    for sample_seed in seed
                ]
            else:
                raise ValueError(
                    "SANA-WM seed list length must be 1 or match latent batch "
                    f"size; got {len(seed)} seeds for batch {batch_size}."
                )
        return torch.Generator(device=device).manual_seed(int(seed))

    @staticmethod
    def _canonical_condition_image_tensor(image: torch.Tensor) -> torch.Tensor:
        """Return image as NCHW RGB float tensor without changing its value range.

        Accepted input layouts:
          - NCHW  (N, 1|3|4, H, W)
          - NHWC  (N, H, W, 1|3|4)
          - CHW   (1|3|4, H, W) — only when the channel dim is unambiguous
          - HWC   (H, W, 1|3|4) — only when the channel dim is unambiguous
          - NCFHW singleton-video (N, C, 1, H, W) → squeezed to NCHW

        Raises ``ValueError`` when the layout is ambiguous (both leading and
        trailing dimension fall in the channel-count set {1, 3, 4}) so callers
        get an explicit error instead of a silently wrong layout.
        """
        _CHANNEL_COUNTS = {1, 3, 4}
        image = image.float()
        if image.dim() == 5 and image.shape[2] == 1:
            image = image.squeeze(2)
        if image.dim() == 3:
            c_first, c_last = image.shape[0], image.shape[-1]
            is_channel_first = c_first in _CHANNEL_COUNTS
            is_channel_last = c_last in _CHANNEL_COUNTS
            if is_channel_first and not is_channel_last:
                image = image.unsqueeze(0)  # CHW → NCHW
            elif is_channel_last and not is_channel_first:
                image = image.permute(2, 0, 1).unsqueeze(0)  # HWC → NCHW
            elif is_channel_first and is_channel_last:
                raise ValueError(
                    f"Ambiguous condition_image tensor shape {tuple(image.shape)}: "
                    f"both leading dim ({c_first}) and trailing dim ({c_last}) look "
                    "like channel counts (1, 3, or 4). Pass an explicit NCHW tensor."
                )
            else:
                raise ValueError(
                    "condition_image tensor must be CHW or HWC with 1, 3, or 4 "
                    f"channels; got shape {tuple(image.shape)}."
                )
        elif image.dim() == 4:
            if image.shape[1] in _CHANNEL_COUNTS:
                pass  # already NCHW
            elif image.shape[-1] in _CHANNEL_COUNTS:
                image = image.permute(0, 3, 1, 2)  # NHWC → NCHW
            else:
                raise ValueError(
                    "condition_image tensor must be NCHW or NHWC with 1, 3, "
                    f"or 4 channels; got {tuple(image.shape)}."
                )
        else:
            raise ValueError(
                "condition_image tensor must have shape CHW, HWC, NCHW, NHWC, "
                f"or NCHW singleton-video; got {tuple(image.shape)}."
            )

        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        elif image.shape[1] == 4:
            image = image[:, :3]
        elif image.shape[1] != 3:
            raise ValueError(
                f"condition_image must have 1, 3, or 4 channels; got {image.shape[1]}."
            )
        return image.contiguous()

    @staticmethod
    def _resize_center_crop_tensor(
        image: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
        """Match official SANA-WM resize-then-center-crop preprocessing."""
        image = SanaWMBeforeDenoisingStage._canonical_condition_image_tensor(image)
        src_h, src_w = int(image.shape[-2]), int(image.shape[-1])
        scale = max(target_h / float(src_h), target_w / float(src_w))
        resized_w = max(target_w, int(round(src_w * scale)))
        resized_h = max(target_h, int(round(src_h * scale)))
        if resized_h != src_h or resized_w != src_w:
            import torch.nn.functional as F

            image = F.interpolate(
                image,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
        left = (resized_w - target_w) // 2
        top = (resized_h - target_h) // 2
        image = image[..., top : top + target_h, left : left + target_w].contiguous()
        return image, {
            "source_size": (src_w, src_h),
            "resized_size": (resized_w, resized_h),
            "crop_offset": (left, top),
            "target_size": (target_w, target_h),
        }

    @staticmethod
    def _preprocess_condition_image(
        condition_image: Any,
        *,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
        """Aspect-preserving resize + center crop, mirroring NVlabs/Sana."""
        import PIL.Image

        if isinstance(condition_image, list):
            if len(condition_image) == 0:
                raise ValueError(
                    "condition_image list is empty; SANA-WM requires a first "
                    "frame conditioning image."
                )
            condition_image = condition_image[0]

        if isinstance(condition_image, PIL.Image.Image):
            import torchvision.transforms.functional as TF

            image = condition_image.convert("RGB")
            src_w, src_h = image.size
            scale = max(target_h / float(src_h), target_w / float(src_w))
            resized_w = max(target_w, int(round(src_w * scale)))
            resized_h = max(target_h, int(round(src_h * scale)))
            resampling_enum = getattr(PIL.Image, "Resampling", None)
            resampling = (
                resampling_enum.LANCZOS
                if resampling_enum is not None
                else PIL.Image.LANCZOS
            )
            image = image.resize((resized_w, resized_h), resampling)
            left = (resized_w - target_w) // 2
            top = (resized_h - target_h) // 2
            image = image.crop((left, top, left + target_w, top + target_h))
            return TF.to_tensor(image).unsqueeze(0), {
                "source_size": (src_w, src_h),
                "resized_size": (resized_w, resized_h),
                "crop_offset": (left, top),
                "target_size": (target_w, target_h),
            }

        if isinstance(condition_image, torch.Tensor):
            return SanaWMBeforeDenoisingStage._resize_center_crop_tensor(
                condition_image,
                target_h=target_h,
                target_w=target_w,
            )

        raise TypeError(
            "condition_image must be a PIL image, tensor, or non-empty list; "
            f"got {type(condition_image).__name__}."
        )

    @staticmethod
    def _transform_intrinsics_for_condition_image(
        intrinsics_vec4: torch.Tensor,
        preprocess_info: (
            dict[str, tuple[int, int]] | list[dict[str, tuple[int, int]]] | None
        ),
    ) -> torch.Tensor:
        """Map source-image intrinsics into the cropped output pixel grid."""
        if not preprocess_info:
            return intrinsics_vec4
        if isinstance(preprocess_info, list):
            transform = (
                SanaWMBeforeDenoisingStage._transform_intrinsics_for_condition_image
            )
            if len(preprocess_info) == 1:
                return transform(intrinsics_vec4, preprocess_info[0])
            if len(preprocess_info) != intrinsics_vec4.shape[0]:
                raise ValueError(
                    "SANA-WM condition-image preprocess metadata length must "
                    "match intrinsics batch size; got "
                    f"{len(preprocess_info)} metadata entries for batch "
                    f"{intrinsics_vec4.shape[0]}."
                )
            return torch.cat(
                [
                    transform(intrinsics_vec4[index : index + 1], info)
                    for index, info in enumerate(preprocess_info)
                ],
                dim=0,
            )
        src_w, src_h = preprocess_info["source_size"]
        resized_w, resized_h = preprocess_info["resized_size"]
        left, top = preprocess_info["crop_offset"]
        sx = resized_w / float(src_w)
        sy = resized_h / float(src_h)
        out = intrinsics_vec4.clone()
        out[..., 0] *= sx
        out[..., 2] = out[..., 2] * sx - left
        out[..., 1] *= sy
        out[..., 3] = out[..., 3] * sy - top
        return out

    @staticmethod
    def _set_condition_image_preprocess_info(
        batch: Req | None,
        preprocess_info: Any,
    ) -> None:
        if batch is None:
            return
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}
        batch.extra[_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY] = preprocess_info

    @staticmethod
    def _splice_first_frame_latent(
        latents: torch.Tensor,
        first_frame_z: torch.Tensor,
    ) -> torch.Tensor:
        B = latents.shape[0]
        first_frame_z = first_frame_z.to(device=latents.device, dtype=latents.dtype)
        if first_frame_z.shape[0] == 1 and B > 1:
            first_frame_z = first_frame_z.expand(B, -1, -1, -1, -1)
        elif first_frame_z.shape[0] != B:
            raise ValueError(
                "SANA-WM first-frame latent batch does not match noise batch: "
                f"{first_frame_z.shape[0]} vs {B}."
            )

        latents = latents.clone()
        latents[:, :, 0:1] = first_frame_z
        log_sana_wm_tensor_stats("latents.after_first_frame_splice", latents)
        return latents


    @torch.no_grad()
    def _splice_first_frame(
        self,
        latents: torch.Tensor,  # (B, 128, T_lat, H_sp, W_sp)
        condition_image,  # PIL Image or torch.Tensor
        dtype: torch.dtype,
        device: torch.device,
        batch: Req | None = None,
    ) -> torch.Tensor:
        """Replace latents[:, :, 0] with VAE-encoded first frame."""
        B, _C, _T_lat, H_sp, W_sp = latents.shape
        target_h = H_sp * self.pipeline_config.vae_stride[1]  # 32
        target_w = W_sp * self.pipeline_config.vae_stride[2]  # 32
        condition_images = self._condition_images_for_batch(condition_image, B)
        first_frame_latents = []
        preprocess_infos = []
        for image in condition_images:
            img_tensor, preprocess_info = self._preprocess_condition_image(
                image,
                target_h=target_h,
                target_w=target_w,
            )
            preprocess_infos.append(preprocess_info)
            first_frame_latents.append(
                self._vae_encode_image(img_tensor, dtype, device)
            )

        preprocess_info_for_batch = (
            preprocess_infos[0] if len(preprocess_infos) == 1 else preprocess_infos
        )
        self._set_condition_image_preprocess_info(batch, preprocess_info_for_batch)
        self.log_info(
            "First-frame condition image preprocessed: source=%s, resized=%s, "
            "crop_offset=%s, target=%s.",
            preprocess_infos[0]["source_size"],
            preprocess_infos[0]["resized_size"],
            preprocess_infos[0]["crop_offset"],
            preprocess_infos[0]["target_size"],
        )
        if len(preprocess_infos) > 1:
            self.log_info(
                "Processed %d batched first-frame images.", len(preprocess_infos)
            )

        first_frame_z = torch.cat(first_frame_latents, dim=0)
        return self._splice_first_frame_latent(latents, first_frame_z)

    @staticmethod
    def _condition_images_for_batch(condition_image: Any, batch_size: int) -> list[Any]:
        if isinstance(condition_image, list):
            if not condition_image:
                raise ValueError(
                    "condition_image list is empty; SANA-WM requires a first "
                    "frame conditioning image."
                )
            if len(condition_image) == 1 or len(condition_image) == batch_size:
                return list(condition_image)
            raise ValueError(
                "SANA-WM condition_image list must contain one image or one "
                f"image per batch item; got {len(condition_image)} images for "
                f"batch {batch_size}."
            )
        return [condition_image]

    @staticmethod
    def _condition_inputs(batch: Req) -> dict[str, Any]:
        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        if not isinstance(condition_inputs, dict):
            raise TypeError(
                "SANA-WM condition_inputs must be a dict, "
                f"got {type(condition_inputs).__name__}."
            )
        return condition_inputs

    @staticmethod
    def _action_num_frames_for_request(batch: Req) -> int | None:
        action = SanaWMBeforeDenoisingStage._condition_inputs(batch).get("action")
        if action is None:
            return None
        return sana_wm_action_num_frames(action)

    @staticmethod
    def _has_explicit_camera_request(batch: Req) -> bool:
        condition_inputs = SanaWMBeforeDenoisingStage._condition_inputs(batch)
        if any(
            key in condition_inputs and condition_inputs[key] is not None
            for key in (
                "camera_conditions",
                "chunk_plucker",
                "camera_to_world",
                "intrinsics",
                "action",
            )
        ):
            return True
        return False

    def _build_camera_conditioning(
        self,
        batch: Req,
        *,
        batch_size: int,
        num_frames: int,
        latent_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}
        if not self.pipeline_config.camera_conditioning:
            return None, None, "disabled"

        extra = batch.extra
        condition_inputs = self._condition_inputs(batch)
        T_lat = latent_shape[2]
        sp_h = latent_shape[3]
        sp_w = latent_shape[4]
        vae_temporal_stride = self.pipeline_config.vae_stride[0]
        camera_compute_dtype = torch.float32

        camera_conditions = condition_inputs.get("camera_conditions")
        chunk_plucker = condition_inputs.get("chunk_plucker")
        preprocess_info = extra.get(_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY)
        action = condition_inputs.get("action")
        arch = getattr(
            getattr(self.pipeline_config, "dit_config", None),
            "arch_config",
            None,
        )
        requires_chunk_plucker = bool(
            getattr(arch, "use_chunk_plucker_post_attn", False)
            or getattr(arch, "use_chunk_plucker_input", False)
        )
        if action is not None and (
            camera_conditions is not None or chunk_plucker is not None
        ):
            raise ValueError(
                "SANA-WM action cannot be combined with prepacked "
                "camera_conditions/chunk_plucker."
            )
        if camera_conditions is not None:
            camera_conditions = (
                camera_conditions
                if isinstance(camera_conditions, torch.Tensor)
                else torch.as_tensor(camera_conditions)
            ).to(device=device, dtype=camera_compute_dtype)
            if camera_conditions.dim() == 2:
                camera_conditions = camera_conditions.unsqueeze(0)
            if camera_conditions.dim() != 3:
                raise ValueError(
                    "camera_conditions must have shape (T,20) or (B,T,20), "
                    f"got {tuple(camera_conditions.shape)}"
                )
            if camera_conditions.shape[0] == 1 and batch_size > 1:
                camera_conditions = camera_conditions.expand(batch_size, -1, -1)
            if camera_conditions.shape[0] != batch_size:
                raise ValueError(
                    "camera_conditions batch dimension must be 1 or match "
                    f"request batch size {batch_size}, got "
                    f"{camera_conditions.shape[0]}."
                )
            if camera_conditions.shape[-1] != 20:
                raise ValueError(
                    "camera_conditions must have last dimension 20, got "
                    f"{tuple(camera_conditions.shape)}"
                )
            if camera_conditions.shape[1] == T_lat:
                source = "prepacked"
                if chunk_plucker is None and requires_chunk_plucker:
                    raise ValueError(
                        "Prepacked latent-frame camera_conditions require "
                        "chunk_plucker for this SANA-WM checkpoint. Pass "
                        "chunk_plucker with shape (B,48,T,H,W), or pass "
                        "original-frame camera_conditions so SGLang can "
                        "derive chunk_plucker."
                    )
            else:
                source = "prebuilt_original_frames"
                original_camera_conditions = pad_or_trim_sana_wm_frames(
                    camera_conditions, num_frames
                )
                camera_conditions = latent_frame_sana_wm_camera_conditions(
                    original_camera_conditions,
                    num_frames=num_frames,
                    latent_frames=T_lat,
                    vae_temporal_stride=vae_temporal_stride,
                )
                if chunk_plucker is None:
                    from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
                        compute_chunk_plucker,
                    )

                    chunk_plucker = compute_chunk_plucker(
                        camera_conditions=original_camera_conditions,
                        HW=(T_lat, sp_h, sp_w),
                        vae_temporal_stride=vae_temporal_stride,
                        patch_size=(1, 1, 1),
                    )
        else:
            camera_to_world = condition_inputs.get("camera_to_world")
            intrinsics = condition_inputs.get("intrinsics")
            if action is not None and camera_to_world is not None:
                raise ValueError(
                    "SANA-WM action and camera_to_world are mutually exclusive."
                )

            if action is not None:
                source = (
                    "action" if intrinsics is not None else "action_default_intrinsics"
                )
                translation_speed, rotation_speed_deg, pitch_limit_deg = (
                    validate_sana_wm_motion_params(
                        translation_speed=condition_inputs.get(
                            "translation_speed",
                            SANA_WM_DEFAULT_TRANSLATION_SPEED,
                        ),
                        rotation_speed_deg=condition_inputs.get(
                            "rotation_speed_deg",
                            SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
                        ),
                        pitch_limit_deg=condition_inputs.get(
                            "pitch_limit_deg",
                            SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
                        ),
                    )
                )
                camera_to_world = coerce_sana_wm_action_camera_to_world(
                    action,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    translation_speed=translation_speed,
                    rotation_speed_deg=rotation_speed_deg,
                    pitch_limit_deg=pitch_limit_deg,
                    device=device,
                    dtype=camera_compute_dtype,
                )
                self.log_info(
                    "SANA-WM action trajectory rolled out: frames=%d, "
                    "translation_speed=%.6g, rotation_speed_deg=%.6g, "
                    "pitch_limit_deg=%.6g",
                    camera_to_world.shape[1],
                    translation_speed,
                    rotation_speed_deg,
                    pitch_limit_deg,
                )
                if intrinsics is None:
                    _, intrinsics_vec4 = default_sana_wm_static_camera(
                        batch_size=batch_size,
                        num_frames=num_frames,
                        pixel_h=batch.height,
                        pixel_w=batch.width,
                        device=device,
                        dtype=camera_compute_dtype,
                    )
                    self.log_info(
                        "No intrinsics provided; using centered pinhole "
                        "intrinsics with default horizontal FOV %.2f deg for "
                        "the action trajectory. Pass request intrinsics for "
                        "camera-accurate geometry.",
                        sana_wm_default_horizontal_fov_deg(),
                    )
                else:
                    intrinsics_vec4 = coerce_sana_wm_intrinsics_vec4(
                        intrinsics,
                        batch_size=batch_size,
                        num_frames=num_frames,
                        device=device,
                        dtype=camera_compute_dtype,
                    )
                    intrinsics_vec4 = self._transform_intrinsics_for_condition_image(
                        intrinsics_vec4,
                        preprocess_info,
                    )
            elif camera_to_world is not None:
                source = (
                    "request"
                    if intrinsics is not None
                    else "request_default_intrinsics"
                )
                camera_to_world = coerce_sana_wm_camera_to_world(
                    camera_to_world,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    device=device,
                    dtype=camera_compute_dtype,
                )
                if intrinsics is None:
                    _, intrinsics_vec4 = default_sana_wm_static_camera(
                        batch_size=batch_size,
                        num_frames=num_frames,
                        pixel_h=batch.height,
                        pixel_w=batch.width,
                        device=device,
                        dtype=camera_compute_dtype,
                    )
                    self.log_info(
                        "No intrinsics provided; using centered pinhole "
                        "intrinsics with default horizontal FOV %.2f deg for "
                        "the request camera trajectory. Pass request "
                        "intrinsics for camera-accurate geometry.",
                        sana_wm_default_horizontal_fov_deg(),
                    )
                else:
                    intrinsics_vec4 = coerce_sana_wm_intrinsics_vec4(
                        intrinsics,
                        batch_size=batch_size,
                        num_frames=num_frames,
                        device=device,
                        dtype=camera_compute_dtype,
                    )
                    intrinsics_vec4 = self._transform_intrinsics_for_condition_image(
                        intrinsics_vec4,
                        preprocess_info,
                    )
            elif intrinsics is not None:
                source = "default_static_request_intrinsics"
                camera_to_world, _ = default_sana_wm_static_camera(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    pixel_h=batch.height,
                    pixel_w=batch.width,
                    device=device,
                    dtype=camera_compute_dtype,
                )
                intrinsics_vec4 = coerce_sana_wm_intrinsics_vec4(
                    intrinsics,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    device=device,
                    dtype=camera_compute_dtype,
                )
                intrinsics_vec4 = self._transform_intrinsics_for_condition_image(
                    intrinsics_vec4,
                    preprocess_info,
                )
                self.log_info(
                    "No camera trajectory provided; using static identity "
                    "poses with request intrinsics."
                )
            else:
                source = "default_static"
                self.log_info(
                    "No camera trajectory provided; using a static identity "
                    "camera with centered pinhole intrinsics at default "
                    "horizontal FOV %.2f deg. Pass camera_to_world/intrinsics "
                    "for camera-controlled output.",
                    sana_wm_default_horizontal_fov_deg(),
                )
                camera_to_world, intrinsics_vec4 = default_sana_wm_static_camera(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    pixel_h=batch.height,
                    pixel_w=batch.width,
                    device=device,
                    dtype=camera_compute_dtype,
                )

            camera_to_world = relative_sana_wm_camera_poses(camera_to_world)
            intrinsics_vec4 = scale_sana_wm_intrinsics_to_latent(
                intrinsics_vec4,
                pixel_h=batch.height,
                pixel_w=batch.width,
                latent_h=sp_h,
                latent_w=sp_w,
            )
            original_camera_conditions = flatten_sana_wm_camera_conditions(
                camera_to_world, intrinsics_vec4
            )
            camera_conditions = latent_frame_sana_wm_camera_conditions(
                original_camera_conditions,
                num_frames=num_frames,
                latent_frames=T_lat,
                vae_temporal_stride=vae_temporal_stride,
            )
            if chunk_plucker is None:
                from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
                    compute_chunk_plucker,
                )

                chunk_plucker = compute_chunk_plucker(
                    camera_conditions=original_camera_conditions,
                    HW=(T_lat, sp_h, sp_w),
                    vae_temporal_stride=vae_temporal_stride,
                    patch_size=(1, 1, 1),
                )

        if chunk_plucker is not None:
            chunk_plucker = (
                chunk_plucker
                if isinstance(chunk_plucker, torch.Tensor)
                else torch.as_tensor(chunk_plucker)
            ).to(device=device, dtype=dtype)
            if chunk_plucker.dim() == 4:
                chunk_plucker = chunk_plucker.unsqueeze(0)
            if chunk_plucker.shape[0] == 1 and batch_size > 1:
                chunk_plucker = chunk_plucker.expand(batch_size, -1, -1, -1, -1)
            if chunk_plucker.dim() != 5:
                raise ValueError(
                    "chunk_plucker must have shape (48,T,H,W) or "
                    f"(B,48,T,H,W), got {tuple(chunk_plucker.shape)}"
                )
            if chunk_plucker.shape[0] != batch_size:
                raise ValueError(
                    "chunk_plucker batch dimension must be 1 or match "
                    f"request batch size {batch_size}, got "
                    f"{chunk_plucker.shape[0]}."
                )
            expected_chunk_shape = (batch_size, 48, T_lat, sp_h, sp_w)
            if tuple(chunk_plucker.shape) != expected_chunk_shape:
                raise ValueError(
                    "chunk_plucker shape mismatch for SANA-WM: expected "
                    f"{expected_chunk_shape}, got {tuple(chunk_plucker.shape)}."
                )

        if camera_conditions is not None:
            camera_conditions = camera_conditions.to(device=device, dtype=dtype)

        return camera_conditions, chunk_plucker, source

    @staticmethod
    def _duplicate_condition_for_cfg(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.cat([value, value], dim=0)

    def _prepare_transformer_static_conditioning(
        self,
        *,
        camera_conditions: torch.Tensor | None,
        chunk_plucker: torch.Tensor | None,
        latent_shape: tuple,
        do_cfg: bool,
        cfg_parallel: bool,
    ) -> dict[str, Any]:
        if self.transformer is None:
            return {}
        if camera_conditions is None and chunk_plucker is None:
            return {}

        model_camera_conditions = camera_conditions
        model_chunk_plucker = chunk_plucker
        if do_cfg and not cfg_parallel:
            model_camera_conditions = self._duplicate_condition_for_cfg(
                camera_conditions
            )
            model_chunk_plucker = self._duplicate_condition_for_cfg(chunk_plucker)

        with self.use_declared_component(
            component_name="transformer",
            module=self.transformer,
            phase="transformer",
        ) as active_transformer:
            transformer = (
                active_transformer
                if active_transformer is not None
                else self.transformer
            )
            prepare_static = getattr(
                transformer, "prepare_sana_wm_static_conditioning", None
            )
            if prepare_static is None:
                return {}
            static_kwargs = prepare_static(
                camera_conditions=model_camera_conditions,
                chunk_plucker=model_chunk_plucker,
                latent_shape=latent_shape,
            )

        if static_kwargs:
            plucker_bytes = 0
            plucker_emb = static_kwargs.get(_SANA_WM_PRECOMPUTED_PLUCKER_EMB_KEY)
            if isinstance(plucker_emb, torch.Tensor):
                plucker_bytes = plucker_emb.numel() * plucker_emb.element_size()
            self.log_info(
                "SANA-WM transformer static conditioning prepared: %s "
                "(plucker_emb_mb=%.2f)",
                sorted(static_kwargs.keys()),
                plucker_bytes / (1024 * 1024),
            )
        return static_kwargs

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ):
        """Set up scheduler timesteps and populate batch.timesteps, .sigmas."""
        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        num_inference_steps = batch.num_inference_steps

        flow_shift = getattr(batch, "flow_shift", None)
        if flow_shift is None:
            flow_shift = getattr(
                self.pipeline_config,
                "inference_flow_shift",
                None,
            )
        if flow_shift is None:
            flow_shift = getattr(self.pipeline_config, "flow_shift", 9.95)
        flow_shift = float(flow_shift)
        kwargs = {}

        # diffusers FlowMatchEulerDiscreteScheduler supports mu/shift
        import inspect

        sig_params = inspect.signature(scheduler.set_timesteps).parameters
        if "shift" in sig_params:
            kwargs["shift"] = flow_shift
        elif "mu" in sig_params:
            # Convert flow_shift to mu: mu ≈ log(shift)
            import math

            kwargs["mu"] = math.log(flow_shift)

        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas.tolist()
        if sigmas:
            self.log_info(
                "FlowMatch timesteps prepared: steps=%d, flow_shift=%.4f, "
                "sigma_start=%.6f, sigma_end=%.6f",
                num_inference_steps,
                flow_shift,
                float(sigmas[0]),
                float(sigmas[-1]),
            )

        batch.timesteps = timesteps
        batch.sigmas = sigmas
        batch.scheduler = scheduler
        return batch

    def _adjust_num_frames_for_request(self, batch: Req) -> int:
        requested_num_frames = batch.num_frames or 49
        action_num_frames = self._action_num_frames_for_request(batch)
        if action_num_frames is not None and action_num_frames != requested_num_frames:
            self.log_info(
                "SANA-WM action trajectory has %d frames; keeping requested "
                "num_frames=%d and padding/trimming the action trajectory.",
                action_num_frames,
                requested_num_frames,
            )
        return self.pipeline_config.adjust_num_frames(requested_num_frames)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
        Pre-process everything needed by DenoisingStage for SANA-WM.

        Expects batch to already have prompt_embeds set by SanaWMTextEncodingStage.
        """
        device = get_local_torch_device()
        dtype = PRECISION_TO_TYPE.get(
            getattr(self.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}
        _clear_sana_wm_precomputed_static_conditioning(batch)

        # --- 0. Adjust num_frames to be compatible with VAE temporal stride ---
        num_frames = self._adjust_num_frames_for_request(batch)
        batch.num_frames = num_frames
        self.log_info(
            "SANA-WM prepare: seed=%s, size=%dx%d, frames=%d, "
            "vae_stride=%s, diagnostics=%s",
            getattr(batch, "seed", None),
            batch.width,
            batch.height,
            num_frames,
            self.pipeline_config.vae_stride,
            "on" if sana_wm_diagnostics_enabled() else "off",
        )

        # --- 1. Generator for reproducibility ---
        batch_size = batch.batch_size or 1
        generator = getattr(batch, "generator", None)
        if not isinstance(generator, (list, tuple, torch.Generator)):
            generator = self._generator_from_seed(
                getattr(batch, "seed", None),
                batch_size=batch_size,
                device=device,
            )
            batch.generator = generator

        # --- 2. Compute latent shape and initialize noise ---
        latent_shape = self.pipeline_config.prepare_latent_shape(
            batch, batch_size, num_frames
        )
        # latent_shape: (B, 128, T_latent, H_sp, W_sp)
        latents = self._prepare_noise_latents(latent_shape, dtype, device, generator)
        log_sana_wm_tensor_stats("latents.initial_noise", latents)

        # Store raw shape for DecodingStage
        batch.raw_latent_shape = latent_shape

        # --- 3. VAE-encode first frame and splice into noise latents ---
        condition_image = getattr(batch, "condition_image", None)
        if condition_image is not None:
            try:
                latents = self._splice_first_frame(
                    latents, condition_image, dtype, device, batch=batch
                )
                self.log_info("First-frame spliced into noise latents.")
            except Exception as e:
                raise RuntimeError(
                    "SANA-WM first-frame conditioning failed; refusing to "
                    "continue with pure-noise latents because that produces "
                    "misleading low-quality output."
                ) from e
        else:
            raise ValueError(
                "SANA-WM is a TI2V world model and requires condition_image "
                "for first-frame conditioning. Provide --image-path, "
                "--condition-image, or the equivalent API image input."
            )

        batch.latents = latents

        # --- 4. Camera conditioning ---
        # The released SANA-WM checkpoint is camera-conditioned. Official
        # inference requires a camera trajectory or action DSL. If the SGLang
        # request omits one, use a static identity trajectory so the UCPE path
        # remains active instead of silently dropping all camera conditioning.
        try:
            camera_conditions, chunk_plucker, camera_source = (
                self._build_camera_conditioning(
                    batch,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    latent_shape=latent_shape,
                    device=device,
                    dtype=dtype,
                )
            )
        except Exception as e:
            raise RuntimeError("SANA-WM camera conditioning failed.") from e

        if camera_conditions is not None:
            batch.extra["camera_conditions"] = camera_conditions
            log_sana_wm_tensor_stats("camera_conditions", camera_conditions)
        if chunk_plucker is not None:
            batch.extra["chunk_plucker"] = chunk_plucker
            log_sana_wm_tensor_stats("chunk_plucker", chunk_plucker)
        self.log_info(
            "SANA-WM camera conditioning: source=%s, raymap=%s, chunk_plucker=%s",
            camera_source,
            None if camera_conditions is None else tuple(camera_conditions.shape),
            None if chunk_plucker is None else tuple(chunk_plucker.shape),
        )

        # --- 5. Ensure prompt_embeds is a list (DenoisingStage expects list[Tensor]) ---
        if isinstance(batch.prompt_embeds, torch.Tensor):
            batch.prompt_embeds = [batch.prompt_embeds]
        if batch.negative_prompt_embeds is not None and isinstance(
            batch.negative_prompt_embeds, torch.Tensor
        ):
            batch.negative_prompt_embeds = [batch.negative_prompt_embeds]

        # --- 6. CFG setup ---
        batch.do_classifier_free_guidance = _sana_wm_should_do_cfg(batch)
        cfg_parallel = bool(
            getattr(server_args, "enable_cfg_parallel", False)
            and batch.do_classifier_free_guidance
        )

        # --- 7. Precompute request-static transformer camera conditioning ---
        batch.extra.update(
            self._prepare_transformer_static_conditioning(
                camera_conditions=camera_conditions,
                chunk_plucker=chunk_plucker,
                latent_shape=latent_shape,
                do_cfg=batch.do_classifier_free_guidance,
                cfg_parallel=cfg_parallel,
            )
        )

        # --- 8. Prepare timesteps and sigmas ---
        batch = self._prepare_timesteps(batch, server_args, device)

        self.log_info(
            "BeforeDenoisingStage done: latent=%s, T_lat=%d, H_sp=%d, W_sp=%d, "
            "num_inference_steps=%d, camera=%s",
            str(latent_shape),
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
            batch.num_inference_steps,
            "yes" if batch.extra.get("camera_conditions") is not None else "no",
        )
        return batch

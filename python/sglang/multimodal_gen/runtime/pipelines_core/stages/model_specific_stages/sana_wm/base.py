# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
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
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

from .utils import (
    action_string_to_c2w,
    compute_resize_crop_geometry,
    parse_action_string,
)

logger = init_logger(__name__)

_SANA_WM_DIAGNOSTICS_ENVS = (
    "SGLANG_SANA_WM_DIAGNOSTICS",
    "SGLANG_SANA_WM_LOG_TENSOR_STATS",
)

_SANA_WM_DEFAULT_VAE_TILE_MIN_FRAMES = 96
_SANA_WM_DEFAULT_VAE_TILE_STRIDE_FRAMES = 64
_SANA_WM_DEFAULT_TRANSLATION_SPEED = (
    0.04  # match official streaming (STREAMING_TRANSLATION_SPEED)
)
_SANA_WM_DEFAULT_ROTATION_SPEED_DEG = 1.2
_SANA_WM_DEFAULT_PITCH_LIMIT_DEG = 85.0
_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY = "sana_wm_condition_image_preprocess"

# Single action-DSL parser for both paths (utils.parse_action_string is the
# canonical, official-matching implementation); the old in-house duplicate body
# is gone, the public name stays for existing callers/tests.
parse_sana_wm_action_string = parse_action_string


def sana_wm_vae_scaling_factor(vae, pipeline_config) -> float:
    """Resolve the VAE scaling factor (batch-reference precedence).

    vae.config.scaling_factor -> vae.scaling_factor ->
    pipeline_config.vae_config.arch_config.scaling_factor -> 1.0; zero is
    treated as unset. Single source for both the batch and realtime
    first-frame encode paths.
    """
    scaling_factor = (
        getattr(getattr(vae, "config", None), "scaling_factor", None)
        or getattr(vae, "scaling_factor", None)
        or getattr(
            getattr(getattr(pipeline_config, "vae_config", None), "arch_config", None),
            "scaling_factor",
            None,
        )
        or 1.0
    )
    if isinstance(scaling_factor, torch.Tensor):
        return float(scaling_factor.item())
    scaling_factor = float(scaling_factor)
    return 1.0 if scaling_factor == 0.0 else scaling_factor


def sana_wm_normalize_vae_latents(
    vae, z: torch.Tensor, pipeline_config
) -> torch.Tensor:
    """Normalize freshly-encoded VAE latents (batch-reference semantics).

    ``(z - latents_mean) * scaling_factor / latents_std`` when the VAE carries
    mean/std buffers (the LTX-2 causal VAE does); legacy shift-then-scale
    otherwise. ``z`` is expected in float32 (drifting this was parity root
    cause #2).
    """
    latents_mean = getattr(vae, "latents_mean", None)
    latents_std = getattr(vae, "latents_std", None)
    scaling_factor = sana_wm_vae_scaling_factor(vae, pipeline_config)
    if sana_wm_diagnostics_enabled():
        logger.info(
            "[SANA-WM diagnostics] VAE encode normalization: "
            "has_latents_mean_std=%s scaling_factor=%.6g",
            isinstance(latents_mean, torch.Tensor)
            and isinstance(latents_std, torch.Tensor),
            scaling_factor,
        )
    if isinstance(latents_mean, torch.Tensor) and isinstance(latents_std, torch.Tensor):
        latents_mean = latents_mean.to(device=z.device, dtype=z.dtype).view(
            1, -1, 1, 1, 1
        )
        latents_std = latents_std.to(device=z.device, dtype=z.dtype).view(
            1, -1, 1, 1, 1
        )
        return (z - latents_mean) * scaling_factor / latents_std

    # Legacy VAE convention: encode applies shift before scaling.
    shift_factor = getattr(vae, "shift_factor", None)
    if shift_factor is not None:
        z = z - (
            shift_factor.to(z.device, z.dtype)
            if isinstance(shift_factor, torch.Tensor)
            else shift_factor
        )
    return z * scaling_factor


def sana_wm_action_to_camera_to_world(
    action: str,
    *,
    translation_speed: float = _SANA_WM_DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = _SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = _SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
) -> torch.Tensor:
    """Roll out upstream SANA-WM action DSL to a camera-to-world trajectory.

    Coordinate convention is OpenCV: +X right, +Y down, +Z forward. Returned
    shape is ``(N+1, 4, 4)``, float32.

    Delegates to ``utils.action_string_to_c2w`` (byte-identical kinematics to
    the NVlabs reference, including the strafe->yaw coupling
    ``yaw += 0.4 * (d - a)`` that this module's previous in-house rollout
    dropped) so the batch and realtime paths share ONE kinematics impl and
    cannot drift.
    """
    return torch.from_numpy(
        action_string_to_c2w(
            action,
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
            pitch_limit_deg=pitch_limit_deg,
        )
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
    """Log compact tensor statistics for reference-quality alignment.

    Enable with ``SGLANG_SANA_WM_DIAGNOSTICS=1``. The fingerprint is a strided
    sum over at most ~4096 values, for spotting deterministic drift.
    """
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

        stats = data.float()
        finite = torch.isfinite(stats)
        finite_ratio = float(finite.float().mean().item())
        finite_stats = stats[finite] if bool(finite.any().item()) else stats.reshape(-1)
        flat = finite_stats.reshape(-1)
        stride = max(1, flat.numel() // 4096)
        fingerprint = float(flat[::stride].sum().item())
        std = float(finite_stats.std(unbiased=False).item())
        logger.info(
            "[SANA-WM diagnostics] %s: shape=%s dtype=%s device=%s "
            "finite=%.6f min=%.6g max=%.6g mean=%.6g std=%.6g "
            "l2=%.6g fingerprint=%.6g",
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
        )


def configure_sana_wm_ltx2_vae_for_long_video(
    vae: Any,
    pipeline_config: SanaWMPipelineConfig,
    *,
    log_info: Any | None = None,
) -> None:
    """Apply SANA-WM's upstream LTX-2 VAE tiling knobs.

    The LTX-2 VAE implements temporal tiled decode but, unlike the generic VAE
    base, does not enable it from ``enable_tiling()`` alone. Without this,
    321-frame 720p decode can enter one large Conv3d path and hit PyTorch's
    32-bit index math limit.
    """

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
    """Decode SANA-WM LTX-2 latents with upstream long-video VAE settings."""

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
    """Gemma-2 text encoding that mirrors NVlabs SANA-WM inference.

    The official script prepends a long ``chi_prompt`` only to the positive
    branch, tokenizes that longer string, then keeps token 0 and the last
    299 tokens. The negative branch remains a normal 300-token padded prompt.
    Keeping this local avoids bending the shared TextEncodingStage around a
    model-specific prompt-window contract.
    """

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
        ) = self.encode_text(
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

        if batch.do_classifier_free_guidance:
            assert isinstance(batch.negative_prompt, str)
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                _neg_seq_lens_list,
            ) = self.encode_text(
                batch.negative_prompt,
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
            log_sana_wm_tensor_stats("text.negative_prompt_embeds", neg_embeds_list[0])

        return batch


class SanaWMDenoisingStage(DenoisingStage):
    """SANA-WM stage-1 sampler matching NVlabs ``flow_euler_ltx``.

    The generic denoising stage uses one scalar timestep for every latent token
    and updates the whole tensor. Official SANA-WM inference uses per-frame
    timesteps: the first-frame condition stays at timestep 0 and is not updated,
    while the remaining latent frames denoise normally.
    """

    @property
    def parallelism_type(self) -> StageParallelismType:
        if self.server_args.enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    @staticmethod
    def _combine_cfg_parallel_noise(
        noise_pred: torch.Tensor,
        guidance_scale: float,
        cfg_rank: int,
    ) -> torch.Tensor:
        if cfg_rank == 0:
            partial = guidance_scale * noise_pred
        elif cfg_rank == 1:
            partial = (1.0 - guidance_scale) * noise_pred
        else:
            partial = torch.zeros_like(noise_pred)
        return cfg_model_parallel_all_reduce(partial)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
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

        latents = batch.latents.to(device=device, dtype=target_dtype)
        init_latents = latents.clone()
        condition_mask = torch.zeros_like(latents)
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

        do_cfg = bool(batch.do_classifier_free_guidance)
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
        diffusers_kwargs = extra.get("diffusers_kwargs", {})
        if not isinstance(diffusers_kwargs, dict):
            diffusers_kwargs = {}
        chunk_kwargs = {}
        for key in ("chunk_index", "chunk_size", "chunk_split_strategy"):
            value = extra.get(key, diffusers_kwargs.get(key))
            if value is not None:
                chunk_kwargs[key] = value
        camera_conditions = _to_device_dtype(
            extra.get("camera_conditions"), device=device, dtype=target_dtype
        )
        chunk_plucker = _to_device_dtype(
            extra.get("chunk_plucker"), device=device, dtype=target_dtype
        )

        cfg_parallel = bool(server_args.enable_cfg_parallel and do_cfg)
        cfg_rank = get_classifier_free_guidance_rank() if cfg_parallel else 0
        if cfg_parallel and get_classifier_free_guidance_world_size() > 2:
            logger.warning_once(
                "SANA-WM CFG parallel uses two guidance branches; extra CFG ranks "
                "run dummy forwards and contribute zeros."
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
                "camera_conditions": camera_conditions,
                "chunk_plucker": chunk_plucker,
            }
        else:
            model_kwargs = {
                "encoder_hidden_states": (
                    torch.cat([neg_embeds, pos_embeds], dim=0) if do_cfg else pos_embeds
                ),
                "encoder_attention_mask": (
                    _cat_optional_tensors(neg_mask, pos_mask) if do_cfg else pos_mask
                ),
                "camera_conditions": (
                    torch.cat([camera_conditions, camera_conditions], dim=0)
                    if do_cfg and camera_conditions is not None
                    else camera_conditions
                ),
                "chunk_plucker": (
                    torch.cat([chunk_plucker, chunk_plucker], dim=0)
                    if do_cfg and chunk_plucker is not None
                    else chunk_plucker
                ),
            }
        model_kwargs.update(chunk_kwargs)

        condition_mask_input = (
            condition_mask
            if cfg_parallel or not do_cfg
            else torch.cat([condition_mask, condition_mask], dim=0)
        )
        timestep_condition_limit = (1.0 - condition_mask_input.float()) * 1000.0

        self.log_info(
            "SANA-WM flow_euler_ltx denoising: latent=%s, steps=%d, cfg=%s, "
            "cfg_parallel=%s, guidance_scale=%.4f, first_frame_locked=yes",
            tuple(latents.shape),
            len(timesteps),
            do_cfg,
            cfg_parallel,
            float(getattr(batch, "guidance_scale", 1.0) or 1.0),
        )
        log_sana_wm_tensor_stats("denoise.input_latents", latents)

        start_time = time.perf_counter()
        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer

            for step_idx, t in enumerate(self.progress_bar(timesteps)):
                if cfg_parallel:
                    latent_model_input = latents
                else:
                    latent_model_input = (
                        torch.cat([latents, latents], dim=0) if do_cfg else latents
                    )

                timestep = t.expand(condition_mask_input.shape).float()
                timestep = torch.minimum(timestep, timestep_condition_limit)
                model_timestep = timestep[:, :1, :, 0, 0]

                with set_forward_context(
                    current_timestep=step_idx,
                    attn_metadata=None,
                    forward_batch=batch,
                ):
                    noise_pred = transformer(
                        hidden_states=latent_model_input.to(target_dtype),
                        timestep=model_timestep,
                        **model_kwargs,
                    )

                if do_cfg:
                    guidance_scale = float(getattr(batch, "guidance_scale", 1.0) or 1.0)
                    if cfg_parallel:
                        noise_pred = self._combine_cfg_parallel_noise(
                            noise_pred, guidance_scale, cfg_rank
                        )
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                        timestep = timestep.chunk(2)[0]

                latents_dtype = latents.dtype
                latents_shape = latents.shape
                batch_size, channels, _, _, _ = latents_shape
                scheduler_output = scheduler.step(
                    -noise_pred.reshape(batch_size, channels, -1).transpose(1, 2),
                    t,
                    latents.reshape(batch_size, channels, -1).transpose(1, 2),
                    per_token_timesteps=timestep.reshape(batch_size, channels, -1)[
                        :, 0
                    ],
                    return_dict=False,
                )[0]
                denoised_latents = scheduler_output.transpose(1, 2).reshape(
                    latents_shape
                )

                tokens_to_denoise = t.float() / 1000.0 - 1e-6 < (1.0 - condition_mask)
                latents = torch.where(tokens_to_denoise, denoised_latents, latents)
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if sana_wm_diagnostics_enabled() and (
                    step_idx == 0 or step_idx == len(timesteps) - 1
                ):
                    log_sana_wm_tensor_stats(
                        f"denoise.step_{step_idx}.noise_pred", noise_pred
                    )
                    log_sana_wm_tensor_stats(
                        f"denoise.step_{step_idx}.latents", latents
                    )

        log_sana_wm_tensor_stats("denoise.output_latents", latents)
        unchanged = (latents[:, :, :1] - init_latents[:, :, :1]).abs().max().item()
        self.log_info(
            "SANA-WM flow_euler_ltx denoising finished in %.4f seconds; "
            "first_frame_max_delta=%.6g",
            time.perf_counter() - start_time,
            float(unchanged),
        )
        batch.latents = server_args.pipeline_config.post_denoising_loop(latents, batch)
        return batch


class SanaWMBeforeDenoisingStage(PipelineStage):
    """
    Monolithic pre-processing stage for SANA-WM TI2V inference.

    Must run after SanaWMTextEncodingStage, which populates batch.prompt_embeds.
    """

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
        if self.vae is None:
            return []
        stage_name = self._component_stage_name(stage_name)
        pipeline_config = getattr(server_args, "pipeline_config", self.pipeline_config)
        vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="vae",
                target_dtype=vae_dtype,
            )
        ]

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

        # Normalize image to [-1, 1] range expected by the VAE
        if image.max() > 1.01:
            image = image / 255.0
        image = (image * 2.0 - 1.0).to(device=device, dtype=vae_dtype)

        # Add temporal dim if absent: (B, C, H, W) -> (B, C, 1, H, W)
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

        z = sana_wm_normalize_vae_latents(vae, z, self.pipeline_config)

        log_sana_wm_tensor_stats("first_frame.latent_normalized", z)
        return z.to(dtype=dtype)  # (1, 128, 1, H_sp, W_sp)

    @staticmethod
    def _extract_vae_latents(encoded: Any) -> torch.Tensor:
        """Return deterministic VAE latents from common Diffusers outputs."""
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
            return SanaWMBeforeDenoisingStage._extract_vae_latents(encoded[0])
        if isinstance(encoded, torch.Tensor):
            return encoded
        raise TypeError(
            "Unsupported VAE encode output for SANA-WM first-frame conditioning: "
            f"{type(encoded).__name__}"
        )

    def _get_vae_scaling_factor(self, vae) -> float:
        return sana_wm_vae_scaling_factor(vae, self.pipeline_config)

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
        """Return image as NCHW RGB float tensor without changing its value range."""
        image = image.float()
        if image.dim() == 5 and image.shape[2] == 1:
            image = image.squeeze(2)
        if image.dim() == 3:
            if image.shape[0] in (1, 3, 4):
                image = image.unsqueeze(0)
            elif image.shape[-1] in (1, 3, 4):
                image = image.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    "condition_image tensor must be CHW or HWC with 1, 3, "
                    f"or 4 channels; got {tuple(image.shape)}."
                )
        elif image.dim() == 4:
            if image.shape[1] in (1, 3, 4):
                pass
            elif image.shape[-1] in (1, 3, 4):
                image = image.permute(0, 3, 1, 2)
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
        resized_w, resized_h, left, top = compute_resize_crop_geometry(
            src_w, src_h, target_h, target_w
        )
        if resized_h != src_h or resized_w != src_w:
            import torch.nn.functional as F

            image = F.interpolate(
                image,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
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
            resized_w, resized_h, left, top = compute_resize_crop_geometry(
                src_w, src_h, target_h, target_w
            )
            resampling_enum = getattr(PIL.Image, "Resampling", None)
            resampling = (
                resampling_enum.LANCZOS
                if resampling_enum is not None
                else PIL.Image.LANCZOS
            )
            image = image.resize((resized_w, resized_h), resampling)
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

        if batch is not None:
            if not hasattr(batch, "extra") or batch.extra is None:
                batch.extra = {}
            batch.extra[_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY] = (
                preprocess_infos[0] if len(preprocess_infos) == 1 else preprocess_infos
            )
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
    def _pad_or_trim_frames(tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
        current = tensor.shape[1]
        if current == num_frames:
            return tensor
        if current > num_frames:
            return tensor[:, :num_frames]
        if current == 0:
            raise ValueError("camera trajectory must contain at least one frame")
        pad = num_frames - current
        last = tensor[:, -1:].repeat(1, pad, *([1] * (tensor.ndim - 2)))
        return torch.cat([tensor, last], dim=1)

    @staticmethod
    def _maybe_load_npy_tensor(value: Any, field_name: str) -> Any:
        if isinstance(value, (str, os.PathLike)):
            import numpy as np

            path = os.fspath(value)
            if not path.endswith(".npy"):
                raise ValueError(
                    f"{field_name} path must point to a .npy file, got {path!r}"
                )
            return torch.from_numpy(np.load(path))
        return value

    @staticmethod
    def _first_mapping_value(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    @staticmethod
    def _first_request_mapping_value(
        extra: dict[str, Any],
        diffusers_kwargs: dict[str, Any],
        *keys: str,
    ) -> Any:
        for mapping in (extra, diffusers_kwargs):
            for key in keys:
                if key in mapping and mapping[key] is not None:
                    return mapping[key]
        return None

    @staticmethod
    def _request_float_value(
        extra: dict[str, Any],
        diffusers_kwargs: dict[str, Any],
        *keys: str,
        default: float,
    ) -> float:
        value = SanaWMBeforeDenoisingStage._first_request_mapping_value(
            extra, diffusers_kwargs, *keys
        )
        return default if value is None else float(value)

    @staticmethod
    def _request_action_value(
        extra: dict[str, Any],
        diffusers_kwargs: dict[str, Any],
    ) -> Any:
        return SanaWMBeforeDenoisingStage._first_request_mapping_value(
            extra, diffusers_kwargs, "action", "sana_wm_action"
        )

    @staticmethod
    def _pad_or_trim_action_trajectory(
        trajectory: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        current = trajectory.shape[0]
        if current == num_frames:
            return trajectory
        if current > num_frames:
            return trajectory[:num_frames]
        pad = num_frames - current
        return torch.cat([trajectory, trajectory[-1:].repeat(pad, 1, 1)], dim=0)

    @staticmethod
    def _coerce_action_camera_to_world(
        value: Any,
        *,
        batch_size: int,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
        pitch_limit_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, str):
            actions = [value]
        elif isinstance(value, (list, tuple)) and all(
            isinstance(x, str) for x in value
        ):
            actions = list(value)
        else:
            raise ValueError(
                "SANA-WM action must be a string or a list of strings, "
                f"got {type(value).__name__}."
            )

        trajectories = [
            SanaWMBeforeDenoisingStage._pad_or_trim_action_trajectory(
                sana_wm_action_to_camera_to_world(
                    action,
                    translation_speed=translation_speed,
                    rotation_speed_deg=rotation_speed_deg,
                    pitch_limit_deg=pitch_limit_deg,
                ),
                num_frames,
            )
            for action in actions
        ]
        camera = torch.stack(trajectories, dim=0).to(device=device, dtype=dtype)
        if camera.shape[0] == 1 and batch_size > 1:
            camera = camera.expand(batch_size, -1, -1, -1)
        elif camera.shape[0] != batch_size:
            raise ValueError(
                f"SANA-WM action batch {camera.shape[0]} does not match {batch_size}"
            )
        return camera

    @staticmethod
    def _action_num_frames_for_request(batch: Req) -> int | None:
        extra = getattr(batch, "extra", None) or {}
        diffusers_kwargs = extra.get("diffusers_kwargs", {})
        if not isinstance(diffusers_kwargs, dict):
            diffusers_kwargs = {}
        action = SanaWMBeforeDenoisingStage._request_action_value(
            extra, diffusers_kwargs
        )
        if action is None:
            return None
        if isinstance(action, str):
            return len(parse_sana_wm_action_string(action)) + 1
        if isinstance(action, (list, tuple)) and all(
            isinstance(x, str) for x in action
        ):
            return max(len(parse_sana_wm_action_string(x)) + 1 for x in action)
        raise ValueError(
            "SANA-WM action must be a string or a list of strings, "
            f"got {type(action).__name__}."
        )

    @staticmethod
    def _coerce_camera_to_world(
        value: Any,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = SanaWMBeforeDenoisingStage._maybe_load_npy_tensor(
            value, "camera_to_world"
        )
        camera = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if camera.dim() == 3:
            camera = camera.unsqueeze(0)
        if camera.dim() != 4 or camera.shape[-2:] != (4, 4):
            raise ValueError(
                "camera_to_world must have shape (F,4,4) or (B,F,4,4), "
                f"got {tuple(camera.shape)}"
            )
        camera = camera.to(device=device, dtype=dtype)
        if camera.shape[0] == 1 and batch_size > 1:
            camera = camera.expand(batch_size, -1, -1, -1)
        elif camera.shape[0] != batch_size:
            raise ValueError(
                f"camera_to_world batch {camera.shape[0]} does not match {batch_size}"
            )
        return SanaWMBeforeDenoisingStage._pad_or_trim_frames(camera, num_frames)

    @staticmethod
    def _intrinsics_matrix_to_vec4(intrinsics: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                intrinsics[..., 0, 0],
                intrinsics[..., 1, 1],
                intrinsics[..., 0, 2],
                intrinsics[..., 1, 2],
            ],
            dim=-1,
        )

    @staticmethod
    def _coerce_intrinsics_vec4(
        value: Any,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = SanaWMBeforeDenoisingStage._maybe_load_npy_tensor(value, "intrinsics")
        intrinsics = (
            value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        )
        intrinsics = intrinsics.to(device=device, dtype=dtype)

        if intrinsics.dim() == 1 and intrinsics.shape[0] == 4:
            intrinsics = intrinsics.view(1, 1, 4)
        elif intrinsics.dim() == 2 and intrinsics.shape == (3, 3):
            intrinsics = SanaWMBeforeDenoisingStage._intrinsics_matrix_to_vec4(
                intrinsics
            ).view(1, 1, 4)
        elif intrinsics.dim() == 2 and intrinsics.shape[-1] == 4:
            intrinsics = intrinsics.unsqueeze(0)
        elif intrinsics.dim() == 3 and intrinsics.shape[-2:] == (3, 3):
            vec4 = SanaWMBeforeDenoisingStage._intrinsics_matrix_to_vec4(intrinsics)
            if intrinsics.shape[0] >= num_frames:
                intrinsics = vec4.unsqueeze(0)
            elif intrinsics.shape[0] == batch_size:
                intrinsics = vec4.unsqueeze(1)
            else:
                raise ValueError(
                    "intrinsics with shape (N,3,3) must use N>=num_frames "
                    f"or N=batch_size, got N={intrinsics.shape[0]}, "
                    f"num_frames={num_frames}, batch_size={batch_size}"
                )
        elif intrinsics.dim() == 3 and intrinsics.shape[-1] == 4:
            pass
        elif intrinsics.dim() == 4 and intrinsics.shape[-2:] == (3, 3):
            intrinsics = SanaWMBeforeDenoisingStage._intrinsics_matrix_to_vec4(
                intrinsics
            )
        else:
            raise ValueError(
                "intrinsics must have shape (4,), (F,4), (B,F,4), "
                "(3,3), (F,3,3), or (B,F,3,3); "
                f"got {tuple(intrinsics.shape)}"
            )

        if intrinsics.shape[0] == 1 and batch_size > 1:
            intrinsics = intrinsics.expand(batch_size, -1, -1)
        elif intrinsics.shape[0] != batch_size:
            raise ValueError(
                f"intrinsics batch {intrinsics.shape[0]} does not match {batch_size}"
            )
        if intrinsics.shape[1] == 1 and num_frames > 1:
            intrinsics = intrinsics.expand(-1, num_frames, -1)
        return SanaWMBeforeDenoisingStage._pad_or_trim_frames(intrinsics, num_frames)

    @staticmethod
    def _relative_camera_poses(camera_to_world: torch.Tensor) -> torch.Tensor:
        input_dtype = camera_to_world.dtype
        camera_to_world = camera_to_world.float()
        first_inv = torch.linalg.inv(camera_to_world[:, :1])
        poses = torch.matmul(first_inv, camera_to_world)
        eye = torch.eye(
            4,
            device=camera_to_world.device,
            dtype=camera_to_world.dtype,
        )
        poses[:, 0] = eye
        return poses.to(dtype=input_dtype)

    @staticmethod
    def _scale_intrinsics_to_latent(
        intrinsics_vec4: torch.Tensor,
        *,
        pixel_h: int,
        pixel_w: int,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        intrinsics_latent = intrinsics_vec4.clone()
        intrinsics_latent[..., [0, 2]] *= latent_w / float(pixel_w)
        intrinsics_latent[..., [1, 3]] *= latent_h / float(pixel_h)
        return intrinsics_latent

    @staticmethod
    def _flatten_camera_conditions(
        camera_to_world: torch.Tensor,
        intrinsics_vec4: torch.Tensor,
    ) -> torch.Tensor:
        c2w_flat = camera_to_world.reshape(
            camera_to_world.shape[0],
            camera_to_world.shape[1],
            16,
        )
        return torch.cat(
            [c2w_flat, intrinsics_vec4],
            dim=-1,
        )

    @staticmethod
    def _latent_frame_camera_conditions(
        camera_conditions: torch.Tensor,
        *,
        num_frames: int,
        latent_frames: int,
        vae_temporal_stride: int,
    ) -> torch.Tensor:
        time_indices = torch.arange(
            0,
            num_frames,
            vae_temporal_stride,
            device=camera_conditions.device,
            dtype=torch.long,
        )
        if time_indices.numel() < latent_frames:
            pad = latent_frames - int(time_indices.numel())
            time_indices = torch.cat(
                [time_indices, time_indices[-1:].repeat(pad)], dim=0
            )
        time_indices = time_indices[:latent_frames]
        return camera_conditions.index_select(1, time_indices)

    def _default_static_camera(
        self,
        *,
        batch_size: int,
        num_frames: int,
        pixel_h: int,
        pixel_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_to_world = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4)
        camera_to_world = camera_to_world.repeat(batch_size, num_frames, 1, 1)
        focal = 0.8 * float(max(pixel_h, pixel_w))
        intrinsics = torch.tensor(
            [focal, focal, pixel_w / 2.0, pixel_h / 2.0],
            device=device,
            dtype=dtype,
        ).view(1, 1, 4)
        intrinsics = intrinsics.repeat(batch_size, num_frames, 1)
        return camera_to_world, intrinsics

    @staticmethod
    def _has_explicit_camera_request(batch: Req) -> bool:
        extra = getattr(batch, "extra", None) or {}
        if any(
            key in extra and extra[key] is not None
            for key in (
                "camera_conditions",
                "chunk_plucker",
                "camera_to_world",
                "intrinsics",
                "action",
                "sana_wm_action",
            )
        ):
            return True
        diffusers_kwargs = extra.get("diffusers_kwargs", {})
        if not isinstance(diffusers_kwargs, dict):
            return False
        return any(
            key in diffusers_kwargs and diffusers_kwargs[key] is not None
            for key in (
                "camera_conditions",
                "chunk_plucker",
                "camera_to_world",
                "camera_to_world_path",
                "camera_path",
                "intrinsics",
                "intrinsics_path",
                "action",
                "sana_wm_action",
            )
        )

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
        T_lat = latent_shape[2]
        sp_h = latent_shape[3]
        sp_w = latent_shape[4]
        vae_temporal_stride = self.pipeline_config.vae_stride[0]
        camera_compute_dtype = torch.float32

        diffusers_kwargs = extra.get("diffusers_kwargs", {})
        if not isinstance(diffusers_kwargs, dict):
            diffusers_kwargs = {}
        camera_conditions = self._first_request_mapping_value(
            extra, diffusers_kwargs, "camera_conditions"
        )
        chunk_plucker = self._first_request_mapping_value(
            extra, diffusers_kwargs, "chunk_plucker"
        )
        preprocess_info = extra.get(_SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY)
        action = self._request_action_value(extra, diffusers_kwargs)
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
                original_camera_conditions = self._pad_or_trim_frames(
                    camera_conditions, num_frames
                )
                camera_conditions = self._latent_frame_camera_conditions(
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
            camera_to_world = extra.get("camera_to_world", None)
            intrinsics = extra.get("intrinsics", None)
            if camera_to_world is None:
                camera_to_world = self._first_mapping_value(
                    diffusers_kwargs,
                    "camera_to_world",
                    "camera_to_world_path",
                    "camera_path",
                )
            if intrinsics is None:
                intrinsics = self._first_mapping_value(
                    diffusers_kwargs,
                    "intrinsics",
                    "intrinsics_path",
                )
            if action is not None and camera_to_world is not None:
                raise ValueError(
                    "SANA-WM action and camera_to_world/camera_path are "
                    "mutually exclusive."
                )

            if action is not None:
                source = (
                    "action" if intrinsics is not None else "action_default_intrinsics"
                )
                translation_speed = self._request_float_value(
                    extra,
                    diffusers_kwargs,
                    "translation_speed",
                    "action_translation_speed",
                    "sana_wm_translation_speed",
                    default=_SANA_WM_DEFAULT_TRANSLATION_SPEED,
                )
                rotation_speed_deg = self._request_float_value(
                    extra,
                    diffusers_kwargs,
                    "rotation_speed_deg",
                    "action_rotation_speed_deg",
                    "sana_wm_rotation_speed_deg",
                    default=_SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
                )
                pitch_limit_deg = self._request_float_value(
                    extra,
                    diffusers_kwargs,
                    "pitch_limit_deg",
                    "action_pitch_limit_deg",
                    "sana_wm_pitch_limit_deg",
                    default=_SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
                )
                camera_to_world = self._coerce_action_camera_to_world(
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
                    _, intrinsics_vec4 = self._default_static_camera(
                        batch_size=batch_size,
                        num_frames=num_frames,
                        pixel_h=batch.height,
                        pixel_w=batch.width,
                        device=device,
                        dtype=camera_compute_dtype,
                    )
                    self.log_info(
                        "No intrinsics provided; using heuristic centered "
                        "intrinsics for the action trajectory."
                    )
                else:
                    intrinsics_vec4 = self._coerce_intrinsics_vec4(
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
                camera_to_world = self._coerce_camera_to_world(
                    camera_to_world,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    device=device,
                    dtype=camera_compute_dtype,
                )
                if intrinsics is None:
                    _, intrinsics_vec4 = self._default_static_camera(
                        batch_size=batch_size,
                        num_frames=num_frames,
                        pixel_h=batch.height,
                        pixel_w=batch.width,
                        device=device,
                        dtype=camera_compute_dtype,
                    )
                    self.log_info(
                        "No intrinsics provided; using heuristic centered "
                        "intrinsics for the request camera trajectory."
                    )
                else:
                    intrinsics_vec4 = self._coerce_intrinsics_vec4(
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
                camera_to_world, _ = self._default_static_camera(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    pixel_h=batch.height,
                    pixel_w=batch.width,
                    device=device,
                    dtype=camera_compute_dtype,
                )
                intrinsics_vec4 = self._coerce_intrinsics_vec4(
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
                    "camera with heuristic centered intrinsics. Pass "
                    "camera_to_world/intrinsics for camera-controlled output."
                )
                camera_to_world, intrinsics_vec4 = self._default_static_camera(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    pixel_h=batch.height,
                    pixel_w=batch.width,
                    device=device,
                    dtype=camera_compute_dtype,
                )

            camera_to_world = self._relative_camera_poses(camera_to_world)
            intrinsics_vec4 = self._scale_intrinsics_to_latent(
                intrinsics_vec4,
                pixel_h=batch.height,
                pixel_w=batch.width,
                latent_h=sp_h,
                latent_w=sp_w,
            )
            original_camera_conditions = self._flatten_camera_conditions(
                camera_to_world, intrinsics_vec4
            )
            camera_conditions = self._latent_frame_camera_conditions(
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

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ):
        """Set up scheduler timesteps and populate batch.timesteps, .sigmas."""
        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        num_inference_steps = batch.num_inference_steps

        flow_shift = getattr(
            self.pipeline_config,
            "inference_flow_shift",
            None,
        )
        if flow_shift is None:
            flow_shift = getattr(self.pipeline_config, "flow_shift", 9.95)
        kwargs = {}

        # diffusers FlowMatchEulerDiscreteScheduler supports mu/shift
        import inspect

        sig_params = inspect.signature(scheduler.set_timesteps).parameters
        if "shift" in sig_params:
            kwargs["shift"] = flow_shift
        elif "mu" in sig_params:
            # Convert flow_shift to mu: mu ~= log(shift)
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

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Pre-process everything needed by DenoisingStage for SANA-WM.

        Expects batch to already have prompt_embeds set by SanaWMTextEncodingStage.
        """
        device = get_local_torch_device()
        dtype = PRECISION_TO_TYPE.get(
            getattr(self.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}

        # Adjust num_frames to be compatible with VAE temporal stride.
        requested_num_frames = batch.num_frames or 49
        action_num_frames = self._action_num_frames_for_request(batch)
        if action_num_frames is not None and action_num_frames < requested_num_frames:
            self.log_info(
                "SANA-WM action trajectory has %d frames; capping requested "
                "num_frames=%d before VAE stride adjustment.",
                action_num_frames,
                requested_num_frames,
            )
            requested_num_frames = action_num_frames
        num_frames = requested_num_frames
        num_frames = self.pipeline_config.adjust_num_frames(num_frames)
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

        batch_size = batch.batch_size or 1
        generator = getattr(batch, "generator", None)
        if not isinstance(generator, (list, tuple, torch.Generator)):
            generator = self._generator_from_seed(
                getattr(batch, "seed", None),
                batch_size=batch_size,
                device=device,
            )
            batch.generator = generator

        latent_shape = self.pipeline_config.prepare_latent_shape(
            batch, batch_size, num_frames
        )
        # latent_shape: (B, 128, T_latent, H_sp, W_sp)
        latents = self._prepare_noise_latents(latent_shape, dtype, device, generator)
        log_sana_wm_tensor_stats("latents.initial_noise", latents)

        batch.raw_latent_shape = latent_shape

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
            if self._has_explicit_camera_request(batch):
                raise RuntimeError(
                    "SANA-WM camera conditioning failed for an explicitly "
                    "provided camera/intrinsics request."
                ) from e
            logger.warning(
                "SANA-WM camera conditioning failed: %s. Disabling camera branch.",
                e,
            )
            camera_conditions, chunk_plucker, camera_source = None, None, "error"

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

        batch = self._prepare_timesteps(batch, server_args, device)

        # Ensure prompt_embeds is a list (DenoisingStage expects list[Tensor]).
        if isinstance(batch.prompt_embeds, torch.Tensor):
            batch.prompt_embeds = [batch.prompt_embeds]
        if batch.negative_prompt_embeds is not None and isinstance(
            batch.negative_prompt_embeds, torch.Tensor
        ):
            batch.negative_prompt_embeds = [batch.negative_prompt_embeds]

        batch.do_classifier_free_guidance = getattr(batch, "guidance_scale", 1.0) > 1.0

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

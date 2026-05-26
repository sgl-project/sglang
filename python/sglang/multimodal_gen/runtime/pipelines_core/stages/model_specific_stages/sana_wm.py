# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM BeforeDenoisingStage.
#
# Consolidates all model-specific pre-processing for SANA-WM TI2V inference:
#   1. Adjust num_frames to satisfy (F-1) % temporal_stride == 0
#   2. Initialize random noise latents (5D: B, 128, T_latent, H_sp, W_sp)
#   3. VAE-encode the first-frame conditioning image and splice into noisy latents
#      (replaces latent[:, :, 0] with the encoded first-frame latent)
#   4. Build the (B, T_latent, 20) latent-frame ``camera_conditions`` raymap
#      consumed by the UCPE camera branch.
#   5. Compute the 48-channel packed Plücker raymap consumed by the
#      ``plucker_embedder`` (one chunk = vae_temporal_stride original frames).
#   6. Prepare FlowMatch timesteps and sigmas (uses flow_shift=9.95).
#
# Text encoding is handled by SanaWMTextEncodingStage so it can mirror the
# official chi-prompt token window without changing the shared text stage.

import os
import time
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
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

logger = init_logger(__name__)

_SANA_WM_DIAGNOSTICS_ENVS = (
    "SGLANG_SANA_WM_DIAGNOSTICS",
    "SGLANG_SANA_WM_LOG_TENSOR_STATS",
)

_SANA_WM_DEFAULT_VAE_TILE_MIN_FRAMES = 96
_SANA_WM_DEFAULT_VAE_TILE_STRIDE_FRAMES = 64


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
    sum over at most ~4096 values, useful for spotting deterministic drift
    without dumping full tensors.
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

    The LTX-2 VAE implements temporal tiled decode, but unlike the generic
    VAE base it does not enable it from ``enable_tiling()`` alone. Without this
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
        scheduler = (
            getattr(batch, "scheduler", None)
            or get_or_create_request_scheduler(batch, self.scheduler)
        )
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
                _first_tensor(
                    server_args.pipeline_config.get_neg_prompt_embeds(batch)
                ),
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
        camera_conditions = _to_device_dtype(
            extra.get("camera_conditions"), device=device, dtype=target_dtype
        )
        chunk_plucker = _to_device_dtype(
            extra.get("chunk_plucker"), device=device, dtype=target_dtype
        )

        self.log_info(
            "SANA-WM flow_euler_ltx denoising: latent=%s, steps=%d, cfg=%s, "
            "guidance_scale=%.4f, first_frame_locked=yes",
            tuple(latents.shape),
            len(timesteps),
            do_cfg,
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
                latent_model_input = (
                    torch.cat([latents, latents], dim=0) if do_cfg else latents
                )
                condition_mask_input = (
                    torch.cat([condition_mask, condition_mask], dim=0)
                    if do_cfg
                    else condition_mask
                )

                timestep = t.expand(condition_mask_input.shape).float()
                timestep = torch.minimum(
                    timestep,
                    (1.0 - condition_mask_input.float()) * 1000.0,
                )
                model_timestep = timestep[:, :1, :, 0, 0]

                model_kwargs = {
                    "encoder_hidden_states": (
                        torch.cat([neg_embeds, pos_embeds], dim=0)
                        if do_cfg
                        else pos_embeds
                    ),
                    "encoder_attention_mask": (
                        _cat_optional_tensors(neg_mask, pos_mask)
                        if do_cfg
                        else pos_mask
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
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    guidance_scale = float(
                        getattr(batch, "guidance_scale", 1.0) or 1.0
                    )
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

                tokens_to_denoise = t.float() / 1000.0 - 1e-6 < (
                    1.0 - condition_mask
                )
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

    # -----------------------------------------------------------------------
    # Helper: VAE-encode and scale an image tensor
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _vae_encode_image(
        self,
        image: torch.Tensor,     # (1, C, H, W) or (1, C, 1, H, W) in [0, 1] float
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a single image frame through the VAE encoder."""
        vae = self.vae
        configure_sana_wm_ltx2_vae_for_long_video(vae, self.pipeline_config)
        vae_dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        vae_dtype = vae_dtype_map.get(
            self.pipeline_config.vae_precision, torch.bfloat16
        )

        # Normalize image to [-1, 1] range expected by the VAE
        if image.max() > 1.01:
            image = image / 255.0
        image = (image * 2.0 - 1.0).to(device=device, dtype=vae_dtype)

        # Add temporal dim if not present: (B, C, H, W) → (B, C, 1, H, W)
        if image.dim() == 4:
            image = image.unsqueeze(2)

        log_sana_wm_tensor_stats("first_frame.pixel_input_normalized", image)
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

    # -----------------------------------------------------------------------
    # Helper: initialize noise latents
    # -----------------------------------------------------------------------

    def _prepare_noise_latents(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # -----------------------------------------------------------------------
    # Helper: splice first-frame image latent into noise latents
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _splice_first_frame(
        self,
        latents: torch.Tensor,  # (B, 128, T_lat, H_sp, W_sp)
        condition_image,        # PIL Image or torch.Tensor
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Replace latents[:, :, 0] with VAE-encoded first frame."""
        import PIL.Image

        if isinstance(condition_image, list):
            if len(condition_image) == 0:
                logger.warning(
                    "condition_image list is empty; skipping first-frame splice."
                )
                return latents
            condition_image = condition_image[0]

        # Convert PIL to tensor if needed
        if isinstance(condition_image, PIL.Image.Image):
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(condition_image.convert("RGB")).unsqueeze(0)
        elif isinstance(condition_image, torch.Tensor):
            img_tensor = condition_image.float()
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.dim() == 5 and img_tensor.shape[2] == 1:
                img_tensor = img_tensor.squeeze(2)
        else:
            logger.warning("condition_image type unsupported; skipping first-frame splice.")
            return latents

        # Resize if needed to match latent spatial dims
        B, C, T_lat, H_sp, W_sp = latents.shape
        target_h = H_sp * self.pipeline_config.vae_stride[1]  # 32
        target_w = W_sp * self.pipeline_config.vae_stride[2]  # 32
        if img_tensor.shape[-2] != target_h or img_tensor.shape[-1] != target_w:
            import torch.nn.functional as F
            img_tensor = F.interpolate(
                img_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            self.log_info(
                "First-frame condition image resized to %dx%d for VAE encode.",
                target_w,
                target_h,
            )

        first_frame_z = self._vae_encode_image(img_tensor, dtype, device)
        # first_frame_z: (1, 128, 1, H_sp, W_sp) — expand to batch
        first_frame_z = first_frame_z.expand(B, -1, -1, -1, -1)

        # Splice: replace the first temporal latent frame
        latents = latents.clone()
        latents[:, :, 0:1] = first_frame_z
        log_sana_wm_tensor_stats("latents.after_first_frame_splice", latents)
        return latents

    # -----------------------------------------------------------------------
    # Helper: camera conditioning
    # -----------------------------------------------------------------------

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
        value = SanaWMBeforeDenoisingStage._maybe_load_npy_tensor(
            value, "intrinsics"
        )
        intrinsics = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
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
            if intrinsics.shape[0] == num_frames:
                intrinsics = vec4.unsqueeze(0)
            elif intrinsics.shape[0] == batch_size:
                intrinsics = vec4.unsqueeze(1)
            else:
                raise ValueError(
                    "intrinsics with shape (N,3,3) must use N=num_frames or "
                    f"N=batch_size, got N={intrinsics.shape[0]}"
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
        return SanaWMBeforeDenoisingStage._pad_or_trim_frames(
            intrinsics, num_frames
        )

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

        camera_conditions = extra.get("camera_conditions", None)
        chunk_plucker = extra.get("chunk_plucker", None)
        diffusers_kwargs = extra.get("diffusers_kwargs", {})
        if not isinstance(diffusers_kwargs, dict):
            diffusers_kwargs = {}
        if camera_conditions is not None:
            camera_conditions = (
                camera_conditions
                if isinstance(camera_conditions, torch.Tensor)
                else torch.as_tensor(camera_conditions)
            ).to(device=device, dtype=camera_compute_dtype)
            if camera_conditions.dim() == 2:
                camera_conditions = camera_conditions.unsqueeze(0)
            if camera_conditions.shape[0] == 1 and batch_size > 1:
                camera_conditions = camera_conditions.expand(batch_size, -1, -1)
            if camera_conditions.shape[-1] != 20:
                raise ValueError(
                    "camera_conditions must have last dimension 20, got "
                    f"{tuple(camera_conditions.shape)}"
                )
            if camera_conditions.shape[1] == T_lat:
                source = "prepacked"
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
            if camera_to_world is not None:
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
            if chunk_plucker.shape[0] == 1 and batch_size > 1:
                chunk_plucker = chunk_plucker.expand(batch_size, -1, -1, -1, -1)

        if camera_conditions is not None:
            camera_conditions = camera_conditions.to(device=device, dtype=dtype)

        return camera_conditions, chunk_plucker, source

    # -----------------------------------------------------------------------
    # Helper: compute timesteps and sigmas for FlowMatch scheduling
    # -----------------------------------------------------------------------

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ):
        """Set up scheduler timesteps and populate batch.timesteps, .sigmas."""
        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        num_inference_steps = batch.num_inference_steps

        # Use flow_shift from pipeline config
        flow_shift = getattr(self.pipeline_config, "flow_shift", 9.95)
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

    # -----------------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
        Pre-process everything needed by DenoisingStage for SANA-WM.

        Expects batch to already have prompt_embeds set by SanaWMTextEncodingStage.
        """
        device = get_local_torch_device()
        dtype = torch.bfloat16  # SANA-WM runs in bf16

        # --- 0. Adjust num_frames to be compatible with VAE temporal stride ---
        num_frames = batch.num_frames or 49
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

        # --- 1. Generator for reproducibility ---
        seed = batch.seed if hasattr(batch, "seed") and batch.seed is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed)
        batch.generator = generator

        # --- 2. Compute latent shape and initialize noise ---
        batch_size = batch.batch_size or 1
        latent_shape = self.pipeline_config.prepare_latent_shape(batch, batch_size, num_frames)
        # latent_shape: (B, 128, T_latent, H_sp, W_sp)
        latents = self._prepare_noise_latents(latent_shape, dtype, device, generator)
        log_sana_wm_tensor_stats("latents.initial_noise", latents)

        # Store raw shape for DecodingStage
        batch.raw_latent_shape = latent_shape

        # --- 3. VAE-encode first frame and splice into noise latents ---
        condition_image = getattr(batch, "condition_image", None)
        if condition_image is not None:
            try:
                latents = self._splice_first_frame(latents, condition_image, dtype, device)
                self.log_info("First-frame spliced into noise latents.")
            except Exception as e:
                logger.warning(f"First-frame splice failed: {e}. Using pure noise latents.")
        else:
            self.log_info("No condition_image provided; using pure noise latents (T2V mode).")

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

        # --- 5. Prepare timesteps and sigmas ---
        batch = self._prepare_timesteps(batch, server_args, device)

        # --- 6. Ensure prompt_embeds is a list (DenoisingStage expects list[Tensor]) ---
        if isinstance(batch.prompt_embeds, torch.Tensor):
            batch.prompt_embeds = [batch.prompt_embeds]
        if (
            batch.negative_prompt_embeds is not None
            and isinstance(batch.negative_prompt_embeds, torch.Tensor)
        ):
            batch.negative_prompt_embeds = [batch.negative_prompt_embeds]

        # --- 7. CFG setup ---
        batch.do_classifier_free_guidance = (
            getattr(batch, "guidance_scale", 1.0) > 1.0
        )

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

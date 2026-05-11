# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, auto
from typing import Any

import numpy as np
import PIL
import torch
from einops import rearrange

from sglang.multimodal_gen.configs.models import (
    DiTConfig,
    EncoderConfig,
    ModelConfig,
    VAEConfig,
)
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.utils import update_config_from_args
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.models.vision_utils import get_default_height_width
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import (
    FlexibleArgumentParser,
    StoreBoolean,
    shallow_asdict,
)

logger = init_logger(__name__)


# NOTE: possible duplication with DataType
# this may focus on the model's original ability
class ModelTaskType(Enum):
    # TODO: check if I2V/TI2V models can work w/wo text

    I2V = auto()  # Image to Video
    T2V = auto()  # Text to Video
    TI2V = auto()  # Text and Image to Video

    T2I = auto()  # Text to Image
    I2I = auto()  # Image to Image
    TI2I = auto()  # Image to Image or Text-Image to Image
    I2M = auto()  # Image to Mesh

    def is_image_gen(self) -> bool:
        return (
            self == ModelTaskType.T2I
            or self == ModelTaskType.I2I
            or self == ModelTaskType.TI2I
        )

    def requires_image_input(self) -> bool:
        return (
            self == ModelTaskType.I2V
            or self == ModelTaskType.I2I
            or self == ModelTaskType.I2M
        )

    def accepts_image_input(self) -> bool:
        return (
            self == ModelTaskType.I2V
            or self == ModelTaskType.I2I
            or self == ModelTaskType.TI2I
            or self == ModelTaskType.TI2V
            or self == ModelTaskType.I2M
        )

    def data_type(self) -> DataType:
        if self == ModelTaskType.I2M:
            return DataType.MESH
        if self.is_image_gen():
            return DataType.IMAGE
        else:
            return DataType.VIDEO


class STA_Mode(str, Enum):
    """STA (Sliding Tile Attention) modes."""

    STA_INFERENCE = "STA_inference"
    STA_SEARCHING = "STA_searching"
    STA_TUNING = "STA_tuning"
    STA_TUNING_CFG = "STA_tuning_cfg"
    NONE = None


def postprocess_text(output: BaseEncoderOutput, _text_inputs) -> torch.tensor:
    raise NotImplementedError


@dataclass(frozen=True)
class TextConditioningOutput:
    """Text embeddings and masks aligned to postprocessed sequence length.

    `prompt_embeds_mask` and `prompt_seq_lens` describe real text tokens after
    model-specific trimming or packing, not the raw tokenizer output.
    """

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None = None
    prompt_seq_lens: list[int] | None = None


def pad_text_embeddings_with_mask(
    text_embeds: list[torch.Tensor],
) -> TextConditioningOutput:
    """Pad variable-length text embeddings and return the valid-token mask."""
    if not text_embeds:
        raise ValueError("text_embeds must contain at least one tensor")

    max_seq_len = max(e.size(0) for e in text_embeds)
    prompt_embeds = torch.stack(
        [
            torch.cat([e, e.new_zeros(max_seq_len - e.size(0), e.size(1))])
            for e in text_embeds
        ]
    )
    seq_lens = [int(e.size(0)) for e in text_embeds]
    seq_lens_tensor = torch.tensor(
        seq_lens,
        device=prompt_embeds.device,
        dtype=torch.long,
    )
    positions = torch.arange(max_seq_len, device=prompt_embeds.device).unsqueeze(0)
    prompt_embeds_mask = positions < seq_lens_tensor.unsqueeze(1)
    return TextConditioningOutput(prompt_embeds, prompt_embeds_mask, seq_lens)


def shard_rotary_emb_for_sp(emb):
    """
    Shard rotary embeddings [S, D] along sequence for SP.
    If S is not divisible by SP degree, pad by repeating the last row.
    """
    # Sequence Parallelism: slice image RoPE to local shard if enabled
    try:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_parallel_rank,
            get_sp_world_size,
        )

        sp_world_size = get_sp_world_size()
    except Exception:
        sp_world_size = 1
    seq_len = emb.shape[0]
    if seq_len % sp_world_size != 0:
        pad_len = sp_world_size - (seq_len % sp_world_size)
        pad = emb[-1:].repeat(pad_len, 1)
        emb = torch.cat([emb, pad], dim=0)
    if sp_world_size > 1:
        try:
            rank = get_sp_parallel_rank()
        except Exception:
            rank = 0
        seq_len = emb.shape[0]
        local_len = seq_len // sp_world_size
        start = rank * local_len
        end = start + local_len
        emb = emb[start:end]
        return emb
    else:
        return emb


def maybe_unpad_latents(latents, batch):
    # If SP padding was applied, remove extra tokens before reshaping
    raw_shape = batch.raw_latent_shape
    if len(raw_shape) == 3:
        # Sequence format [B, S, D]: use seq_len directly
        target_tokens = raw_shape[1]
    else:
        # Spatial format [B, C, H, W] or [B, C, T, H, W]: use width * height
        width, height = raw_shape[-1], raw_shape[-2]
        target_tokens = width * height
    if latents.shape[1] > target_tokens:
        latents = latents[:, :target_tokens, :]
    return latents


@dataclass
class PipelineConfig:
    """The base configuration class for a generation pipeline."""

    task_type: ModelTaskType = ModelTaskType.I2I
    skip_input_image_preprocess: bool = False

    model_path: str = ""
    pipeline_config_path: str | None = None

    # precision and autocast
    enable_autocast: bool = True

    # generation parameters
    # controls the timestep embedding generation
    should_use_guidance: bool = True
    embedded_cfg_scale: float = 6.0
    cfg_policy: CFGPolicy = field(default_factory=CFGPolicy)
    generator_device: str | None = None
    flow_shift: float | None = None
    disable_autocast: bool = False

    # Model configuration
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    dit_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = True
    vae_slicing: bool = False
    vae_sp: bool = True

    # Image encoder configuration
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    image_encoder_precision: str = "fp32"

    # Text encoder configuration
    DEFAULT_TEXT_ENCODER_PRECISIONS = ("fp32",)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    # See PRECISION_TO_TYPE for detailed mapping
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}])

    # image encoding
    image_encoder_extra_args: dict = field(default_factory=lambda: {})

    def postprocess_image(self, image):
        return image.last_hidden_state

    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None,)
    )

    # get prompt_embeds from encoder output
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.tensor], ...] = (
        field(default_factory=lambda: (postprocess_text,))
    )

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # DMD parameters
    dmd_denoising_steps: list[int] | None = field(default=None)

    # Wan2.2 TI2V parameters
    boundary_ratio: float | None = None

    # Compilation
    # enable_torch_compile: bool = False

    # calculate the adjust size for condition image
    # width: original condition image width
    # height: original condition image height
    def calculate_condition_image_size(self, image, width, height) -> tuple[int, int]:
        vae_scale_factor = self.vae_config.arch_config.spatial_compression_ratio
        height, width = get_default_height_width(image, vae_scale_factor, height, width)
        return width, height

    ## For timestep preparation stage

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return sigmas

    def get_classifier_free_guidance_scale(self, batch, guidance_scale: float) -> float:
        return guidance_scale

    def postprocess_cfg_noise(
        self,
        batch,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Model-specific CFG variants can override this hook
        # e.g. Qwen-Image's true-CFG norm matching.
        return noise_pred

    ## For ImageVAEEncodingStage
    def preprocess_condition_image(
        self, image, target_width, target_height, _vae_image_processor
    ):
        """
        preprocess the condition image, returns (image, final_image_width, final_image_height)
        """
        return image.resize(
            (target_width, target_height), PIL.Image.Resampling.LANCZOS
        ), (target_width, target_height)

    def prepare_calculated_size(self, image):
        return self.calculate_condition_image_size(image, image.width, image.height)

    def prepare_image_processor_kwargs(self, batch, neg=False):
        return {}

    def postprocess_image_latent(self, latent_condition, batch):
        vae_arch_config = self.vae_config.arch_config
        spatial_compression_ratio = vae_arch_config.spatial_compression_ratio
        temporal_compression_ratio = vae_arch_config.temporal_compression_ratio
        num_frames = batch.num_frames
        latent_height = batch.height // spatial_compression_ratio
        latent_width = batch.width // spatial_compression_ratio
        mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            repeats=temporal_compression_ratio,
            dim=2,
        )
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
        )
        mask_lat_size = mask_lat_size.view(
            1,
            -1,
            temporal_compression_ratio,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        image_latents = torch.concat([mask_lat_size, latent_condition], dim=1)
        return image_latents

    def slice_noise_pred(self, noise, latents):
        return noise

    def adjust_num_frames(self, num_frames):
        return num_frames

    # tokenize the prompt
    def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs) -> dict:
        return tokenizer(prompt, **tok_kwargs)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = batch.height // self.vae_config.arch_config.spatial_compression_ratio
        width = batch.width // self.vae_config.arch_config.spatial_compression_ratio

        # Calculate latent shape
        shape = (
            batch_size,
            self.dit_config.num_channels_latents,
            num_frames,
            height,
            width,
        )

        return shape

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        return prompt_dtype

    def allow_set_num_frames(self):
        return False

    def supports_dynamic_batching(self):
        """Return whether this pipeline can opt in to dynamic batching.

        The scheduler still checks each request before merging it into a batch.
        """
        return self.task_type in (ModelTaskType.T2I, ModelTaskType.T2V)

    def estimate_request_cost(self, batch) -> float:
        """Return the relative cost used for batching admission caps.

        This is compared with `max_cost` from the batching config; it is not a
        memory estimate. The default cost is latent tokens times frames times
        outputs; pipelines can override it for model-specific admission.
        """
        latent_tokens = float(batch.n_tokens or 0)
        if latent_tokens <= 0:
            width = int(batch.width or 0)
            height = int(batch.height or 0)
            if width > 0 and height > 0:
                vae_scale = getattr(
                    self.vae_config.arch_config, "vae_scale_factor", None
                )
                if vae_scale is None and hasattr(
                    self.vae_config, "get_vae_scale_factor"
                ):
                    vae_scale = self.vae_config.get_vae_scale_factor()
                vae_scale = max(1, int(vae_scale or 1))
                latent_tokens = math.ceil(width / vae_scale) * math.ceil(
                    height / vae_scale
                )

        num_frames = max(1, int(batch.num_frames or 1))
        num_outputs = max(1, int(batch.num_outputs_per_prompt or 1))
        return latent_tokens * num_frames * num_outputs

    def get_decode_scale_and_shift(self, device, dtype, vae):
        vae_arch_config = self.vae_config.arch_config
        scaling_factor = getattr(vae_arch_config, "scaling_factor", None)
        if scaling_factor is None:
            scaling_factor = getattr(vae, "scaling_factor", None)

        shift_factor = getattr(vae_arch_config, "shift_factor", None)
        if shift_factor is None:
            shift_factor = getattr(vae, "shift_factor", None)
        return scaling_factor, shift_factor

    # called after latents are prepared
    def maybe_pack_latents(self, latents, batch_size, batch):
        return latents

    def maybe_prepare_latent_ids(self, latents):
        return None

    # called after vae encode
    def postprocess_vae_encode(self, image_latents, vae):
        return image_latents

    # called after postprocess_vae_encode, before generic scale/shift
    def normalize_vae_encode(self, image_latents, vae):
        return None

    # called after scale_and_shift, before vae decoding
    def preprocess_decoding(self, latents, server_args=None, vae=None):
        return latents

    @staticmethod
    def _gather_sp_tensor(tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
        """All-gather an SP-sharded tensor along the specified logical dimension."""
        return sequence_model_parallel_all_gather(tensor.contiguous(), dim=dim)

    @staticmethod
    def _trim_sp_gather_padding(
        tensor: torch.Tensor, *, orig_len: int | None, dim: int
    ) -> torch.Tensor:
        """Trim padding introduced before SP sharding back to the original length."""
        if orig_len is None:
            return tensor
        orig_len = int(orig_len)
        if orig_len <= 0 or tensor.shape[dim] <= orig_len:
            return tensor
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(orig_len)
        return tensor[tuple(slices)]

    def gather_latents_for_sp(self, latents, batch=None):
        # For video latents [B, C, T_local, H, W], gather along time dim=2
        return self._gather_sp_tensor(latents, dim=2)

    def can_shard_audio_latents_for_sp(self, audio_latents) -> bool:
        """Return whether this pipeline uses packed audio latents that can be SP-sharded."""
        return False

    def shard_audio_latents_for_sp(self, batch, audio_latents):
        """Shard packed audio latents for SP. Pipelines without packed audio latents should return the input unchanged."""
        return audio_latents, False

    def gather_audio_latents_for_sp(self, audio_latents, batch):
        """Gather SP-sharded audio latents back to full sequence length."""
        return audio_latents

    def prepare_video_rope_coords_for_sp(
        self,
        model,
        batch,
        latent_model_input,
        *,
        num_frames,
        height,
        width,
    ):
        """Prepare model-side video RoPE coordinates for the local SP shard when the pipeline requires them."""
        return None

    def prepare_audio_rope_coords_for_sp(
        self,
        model,
        batch,
        audio_latent_model_input,
        *,
        num_frames,
    ):
        """Prepare model-side audio RoPE coordinates for the local SP shard when the pipeline requires them."""
        return None

    def gather_noise_pred_for_sp(self, batch, noise_pred):
        noise_pred = self.gather_latents_for_sp(noise_pred)
        raw_latent_shape = getattr(batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and noise_pred.dim() == 3:
            noise_pred = self._trim_sp_gather_padding(
                noise_pred, orig_len=raw_latent_shape[1], dim=1
            )
        return noise_pred

    def preprocess_vae_image(self, batch, vae_image_processor):
        pass

    def shard_latents_for_sp(self, batch, latents):
        # general logic for video models
        sp_world_size, rank_in_sp_group = get_sp_world_size(), get_sp_parallel_rank()
        if batch.enable_sequence_shard and sp_world_size > 1:
            return latents, False
        if latents.dim() != 5:
            return latents, False
        time_dim = latents.shape[2]

        # Pad to next multiple of SP degree if needed
        if time_dim > 0 and time_dim % sp_world_size != 0:
            logger.debug(
                "Padding latents to next multiple of SP degree, performance is sub-optimal"
            )
            pad_len = sp_world_size - (time_dim % sp_world_size)
            pad = torch.zeros(
                (*latents.shape[:2], pad_len, *latents.shape[3:]),
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, pad], dim=2)

        assert latents.shape[2] % sp_world_size == 0
        sharded_tensor = rearrange(
            latents, "b c (n t) h w -> b c n t h w", n=sp_world_size
        ).contiguous()
        sharded_tensor = sharded_tensor[:, :, rank_in_sp_group, :, :, :]
        return sharded_tensor, True

    def get_text_encoder_attention_mask(
        self, text_inputs: dict, encoder_index: int
    ) -> "torch.Tensor | None":
        """Return the attention mask for the given text encoder.

        Override to suppress (return None) or modify the mask per model.
        """
        return text_inputs.get("attention_mask")

    def build_text_conditioning_mask(
        self,
        text_inputs: dict,
        text_encoder_attention_mask: "torch.Tensor | None",
        prompt_embeds: "torch.Tensor",
        encoder_index: int,
    ) -> "torch.Tensor":
        """Return a mask aligned with post-processed prompt embeddings.

        True values mark valid text tokens. Dynamic batching must carry
        post-processed semantic text lengths explicitly; if a model-specific
        postprocessor changes the sequence length, it must return
        TextConditioningOutput with an embedding-aligned mask.
        """
        if prompt_embeds.ndim < 2:
            raise ValueError(
                "prompt_embeds must have shape [batch, seq, ...] to build text conditioning mask"
            )

        if prompt_embeds.ndim == 2:
            batch_size, embed_seq_len = 1, prompt_embeds.shape[0]
        else:
            batch_size, embed_seq_len = prompt_embeds.shape[:2]
        device = prompt_embeds.device
        if text_encoder_attention_mask is None:
            return torch.ones(
                (batch_size, embed_seq_len), dtype=torch.bool, device=device
            )

        raw_mask = text_encoder_attention_mask.to(device=device).bool()
        if raw_mask.ndim != 2 or raw_mask.shape[0] != batch_size:
            raise ValueError(
                "text attention mask must have shape [batch, seq] matching prompt_embeds batch"
            )

        if raw_mask.shape[1] == embed_seq_len:
            return raw_mask

        if prompt_embeds.ndim == 2 and raw_mask.shape[0] == 1:
            return torch.ones((1, embed_seq_len), dtype=torch.bool, device=device)

        raise ValueError(
            "text attention mask length does not match postprocessed prompt embeddings. "
            "Postprocess functions that trim, pack, or otherwise change text sequence "
            "length must return TextConditioningOutput with an embedding-aligned mask."
        )

    @staticmethod
    def seq_lens_from_text_conditioning_mask(mask: "torch.Tensor") -> list[int]:
        if mask.ndim != 2:
            raise ValueError("text conditioning mask must have shape [batch, seq]")
        return torch.count_nonzero(mask, dim=1).tolist()

    def require_text_seq_lens(
        self,
        batch,
        encoder_index: int,
        *,
        negative: bool = False,
        expected_batch_size: int | None = None,
    ) -> list[int]:
        """Return postprocessed text lengths captured during text encoding.

        Dynamic batches use these lengths for model masks, RoPE, and cache
        sizing after text embeddings have been padded.
        """
        seq_lens_by_encoder = (
            batch.negative_prompt_seq_lens if negative else batch.prompt_seq_lens
        )
        kind = "negative" if negative else "positive"
        if seq_lens_by_encoder is None or encoder_index >= len(seq_lens_by_encoder):
            raise ValueError(
                f"Missing {kind} prompt_seq_lens for text encoder {encoder_index}; "
                "dynamic text conditioning requires explicit sequence lengths."
            )

        seq_lens = [int(x) for x in seq_lens_by_encoder[encoder_index]]
        if expected_batch_size is not None and len(seq_lens) != int(
            expected_batch_size
        ):
            raise ValueError(
                f"{kind} prompt_seq_lens for text encoder {encoder_index} has "
                f"{len(seq_lens)} entries, expected {expected_batch_size}."
            )
        return seq_lens

    def get_text_encoder_pooler_output(
        self, outputs: "BaseEncoderOutput", encoder_index: int
    ) -> "torch.Tensor | None":
        """Return the pooler output for the given text encoder, or None to skip.

        Override for models that need pooled embeddings (e.g. FLUX v1, SD3).
        """
        return None

    def select_vae_weight_files(
        self,
        safetensors_list: list[str],
        component_model_path: str,
        component_name: str,
        vae_precision: str,
    ) -> list[str]:
        return safetensors_list

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds

    def post_denoising_loop(self, latents, batch):
        latents = maybe_unpad_latents(latents, batch)
        return latents

    def post_decoding(self, frames, server_args):
        return frames

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def _unpad_and_unpack_latents(self, latents, audio_latents, batch, vae, audio_vae):
        raise NotImplementedError("not yet implemented")

    def gather_denoising_env_static_for_sp(self, batch, cond_kwargs: dict | None):
        return cond_kwargs

    @staticmethod
    def add_cli_args(
        parser: FlexibleArgumentParser, prefix: str = ""
    ) -> FlexibleArgumentParser:
        prefix_with_dot = f"{prefix}." if (prefix.strip() != "") else ""

        # model_path will be conflicting with the model_path in ServerArgs,
        # so we add it separately if prefix is not empty
        if prefix_with_dot != "":
            parser.add_argument(
                f"--{prefix_with_dot}model-path",
                type=str,
                dest=f"{prefix_with_dot.replace('-', '_')}model_path",
                default=PipelineConfig.model_path,
                help="Path to the pretrained model",
            )

        parser.add_argument(
            f"--{prefix_with_dot}pipeline-config-path",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}pipeline_config_path",
            default=PipelineConfig.pipeline_config_path,
            help="Path to the pipeline config",
        )
        parser.add_argument(
            f"--{prefix_with_dot}embedded-cfg-scale",
            type=float,
            dest=f"{prefix_with_dot.replace('-', '_')}embedded_cfg_scale",
            default=PipelineConfig.embedded_cfg_scale,
            help="Embedded CFG scale",
        )
        parser.add_argument(
            f"--{prefix_with_dot}flow-shift",
            type=float,
            dest=f"{prefix_with_dot.replace('-', '_')}flow_shift",
            default=PipelineConfig.flow_shift,
            help="Flow shift parameter",
        )
        parser.add_argument(
            f"--{prefix_with_dot}resolution",
            type=int,
            dest=f"{prefix_with_dot.replace('-', '_')}resolution",
            default=None,
            help="Override the selected pipeline config's resolution setting. Only applies to pipelines that define a resolution field.",
        )

        # DiT configuration
        parser.add_argument(
            f"--{prefix_with_dot}dit-precision",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}dit_precision",
            default=PipelineConfig.dit_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for the DiT model",
        )

        # VAE configuration
        parser.add_argument(
            f"--{prefix_with_dot}vae-precision",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}vae_precision",
            default=PipelineConfig.vae_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for VAE",
        )
        parser.add_argument(
            f"--{prefix_with_dot}vae-tiling",
            action=StoreBoolean,
            dest=f"{prefix_with_dot.replace('-', '_')}vae_tiling",
            default=PipelineConfig.vae_tiling,
            help="Enable VAE tiling",
        )
        parser.add_argument(
            f"--{prefix_with_dot}vae-slicing",
            action=StoreBoolean,
            dest=f"{prefix_with_dot.replace('-', '_')}vae_slicing",
            default=PipelineConfig.vae_slicing,
            help="Enable VAE slicing",
        )
        parser.add_argument(
            f"--{prefix_with_dot}vae-sp",
            action=StoreBoolean,
            dest=f"{prefix_with_dot.replace('-', '_')}vae_sp",
            help="Enable VAE spatial parallelism",
        )

        # Text encoder configuration
        parser.add_argument(
            f"--{prefix_with_dot}text-encoder-precisions",
            nargs="+",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}text_encoder_precisions",
            default=PipelineConfig.DEFAULT_TEXT_ENCODER_PRECISIONS,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for each text encoder",
        )

        # Image encoder configuration
        parser.add_argument(
            f"--{prefix_with_dot}image-encoder-precision",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}image_encoder_precision",
            default=PipelineConfig.image_encoder_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for image encoder",
        )

        # DMD parameters
        parser.add_argument(
            f"--{prefix_with_dot}dmd-denoising-steps",
            type=parse_int_list,
            default=PipelineConfig.dmd_denoising_steps,
            help="Comma-separated list of denoising steps (e.g., '1000,757,522')",
        )

        # Add VAE configuration arguments
        from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig

        VAEConfig.add_cli_args(parser, prefix=f"{prefix_with_dot}vae-config")

        # Add DiT configuration arguments
        from sglang.multimodal_gen.configs.models.dits.base import DiTConfig

        DiTConfig.add_cli_args(parser, prefix=f"{prefix_with_dot}dit-config")

        # Add T5 configuration arguments
        from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config

        T5Config.add_cli_args(parser, prefix=f"{prefix_with_dot}t5-config")

        return parser

    def update_config_from_dict(self, args: dict[str, Any], prefix: str = "") -> None:
        prefix_with_dot = f"{prefix}." if (prefix.strip() != "") else ""
        update_config_from_args(self, args, prefix, pop_args=True)
        update_config_from_args(
            self.vae_config, args, f"{prefix_with_dot}vae_config", pop_args=True
        )
        update_config_from_args(
            self.dit_config, args, f"{prefix_with_dot}dit_config", pop_args=True
        )
        for text_encoder_config in self.text_encoder_configs:
            if isinstance(text_encoder_config, T5Config):
                update_config_from_args(
                    text_encoder_config,
                    args,
                    f"{prefix_with_dot}t5_config",
                    pop_args=True,
                )

    @classmethod
    def from_kwargs(
        cls, kwargs: dict[str, Any], config_cli_prefix: str = ""
    ) -> "PipelineConfig":
        """
        Load PipelineConfig from kwargs Dictionary, as part of the ServerArg initialization process
        kwargs: dictionary of kwargs
        config_cli_prefix: prefix of CLI arguments for this PipelineConfig instance
        """
        from sglang.multimodal_gen.registry import get_model_info

        prefix_with_dot = (
            f"{config_cli_prefix}." if (config_cli_prefix.strip() != "") else ""
        )
        model_path: str | None = kwargs.get(
            prefix_with_dot + "model_path", None
        ) or kwargs.get("model_path")
        pipeline_config_or_path: str | PipelineConfig | dict[str, Any] | None = (
            kwargs.get(prefix_with_dot + "pipeline_config", None)
            or kwargs.get("pipeline_config")
        )
        if model_path is None:
            raise ValueError("model_path is required in kwargs")

        # Check if model_path is a safetensors file and pipeline_class_name is specified
        pipeline_class_name = kwargs.get(
            prefix_with_dot + "pipeline_class_name"
        ) or kwargs.get("pipeline_class_name")
        is_safetensors_file = os.path.isfile(model_path) and model_path.endswith(
            ".safetensors"
        )

        # 1. Get the pipeline config class from the registry
        from sglang.multimodal_gen.configs.pipeline_configs.flux import (
            Flux2PipelineConfig,
        )
        from sglang.multimodal_gen.registry import get_pipeline_config_classes

        # If model_path is a safetensors file and pipeline_class_name is specified,
        # try to get PipelineConfig from the registry first
        if is_safetensors_file and pipeline_class_name:
            config_classes = get_pipeline_config_classes(pipeline_class_name)
            if config_classes is not None:
                pipeline_config_cls, _ = config_classes
                logger.info(
                    f"Detected safetensors file with {pipeline_class_name}, "
                    f"using {pipeline_config_cls.__name__} directly without model_index.json"
                )
            else:
                model_info = get_model_info(
                    model_path,
                    backend=kwargs.get("backend"),
                    model_id=kwargs.get("model_id"),
                )
                if model_info is None:
                    from sglang.multimodal_gen.registry import (
                        _PIPELINE_CONFIG_REGISTRY,
                        _discover_and_register_pipelines,
                    )

                    _discover_and_register_pipelines()
                    available_pipelines = list(_PIPELINE_CONFIG_REGISTRY.keys())
                    raise ValueError(
                        f"Could not get model info for '{model_path}'. "
                        f"If using a safetensors file, please specify a valid pipeline_class_name. "
                        f"Available pipelines with config classes: {available_pipelines}"
                    )
                pipeline_config_cls = model_info.pipeline_config_cls
        else:
            model_info = get_model_info(
                model_path,
                backend=kwargs.get("backend"),
                model_id=kwargs.get("model_id"),
            )
            if model_info is None:
                raise ValueError(
                    f"Could not get model info for '{model_path}'. "
                    f"If using a safetensors file, please specify pipeline_class_name"
                )
            # 1.5. Adjust pipeline config for fine-tuned VAE if needed
            pipeline_config_cls = model_info.pipeline_config_cls
        vae_path = kwargs.get(prefix_with_dot + "vae_path") or kwargs.get("vae_path")
        if vae_path is None:
            component_paths = kwargs.get(
                prefix_with_dot + "component_paths"
            ) or kwargs.get("component_paths")
            if isinstance(component_paths, dict):
                vae_path = component_paths.get("vae")

        # Check if this is a Flux2 model with fal/FLUX.2-Tiny-AutoEncoder
        if (
            isinstance(pipeline_config_cls, type)
            and issubclass(pipeline_config_cls, Flux2PipelineConfig)
            and vae_path is not None
            and "FLUX.2-Tiny-AutoEncoder" in vae_path
        ):
            from sglang.multimodal_gen.configs.pipeline_configs.flux_finetuned import (
                Flux2FinetunedPipelineConfig,
            )

            pipeline_config_cls = Flux2FinetunedPipelineConfig

        pipeline_config = pipeline_config_cls()

        # 2. Load PipelineConfig from a json file or a PipelineConfig object if provided
        if isinstance(pipeline_config_or_path, str):
            pipeline_config.load_from_json(pipeline_config_or_path)
            kwargs[prefix_with_dot + "pipeline_config_path"] = pipeline_config_or_path
        elif isinstance(pipeline_config_or_path, PipelineConfig):
            pipeline_config = pipeline_config_or_path
        elif isinstance(pipeline_config_or_path, dict):
            pipeline_config.update_pipeline_config(pipeline_config_or_path)

        # 3. Update PipelineConfig from CLI arguments if provided
        kwargs[prefix_with_dot + "model_path"] = model_path
        pipeline_config.update_config_from_dict(kwargs, config_cli_prefix)
        return pipeline_config

    def check_pipeline_config(self) -> None:
        if self.vae_sp and not self.vae_tiling:
            raise ValueError(
                "Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True."
            )

        if len(self.text_encoder_configs) != len(self.text_encoder_precisions):
            raise ValueError(
                f"Length of text encoder configs ({len(self.text_encoder_configs)}) must be equal to length of text encoder precisions ({len(self.text_encoder_precisions)})"
            )

        if len(self.text_encoder_configs) != len(self.preprocess_text_funcs):
            raise ValueError(
                f"Length of text encoder configs ({len(self.text_encoder_configs)}) must be equal to length of text preprocessing functions ({len(self.preprocess_text_funcs)})"
            )

        if len(self.preprocess_text_funcs) != len(self.postprocess_text_funcs):
            raise ValueError(
                f"Length of text postprocess functions ({len(self.postprocess_text_funcs)}) must be equal to length of text preprocessing functions ({len(self.preprocess_text_funcs)})"
            )

    def dump_to_json(self, file_path: str):
        output_dict = shallow_asdict(self)
        del_keys = []
        for key, value in output_dict.items():
            if isinstance(value, ModelConfig):
                model_dict = asdict(value)
                # Model Arch Config should be hidden away from the users
                model_dict.pop("arch_config")
                output_dict[key] = model_dict
            elif isinstance(value, tuple) and all(
                isinstance(v, ModelConfig) for v in value
            ):
                model_dicts = []
                for v in value:
                    model_dict = asdict(v)
                    # Model Arch Config should be hidden away from the users
                    model_dict.pop("arch_config")
                    model_dicts.append(model_dict)
                output_dict[key] = model_dicts
            elif isinstance(value, tuple) and all(callable(f) for f in value):
                # Skip dumping functions
                del_keys.append(key)

        for key in del_keys:
            output_dict.pop(key, None)

        with open(file_path, "w") as f:
            json.dump(output_dict, f, indent=2)

    def load_from_json(self, file_path: str):
        with open(file_path) as f:
            input_pipeline_dict = json.load(f)
        self.update_pipeline_config(input_pipeline_dict)

    def update_pipeline_config(self, source_pipeline_dict: dict[str, Any]) -> None:
        for f in fields(self):
            key = f.name
            if key in source_pipeline_dict:
                current_value = getattr(self, key)
                new_value = source_pipeline_dict[key]

                # If it's a nested ModelConfig, update it recursively
                if isinstance(current_value, ModelConfig):
                    current_value.update_model_config(new_value)
                elif isinstance(current_value, tuple) and all(
                    isinstance(v, ModelConfig) for v in current_value
                ):
                    assert len(current_value) == len(
                        new_value
                    ), "Users shouldn't delete or add text encoder config objects in your json"
                    for target_config, source_config in zip(
                        current_value, new_value, strict=True
                    ):
                        target_config.update_model_config(source_config)
                else:
                    setattr(self, key, new_value)

        if hasattr(self, "__post_init__"):
            self.__post_init__()


@dataclass
class ImagePipelineConfig(PipelineConfig):
    """Base config for image generation pipelines with token-like latents [B, S, D]."""

    def _prepare_sigmas(self, sigmas, num_inference_steps):
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        return sigmas

    def shard_latents_for_sp(self, batch, latents):
        # latents: [B, H * W, C]
        sp_world_size, rank_in_sp_group = get_sp_world_size(), get_sp_parallel_rank()
        if batch.enable_sequence_shard:
            return latents, False
        seq_len = latents.shape[1]

        # TODO: reuse code in PipelineConfig::shard_latents_for_sp
        # Pad to next multiple of SP degree if needed
        if seq_len % sp_world_size != 0:
            pad_len = sp_world_size - (seq_len % sp_world_size)
            pad = torch.zeros(
                (*latents.shape[:1], pad_len, *latents.shape[2:]),
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, pad], dim=1)

        sharded_tensor = rearrange(
            latents, "b (n s) d -> b n s d", n=sp_world_size
        ).contiguous()
        sharded_tensor = sharded_tensor[:, rank_in_sp_group, :, :]
        return sharded_tensor, True

    def gather_latents_for_sp(self, latents, batch=None):
        # For image latents [B, S_local, D], gather along sequence dim=1
        return self._gather_sp_tensor(latents, dim=1)

    def _unpad_and_unpack_latents(self, latents, batch):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        channels = self.dit_config.arch_config.in_channels
        batch_size = latents.shape[0]

        height = 2 * (int(batch.height) // (vae_scale_factor * 2))
        width = 2 * (int(batch.width) // (vae_scale_factor * 2))

        latents = maybe_unpad_latents(latents, batch)

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents, batch_size, channels, height, width


@dataclass
class SpatialImagePipelineConfig(ImagePipelineConfig):
    """Base config for spatial image pipelines (e.g. GLM-Image) with 4D latents (B, C, H', W').

    Overrides shard_latents_for_sp / gather_latents_for_sp to shard along the height dimension
    so that each SP rank gets (B, C, H'_local, W') instead of using the token-style (B, S, C) path.
    """

    def shard_latents_for_sp(self, batch, latents):
        # 4D latents (B, C, H', W') -> shard along H' (dim=2); otherwise fall back to base (B, S, C)
        sp_world_size = get_sp_world_size()
        if sp_world_size <= 1:
            return latents, False
        if latents.dim() != 4:
            return super().shard_latents_for_sp(batch, latents)

        # (B, C, H', W')
        _, _, h_lat, w_lat = latents.shape
        if h_lat % sp_world_size != 0:
            pad_len = sp_world_size - (h_lat % sp_world_size)
            pad = torch.zeros(
                (latents.shape[0], latents.shape[1], pad_len, latents.shape[3]),
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, pad], dim=2)
            h_lat = latents.shape[2]
        rank_in_sp_group = get_sp_parallel_rank()
        chunk_size = h_lat // sp_world_size
        h0 = rank_in_sp_group * chunk_size
        h1 = h0 + chunk_size
        sharded = latents[:, :, h0:h1, :].contiguous()
        return sharded, True

    def gather_latents_for_sp(self, latents, batch=None):
        if get_sp_world_size() <= 1:
            return latents
        if latents.dim() != 4:
            return super().gather_latents_for_sp(latents, batch=batch)
        # Gather along dim=2 (H') to match shard_latents_for_sp
        return self._gather_sp_tensor(latents, dim=2)


@dataclass
class SlidingTileAttnConfig(PipelineConfig):
    """Configuration for sliding tile attention."""

    # Override any BaseConfig defaults as needed
    # Add sliding tile specific parameters
    window_size: int = 16
    stride: int = 8

    # You can provide custom defaults for inherited fields
    height: int = 576
    width: int = 1024

    # Additional configuration specific to sliding tile attention
    pad_to_square: bool = False
    use_overlap_optimization: bool = True


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string of integers into a list."""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]

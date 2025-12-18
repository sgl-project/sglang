# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import json
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
from sglang.multimodal_gen.configs.utils import update_config_from_args
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.models.vision_utils import get_default_height_width
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import (
    FlexibleArgumentParser,
    StoreBoolean,
    shallow_asdict,
)

logger = init_logger(__name__)


# NOTE: possible duplication with DataType, WorkloadType
# this may focus on the model's original ability
class ModelTaskType(Enum):
    I2V = auto()  # Image to Video
    T2V = auto()  # Text to Video
    TI2V = auto()  # Text and Image to Video
    T2I = auto()  # Text to Image
    I2I = auto()  # Image to Image

    def is_image_gen(self):
        return self == ModelTaskType.T2I or self == ModelTaskType.I2I


class STA_Mode(str, Enum):
    """STA (Sliding Tile Attention) modes."""

    STA_INFERENCE = "STA_inference"
    STA_SEARCHING = "STA_searching"
    STA_TUNING = "STA_tuning"
    STA_TUNING_CFG = "STA_tuning_cfg"
    NONE = None


def preprocess_text(prompt: str) -> str:
    return prompt


def postprocess_text(output: BaseEncoderOutput, _text_inputs) -> torch.tensor:
    raise NotImplementedError


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
    target_tokens = batch.raw_latent_shape[-1] * batch.raw_latent_shape[-2]
    if latents.shape[1] > target_tokens:
        latents = latents[:, :target_tokens, :]
    return latents


# config for a single pipeline
@dataclass
class PipelineConfig:
    """The base configuration class for a generation pipeline."""

    task_type: ModelTaskType = ModelTaskType.I2I

    model_path: str = ""
    pipeline_config_path: str | None = None

    # generation parameters
    # controls the timestep embedding generation
    should_use_guidance: bool = True
    embedded_cfg_scale: float = 6.0
    flow_shift: float | None = None
    disable_autocast: bool = False

    # Model configuration
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    dit_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = True
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

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text,)
    )

    # get prompt_embeds from encoder output
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.tensor], ...] = (
        field(default_factory=lambda: (postprocess_text,))
    )

    # StepVideo specific parameters
    pos_magic: str | None = None
    neg_magic: str | None = None
    timesteps_scale: bool | None = None

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

    def prepare_image_processor_kwargs(self, batch):
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

    # called after scale_and_shift, before vae decoding
    def preprocess_decoding(self, latents, server_args=None, vae=None):
        return latents

    def gather_latents_for_sp(self, latents):
        # For video latents [B, C, T_local, H, W], gather along time dim=2
        latents = sequence_model_parallel_all_gather(latents, dim=2)
        return latents

    def shard_latents_for_sp(self, batch, latents):
        # general logic for video models
        sp_world_size, rank_in_sp_group = get_sp_world_size(), get_sp_parallel_rank()
        if latents.dim() != 5:
            return latents, False
        time_dim = latents.shape[2]
        if time_dim > 0 and time_dim % sp_world_size == 0:
            sharded_tensor = rearrange(
                latents, "b c (n t) h w -> b c n t h w", n=sp_world_size
            ).contiguous()
            sharded_tensor = sharded_tensor[:, :, rank_in_sp_group, :, :, :]
            return sharded_tensor, True
        return latents, False

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds

    def post_denoising_loop(self, latents, batch):
        latents = maybe_unpad_latents(latents, batch)
        return latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

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
        parser.add_argument(
            f"--{prefix_with_dot}pos_magic",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}pos_magic",
            default=PipelineConfig.pos_magic,
            help="Positive magic prompt for sampling, used in stepvideo",
        )
        parser.add_argument(
            f"--{prefix_with_dot}neg_magic",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}neg_magic",
            default=PipelineConfig.neg_magic,
            help="Negative magic prompt for sampling, used in stepvideo",
        )
        parser.add_argument(
            f"--{prefix_with_dot}timesteps_scale",
            type=bool,
            dest=f"{prefix_with_dot.replace('-', '_')}timesteps_scale",
            default=PipelineConfig.timesteps_scale,
            help="Bool for applying scheduler scale in set_timesteps, used in stepvideo",
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

    @classmethod
    def from_kwargs(
        cls, kwargs: dict[str, Any], config_cli_prefix: str = ""
    ) -> "PipelineConfig":
        """
        Load PipelineConfig from kwargs Dictionary.
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

        # 1. Get the pipeline config class from the registry
        from sglang.multimodal_gen.configs.pipeline_configs.flux import (
            Flux2PipelineConfig,
        )

        model_info = get_model_info(model_path)

        # 1.5. Adjust pipeline config for fine-tuned VAE if needed
        pipeline_config_cls = model_info.pipeline_config_cls
        vae_path = kwargs.get(prefix_with_dot + "vae_path") or kwargs.get("vae_path")

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
        sp_world_size, rank_in_sp_group = get_sp_world_size(), get_sp_parallel_rank()
        seq_len = latents.shape[1]

        # Pad to next multiple of SP degree if needed
        if seq_len % sp_world_size != 0:
            pad_len = sp_world_size - (seq_len % sp_world_size)
            pad = torch.zeros(
                (latents.shape[0], pad_len, latents.shape[2]),
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, pad], dim=1)
            # Record padding length for later unpad
            batch.sp_seq_pad = int(getattr(batch, "sp_seq_pad", 0)) + pad_len

        sharded_tensor = rearrange(
            latents, "b (n s) d -> b n s d", n=sp_world_size
        ).contiguous()
        sharded_tensor = sharded_tensor[:, rank_in_sp_group, :, :]
        return sharded_tensor, True

    def gather_latents_for_sp(self, latents):
        # For image latents [B, S_local, D], gather along sequence dim=1
        latents = sequence_model_parallel_all_gather(latents, dim=1)
        return latents

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

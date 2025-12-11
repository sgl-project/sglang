# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
import hashlib
import json
import math
import os.path
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import StoreBoolean, align_to

logger = init_logger(__name__)


def _json_safe(obj: Any):
    """
    Recursively convert objects to JSON-serializable forms.
    - Enums -> their name
    - Sets/Tuples -> lists
    - Dicts/Lists -> recursively processed
    """
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


def generate_request_id() -> str:
    return str(uuid.uuid4())


def _sanitize_filename(name: str, replacement: str = "_", max_length: int = 150) -> str:
    """Create a filesystem- and ffmpeg-friendly filename.

    - Normalize to ASCII (drop accents and unsupported chars)
    - Replace spaces with underscores
    - Replace any char not in [A-Za-z0-9_.-] with replacement
    - Collapse multiple underscores
    - Trim leading/trailing dots/underscores and limit length
    """
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace(" ", "_")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", replacement, ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("._")
    if not ascii_name:
        ascii_name = "output"
    if max_length and len(ascii_name) > max_length:
        ascii_name = ascii_name[:max_length]
    return ascii_name


class DataType(Enum):
    IMAGE = auto()
    VIDEO = auto()

    def get_default_extension(self) -> str:
        if self == DataType.IMAGE:
            return "jpg"
        else:
            return "mp4"


@dataclass
class SamplingParams:
    """
    Sampling parameters for generation.
    """

    data_type: DataType = DataType.VIDEO

    request_id: str | None = None

    # All fields below are copied from ForwardBatch

    # Image inputs
    image_path: str | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )
    prompt_path: str | None = None
    output_path: str = "outputs/"
    output_file_name: str | None = None

    # Batch info
    num_outputs_per_prompt: int = 1
    seed: int = 1024
    generator_device: str = "cuda"  # Device for random generator: "cuda" or "cpu"

    # Original dimensions (before VAE scaling)
    num_frames: int = 125
    num_frames_round_down: bool = (
        False  # Whether to round down num_frames if it's not divisible by num_gpus
    )
    height: int | None = None
    width: int | None = None
    # NOTE: this is temporary, we need a way to know if width or height is not provided, or do the image resize earlier
    height_not_provided: bool = False
    width_not_provided: bool = False
    fps: int = 24

    # Denoising parameters
    num_inference_steps: int = None
    guidance_scale: float = None
    guidance_rescale: float = 0.0
    boundary_ratio: float | None = None

    # TeaCache parameters
    enable_teacache: bool = False

    # Profiling
    profile: bool = False
    num_profiled_timesteps: int = 5
    profile_all_stages: bool = False

    # Debugging
    debug: bool = False
    perf_dump_path: str | None = None

    # Misc
    save_output: bool = True
    return_frames: bool = False
    return_trajectory_latents: bool = False  # returns all latents for each timestep
    return_trajectory_decoded: bool = False  # returns decoded latents for each timestep
    # if True, disallow user params to override subclass-defined protected fields
    no_override_protected_fields: bool = False
    # whether to adjust num_frames for multi-GPU friendly splitting (default: True)
    adjust_frames: bool = True

    def _set_output_file_ext(self):
        # add extension if needed
        if not any(
            self.output_file_name.endswith(ext)
            for ext in [".mp4", ".jpg", ".png", ".webp"]
        ):
            self.output_file_name = (
                f"{self.output_file_name}.{self.data_type.get_default_extension()}"
            )

    def _set_output_file_name(self):
        # settle output_file_name
        if (
            self.output_file_name is None
            and self.prompt
            and isinstance(self.prompt, str)
        ):
            # generate a random filename
            # get a hash of current params
            params_dict = dataclasses.asdict(self)
            # Avoid recursion
            params_dict["output_file_name"] = ""

            # Convert to a stable JSON string
            params_str = json.dumps(_json_safe(params_dict), sort_keys=True)
            # Create a hash
            hasher = hashlib.sha256()
            hasher.update(params_str.encode("utf-8"))
            param_hash = hasher.hexdigest()[:8]

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base = f"{self.prompt[:100]}_{timestamp}_{param_hash}"
            self.output_file_name = base

        if self.output_file_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.output_file_name = f"output_{timestamp}"

        self.output_file_name = _sanitize_filename(self.output_file_name)

        # Ensure a proper extension is present
        self._set_output_file_ext()

    def __post_init__(self) -> None:
        assert self.num_frames >= 1
        self.data_type = DataType.VIDEO if self.num_frames > 1 else DataType.IMAGE

        if self.width is None:
            self.width_not_provided = True
        if self.height is None:
            self.height_not_provided = True

    def check_sampling_param(self):
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def _adjust(
        self,
        server_args: ServerArgs,
    ):
        """
        final adjustment, called after merged with user params
        """
        pipeline_config = server_args.pipeline_config
        if not isinstance(self.prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(self.prompt)}")

        # Process negative prompt
        if self.negative_prompt is not None and not self.negative_prompt.isspace():
            # avoid stripping default negative prompt: ' ' for qwen-image
            self.negative_prompt = self.negative_prompt.strip()

        # Validate dimensions
        if self.num_frames <= 0:
            raise ValueError(
                f"height, width, and num_frames must be positive integers, got "
                f"height={self.height}, width={self.width}, "
                f"num_frames={self.num_frames}"
            )

        if pipeline_config.task_type.is_image_gen():
            # settle num_frames
            logger.debug(f"Setting num_frames to 1 because this is an image-gen model")
            self.num_frames = 1
            self.data_type = DataType.IMAGE
        elif self.adjust_frames:
            # NOTE: We must apply adjust_num_frames BEFORE the SP alignment logic below.
            # If we apply it after, adjust_num_frames might modify the frame count
            # and break the divisibility constraint (alignment) required by num_gpus.
            self.num_frames = server_args.pipeline_config.adjust_num_frames(
                self.num_frames
            )

            # Adjust number of frames based on number of GPUs for video task
            use_temporal_scaling_frames = (
                pipeline_config.vae_config.use_temporal_scaling_frames
            )
            num_frames = self.num_frames
            num_gpus = server_args.num_gpus
            temporal_scale_factor = (
                pipeline_config.vae_config.arch_config.temporal_compression_ratio
            )

            if use_temporal_scaling_frames:
                orig_latent_num_frames = (num_frames - 1) // temporal_scale_factor + 1
            else:  # stepvideo only
                orig_latent_num_frames = self.num_frames // 17 * 3

            if orig_latent_num_frames % server_args.num_gpus != 0:
                # Adjust latent frames to be divisible by number of GPUs
                if self.num_frames_round_down:
                    # Ensure we have at least 1 batch per GPU
                    new_latent_num_frames = (
                        max(1, (orig_latent_num_frames // num_gpus)) * num_gpus
                    )
                else:
                    new_latent_num_frames = (
                        math.ceil(orig_latent_num_frames / num_gpus) * num_gpus
                    )

                if use_temporal_scaling_frames:
                    # Convert back to number of frames, ensuring num_frames-1 is a multiple of temporal_scale_factor
                    new_num_frames = (
                        new_latent_num_frames - 1
                    ) * temporal_scale_factor + 1
                else:  # stepvideo only
                    # Find the least common multiple of 3 and num_gpus
                    divisor = math.lcm(3, num_gpus)
                    # Round up to the nearest multiple of this LCM
                    new_latent_num_frames = (
                        (new_latent_num_frames + divisor - 1) // divisor
                    ) * divisor
                    # Convert back to actual frames using the StepVideo formula
                    new_num_frames = new_latent_num_frames // 3 * 17

                logger.info(
                    "Adjusting number of frames from %s to %s based on number of GPUs (%s)",
                    self.num_frames,
                    new_num_frames,
                    server_args.num_gpus,
                )
                self.num_frames = new_num_frames

        self._set_output_file_name()
        self.log(server_args=server_args)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "SamplingParams":
        from sglang.multimodal_gen.registry import get_model_info

        model_info = get_model_info(model_path)
        sampling_params: SamplingParams = model_info.sampling_param_cls(**kwargs)
        return sampling_params

    @staticmethod
    def from_user_sampling_params_args(model_path: str, server_args, *args, **kwargs):
        sampling_params = SamplingParams.from_pretrained(model_path)

        user_sampling_params = SamplingParams(*args, **kwargs)
        # TODO: refactor
        sampling_params._merge_with_user_params(user_sampling_params)
        sampling_params._adjust(server_args)

        return sampling_params

    def output_size_str(self) -> str:
        return f"{self.width}x{self.height}"

    def seconds(self) -> float:
        return self.num_frames / self.fps

    @staticmethod
    def add_cli_args(parser: Any) -> Any:
        """Add CLI arguments for SamplingParam fields"""
        parser.add_argument("--data-type", type=str, nargs="+", default=DataType.VIDEO)
        parser.add_argument(
            "--num-frames-round-down",
            action="store_true",
            default=SamplingParams.num_frames_round_down,
        )
        parser.add_argument(
            "--enable-teacache",
            action="store_true",
            default=SamplingParams.enable_teacache,
        )

        # profiling
        parser.add_argument(
            "--profile",
            action="store_true",
            default=SamplingParams.profile,
            help="Enable torch profiler for denoising stage",
        )
        parser.add_argument(
            "--num-profiled-timesteps",
            type=int,
            default=SamplingParams.num_profiled_timesteps,
            help="Number of timesteps to profile after warmup",
        )
        parser.add_argument(
            "--profile-all-stages",
            action="store_true",
            dest="profile_all_stages",
            default=SamplingParams.profile_all_stages,
            help="Used with --profile, profile all pipeline stages",
        )

        parser.add_argument(
            "--debug",
            action="store_true",
            default=SamplingParams.debug,
            help="",
        )

        parser.add_argument(
            "--prompt",
            type=str,
            default=SamplingParams.prompt,
            help="Text prompt for generation",
        )
        parser.add_argument(
            "--negative-prompt",
            type=str,
            default=SamplingParams.negative_prompt,
            help="Negative text prompt for generation",
        )
        parser.add_argument(
            "--prompt-path",
            type=str,
            default=SamplingParams.prompt_path,
            help="Path to a text file containing the prompt",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=SamplingParams.output_path,
            help="Path to save the generated image/video",
        )
        parser.add_argument(
            "--output-file-name",
            type=str,
            default=SamplingParams.output_file_name,
            help="Name of the output file",
        )
        parser.add_argument(
            "--num-outputs-per-prompt",
            type=int,
            default=SamplingParams.num_outputs_per_prompt,
            help="Number of outputs to generate per prompt",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=SamplingParams.seed,
            help="Random seed for generation",
        )
        parser.add_argument(
            "--generator-device",
            type=str,
            default=SamplingParams.generator_device,
            choices=["cuda", "cpu"],
            help="Device for random generator (cuda or cpu). Default: cuda",
        )
        parser.add_argument(
            "--num-frames",
            type=int,
            default=SamplingParams.num_frames,
            help="Number of frames to generate",
        )
        parser.add_argument(
            "--height",
            type=int,
            default=SamplingParams.height,
            help="Height of generated output",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=SamplingParams.width,
            help="Width of generated output",
        )
        # resolution shortcuts
        parser.add_argument(
            "--4k",
            action="store_true",
            dest="resolution_4k",
            help="Set resolution to 4K (3840x2160)",
        )
        parser.add_argument(
            "--2k",
            action="store_true",
            dest="resolution_2k",
            help="Set resolution to 2K (2560x1440)",
        )
        parser.add_argument(
            "--1080p",
            action="store_true",
            dest="resolution_1080p",
            help="Set resolution to 1080p (1920x1080)",
        )
        parser.add_argument(
            "--720p",
            action="store_true",
            dest="resolution_720p",
            help="Set resolution to 720p (1280x720)",
        )

        parser.add_argument(
            "--fps",
            type=int,
            default=SamplingParams.fps,
            help="Frames per second for saved output",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=SamplingParams.num_inference_steps,
            help="Number of denoising steps",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=SamplingParams.guidance_scale,
            help="Classifier-free guidance scale",
        )
        parser.add_argument(
            "--guidance-rescale",
            type=float,
            default=SamplingParams.guidance_rescale,
            help="Guidance rescale factor",
        )
        parser.add_argument(
            "--boundary-ratio",
            type=float,
            default=SamplingParams.boundary_ratio,
            help="Boundary timestep ratio",
        )
        parser.add_argument(
            "--save-output",
            action="store_true",
            default=SamplingParams.save_output,
            help="Whether to save the output to disk",
        )
        parser.add_argument(
            "--no-save-output",
            action="store_false",
            dest="save_output",
            help="Don't save the output to disk",
        )
        parser.add_argument(
            "--return-frames",
            action="store_true",
            default=SamplingParams.return_frames,
            help="Whether to return the raw frames",
        )
        parser.add_argument(
            "--image-path",
            type=str,
            default=SamplingParams.image_path,
            help="Path to input image for image-to-video generation",
        )
        parser.add_argument(
            "--moba-config-path",
            type=str,
            default=None,
            help="Path to a JSON file containing V-MoBA specific configurations.",
        )
        parser.add_argument(
            "--return-trajectory-latents",
            action="store_true",
            default=SamplingParams.return_trajectory_latents,
            help="Whether to return the trajectory",
        )
        parser.add_argument(
            "--return-trajectory-decoded",
            action="store_true",
            default=SamplingParams.return_trajectory_decoded,
            help="Whether to return the decoded trajectory",
        )
        parser.add_argument(
            "--no-override-protected-fields",
            action="store_true",
            default=SamplingParams.no_override_protected_fields,
            help=(
                "If set, disallow user params to override fields defined in subclasses."
            ),
        )
        parser.add_argument(
            "--adjust-frames",
            action=StoreBoolean,
            default=SamplingParams.adjust_frames,
            help=(
                "Enable/disable adjusting num_frames to evenly split latent frames across GPUs "
                "and satisfy model temporal constraints. If disabled, tokens might be padded for SP."
                "Default: true. Examples: --adjust-frames, --adjust-frames true, --adjust-frames false."
            ),
        )
        return parser

    @classmethod
    def get_cli_args(cls, args: argparse.Namespace):
        # handle resolution shortcuts
        if hasattr(args, "resolution_4k") and args.resolution_4k:
            args.width = 3840
            args.height = 2160
        elif hasattr(args, "resolution_2k") and args.resolution_2k:
            args.width = 2560
            args.height = 1440
        elif hasattr(args, "resolution_1080p") and args.resolution_1080p:
            args.width = 1920
            args.height = 1080
        elif hasattr(args, "resolution_720p") and args.resolution_720p:
            args.width = 1280
            args.height = 720

        attrs = [attr.name for attr in dataclasses.fields(cls)]
        args.height_not_provided = False
        args.width_not_provided = False
        return {attr: getattr(args, attr) for attr in attrs}

    def output_file_path(self):
        return os.path.join(self.output_path, self.output_file_name)

    def _merge_with_user_params(self, user_params: "SamplingParams"):
        """
        Merges parameters from a user-provided SamplingParams object.
        """
        if user_params is None:
            return

        predefined_fields = set(type(self).__annotations__.keys())

        # global switch: if True, allow overriding protected fields
        allow_override_protected = not user_params.no_override_protected_fields
        for field in dataclasses.fields(user_params):
            field_name = field.name
            user_value = getattr(user_params, field_name)
            default_class_value = getattr(SamplingParams, field_name)

            # A field is considered user-modified if its value is different from the default
            is_user_modified = user_value != default_class_value
            is_protected_field = field_name in predefined_fields
            if is_user_modified and (
                allow_override_protected or not is_protected_field
            ):
                setattr(self, field_name, user_value)
        self.height_not_provided = user_params.height_not_provided
        self.width_not_provided = user_params.width_not_provided
        self.__post_init__()

    @property
    def n_tokens(self) -> int:
        # Calculate latent sizes
        if self.height and self.width:
            latents_size = [
                (self.num_frames - 1) // 4 + 1,
                self.height // 8,
                self.width // 8,
            ]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        else:
            n_tokens = -1
        return n_tokens

    def output_file_path(self):
        return os.path.join(self.output_path, self.output_file_name)

    def log(self, server_args: ServerArgs):
        # TODO: in some cases (e.g., TI2I), height and weight might be undecided at this moment
        if self.height:
            target_height = align_to(self.height, 16)
        else:
            target_height = -1
        if self.width:
            target_width = align_to(self.width, 16)
        else:
            target_width = -1

        # Log sampling parameters
        debug_str = f"""Sampling params:
                       width: {target_width}
                      height: {target_height}
                  num_frames: {self.num_frames}
                      prompt: {self.prompt}
                  neg_prompt: {self.negative_prompt}
                        seed: {self.seed}
                 infer_steps: {self.num_inference_steps}
      num_outputs_per_prompt: {self.num_outputs_per_prompt}
              guidance_scale: {self.guidance_scale}
     embedded_guidance_scale: {server_args.pipeline_config.embedded_cfg_scale}
                    n_tokens: {self.n_tokens}
                  flow_shift: {server_args.pipeline_config.flow_shift}
                  image_path: {self.image_path}
                 save_output: {self.save_output}
            output_file_path: {self.output_file_path()}
        """  # type: ignore[attr-defined]
        logger.info(debug_str)


@dataclass
class CacheParams:
    cache_type: str = "none"

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
import hashlib
import json
import os.path
import re
import time
import unicodedata
import uuid
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import align_to

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
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    boundary_ratio: float | None = None

    # TeaCache parameters
    enable_teacache: bool = False

    # Profiling
    profile: bool = False
    num_profiled_timesteps: int = 2

    # Debugging
    debug: bool = False

    # Misc
    save_output: bool = True
    return_frames: bool = False
    return_trajectory_latents: bool = False  # returns all latents for each timestep
    return_trajectory_decoded: bool = False  # returns decoded latents for each timestep

    def set_output_file_ext(self):
        # add extension if needed
        if not any(
            self.output_file_name.endswith(ext)
            for ext in [".mp4", ".jpg", ".png", ".webp"]
        ):
            self.output_file_name = (
                f"{self.output_file_name}.{self.data_type.get_default_extension()}"
            )

    def set_output_file_name(self):
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
        self.set_output_file_ext()

    def __post_init__(self) -> None:
        assert self.num_frames >= 1
        self.data_type = DataType.VIDEO if self.num_frames > 1 else DataType.IMAGE

        if self.width is None:
            self.width_not_provided = True
            self.width = 1280
        if self.height is None:
            self.height_not_provided = True
            self.height = 720

    def check_sampling_param(self):
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def update(self, source_dict: dict[str, Any]) -> None:
        for key, value in source_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.exception("%s has no attribute %s", type(self).__name__, key)

        self.__post_init__()

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "SamplingParams":
        from sglang.multimodal_gen.registry import get_model_info

        model_info = get_model_info(model_path)
        logger.debug(f"Found model info: {model_info}")
        if model_info is not None:
            sampling_params: SamplingParams = model_info.sampling_param_cls(**kwargs)
        else:
            logger.warning(
                "Couldn't find an optimal sampling param for %s. Using the default sampling param.",
                model_path,
            )
            sampling_params = cls(**kwargs)
        return sampling_params

    def from_user_sampling_params(self, user_params):
        sampling_params = deepcopy(self)
        sampling_params._merge_with_user_params(user_params)
        return sampling_params

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
        parser.add_argument(
            "--profile",
            action="store_true",
            default=SamplingParams.profile,
            help="Enable torch profiler for denoising stage",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=SamplingParams.debug,
            help="",
        )
        parser.add_argument(
            "--num-profiled-timesteps",
            type=int,
            default=SamplingParams.num_profiled_timesteps,
            help="Number of timesteps to profile after warmup",
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
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        args.height_not_provided = False
        args.width_not_provided = False
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def output_file_path(self):
        return os.path.join(self.output_path, self.output_file_name)

    def _merge_with_user_params(self, user_params):
        """
        Merges parameters from a user-provided SamplingParams object.

        This method updates the current object with values from `user_params`,
        but skips any fields that are explicitly defined in the current object's
        subclass. This is to preserve model-specific optimal parameters.
        It also skips fields that the user has not changed from the default
        in `user_params`.
        """
        if user_params is None:
            return

        # Get fields defined directly in the subclass (not inherited)
        subclass_defined_fields = set(type(self).__annotations__.keys())

        # Compare against current instance to avoid constructing a default instance
        default_params = SamplingParams()

        for field in dataclasses.fields(user_params):
            field_name = field.name
            user_value = getattr(user_params, field_name)
            default_value = getattr(default_params, field_name)

            # A field is considered user-modified if its value is different from
            # the default, with an exception for `output_file_name` which is
            # auto-generated with a random component.
            is_user_modified = (
                user_value != default_value
                if field_name != "output_file_name"
                else user_params.output_file_path is not None
            )
            if is_user_modified and field_name not in subclass_defined_fields:
                if hasattr(self, field_name):
                    setattr(self, field_name, user_value)

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
                      height: {target_height}
                       width: {target_width}
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

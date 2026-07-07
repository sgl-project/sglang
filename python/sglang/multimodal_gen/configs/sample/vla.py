# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    _sanitize_filename,
)
from sglang.multimodal_gen.utils import StoreBoolean, expand_path_fields

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


@dataclass
class VLASamplingParams:
    """Sampling parameters for VLA/action-generation policies."""

    data_type: DataType = DataType.ACTION
    request_id: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    prompt: str | list[str] | None = field(
        default="", metadata={"batch_sig_exclude": True}
    )
    num_outputs_per_prompt: int = 1
    seed: int | list[int] = field(default=42, metadata={"batch_sig_exclude": True})
    generator_device: str | None = None
    num_inference_steps: int = 10

    output_path: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    output_file_name: str | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    save_output: bool = False
    return_file_paths_only: bool = False

    profile: bool = field(default=False, metadata={"batch_sig_exclude": True})
    num_profiled_timesteps: int = field(default=5, metadata={"batch_sig_exclude": True})
    profile_all_stages: bool = field(
        default=False, metadata={"batch_sig_exclude": True}
    )
    debug: bool = field(default=False, metadata={"batch_sig_exclude": True})
    perf_dump_path: str | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    suppress_logs: bool = field(default=False, metadata={"batch_sig_exclude": True})

    enable_sequence_shard: bool | None = None
    max_sequence_length: int | None = None
    no_override_protected_fields: bool = field(
        default=False, metadata={"batch_sig_exclude": True}
    )

    def __post_init__(self) -> None:
        self.data_type = DataType.ACTION
        self._validate()

        env_steps = os.environ.get("SGLANG_TEST_NUM_INFERENCE_STEPS")
        if env_steps is not None and self.num_inference_steps is not None:
            self.num_inference_steps = int(env_steps)

    def build_request_extra(self) -> dict[str, Any]:
        extra = {}
        diffusers_kwargs = getattr(self, "diffusers_kwargs", None)
        if diffusers_kwargs:
            extra["diffusers_kwargs"] = diffusers_kwargs
        explicit_fields = getattr(self, "_explicit_fields", None)
        if explicit_fields is not None:
            extra["explicit_fields"] = sorted(explicit_fields)
        return extra

    def apply_request_extra(self, req: Any) -> None:
        req.extra.update(self.build_request_extra())

    def _validate(self):
        if (
            not isinstance(self.num_outputs_per_prompt, int)
            or self.num_outputs_per_prompt <= 0
        ):
            raise ValueError(
                "num_outputs_per_prompt must be a positive int, "
                f"got {self.num_outputs_per_prompt!r}"
            )

        if isinstance(self.seed, list):
            if not self.seed:
                raise ValueError("seed list must not be empty")
            for seed in self.seed:
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ValueError(
                        f"seed list must contain non-negative ints, got {self.seed!r}"
                    )
        elif (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                f"seed must be a non-negative int or list of ints, got {self.seed!r}"
            )

        if (
            not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps <= 0
        ):
            raise ValueError(
                "num_inference_steps must be a positive int, "
                f"got {self.num_inference_steps!r}"
            )

        if self.generator_device not in (None, "cuda", "musa", "cpu"):
            raise ValueError(
                "generator_device must be one of None, 'cuda', 'musa', or 'cpu', "
                f"got {self.generator_device!r}"
            )

    def _validate_with_pipeline_config(self, pipeline_config):
        if pipeline_config.task_type.data_type() != DataType.ACTION:
            raise ValueError(
                f"VLASamplingParams requires an ACTION pipeline, got {pipeline_config.task_type.name}"
            )

    def _adjust(self, server_args: "ServerArgs"):
        expand_path_fields(self)
        self.data_type = DataType.ACTION
        self.return_file_paths_only = False
        if self.output_path is None and server_args.output_path is not None:
            self.output_path = server_args.output_path
        if self.output_path is None:
            self.save_output = False
        if self.save_output and not server_args.comfyui_mode:
            self._set_output_file_name()

    def _set_output_file_ext(self):
        if self.output_file_name and not self.output_file_name.endswith(".json"):
            self.output_file_name = f"{self.output_file_name}.json"

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "vla_action"
        self.output_file_name = _sanitize_filename(self.output_file_name)
        self._set_output_file_ext()

    def output_file_path(self):
        if self.output_path is None or self.output_file_name is None:
            return None
        return os.path.join(self.output_path, self.output_file_name)

    def _merge_with_user_params(
        self,
        user_params: "VLASamplingParams",
        explicit_fields: set[str] | None = None,
    ):
        if user_params is None:
            return

        predefined_fields = set(type(self).__annotations__.keys())
        allow_override_protected = not user_params.no_override_protected_fields
        for field_info in dataclasses.fields(user_params):
            field_name = field_info.name
            user_value = getattr(user_params, field_name)
            if field_info.default is not dataclasses.MISSING:
                default_class_value = field_info.default
            elif field_info.default_factory is not dataclasses.MISSING:
                default_class_value = field_info.default_factory()
            else:
                default_class_value = dataclasses.MISSING

            if explicit_fields is not None:
                is_user_modified = field_name in explicit_fields
            else:
                is_user_modified = user_value != default_class_value
            is_protected_field = field_name in predefined_fields
            if is_user_modified and (
                allow_override_protected or not is_protected_field
            ):
                setattr(self, field_name, user_value)

        if explicit_fields is not None:
            self._explicit_fields = set(explicit_fields)
        self.__post_init__()

    @staticmethod
    def add_cli_args(parser: Any) -> Any:
        def add_argument(*name_or_flags, **kwargs):
            kwargs.setdefault("default", argparse.SUPPRESS)
            return parser.add_argument(*name_or_flags, **kwargs)

        add_argument(
            "--prompt",
            type=str,
            nargs="+",
            help="Language instruction(s) for the VLA policy.",
        )
        add_argument(
            "--num-inference-steps",
            type=int,
            help="Number of action denoising steps.",
        )
        add_argument(
            "--num-outputs-per-prompt",
            type=int,
            help="Number of candidate actions to generate per observation.",
        )
        add_argument(
            "--seed",
            type=int,
            nargs="+",
            help="Random seed for action noise generation.",
        )
        add_argument(
            "--generator-device",
            type=str,
            choices=["cuda", "musa", "cpu"],
            help="Device for random generator. Default: use the model-specific setting.",
        )
        add_argument(
            "--profile",
            action="store_true",
            help="Enable torch profiler for action denoising.",
        )
        add_argument(
            "--num-profiled-timesteps",
            type=int,
            help="Number of denoising timesteps to profile after warmup.",
        )
        add_argument(
            "--profile-all-stages",
            action="store_true",
            dest="profile_all_stages",
            help="Used with --profile, profile all pipeline stages.",
        )
        add_argument("--debug", action="store_true")
        add_argument(
            "--enable-sequence-shard",
            action=StoreBoolean,
            help="Enable sequence dimension shard with sequence parallelism.",
        )
        add_argument(
            "--max-sequence-length",
            type=int,
            help="Maximum prefix sequence length.",
        )
        add_argument(
            "--no-override-protected-fields",
            action="store_true",
            help="If set, disallow user params to override subclass-defined fields.",
        )
        return parser

    @classmethod
    def get_cli_args(cls, args: argparse.Namespace):
        sampling_params_fields = {attr.name for attr in dataclasses.fields(cls)}
        args_attrs = set(vars(args).keys())
        attrs = sampling_params_fields & args_attrs
        cli_args = {
            attr: getattr(args, attr)
            for attr in attrs
            if hasattr(args, attr) and getattr(args, attr) is not None
        }
        if isinstance(cli_args.get("seed"), list) and len(cli_args["seed"]) == 1:
            cli_args["seed"] = cli_args["seed"][0]
        return cli_args

    def output_size_str(self) -> str:
        return "action"

    def seconds(self) -> float:
        return 0.0

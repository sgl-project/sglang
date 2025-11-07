# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of sgl-diffusion Inference."""
import argparse
import dataclasses
import inspect
import json
import random
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from typing import Any, Optional

from sglang.multimodal_gen.configs.configs import PreprocessConfig
from sglang.multimodal_gen.configs.pipelines import FluxPipelineConfig
from sglang.multimodal_gen.configs.pipelines.base import PipelineConfig, STA_Mode
from sglang.multimodal_gen.configs.pipelines.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.common import (
    is_port_available,
    is_valid_ipv6_address,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.utils import FlexibleArgumentParser, StoreBoolean

logger = init_logger(__name__)

ZMQ_TCP_PORT_DELTA = 233


def _is_torch_tensor(obj: Any) -> tuple[bool, Any]:
    """Return (is_tensor, torch_module_or_None) without importing torch at module import time."""
    try:
        import torch  # type: ignore

        return isinstance(obj, torch.Tensor), torch
    except Exception:
        return False, None


def _sanitize_for_logging(obj: Any, key_hint: str | None = None) -> Any:
    """Recursively convert objects to JSON-serializable forms for concise logging.

    Rules:
    - Drop any field/dict key named 'param_names_mapping'.
    - Render Enums using their value.
    - Render torch.Tensor as a compact summary; if key name is 'scaling_factor', include stats.
    - Dataclasses are expanded to dicts and sanitized recursively.
    - Callables/functions are rendered as their qualified name.
    - Fallback to str(...) for unknown types.
    """
    # Handle simple types quickly
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Enum -> value for readability
    if isinstance(obj, Enum):
        return obj.value

    # torch.Tensor handling (lazy import)
    is_tensor, torch_mod = _is_torch_tensor(obj)
    if is_tensor:
        try:
            ten = obj.detach().cpu()
            if key_hint == "scaling_factor":
                # Provide a compact, single-line summary for scaling_factor
                stats = {
                    "shape": list(ten.shape),
                    "dtype": str(ten.dtype),
                }
                # Stats might fail for some dtypes; guard individually
                try:
                    stats["min"] = float(ten.min().item())
                except Exception:
                    pass
                try:
                    stats["max"] = float(ten.max().item())
                except Exception:
                    pass
                try:
                    stats["mean"] = float(ten.float().mean().item())
                except Exception:
                    pass
                return {"tensor": "scaling_factor", **stats}
            # Generic tensor summary
            return {"tensor": True, "shape": list(ten.shape), "dtype": str(ten.dtype)}
        except Exception:
            return "<tensor>"

    # Dataclasses -> dict
    if dataclasses.is_dataclass(obj):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            if not f.repr:
                continue
            name = f.name
            if "names_mapping" in name:  # drop noisy mappings
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            result[name] = _sanitize_for_logging(value, key_hint=name)
        return result

    # Dicts -> sanitize keys/values; drop 'param_names_mapping'
    if isinstance(obj, dict):
        result_dict: dict[str, Any] = {}
        for k, v in obj.items():
            try:
                key_str = str(k)
            except Exception:
                key_str = "<key>"
            if key_str == "param_names_mapping":
                continue
            result_dict[key_str] = _sanitize_for_logging(v, key_hint=key_str)
        return result_dict

    # Sequences/Sets -> list
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_logging(x) for x in obj]

    # Functions / Callables -> qualified name
    try:
        if inspect.isroutine(obj) or inspect.isclass(obj):
            module = getattr(obj, "__module__", "")
            qn = getattr(obj, "__qualname__", getattr(obj, "__name__", "<callable>"))
            return f"{module}.{qn}" if module else qn
    except Exception:
        pass

    # Fallback: string representation
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


class ExecutionMode(str, Enum):
    """
    Enumeration for different pipeline modes.

    Inherits from str to allow string comparison for backward compatibility.
    """

    INFERENCE = "inference"
    PREPROCESS = "preprocess"
    FINETUNING = "finetuning"
    DISTILLATION = "distillation"

    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Convert string to ExecutionMode enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [mode.value for mode in cls]


class WorkloadType(str, Enum):
    """
    Enumeration for different workload types.

    Inherits from str to allow string comparison for backward compatibility.
    """

    I2V = "i2v"  # Image to Video
    T2V = "t2v"  # Text to Video
    T2I = "t2i"  # Text to Image
    I2I = "i2i"  # Image to Image

    @classmethod
    def from_string(cls, value: str) -> "WorkloadType":
        """Convert string to WorkloadType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid workload type: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [workload.value for workload in cls]


# args for sgl_diffusion framework
@dataclasses.dataclass
class ServerArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Attention
    attention_backend: str = None

    # Running mode
    mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy
    cache_strategy: str = "none"

    # Distributed executor backend
    distributed_executor_backend: str = "mp"
    nccl_port: Optional[int] = None

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_degree: int = -1
    # sequence parallelism
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
    # data parallelism
    # number of data parallelism groups
    dp_size: int = 1
    # number of gpu in a dp group
    dp_degree: int = 1
    # cfg parallel
    enable_cfg_parallel: bool = False

    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # timeout for torch.distributed

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)
    preprocess_config: PreprocessConfig | None = None

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = False
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed inference
    # TODO: do not hard code
    master_port: int | None = None

    # http server endpoint config, would be ignored in local mode
    host: str | None = None
    port: int | None = None

    scheduler_port: int = 5555

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_file_path: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
        }
    )
    override_transformer_cls_name: str | None = None

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = None

    # Logging
    log_level: str = "info"

    @property
    def broker_port(self) -> int:
        return self.port + 1

    @property
    def is_local_mode(self) -> bool:
        """
        If no server is running when a generation task begins, 'local_mode' will be enabled: a dedicated server will be launched
        """
        return self.host is None or self.port is None

    def __post_init__(self):
        self.scheduler_port = self.settle_port(self.scheduler_port)
        # TODO: remove hard code
        self.master_port = self.settle_port(self.master_port or 30005, 37)
        if self.moba_config_path:
            try:
                with open(self.moba_config_path) as f:
                    self.moba_config = json.load(f)
                logger.info("Loaded V-MoBA config from %s", self.moba_config_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(
                    "Failed to load V-MoBA config from %s: %s", self.moba_config_path, e
                )
                raise
        self.check_server_args()

        configure_logger(server_args=self)

        # log clean server_args
        try:
            safe_args = _sanitize_for_logging(self, key_hint="server_args")
            logger.info("server_args: %s", json.dumps(safe_args, ensure_ascii=False))
        except Exception:
            # Fallback to default repr if sanitization fails
            logger.info(f"server_args: {self}")

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )

        # attention
        parser.add_argument(
            "--attention-backend",
            type=str,
            default=None,
            choices=[e.name.lower() for e in AttentionBackendEnum],
            help="The attention backend to use. If not specified, the backend is automatically selected based on hardware and installed packages.",
        )

        # Running mode
        parser.add_argument(
            "--mode",
            type=str,
            choices=ExecutionMode.choices(),
            default=ServerArgs.mode.value,
            help="The mode to run sgl-diffusion",
        )

        # Workload type
        parser.add_argument(
            "--workload-type",
            type=str,
            choices=WorkloadType.choices(),
            default=ServerArgs.workload_type.value,
            help="The workload type",
        )

        # distributed_executor_backend
        parser.add_argument(
            "--distributed-executor-backend",
            type=str,
            choices=["mp"],
            default=ServerArgs.distributed_executor_backend,
            help="The distributed executor backend to use",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=ServerArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=ServerArgs.revision,
            help="The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=ServerArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sp-degree",
            type=int,
            default=ServerArgs.sp_degree,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--ulysses-degree",
            type=int,
            default=ServerArgs.ulysses_degree,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--ring-degree",
            type=int,
            default=ServerArgs.ring_degree,
            help="Ring sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--enable-cfg-parallel",
            action="store_true",
            default=ServerArgs.enable_cfg_parallel,
            help="Enable cfg parallel.",
        )
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            "--dp",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )

        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=ServerArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=ServerArgs.hsdp_shard_dim,
            help="The data parallelism shards.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Output type
        parser.add_argument(
            "--output-type",
            type=str,
            default=ServerArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )

        # Prompt text file for batch processing
        parser.add_argument(
            "--prompt-file-path",
            type=str,
            default=ServerArgs.prompt_file_path,
            help="Path to a text file containing prompts (one per line) for batch processing",
        )

        # STA (Sliding Tile Attention) parameters
        parser.add_argument(
            "--STA-mode",
            type=str,
            default=ServerArgs.STA_mode.value,
            choices=[mode.value for mode in STA_Mode],
            help="STA mode contains STA_inference, STA_searching, STA_tuning, STA_tuning_cfg, None",
        )
        parser.add_argument(
            "--skip-time-steps",
            type=int,
            default=ServerArgs.skip_time_steps,
            help="Number of time steps to warmup (full attention) for STA",
        )
        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=ServerArgs.enable_torch_compile,
            help="Use torch.compile to speed up DiT inference."
            + "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )

        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help="Use FSDP for inference by sharding the model weights. Latency is very low due to prefetch--enable if run out of memory.",
        )
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help='Pin memory for CPU offload. Only added as a temp workaround if it throws "CUDA error: invalid argument". '
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help="Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # VSA parameters
        parser.add_argument(
            "--VSA-sparsity",
            type=float,
            default=ServerArgs.VSA_sparsity,
            help="Validation sparsity for VSA",
        )

        # Master port for distributed inference
        parser.add_argument(
            "--master-port",
            type=int,
            default=ServerArgs.master_port,
            help="Master port for distributed inference. If not set, a random free port will be used.",
        )
        parser.add_argument(
            "--scheduler-port",
            type=int,
            default=ServerArgs.scheduler_port,
            help="Port for the scheduler server.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="Host for the HTTP API server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="Port for the HTTP API server.",
        )

        # Stage verification
        parser.add_argument(
            "--enable-stage-verification",
            action=StoreBoolean,
            default=ServerArgs.enable_stage_verification,
            help="Enable input/output verification for pipeline stages",
        )
        parser.add_argument(
            "--override-transformer-cls-name",
            type=str,
            default=ServerArgs.override_transformer_cls_name,
            help="Override transformer cls name",
        )
        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

        # Add preprocessing configuration arguments
        PreprocessConfig.add_cli_args(parser)

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        return parser

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def scheduler_endpoint(self):
        """
        Internal endpoint for scheduler

        """
        scheduler_host = self.host or "localhost"
        return f"tcp://{scheduler_host}:{self.scheduler_port}"

    def settle_port(self, port: int, port_inc: int = 42) -> int:
        while True:
            if is_port_available(port):
                return port
            if port < 60000:
                port += port_inc
            else:
                port -= port_inc + 1

    def post_init_serve(self):
        """
        Post init when in serve mode
        """
        if self.host is None:
            self.host = "localhost"
        if self.port is None:
            self.port = 3000
        self.port = self.settle_port(self.port)

    @classmethod
    def from_cli_args(
        cls, args: argparse.Namespace, unknown_args: list[str] | None = None
    ) -> "ServerArgs":
        if unknown_args is None:
            unknown_args = []
        provided_args = cls.get_provided_args(args, unknown_args)

        # Handle config file
        config_file = provided_args.get("config")
        if config_file:
            config_args = cls.load_config_file(config_file)
            # Provided args override config file args
            provided_args = {**config_args, **provided_args}

        # Handle special cases
        # if "tp_size" in provided_args:
        #     provided_args["tp"] = provided_args.pop("tp_size")

        return cls.from_dict(provided_args)

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> "ServerArgs":
        """Create a ServerArgs object from a dictionary."""
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        server_args_kwargs: dict[str, Any] = {}

        for attr in attrs:
            if attr == "pipeline_config":
                pipeline_config = PipelineConfig.from_kwargs(kwargs)
                logger.debug(f"Using PipelineConfig: {type(pipeline_config)}")
                server_args_kwargs["pipeline_config"] = pipeline_config
            elif attr == "preprocess_config":
                preprocess_config = PreprocessConfig.from_kwargs(kwargs)
                server_args_kwargs["preprocess_config"] = preprocess_config
            elif attr in kwargs:
                server_args_kwargs[attr] = kwargs[attr]

        return cls(**server_args_kwargs)

    @staticmethod
    def load_config_file(config_file: str) -> dict[str, Any]:
        """Load a config file."""
        if config_file.endswith(".json"):
            with open(config_file, "r") as f:
                return json.load(f)
        elif config_file.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ServerArgs":
        # Convert mode string to enum if necessary
        if "mode" in kwargs and isinstance(kwargs["mode"], str):
            kwargs["mode"] = ExecutionMode.from_string(kwargs["mode"])

        # Convert workload_type string to enum if necessary
        if "workload_type" in kwargs and isinstance(kwargs["workload_type"], str):
            kwargs["workload_type"] = WorkloadType.from_string(kwargs["workload_type"])

        kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
        kwargs["preprocess_config"] = PreprocessConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    @staticmethod
    def get_provided_args(
        args: argparse.Namespace, unknown_args: list[str]
    ) -> dict[str, Any]:
        """Get the arguments provided by the user."""
        provided_args = {}
        # We need to check against the raw command-line arguments to see what was
        # explicitly provided by the user, vs. what's a default value from argparse.
        raw_argv = sys.argv + unknown_args

        # Create a set of argument names that were present on the command line.
        # This handles both styles: '--arg=value' and '--arg value'.
        provided_arg_names = set()
        for arg in raw_argv:
            if arg.startswith("--"):
                # For '--arg=value', this gets 'arg'; for '--arg', this also gets 'arg'.
                arg_name = arg.split("=", 1)[0].replace("-", "_").lstrip("_")
                provided_arg_names.add(arg_name)

        # Populate provided_args if the argument from the namespace was on the command line.
        for k, v in vars(args).items():
            if k in provided_arg_names:
                provided_args[k] = v

        return provided_args

    def check_server_sp_args(self):

        if self.pipeline_config.is_image_gen:
            if (
                (self.sp_degree and self.sp_degree > 1)
                or (self.ulysses_degree and self.ulysses_degree > 1)
                or (self.ring_degree and self.ring_degree > 1)
            ):
                raise ValueError(
                    "SP is not supported for image generation models for now"
                )
            self.sp_degree = self.ulysses_degree = self.ring_degree = 1

        if (
            self.ring_degree is not None
            and self.ring_degree > 1
            and self.attention_backend != "fa3"
        ):
            raise ValueError("Ring Attention is only supported for fa3 backend for now")

        if self.sp_degree == -1:
            # assume we leave all remaining gpus to sp
            num_gpus_per_group = self.dp_size * self.tp_size
            if self.enable_cfg_parallel:
                num_gpus_per_group *= 2
            if self.num_gpus % num_gpus_per_group != 0:
                raise ValueError(f"{self.num_gpus=} % {num_gpus_per_group} != 0")
            self.sp_degree = self.num_gpus // num_gpus_per_group

        if (
            self.ulysses_degree is None
            and self.ring_degree is None
            and self.sp_degree != 1
        ):
            self.ulysses_degree = self.sp_degree
            logger.info(
                f"Automatically set ulysses_degree=sp_degree={self.ulysses_degree} for best performance"
            )

        if self.ulysses_degree is None:
            self.ulysses_degree = 1
            logger.info(
                f"Ulysses degree not set, " f"using default value {self.ulysses_degree}"
            )

        if self.ring_degree is None:
            self.ring_degree = 1
            logger.info(
                f"Ring degree not set, " f"using default value {self.ring_degree}"
            )

        if self.sp_degree == -1:
            self.sp_degree = self.ring_degree * self.ulysses_degree
            logger.info(
                f"sequence_parallel_degree is not provided, using ring_degree * ulysses_degree = {self.sp_degree}"
            )

        if self.sp_degree != self.ring_degree * self.ulysses_degree:
            raise ValueError(
                f"sequence_parallel_degree is not equal to ring_degree * ulysses_degree, {self.sp_degree} != {self.ring_degree} * {self.ulysses_degree}"
            )

    def check_server_dp_args(self):
        assert self.num_gpus % self.dp_size == 0, f"{self.num_gpus=}, {self.dp_size=}"
        assert self.dp_size >= 1, "--dp-size must be natural number"
        self.dp_degree = self.num_gpus // self.dp_size
        logger.info(f"Setting dp_degree to: {self.dp_degree}")

    def check_server_args(self) -> None:
        """Validate inference arguments for consistency"""
        if current_platform.is_mps():
            self.use_fsdp_inference = False

        # autocast
        is_flux = (
            isinstance(self.pipeline_config, FluxPipelineConfig)
            or isinstance(self.pipeline_config, QwenImagePipelineConfig)
            or isinstance(self.pipeline_config, QwenImageEditPipelineConfig)
        )
        if is_flux:
            self.disable_autocast = True

        # Validate mode consistency
        assert isinstance(
            self.mode, ExecutionMode
        ), f"Mode must be an ExecutionMode enum, got {type(self.mode)}"
        assert (
            self.mode in ExecutionMode.choices()
        ), f"Invalid execution mode: {self.mode}"

        # Validate workload type
        assert isinstance(
            self.workload_type, WorkloadType
        ), f"Workload type must be a WorkloadType enum, got {type(self.workload_type)}"
        assert (
            self.workload_type in WorkloadType.choices()
        ), f"Invalid workload type: {self.workload_type}"

        if self.tp_size == -1:
            self.tp_size = 1

        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_gpus

        assert (
            self.sp_degree <= self.num_gpus and self.num_gpus % self.sp_degree == 0
        ), "num_gpus must >= and be divisible by sp_size"
        assert (
            self.hsdp_replicate_dim <= self.num_gpus
            and self.num_gpus % self.hsdp_replicate_dim == 0
        ), "num_gpus must >= and be divisible by hsdp_replicate_dim"
        assert (
            self.hsdp_shard_dim <= self.num_gpus
            and self.num_gpus % self.hsdp_shard_dim == 0
        ), "num_gpus must >= and be divisible by hsdp_shard_dim"

        if self.num_gpus < max(self.tp_size, self.sp_degree):
            self.num_gpus = max(self.tp_size, self.sp_degree)

        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in ServerArgs")

        self.pipeline_config.check_pipeline_config()

        # Add preprocessing config validation if needed
        if self.mode == ExecutionMode.PREPROCESS:
            if self.preprocess_config is None:
                raise ValueError(
                    "preprocess_config is not set in ServerArgs when mode is PREPROCESS"
                )
            if self.preprocess_config.model_path == "":
                self.preprocess_config.model_path = self.model_path
            if not self.pipeline_config.vae_config.load_encoder:
                self.pipeline_config.vae_config.load_encoder = True
            self.preprocess_config.check_preprocess_config()

        # parallelism
        self.check_server_dp_args()
        # allocate all remaining gpus for sp-size
        self.check_server_sp_args()

        if self.enable_cfg_parallel:
            if self.num_gpus == 1:
                raise ValueError(
                    "CFG Parallelism is enabled via `--enable-cfg-parallel`, while -num-gpus==1"
                )


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # Master port for distributed inference
    master_port: int | None = None

    @staticmethod
    def from_server_args(
        server_args: ServerArgs, dp_rank: Optional[int] = None
    ) -> "PortArgs":
        if server_args.nccl_port is None:
            nccl_port = server_args.scheduler_port + random.randint(100, 1000)
            while True:
                if is_port_available(nccl_port):
                    break
                if nccl_port < 60000:
                    nccl_port += 42
                else:
                    nccl_port -= 43
        else:
            nccl_port = server_args.nccl_port

        # Normal case, use IPC within a single node
        return PortArgs(
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            master_port=server_args.master_port,
        )


# TODO: not sure what _current_server_args is for, using a _global_server_args instead
_current_server_args = None
_global_server_args = None


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    global _current_server_args
    _current_server_args = server_args
    return server_args


@contextmanager
def set_current_server_args(server_args: ServerArgs):
    """
    Temporarily set the current sgl_diffusion config.
    Used during model initialization.
    We save the current sgl_diffusion config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the sgl_diffusion config to determine how to dispatch.
    """
    global _current_server_args
    old_server_args = _current_server_args
    try:
        _current_server_args = server_args
        yield
    finally:
        _current_server_args = old_server_args


def set_global_server_args(server_args: ServerArgs):
    """
    Set the global sgl_diffusion config for each process
    """
    global _global_server_args
    _global_server_args = server_args


def get_current_server_args() -> ServerArgs:
    if _current_server_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the sgl_diffusion config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current sgl_diffusion args is not set.")
    return _current_server_args


def get_global_server_args() -> ServerArgs:
    if _global_server_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the sgl_diffusion config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Global sgl_diffusion args is not set.")
    return _global_server_args


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]

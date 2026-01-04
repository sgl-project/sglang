# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of sglang-diffusion Inference."""

import argparse
import dataclasses
import inspect
import json
import os
import random
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from typing import Any, Optional

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig, STA_Mode
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


@dataclasses.dataclass
class ServerArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Attention
    attention_backend: str = None

    # Distributed executor backend
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

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline

    # VAE parameters
    vae_path: str | None = None  # Custom VAE path (e.g., for distilled autoencoder)
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    # CPU offload parameters
    dit_cpu_offload: bool | None = None
    dit_layerwise_offload: bool | None = None
    text_encoder_cpu_offload: bool | None = None
    image_encoder_cpu_offload: bool | None = None
    vae_cpu_offload: bool | None = None
    use_fsdp_inference: bool = False
    pin_cpu_memory: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False
    enable_warmup: bool = False

    disable_autocast: bool | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed inference
    # TODO: do not hard code
    master_port: int | None = None

    # http server endpoint config
    host: str | None = "127.0.0.1"
    port: int | None = 30000

    # TODO: webui and their endpoint, check if webui_port is available.
    webui: bool = False
    webui_port: int | None = 12312

    scheduler_port: int = 5555

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

    def adjust_offload(self):
        if self.pipeline_config.task_type.is_image_gen():
            logger.info(
                "Disabling some offloading (except dit, text_encoder) for image generation model"
            )
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = False
            if self.vae_cpu_offload is None:
                self.vae_cpu_offload = False
        else:
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = False
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = True
            if self.vae_cpu_offload is None:
                self.vae_cpu_offload = False

    def __post_init__(self):
        # configure logger before use
        configure_logger(server_args=self)

        self.adjust_offload()

        if self.attention_backend in ["fa3", "fa4"]:
            self.attention_backend = "fa"

        # network initialization: port and host
        self.port = self.settle_port(self.port)
        # Add randomization to avoid race condition when multiple servers start simultaneously
        initial_scheduler_port = self.scheduler_port + random.randint(0, 100)
        self.scheduler_port = self.settle_port(initial_scheduler_port)
        # TODO: remove hard code
        initial_master_port = (self.master_port or 30005) + random.randint(0, 100)
        self.master_port = self.settle_port(initial_master_port, 37)
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
            "--vae-path",
            type=str,
            default=ServerArgs.vae_path,
            help="Custom path to VAE model (e.g., for distilled autoencoder). If not specified, VAE will be loaded from the main model path.",
        )

        # attention
        parser.add_argument(
            "--attention-backend",
            type=str,
            default=None,
            choices=[e.name.lower() for e in AttentionBackendEnum] + ["fa3", "fa4"],
            help="The attention backend to use. If not specified, the backend is automatically selected based on hardware and installed packages.",
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
            "--enable-warmup",
            action=StoreBoolean,
            default=ServerArgs.enable_warmup,
            help="Perform a 1-step end-to-end warmup request before the actual request. "
            "Recommended to enable when benchmarking to ensure fair comparison and best performance."
            "When enabled, look for the line ending with `with warmup excluded` for actual processing time.",
        )
        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--dit-layerwise-offload",
            action=StoreBoolean,
            default=ServerArgs.dit_layerwise_offload,
            help="Enable layerwise CPU offload with async H2D prefetch overlap for supported DiT models (e.g., Wan). "
            "Cannot be used together with cache-dit (SGLANG_CACHE_DIT_ENABLED), dit_cpu_offload, or use_fsdp_inference.",
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
        parser.add_argument(
            "--webui",
            action=StoreBoolean,
            default=ServerArgs.webui,
            help="Whether to use webui for better display",
        )

        parser.add_argument(
            "--webui-port",
            type=int,
            default=ServerArgs.webui_port,
            help="Whether to use webui for better display",
        )

        # LoRA
        parser.add_argument(
            "--lora-path",
            type=str,
            default=ServerArgs.lora_path,
            help="The path to the LoRA adapter weights (can be local file path or HF hub id) to launch with",
        )
        parser.add_argument(
            "--lora-nickname",
            type=str,
            default=ServerArgs.lora_nickname,
            help="The nickname for the LoRA adapter to launch with",
        )
        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

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

    @property
    def scheduler_endpoint(self):
        """
        Internal endpoint for scheduler.
        Prefers the configured host but normalizes localhost -> 127.0.0.1 to avoid ZMQ issues.
        """
        scheduler_host = self.host
        if scheduler_host is None or scheduler_host == "localhost":
            scheduler_host = "127.0.0.1"
        return f"tcp://{scheduler_host}:{self.scheduler_port}"

    def settle_port(
        self, port: int, port_inc: int = 42, max_attempts: int = 100
    ) -> int:
        """
        Find an available port with retry logic.

        Args:
            port: Initial port to check
            port_inc: Port increment for each attempt
            max_attempts: Maximum number of attempts to find an available port

        Returns:
            An available port number

        Raises:
            RuntimeError: If no available port is found after max_attempts
        """
        attempts = 0
        original_port = port

        while attempts < max_attempts:
            if is_port_available(port):
                if attempts > 0:
                    logger.info(
                        f"Port {original_port} was unavailable, using port {port} instead"
                    )
                return port

            attempts += 1
            if port < 60000:
                port += port_inc
            else:
                # Wrap around with randomization to avoid collision
                port = 5000 + random.randint(0, 1000)

        raise RuntimeError(
            f"Failed to find available port after {max_attempts} attempts "
            f"(started from port {original_port})"
        )

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
        kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
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
            logger.debug(
                f"Ulysses degree not set, using default value {self.ulysses_degree}"
            )

        if self.ring_degree is None:
            self.ring_degree = 1
            logger.debug(f"Ring degree not set, using default value {self.ring_degree}")

        if self.ring_degree > 1:
            if self.attention_backend is not None and self.attention_backend != "fa":
                raise ValueError(
                    "Ring Attention is only supported for flash attention backend for now"
                )
            else:
                self.attention_backend = "fa"
                logger.info(
                    "Ring Attention is currently only supported for flash attention, attention_backend has been automatically set to flash attention"
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
        # NOTE: disable temporarily
        # self.dp_degree = self.num_gpus // self.dp_size
        logger.debug(f"Setting dp_degree to: {self.dp_degree}")
        if self.dp_size > 1:
            raise ValueError("DP is not yet supported")

    def check_server_args(self) -> None:
        """Validate inference arguments for consistency"""
        if current_platform.is_mps():
            self.use_fsdp_inference = False
            self.dit_layerwise_offload = False

        if self.dit_layerwise_offload:
            if self.use_fsdp_inference:
                logger.warning(
                    "dit_layerwise_offload is enabled, automatically disabling use_fsdp_inference."
                )
                self.use_fsdp_inference = False
            if self.dit_cpu_offload:
                logger.warning(
                    "dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload."
                )
                self.dit_cpu_offload = False
            if os.getenv("SGLANG_CACHE_DIT_ENABLED", "").lower() == "true":
                raise ValueError(
                    "dit_layerwise_offload cannot be enabled together with cache-dit. "
                    "cache-dit may reuse skipped blocks whose weights have been released by layerwise offload, "
                    "causing shape mismatch errors. "
                    "Please disable either --dit-layerwise-offload or SGLANG_CACHE_DIT_ENABLED."
                )

        # autocast
        if self.disable_autocast is None:
            self.disable_autocast = not self.pipeline_config.enable_autocast
        else:
            self.disable_autocast = False

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
        if self.attention_backend is None:
            self._set_default_attention_backend()

        # parallelism
        self.check_server_dp_args()
        # allocate all remaining gpus for sp-size
        self.check_server_sp_args()

        if self.enable_cfg_parallel:
            if self.num_gpus == 1:
                raise ValueError(
                    "CFG Parallelism is enabled via `--enable-cfg-parallel`, while -num-gpus==1"
                )

        if os.getenv("SGLANG_CACHE_DIT_ENABLED", "").lower() == "true":
            has_sp = self.sp_degree > 1
            has_tp = self.tp_size > 1
            if has_sp and has_tp:
                raise ValueError(
                    "cache-dit does not support hybrid parallelism (SP + TP). "
                    "Please use either sequence parallelism or tensor parallelism, not both."
                )

    def _set_default_attention_backend(self) -> None:
        """Configure ROCm defaults when users do not specify an attention backend."""
        if current_platform.is_rocm():
            default_backend = AttentionBackendEnum.AITER.name.lower()
            self.attention_backend = default_backend
            logger.info(
                "Attention backend not specified. Using '%s' by default on ROCm "
                "to match SGLang SRT defaults.",
                default_backend,
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


def get_current_server_args() -> ServerArgs | None:
    if _current_server_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the sgl_diffusion config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current sgl_diffusion args is not set.")
    return _current_server_args


def get_global_server_args() -> ServerArgs | None:
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

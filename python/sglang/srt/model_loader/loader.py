# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/model_executor/model_loader/loader.py

from __future__ import annotations

# ruff: noqa: SIM117
import collections
import dataclasses
import fnmatch
import gc
import glob
import json
import logging
import math
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import huggingface_hub
import numpy as np
import torch

from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    get_remote_instance_transfer_engine_info_per_rank,
    register_memory_region,
)
from sglang.srt.server_args import get_global_server_args

# Try to import accelerate (optional dependency)
try:
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_max_memory

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    infer_auto_device_map = None
    init_empty_weights = None
    get_max_memory = None

from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.connector import (
    ConnectorType,
    create_remote_connector,
    get_connector_type,
)
from sglang.srt.connector.utils import parse_model_name
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    trigger_transferring_weights_request,
)
from sglang.srt.model_loader.utils import (
    get_model_architecture,
    post_load_weights,
    set_default_torch_dtype,
)

# Constants for memory management
DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION = (
    0.8  # Reserve 20% GPU memory headroom for ModelOpt calibration
)
from sglang.srt.environ import envs
from sglang.srt.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_gguf_extra_tensor_names,
    get_quant_config,
    gguf_quant_weights_iterator,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
    set_runai_streamer_env,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_npu,
    is_pin_memory_available,
    rank0_log,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

_is_npu = is_npu()
# ModelOpt: QUANT_CFG_CHOICES is imported from modelopt_utils.py
# which contains the complete mapping of quantization config choices

logger = logging.getLogger(__name__)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_infos: Dict[str, Dict] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_data = p.data
            device_data = p.data.to(target_device)
            original_infos[name] = dict(
                device=p.device,
                original_data=original_data,
                device_data=device_data,
            )
            p.data = device_data
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_infos:
                original_info = original_infos[name]
                device_data = original_info["device_data"]
                original_data = original_info["original_data"]
                original_device: torch.device = original_info["device"]

                if (
                    (device_data.device == p.data.device)
                    and (device_data.data_ptr() == p.data.data_ptr())
                    and (device_data.shape == p.data.shape)
                    and (device_data.dtype == p.data.dtype)
                ):
                    original_data.copy_(p.data.to(original_data.device))
                    p.data = original_data
                elif original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


logger = logging.getLogger(__name__)


def _get_quantization_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    if model_config.quantization is not None:
        quant_config = get_quant_config(
            model_config, load_config, packed_modules_mapping, remap_prefix
        )
        # (yizhang2077) workaround for nvidia/Llama-4-Maverick-17B-128E-Eagle3
        if quant_config is None:
            return None
        if not _is_npu:
            major, minor = get_device_capability()

            if major is not None and minor is not None:
                assert 0 <= minor < 10
                capability = major * 10 + minor
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. "
                        f"Minimum capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )
        return quant_config
    return None


def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class, _ = get_model_architecture(model_config)
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})
    remap_prefix = getattr(model_class, "remap_prefix", None)
    if _is_npu:
        packed_modules_mapping.update(
            {
                "visual": {
                    "qkv_proj": ["qkv"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                },
                "vision_model": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "proj": ["out_proj"],
                },
                "model": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                    "fused_qkv_a_proj_with_mqa": [
                        "q_a_proj",
                        "kv_a_proj_with_mqa",
                    ],
                },
            }
        )

    quant_config = _get_quantization_config(
        model_config, load_config, packed_modules_mapping, remap_prefix
    )
    hf_to_sglang_mapper = getattr(model_class, "hf_to_sglang_mapper", None)
    # pass mappings by reference to quant_config
    if hf_to_sglang_mapper is not None and quant_config is not None:
        quant_config.apply_weight_name_mapper(hf_to_sglang_mapper)

    # Build kwargs conditionally
    kwargs = {
        "config": model_config.hf_config,
        "quant_config": quant_config,
    }

    # Only add sparse head kwargs if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()
    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        kwargs["sparse_head"] = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.get()
        kwargs["model_path"] = model_config.model_path

    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    return model_class(**kwargs)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        model_config: Optional["ModelConfig"] = None
        """The model configuration (for checking architecture, etc)."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

    def _maybe_download_from_modelscope(
        self, model: str, revision: Optional[str]
    ) -> Optional[str]:
        """Download model from ModelScope hub if SGLANG_USE_MODELSCOPE is True.

        Returns the path to the downloaded model, or None if the model is not
        downloaded from ModelScope."""
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return None

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            self._maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == LoadFormat.SAFETENSORS
            or load_format == LoadFormat.FASTSAFETENSORS
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.MISTRAL:
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        elif load_format == LoadFormat.DUMMY:
            raise ValueError(
                f"DUMMY load_format should use DummyModelLoader and not call _prepare_weights"
            )
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        server_args = get_global_server_args()
        if server_args and server_args.model_checksum is not None:
            from sglang.srt.utils.model_file_verifier import verify

            checksums_source = server_args.model_checksum or model_name_or_path
            verify(model_path=hf_folder, checksums_source=checksums_source)

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt
        )

        if use_safetensors and source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files,
                hf_folder,
                "model.safetensors.index.json",
                source.model_config.hf_config,
            )

        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif use_safetensors:
            weight_loader_disable_mmap = (
                get_global_server_args().weight_loader_disable_mmap
            )

            if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                )
            elif extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_safetensors_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                    disable_mmap=weight_loader_disable_mmap,
                )
            else:
                weights_iterator = safetensors_weights_iterator(
                    hf_weights_files, disable_mmap=weight_loader_disable_mmap
                )

        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(hf_weights_files)

        if self.load_config.draft_model_idx is not None:
            import re

            pattern = r"model.mtp.layers.(\d+)."
            filtered_weights = []
            for name, tensor in weights_iterator:
                group = re.match(pattern, name)
                if group is not None:
                    idx = int(group.group(1))
                    if idx != self.load_config.draft_model_idx:
                        continue
                    new_name = name.replace(group.group(), "model.mtp.layers.0.")
                else:
                    new_name = name
                filtered_weights.append((source.prefix + new_name, tensor))
            return tuple(filtered_weights)

        if self.counter_before_loading_weights == 0.0:
            logger.info("Beginning to load weights")
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:

        primary_weights = DefaultModelLoader.Source.init_new(model_config, model)
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source], getattr(model, "secondary_weights", ())
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model_path, model_config.revision, fall_back_to_pt=True
        )

    def _load_modelopt_base_model(self, model_config: ModelConfig) -> nn.Module:
        """Load and prepare the base model for ModelOpt quantization.

        This method handles the common model loading logic shared between
        DefaultModelLoader (conditional) and ModelOptModelLoader (dedicated).
        """
        if not HAS_ACCELERATE:
            raise ImportError(
                "accelerate is required for ModelOpt quantization. "
                "Please install it with: pip install accelerate"
            )

        hf_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=True,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
        with init_empty_weights():
            torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
            model = AutoModelForCausalLM.from_config(
                hf_config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        max_memory = get_max_memory()
        inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

        on_cpu = "cpu" in inferred_device_map.values()
        model_kwargs = {"torch_dtype": "auto"}
        device_map = "auto"

        if on_cpu:
            for device in max_memory.keys():
                if isinstance(device, int):
                    max_memory[device] *= DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION

            logger.warning(
                "Model does not fit to the GPU mem. "
                f"We apply the following memory limit for calibration: \n{max_memory}\n"
                f"If you hit GPU OOM issue, please adjust the memory fraction "
                f"(currently {DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION}) or "
                "reduce the calibration `batch_size` manually."
            )
            model_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            device_map=device_map,
            **model_kwargs,
            trust_remote_code=True,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
        # Handle both legacy modelopt_quant and unified quantization flags
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Legacy approach
            quant_choice_str = model_config.modelopt_quant
            rank0_log(f"ModelOpt quantization requested (legacy): {quant_choice_str}")
        else:
            # Unified approach - extract quantization type
            quant_choice_str = model_config._get_modelopt_quant_type()
            rank0_log(
                f"ModelOpt quantization requested (unified): {model_config.quantization} -> {quant_choice_str}"
            )

        if not isinstance(quant_choice_str, str):
            raise TypeError(
                f"Quantization type must be a string (e.g., 'fp8'), "
                f"got {type(quant_choice_str)}"
            )

        return model

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Load base model using shared method
            model = self._load_modelopt_base_model(model_config)
            # Note: DefaultModelLoader doesn't do additional quantization processing
            # For full ModelOpt quantization, use ModelOptModelLoader
            return model.eval()

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                )

            self.load_weights_and_postprocess(
                model, self._get_all_weights(model_config, model), target_device
            )

        self.counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        return model.eval()

    @staticmethod
    def load_weights_and_postprocess(model, weights, target_device):
        model.load_weights(weights)

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                if _is_npu:
                    torch.npu.empty_cache()


class LayeredModelLoader(DefaultModelLoader):
    """Model loader that loads weights layer by layer so that one can quantize a
    layer before loading another to make the peak memory envelope smaller."""

    def __init__(self, load_config: LoadConfig):
        # Back to the default load format
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
        from sglang.srt.server_args import get_global_server_args

        torchao_config = get_global_server_args().torchao_config
        target_device = torch.device(device_config.device)

        with set_default_torch_dtype(model_config.dtype):
            # Create model on meta device
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                )

            # Check model's layered load support
            if not hasattr(model, "load_weights_to_module"):
                raise ValueError(
                    "LayeredModelLoader requires the model to have a "
                    "`load_weights_to_module` method. "
                    f"{model_config.model_path} does not support it."
                )

            # Get all weights from disk
            weights = self._get_all_weights(model_config, model)

            # Helper function to recursively fill the weights of a module
            def fill_module(module, fqn: List[str], weights):
                """
                fqn: list of strings representing the fully qualified name of `module`.
                """
                # Layer by layer
                for name, submod in module.named_children():
                    fill_module(submod, fqn + [name], weights)

                # First materialize on target device
                module.to_empty(device=target_device, recurse=False)
                fqn_path = ".".join(fqn)
                # Fill weights
                model.load_weights_to_module(
                    fqn_path,
                    weights,
                )
                # Quantize weights if applicable
                if torchao_config and "proj" in fqn_path:
                    # Note: `None` here is needed to indicate no filter, see
                    # `apply_torchao_config_to_model` for details.
                    apply_torchao_config_to_model(module, torchao_config, None)

            # Start calling on root module
            fill_module(model, [], weights)

        if torchao_config:
            model.torchao_applied = True

        return model.eval()


class QuantizedRLModelLoader(DefaultModelLoader):
    """
    Model loader for RL training with FP8 quantization (profile-free, native SGLang).

    Workflow:
      1. Initial load: Load base model → Record state → Apply FP8 quantization
      2. Training Actor in full precision
      3. Reload: Trainer sends full precision weights → Quantize to FP8 → Copy to original memory
      4. Use torch.as_strided to preserve memory locations across reloads

    Usage:
      --model-path Qwen/Qwen2.5-7B --quantization fp8 --load-format flash_rl
    """

    # Parameter attributes to record for weight reloading
    RECORDED_LOADER_KEYS = [
        "weight_loader",
        "load_qkv_weight",
        "load_column_parallel_weight",
        "load_row_parallel_weight",
        "load_merged_column_weight",
        "output_dim",
        "input_dim",
        "_assert_and_load",
    ]

    # Parameters to skip during FP8 quantization (matches FlashRL's exclude_list)
    SKIP_QUANTIZATION_PARAMS = [
        "weight_scale",
        "input_scale",
        "output_scale",
        ".bias",
        "lm_head.weight",
        "model.norm.weight",
        "embed_tokens",  # BF16 params
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
        "projector",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",  # LayerNorms
    ]

    # Stacked parameters (Qwen2): shards loaded separately, then combined
    STACKED_PARAMS_MAPPING = [
        ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
        ("gate_up_proj", ["gate_proj", "up_proj"]),
    ]
    _QKV_SHARD_ALIASES = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
    }

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("[QuantizedRL] Profile-free FP8 quantization enabled")
        self._initial_load_complete = False

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ):
        """Standard weight preparation using base model path."""
        logger.info(f"[QuantizedRL] Loading from base model: {model_name_or_path}")
        temp_config = LoadConfig(load_format=LoadFormat.AUTO)
        temp_loader = DefaultModelLoader(temp_config)
        return temp_loader._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt
        )

    @staticmethod
    def _bind_method_to_cls(func, obj):
        """Bind function to object instance (for weight_loader methods)."""
        import types

        if hasattr(func, "__self__") or not callable(func):
            return func
        return types.MethodType(func, obj)

    def load_weights_and_postprocess(self, model, weights, target_device):
        """
        Initial load: Load BF16 → Record state → Apply FP8 quantization.
        Called ONCE during model initialization.
        """
        logger.info("[QuantizedRL] Initial load with FP8 quantization")

        original_load_weights = model.load_weights

        def load_weights_proxy(weights):
            if QuantizedRLModelLoader.is_reload_scenario(model):
                logger.info("[QuantizedRL] Using fast path reload in load_weights")
                QuantizedRLModelLoader.rebinding_and_load_weights(
                    model, original_load_weights, weights
                )
            else:
                original_load_weights(weights)

        model.load_weights = load_weights_proxy

        model.load_weights(weights)
        original_weights = dict(model.named_parameters())

        # Record pre-quantization state (shape/stride) for torch.as_strided reset

        model.original_weights_rebuild_keys = {}
        for name, p in original_weights.items():
            model.original_weights_rebuild_keys[name] = {
                "shape": p.shape,
                "stride": p.stride(),
                "dtype": p.dtype,
                "nbytes": p.untyped_storage().nbytes(),
            }

        # Record parameter attributes (weight_loader, etc.) before quantization
        recorded_loader = {
            k: dict() for k in QuantizedRLModelLoader.RECORDED_LOADER_KEYS
        }
        for name, p in original_weights.items():
            for key in QuantizedRLModelLoader.RECORDED_LOADER_KEYS:
                if hasattr(p, key):
                    attr = getattr(p, key)
                    if not callable(attr):
                        recorded_loader[key][name] = attr
                    elif hasattr(attr, "__self__") and p is attr.__self__:
                        recorded_loader[key][name] = attr.__func__  # Store unbound
                    else:
                        recorded_loader[key][name] = attr
        model.recorded_loader = recorded_loader

        # Apply FP8 quantization (creates new Parameters, loses attributes)
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        model.flash_rl_initial_load_complete = True
        self._initial_load_complete = True
        logger.info("[QuantizedRL] Initial load complete")

    @staticmethod
    def is_reload_scenario(model):
        """Check if model is ready for reloading (initial load completed)."""
        return (
            hasattr(model, "original_weights_rebuild_keys")
            and hasattr(model, "recorded_loader")
            and getattr(model, "flash_rl_initial_load_complete", False)
        )

    @staticmethod
    def _is_stacked_param(name):
        """Check if parameter is stacked (qkv_proj, gate_up_proj)."""
        for stacked_name, _ in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING:
            if stacked_name in name:
                return True
        return False

    @staticmethod
    def _resolve_stacked_info(name: str) -> Tuple[str, Optional[str], Optional[Any]]:
        for target, shard_names in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING:
            for idx, shard in enumerate(shard_names):
                if shard in name:
                    shard_id = (
                        QuantizedRLModelLoader._QKV_SHARD_ALIASES.get(shard, shard)
                        if target == "qkv_proj"
                        else idx
                    )
                    return name.replace(shard, target), target, shard_id
        return name, None, None

    @staticmethod
    def _store_quantized_scale(
        scale_store: Dict[str, Union[torch.Tensor, Dict[Any, torch.Tensor]]],
        name: str,
        scale: torch.Tensor,
    ) -> None:
        param_name, stacked_key, shard_id = (
            QuantizedRLModelLoader._resolve_stacked_info(name)
        )
        if stacked_key is None:
            scale_store[param_name] = scale
        else:
            shard_dict = scale_store.setdefault(param_name, {})
            assert isinstance(shard_dict, dict)
            shard_dict[shard_id] = scale

    @staticmethod
    def _apply_scale_update(
        all_params: Dict[str, torch.nn.Parameter],
        param_name: str,
        scale_info: Union[torch.Tensor, Dict[Any, torch.Tensor], None],
    ) -> None:
        if scale_info is None:
            return
        # Get tp rank and size
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        def _get_tp_sharded_scale(full_scale_tensor):
            """Get tp sharded scale from full scale tensor"""
            if tp_size == 1:
                return full_scale_tensor

            full_dim = full_scale_tensor.shape[0]
            shard_dim = full_dim // tp_size
            start_idx = tp_rank * shard_dim
            end_idx = start_idx + shard_dim
            return full_scale_tensor[start_idx:end_idx]

        if param_name.endswith(".weight"):
            scale_param_name = f"{param_name[:-7]}.weight_scale"
        else:
            scale_param_name = f"{param_name}.weight_scale"

        scale_param = all_params.get(scale_param_name)
        if scale_param is None:
            logger.warning(
                "[QuantizedRL] Scale parameter not found: %s", scale_param_name
            )
            return
        if isinstance(scale_info, torch.Tensor):
            new_scale = scale_info.t().contiguous()
            if scale_param.data.shape == new_scale.shape:
                scale_param.data.copy_(new_scale)
            else:
                logger.warning(
                    "[QuantizedRL] Scale shape mismatch for %s: expected %s, got %s",
                    scale_param_name,
                    scale_param.data.shape,
                    new_scale.shape,
                )
        else:
            stacked_key = next(
                (
                    target
                    for target, _ in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING
                    if target in param_name
                ),
                None,
            )
            shard_names = next(
                (
                    names
                    for target, names in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING
                    if target == stacked_key
                ),
                [],
            )
            rows_per_shard = scale_param.data.shape[-1] // max(len(shard_names), 1)
            if rows_per_shard * len(shard_names) != scale_param.data.shape[-1]:
                logger.warning(
                    f"Scale param shape {scale_param.data.shape[-1]} not divisible by {len(shard_names)}"
                )
            offset = 0
            for idx, shard in enumerate(shard_names):
                shard_id = (
                    QuantizedRLModelLoader._QKV_SHARD_ALIASES.get(shard, shard)
                    if stacked_key == "qkv_proj"
                    else idx
                )
                shard_scale = scale_info.get(shard_id)
                shard_scale = _get_tp_sharded_scale(shard_scale)
                if shard_scale is None:
                    offset += rows_per_shard
                    continue
                shard_rows = shard_scale.shape[0]
                start = offset
                end = start + shard_rows
                scale_param.data[..., start:end] = shard_scale.t().contiguous()
                offset = end

    @staticmethod
    def rebinding_and_load_weights(model, first_time_load_weights, weights):
        """
        Reload: VERL sends BF16 → Quantize to FP8 → Copy to original memory.

        Flow: Reset params → Restore attributes → Quantize in iterator → Load → Copy back
        """
        logger.info("[QuantizedRL] Reload: Updating weights with FP8 quantization")

        weights_list = list(weights)
        updated_param_names, is_last_update = (
            QuantizedRLModelLoader._get_updated_params(weights_list, model)
        )

        # Save current FP8 parameter data pointers
        existing_params = dict(model.named_parameters())
        current_param_data = {}
        for name in updated_param_names:
            if name in existing_params:
                current_param_data[name] = existing_params[name].data

        # Reset to pre-quantization shape using torch.as_strided
        # Keeps same storage, just changes view - critical for memory preservation
        for name, rebuild_info in model.original_weights_rebuild_keys.items():
            if name in updated_param_names and name in existing_params:
                existing_params[name].data = torch.as_strided(
                    # Note: avoid clone here
                    existing_params[name].data.clone(),
                    rebuild_info["shape"],
                    rebuild_info["stride"],
                )

        # Restore weight loader attributes (only if missing)
        for k, loader_dict in model.recorded_loader.items():
            for param_name, loader in loader_dict.items():
                if param_name in updated_param_names and param_name in existing_params:
                    param = existing_params[param_name]
                    if not hasattr(param, k):
                        if callable(loader):
                            if hasattr(loader, "__self__"):
                                setattr(param, k, loader)
                            else:
                                setattr(
                                    param,
                                    k,
                                    QuantizedRLModelLoader._bind_method_to_cls(
                                        loader, param
                                    ),
                                )
                        else:
                            setattr(param, k, loader)

        del existing_params

        # Quantize BF16 weights to FP8 in iterator (before weight_loader)
        # Store scales for later update
        quantized_scales: Dict[str, Union[torch.Tensor, Dict[Any, torch.Tensor]]] = {}

        def quantize_weights_iterator(weights_iter):
            """Quantize individual shards before weight_loader stacks them."""
            from sglang.srt.layers.quantization.fp8_kernel import (
                per_token_group_quant_fp8,
            )

            for name, weight in weights_iter:
                if any(
                    skip in name
                    for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
                ):
                    logger.info(f"[QuantizedRL] Skip: {name} ({weight.dtype})")
                    yield (name, weight)
                elif weight.dtype in [torch.bfloat16, torch.float32, torch.float16]:
                    qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])
                    logger.info(f"[QuantizedRL] Quantize: {name} {weight.dtype}→FP8")
                    QuantizedRLModelLoader._store_quantized_scale(
                        quantized_scales, name, scale
                    )
                    yield (name, qweight)
                else:
                    logger.info(f"[QuantizedRL] Keep: {name} ({weight.dtype})")
                    yield (name, weight)

        # Load quantized weights (weight_loader stacks FP8 shards)
        first_time_load_weights(quantize_weights_iterator(iter(weights_list)))

        # Copy back to original FP8 memory locations and update scales
        all_params = dict(model.named_parameters())

        for name in updated_param_names:
            if name not in all_params or name not in current_param_data:
                continue
            if any(
                skip in name for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
            ):
                continue

            new_param = all_params[name]
            old_fp8_data = current_param_data[name]

            # Handle embeddings/lm_head (BF16) and quantized weights (FP8)
            if "embed_tokens" in name or "lm_head" in name:
                old_fp8_data.copy_(new_param.data)
                new_param.data = old_fp8_data
            elif (
                new_param.dtype == torch.float8_e4m3fn
                and old_fp8_data.dtype == torch.float8_e4m3fn
            ):
                # FP8: Use strided view for transposed storage
                strided_data = torch.as_strided(
                    new_param.data, old_fp8_data.shape, old_fp8_data.stride()
                )
                old_fp8_data.copy_(strided_data)
                new_param.data = old_fp8_data
                QuantizedRLModelLoader._apply_scale_update(
                    all_params,
                    name,
                    quantized_scales.get(name),
                )
            elif new_param.dtype == old_fp8_data.dtype:
                # Same dtype (LayerNorm, etc.): Direct copy
                old_fp8_data.copy_(new_param.data)
                new_param.data = old_fp8_data
            else:
                raise RuntimeError(
                    f"Unexpected dtype mismatch for {name}: "
                    f"new={new_param.dtype}, old={old_fp8_data.dtype}"
                )

        # Cleanup
        del current_param_data
        if is_last_update:
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("[QuantizedRL] Reload complete")
        return updated_param_names, is_last_update

    @staticmethod
    def _get_updated_params(weights_list, model):
        """Identify which parameters need updating from incoming weights."""
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(model.named_parameters())
        updated_params = set()
        is_last_update = False

        for name, _ in weights_list:
            if name == "lm_head.weight":
                is_last_update = True

            if any(
                skip in name for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
            ):
                continue

            from sglang.srt.layers.utils import get_layer_id

            # Skip params outside layer range (for pipeline parallelism)
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(model, "start_layer")
                and (layer_id < model.start_layer or layer_id >= model.end_layer)
            ):
                continue

            # Skip tied embeddings and vision tower params
            if (
                hasattr(model, "config")
                and model.config.tie_word_embeddings
                and "lm_head.weight" in name
            ):
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # Map stacked param shards (q/k/v_proj → qkv_proj)
            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    updated_params.add(name)
                    mapped = True
                    break

            if not mapped:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    updated_params.add(name)

        return list(updated_params), is_last_update


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        if get_bool_env_var("SGL_CPU_QUANTIZATION"):
            return load_model_with_cpu_quantization(
                self, model_config=model_config, device_config=device_config
            )

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # Skip FusedMoE layers already quantized during init (FP8 or FP4)
                    if (
                        hasattr(module, "is_weights_quantized")
                        and module.is_weights_quantized()
                    ):
                        continue
                    quant_method.process_weights_after_loading(module)

            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)

            post_load_weights(model, model_config)

        return model.eval()


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/runtime/engine/save_sharded_state.py` for creating a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: Dict[Any, List[Tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: Dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: Optional[str]):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        from safetensors.torch import safe_open

        from sglang.srt.distributed import get_tensor_model_parallel_rank

        local_model_path = self._prepare_weights(
            model_config.model_path, model_config.revision
        )

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
            rank = get_tensor_model_parallel_rank()
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )
            filepaths = glob.glob(pattern)
            if not filepaths:
                # TODO: support un-sharded checkpoints too
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!"
                )
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s",
                                tensor.shape,
                                key,
                                param_shape,
                            )
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

            post_load_weights(model, model_config)

        return model.eval()

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from safetensors.torch import save_file

        from sglang.srt.distributed import get_tensor_model_parallel_rank

        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        total_size = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )


class BitsAndBytesModelLoader(BaseModelLoader):
    """Model loader to load model weights with BitAndBytes quantization."""

    possible_config_file_names = ["adapter_config.json"]

    default_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".fc1.",
        ".fc2.",
        ".dense.",
        ".query_key_value.",
        ".qkv_proj.",
        ".dense_h_to_4h.",
        ".dense_4h_to_h.",
        ".out_proj.",
    ]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # we don't need to quantize the whole model, only the target modules
        # that are specified in the adapter config file. If the adapter config
        # file is not provided, we will quantize the default modules.
        if (
            not load_config.model_loader_extra_config
            or "qlora_adapter_name_or_path" not in load_config.model_loader_extra_config
        ):
            self.target_modules = []
            return

        qlora_adapter = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"
        ]

        config_file_path = self._get_config_file(qlora_adapter)

        with open(config_file_path, "r") as f:
            config = json.load(f)
            self.target_modules = config["target_modules"]

    def _get_config_file(self, qlora_adapter: str) -> str:
        is_local = os.path.isdir(qlora_adapter)
        config_file_path = None
        if is_local:
            for file in self.possible_config_file_names:
                config_file_path = os.path.join(qlora_adapter, file)
                if os.path.exists(config_file_path):
                    break
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=qlora_adapter)
            for file in self.possible_config_file_names:
                if file in repo_files:
                    config_file_path = hf_hub_download(
                        repo_id=qlora_adapter, filename=file
                    )
                    break

        if not config_file_path:
            raise ValueError(f"Cannot find adapter config file in {qlora_adapter}")

        return config_file_path

    def _get_weight_files(
        self,
        model_name_or_path: str,
        allowed_patterns: List[str],
        revision: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """Retrieve weight files. Download the files if necessary.

        Return the weight files and the file pattern."""
        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            for pattern in allowed_patterns:
                weight_files = glob.glob(os.path.join(model_name_or_path, pattern))
                if weight_files:
                    return weight_files, pattern
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=model_name_or_path)
            for pattern in allowed_patterns:
                matching_files = fnmatch.filter(repo_files, pattern)
                if matching_files:
                    hf_folder = download_weights_from_hf(
                        model_name_or_path,
                        self.load_config.download_dir,
                        [pattern],
                        revision,
                        ignore_patterns=self.load_config.ignore_patterns,
                    )
                    return glob.glob(os.path.join(hf_folder, pattern)), pattern

        raise RuntimeError(f"No model weights found in: `{model_name_or_path}`")

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[List[str], bool]:
        """Prepare weight files for the model."""

        allowed_patterns = ["*.safetensors", "*.bin", "*.pt"]

        hf_weights_files, matched_pattern = self._get_weight_files(
            model_name_or_path, allowed_patterns, revision
        )

        if matched_pattern != "*.safetensors":
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_weights_files, matched_pattern == "*.safetensors"

    def _hf_weight_iter(self, hf_weights_files, use_safetensors: bool):
        if use_safetensors:
            return safetensors_weights_iterator(hf_weights_files)
        else:
            return pt_weights_iterator(hf_weights_files)

    def _get_quantized_weights_iterator(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        pre_quant: bool,
        load_8bit: bool,
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str, Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes

            if bitsandbytes.__version__ < "0.44.0":
                raise ImportError(
                    "bitsandbytes version is wrong. Please "
                    "install bitsandbytes>=0.44.0."
                )
        except ImportError as err:
            raise ImportError(
                "Please install bitsandbytes>=0.44.0 via "
                "`pip install bitsandbytes>=0.44.0` to use "
                "bitsandbytes quantizer."
            ) from err

        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision
        )

        quant_state_dict: Dict[str, Any] = {}

        if pre_quant:
            if load_8bit:
                return (
                    self._quantized_8bit_generator(
                        hf_weights_files, use_safetensors, quant_state_dict
                    ),
                    quant_state_dict,
                )
            else:
                return (
                    self._quantized_4bit_generator(
                        hf_weights_files, use_safetensors, quant_state_dict
                    ),
                    quant_state_dict,
                )

        return (
            self._unquantized_generator(
                hf_weights_files, use_safetensors, quant_state_dict
            ),
            quant_state_dict,
        )

    def _is_8bit_weight_name(self, weight_name: str):
        quantized_suffix = {".scb", ".weight_format"}
        return any(weight_name.lower().endswith(suffix) for suffix in quantized_suffix)

    def _is_4bit_weight_name(self, weight_name: str):
        quantized_suffix = {
            "absmax",
            "quant_map",
            "nested_absmax",
            "nested_quant_map",
            "bitsandbytes",
        }
        suffix = weight_name.split(".")[-1]
        return any(q_suffix in suffix for q_suffix in quantized_suffix)

    def _quantized_8bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if not weight_name.lower().endswith(".scb"):
                continue

            weight_key = weight_name.lower().replace(".scb", ".weight")
            quant_state_dict[weight_key] = weight_tensor

        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if self._is_8bit_weight_name(weight_name):
                continue

            if weight_name in quant_state_dict:
                set_weight_attrs(weight_tensor, {"load_in_8bit": True})
                yield weight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _quantized_4bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        from bitsandbytes.functional import QuantState

        # First iterate over all quant state weights
        weight_iterator = self._hf_weight_iter(hf_weights_files, use_safetensors)
        temp_state_dict = {}
        for weight_name, weight_tensor in weight_iterator:
            if not self._is_4bit_weight_name(weight_name):
                continue
            # bitsandbytes library requires
            # weight.quant_state.bitsandbytes__* in CPU
            if "quant_state.bitsandbytes" in weight_name:
                temp_state_dict[weight_name] = weight_tensor.cpu().data
            else:
                temp_state_dict[weight_name] = weight_tensor

        # Closure to parse quant_state for each prequant weight
        def _parse_quant_state(param_name: str, temp_state_dict: Dict) -> QuantState:
            quant_state = {}
            for k in temp_state_dict:
                if param_name + "." in k:
                    quant_state[k] = temp_state_dict[k]

            return QuantState.from_dict(quant_state, device="cuda")

        # Second iterate over all prequant and normal weights
        # pre quantized weights would have a quant_state
        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):

            if self._is_4bit_weight_name(weight_name):
                continue

            if (f"{weight_name}.quant_state.bitsandbytes__nf4" in temp_state_dict) or (
                f"{weight_name}.quant_state.bitsandbytes__fp4" in temp_state_dict
            ):
                quant_state = _parse_quant_state(weight_name, temp_state_dict)
                quant_state_dict[weight_name] = quant_state
                yield weight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _unquantized_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        from bitsandbytes.functional import quantize_4bit

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):

            if any(
                target_module in weight_name for target_module in self.target_modules
            ) and weight_name.endswith(".weight"):
                weight_name = weight_name.replace(".weight", ".qweight")

                if any(
                    module in weight_name
                    for module in self.column_parallel_weights_modules
                ):

                    total_size = weight_tensor.size(-1)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[..., start_index:end_index]

                else:
                    total_size = weight_tensor.size(0)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[start_index:end_index, ...]

                # bitsandbytes requires data in GPU
                if weight_sub_tensor.is_cuda:
                    loaded_weight = weight_sub_tensor
                else:
                    loaded_weight = weight_sub_tensor.cuda()

                # remove the following after the issue is fixed:
                # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1342
                if loaded_weight.is_contiguous() is False:
                    loaded_weight = loaded_weight.contiguous()

                with set_default_torch_dtype(torch.float32):
                    processed_weight, quant_state = quantize_4bit(
                        loaded_weight, compress_statistics=True, quant_type="nf4"
                    )

                quant_state_dict[weight_name] = quant_state
            else:
                processed_weight = weight_tensor

            yield weight_name, processed_weight

    def _load_weights(self, model_config: ModelConfig, model: nn.Module) -> None:
        if not hasattr(model, "load_weights"):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(model).__name__}."
            )

        if not hasattr(model, "bitsandbytes_stacked_params_mapping"):
            raise AttributeError(
                f"Model {type(model).__name__} does not support BitsAndBytes "
                "quantization yet."
            )

        if len(self.target_modules) == 0:
            if hasattr(model, "default_bitsandbytes_target_modules"):
                self.target_modules = model.default_bitsandbytes_target_modules
            else:
                self.target_modules = self.default_target_modules

        if hasattr(model, "column_parallel_weights_modules"):
            self.column_parallel_weights_modules = model.column_parallel_weights_modules
        else:
            self.column_parallel_weights_modules = []

        self.model_type = type(model).__name__

        logger.info(
            "Loading weights with BitsAndBytes quantization. " " May take a while ..."
        )

        quant_config = getattr(model_config.hf_config, "quantization_config", None)

        pre_quant = False
        if quant_config is not None:
            quant_method = quant_config.get("quant_method")
            if quant_method == "bitsandbytes":
                pre_quant = True
            else:
                raise ValueError(
                    f"BitsAndBytes loader does not support {quant_method} "
                    "quantization"
                )

        # The quant_states in pre_quantized models cannot work with a split
        # weight tensor. So TP does not work with pre_quantized bnb models.
        if pre_quant and get_tensor_model_parallel_world_size() > 1:
            raise ValueError(
                "Prequant BitsAndBytes models with TP is not supported."
                "Please try with PP."
            )

        load_8bit = False
        if pre_quant:
            load_8bit = quant_config.get("load_in_8bit", False)

        qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(
            model_config.model_path, model_config.revision, pre_quant, load_8bit
        )

        model.load_weights(qweight_iterator)

        torch.cuda.empty_cache()

        param_dict = dict(model.named_parameters())
        stacked_quant_state_dict: Dict[str, Dict[int, Any]] = {}
        model_type = model_config.hf_config.model_type
        for quant_param_name in quant_state_dict:
            non_stacked_param_name = quant_param_name
            if model_type == "mllama" and "vision_model" in quant_param_name:
                # adapt to VisionAttention
                quant_param_name = quant_param_name.replace(
                    "self_attn.o_proj", "self_attn.proj"
                )
            shard_index = 0
            for shard_name, (
                weight_name,
                index,
            ) in model.bitsandbytes_stacked_params_mapping.items():
                if (
                    model_type in ["qwen2_vl", "qwen2_5_vl"]
                    and "visual" in quant_param_name
                ):
                    break
                if shard_name in quant_param_name:
                    shard_index = index
                    quant_param_name = quant_param_name.replace(shard_name, weight_name)
                    break

            if (
                model_type in ["qwen2_vl", "qwen2_5_vl"]
                and "visual" in quant_param_name
            ):
                quant_param_name = quant_param_name.replace(
                    r"attn.qkv.", r"attn.qkv_proj."
                )

            if quant_param_name not in param_dict:
                raise ValueError(
                    f"Parameter {quant_param_name} not found in the model."
                )

            if quant_param_name not in stacked_quant_state_dict:
                stacked_quant_state_dict[quant_param_name] = {}

            stacked_quant_state_dict[quant_param_name][shard_index] = quant_state_dict[
                non_stacked_param_name
            ]

        # save quant_states and offsets as the attributes of the parameters
        for param_name, param in param_dict.items():
            if param_name in stacked_quant_state_dict:
                quant_states = stacked_quant_state_dict[param_name]
                set_weight_attrs(param, {"bnb_quant_state": quant_states})

                pack_ratio = getattr(param, "pack_factor", -1)
                if pack_ratio == -1:
                    raise ValueError(f"pack_factor not set for parameter {param_name}.")

                num_elements = [0] * len(quant_states)
                for seq, quant_state in quant_states.items():
                    num_elements[seq] = math.prod(quant_state.shape) // pack_ratio

                offsets = np.concatenate(([0], np.cumsum(num_elements)))
                # Make torch infer_schema happy(Compatible with vLLM)
                offsets = torch.tensor(offsets).cpu()
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})

                if load_8bit:
                    set_weight_attrs(
                        param, {"matmul_state": [None] * len(quant_states)}
                    )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                )

                self._load_weights(model_config, model)

        return model.eval()


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_name_or_path: str):
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError(f"{model_name_or_path} is not a file.")

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """

        # only load the gguf module when needed
        try:
            import gguf

            # FIXME: add version check for gguf
        except ImportError as err:
            raise ImportError(
                "Please install gguf via `pip install gguf` to use gguf quantizer."
            ) from err

        config = model_config.hf_config
        model_type = config.model_type
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        num_layers = config.num_hidden_layers
        name_map = gguf.get_tensor_name_map(arch, num_layers)
        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(config)
        state_dict = dummy_model.state_dict()

        gguf_to_hf_name_map = {}
        for hf_name in state_dict:
            name, suffix = hf_name.rsplit(".", 1)
            gguf_name = name_map.get_name(name)
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
        return gguf_to_hf_name_map

    def _get_weights_iterator(
        self, model_name_or_path: str, gguf_to_hf_name_map: Dict[str, str]
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return gguf_quant_weights_iterator(model_name_or_path, gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        local_model_path = self._prepare_weights(model_config.model_path)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        if "lm_head.weight" in get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        ):
            model_config.hf_config.update({"tie_word_embeddings": True})

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config)
            model.load_weights(
                self._get_weights_iterator(local_model_path, gguf_weights_map)
            )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model


class RemoteInstanceModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote sglang instance."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.remote_instance_transfer_engine_weight_info = None

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        logger.info("Loading weights from remote instance ...")
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE_INSTANCE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config)

        if (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            model_weights = f"instance://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_send_weights_group_ports[load_config.tp_rank]}"
            with create_remote_connector(model_weights, device_config.device) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.INSTANCE:
                    self.load_model_from_remote_instance_by_nccl(
                        model, client, model_config, device_config
                    )
                else:
                    raise ValueError(
                        f"Unsupported connector type {connector_type} for "
                        f"remote tensor model loading."
                    )
        elif (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        ):
            if load_config.remote_instance_weight_loader_transfer_engine is None:
                raise RuntimeError(
                    "Transfer engine is not initialized for remote instance "
                    "model loader with `transfer_engine` backend. "
                )
            logger.info(
                "TransferEngine registering memory regions (this may take a few seconds)..."
            )
            # register memory region
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                model, load_config.remote_instance_weight_loader_transfer_engine
            )
            logger.info(
                "TransferEngine memory regions have been successfully registered."
            )

            # transfer weights
            success = self.load_model_from_remote_instance_by_transfer_engine(
                model,
                load_config.remote_instance_weight_loader_transfer_engine,
                f"http://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_seed_instance_service_port}",
                load_config.tp_rank,
            )
            if not success:
                raise RuntimeError(
                    "Failed to load weights from remote instance via transfer engine."
                )
        else:
            raise ValueError("Invalid remote instance weight loader backend.")

        return model.eval()

    def load_model_from_remote_instance_by_nccl(
        self, model, client, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        load_config = self.load_config
        instance_ip = socket.gethostbyname(socket.gethostname())
        start_build_group_tic = time.time()
        client.build_group(
            gpu_id=device_config.gpu_id,
            tp_rank=load_config.tp_rank,
            instance_ip=instance_ip,
        )
        torch.cuda.synchronize()
        end_build_group_tic = time.time()
        logger.debug(
            f"finish building group for remote instance, time used: {(end_build_group_tic - start_build_group_tic):.4f}s"
        )

        if load_config.tp_rank == 0:
            t = threading.Thread(
                target=trigger_transferring_weights_request,
                args=(
                    load_config.remote_instance_weight_loader_seed_instance_ip,
                    load_config.remote_instance_weight_loader_seed_instance_service_port,
                    load_config.remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()

        start_get_weights_tic = time.time()
        with set_default_torch_dtype(model_config.dtype):
            for _, tensor in model.named_parameters():
                torch.distributed.broadcast(
                    tensor.data,
                    src=0,
                    group=client._model_update_group,
                )
            torch.cuda.synchronize()

            if hasattr(model, "post_load_weights"):
                model.post_load_weights()
        end_get_weights_tic = time.time()
        logger.debug(
            f"finish getting all weights from remote instance, time used: {(end_get_weights_tic - start_get_weights_tic):.4f}s"
        )
        # destroy the process group after loading weights
        torch.distributed.distributed_c10d.destroy_process_group(
            client._model_update_group
        )
        torch.cuda.empty_cache()

    def load_model_from_remote_instance_by_transfer_engine(
        self, model, transfer_engine, seed_url, tp_rank
    ) -> bool:
        # get remote weights metadata from source instance
        seed_transfer_engine_session_id, seed_transfer_engine_weight_info = (
            get_remote_instance_transfer_engine_info_per_rank(seed_url, tp_rank)
        )
        if (
            seed_transfer_engine_session_id is None
            or seed_transfer_engine_weight_info is None
        ):
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        # prepare local/remote RDMA keys
        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in model.named_parameters():
            weight_info = seed_transfer_engine_weight_info.get(name, None)
            if weight_info is None:
                logger.error(f"Cannot find weight info for {name}.")
                return False

            seed_ptr, seed_numel, seed_element_size = weight_info
            if (
                seed_numel != tensor.numel()
                or seed_element_size != tensor.element_size()
            ):
                logger.error(
                    f"Weight info does not match for {name}, "
                    f"expected ({seed_numel}, {seed_element_size}), "
                    f"got ({tensor.numel()}, {tensor.element_size()})"
                )
                return False
            client_ptr = tensor.data_ptr()
            client_len = tensor.numel() * tensor.element_size()
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(client_ptr)
            client_len_list.append(client_len)

        # load weights from source instance through TransferEngine
        ret = transfer_engine.batch_transfer_sync_read(
            seed_transfer_engine_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        if ret < 0:
            logger.error(f"batch transfer failed, error: {ret}")
            return False

        if hasattr(model, "post_load_weights"):
            model.post_load_weights()

        return True


class RemoteModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote database."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # TODO @DellCurry: move to s3 connector only
        set_runai_streamer_env(load_config)

    def _get_weights_iterator_kv(
        self,
        client,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        assert get_connector_type(client) == ConnectorType.KV
        rank = get_tensor_model_parallel_rank()
        return client.weight_iterator(rank)

    def _get_weights_iterator_fs(
        self,
        client,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        assert get_connector_type(client) == ConnectorType.FS
        return client.weight_iterator()

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        model_path: str,
        url: str,
    ) -> None:
        with create_remote_connector(url) as client:
            assert get_connector_type(client) == ConnectorType.KV
            model_name = parse_model_name(url)
            rank = get_tensor_model_parallel_rank()
            state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
            for key, tensor in state_dict.items():
                r_key = f"{model_name}/keys/rank_{rank}/{key}"
                client.set(r_key, tensor)

            for root, _, files in os.walk(model_path):
                for file_name in files:
                    # ignore hidden files
                    if file_name.startswith("."):
                        continue
                    if os.path.splitext(file_name)[1] in (".json", ".py"):
                        file_path = os.path.join(root, file_name)
                        with open(file_path, encoding="utf-8") as file:
                            file_content = file.read()
                            f_key = f"{model_name}/files/{file_name}"
                            client.setstr(f_key, file_content)

    def _load_model_from_remote_kv(
        self, model: nn.Module, model_config: ModelConfig, client
    ):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
        weights_iterator = self._get_weights_iterator_kv(client)
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        for key, tensor in weights_iterator:
            # If loading with LoRA enabled, additional padding may
            # be added to certain parameters. We only load into a
            # narrowed view of the parameter data.
            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into " "parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            param_data.copy_(tensor)
            state_dict.pop(key)
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

        post_load_weights(model, model_config)

    def _load_model_from_remote_fs(
        self, model, client, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            model.load_weights(self._get_weights_iterator_fs(client))

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        logger.info("Loading weights from remote storage ...")
        start = time.perf_counter()
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        model_weights = model_config.model_path
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config)

            with create_remote_connector(
                model_weights, device=device_config.device
            ) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.KV:
                    self._load_model_from_remote_kv(model, model_config, client)
                elif connector_type == ConnectorType.FS:
                    self._load_model_from_remote_fs(
                        model, client, model_config, device_config
                    )

        end = time.perf_counter()
        logger.info("Loaded weights from remote storage in %.2f seconds.", end - start)
        return model.eval()


def load_model_with_cpu_quantization(
    self,
    *,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    target_device = torch.device(device_config.device)
    with set_default_torch_dtype(model_config.dtype):
        model = _initialize_model(
            model_config,
            self.load_config,
        )

        if not isinstance(self, DummyModelLoader):
            model.load_weights(self._get_all_weights(model_config, model))

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        model.to(target_device)

    return model.eval()


class ModelOptModelLoader(DefaultModelLoader):
    """
    Model loader that applies NVIDIA Model Optimizer quantization
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # Any ModelOpt specific initialization if needed

    def _setup_modelopt_quantization(
        self,
        model,
        tokenizer,
        quant_cfg,
        quantized_ckpt_restore_path: str | None = None,
        quantized_ckpt_save_path: str | None = None,
        export_path: str | None = None,
    ) -> None:
        """
        Set up ModelOpt quantization for the given model.

        Args:
            model: The model to quantize
            tokenizer: The tokenizer associated with the model
            quant_cfg: The quantization configuration
            quantized_ckpt_restore_path: Path to restore quantized checkpoint from
            quantized_ckpt_save_path: Path to save quantized checkpoint to
            export_path: Path to export the quantized model in HuggingFace format

        Raises:
            ImportError: If ModelOpt is not available
            Exception: If quantization setup fails
        """
        try:
            import modelopt.torch.opt as mto
            import modelopt.torch.quantization as mtq
            from modelopt.torch.quantization.utils import is_quantized
        except ImportError as e:
            raise ImportError(
                "ModelOpt is not available. Please install modelopt."
            ) from e

        if is_quantized(model):
            rank0_log("Model is already quantized, skipping quantization setup.")
            return
        # Restore from checkpoint if provided
        if quantized_ckpt_restore_path:
            try:
                mto.restore(model, quantized_ckpt_restore_path)
                rank0_log(
                    f"Restored quantized model from {quantized_ckpt_restore_path}"
                )

                # Export model if path provided (even when restoring from checkpoint)
                self._maybe_export_modelopt(model, export_path)
                return
            except Exception as e:
                logger.warning(
                    f"Failed to restore from {quantized_ckpt_restore_path}: {e}"
                )
                rank0_log("Proceeding with calibration-based quantization...")

        # Set up calibration-based quantization
        try:
            # Left padding tends to work better for batched generation with decoder-only LMs
            with suppress(Exception):
                tokenizer.padding_side = "left"

            from modelopt.torch.utils.dataset_utils import (
                create_forward_loop,
                get_dataset_dataloader,
            )

            # Create calibration dataloader
            calib_dataloader = get_dataset_dataloader(
                dataset_name="cnn_dailymail",  # TODO: Consider making this configurable
                tokenizer=tokenizer,
                batch_size=36,  # TODO: Consider making this configurable
                num_samples=512,  # TODO: Consider making this configurable
                device=model.device,
                include_labels=False,
            )

            calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

            # Apply quantization
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

            if (
                not model_parallel_is_initialized()
                or get_tensor_model_parallel_rank() == 0
            ):
                mtq.print_quant_summary(model)

            # Save checkpoint if path provided
            if quantized_ckpt_save_path:
                try:
                    mto.save(model, quantized_ckpt_save_path)
                    rank0_log(f"Quantized model saved to {quantized_ckpt_save_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to save quantized checkpoint to {quantized_ckpt_save_path}: {e}"
                    )

            # Export model if path provided
            self._maybe_export_modelopt(model, export_path)

        except Exception as e:
            raise Exception(f"Failed to set up ModelOpt quantization: {e}") from e

    def _maybe_export_modelopt(self, model, export_path: str | None) -> None:
        """Export model to HuggingFace format if export_path is provided."""
        if export_path:
            try:
                # Get the original model path from the model config
                original_model_path = getattr(self, "_original_model_path", None)
                self._export_modelopt_checkpoint(
                    model, export_path, original_model_path
                )
                rank0_log(
                    f"Quantized model exported to HuggingFace format at {export_path}"
                )
            except Exception as e:
                rank0_log(
                    f"Warning: Failed to export quantized model to {export_path}: {e}"
                )

    def _export_modelopt_checkpoint(
        self,
        model,
        export_path: str,
        model_path: str = None,
        trust_remote_code: bool = True,
    ) -> None:
        """
        Export the quantized model to HuggingFace format using ModelOpt export API.

        Args:
            model: The quantized model to export
            export_path: Directory path to export the model to
            model_path: Path to the original model (for tokenizer export)
            trust_remote_code: Whether to trust remote code for tokenizer loading

        Raises:
            ImportError: If ModelOpt export functionality is not available
            Exception: If export fails
        """
        try:
            from modelopt.torch.export import export_hf_checkpoint
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ModelOpt export functionality is not available. "
                "Please ensure you have the latest version of modelopt installed."
            ) from e

        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)

        # Export the quantized model
        export_hf_checkpoint(model, export_dir=export_path)

        # Export the tokenizer if model_path is provided
        if model_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
                tokenizer.save_pretrained(export_path)
                rank0_log(f"Tokenizer exported to {export_path}")
            except Exception as e:
                rank0_log(f"Warning: Failed to export tokenizer: {e}")

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        logger.info("ModelOptModelLoader: Loading base model...")

        # Store the original model path for tokenizer export
        self._original_model_path = model_config.model_path

        # Check if model is already quantized
        if model_config._is_already_quantized():
            logger.info("Model is already quantized, loading directly...")
            # Use default loading for pre-quantized models
            return super().load_model(
                model_config=model_config, device_config=device_config
            )

        # TODO: Quantize-and-serve mode has been disabled at the ModelConfig level
        # All quantization now uses the standard workflow (quantize + export/save)
        logger.info("Standard quantization mode: Will quantize and export/save")
        return self._standard_quantization_workflow(model_config, device_config)

    def _standard_quantization_workflow(
        self, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        """Standard quantization workflow: quantize, save checkpoint, export, then return model."""
        # Use shared method from parent class to load base model for quantization
        model = self._load_modelopt_base_model(model_config)

        # Import ModelOpt modules
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            logger.error(
                "NVIDIA Model Optimizer (modelopt) library not found. "
                "Please install it to use ModelOpt quantization."
            )
            raise

        # Handle both old modelopt_quant and new unified quantization flags
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Legacy modelopt_quant flag
            quant_choice_str = model_config.modelopt_quant
        else:
            # Unified quantization flag - extract the type (fp8/fp4)
            quant_choice_str = model_config._get_modelopt_quant_type()

        quant_cfg_name = QUANT_CFG_CHOICES.get(quant_choice_str)
        if not quant_cfg_name:
            raise ValueError(
                f"Invalid quantization choice: '{quant_choice_str}'. "
                f"Available choices: {list(QUANT_CFG_CHOICES.keys())}"
            )

        try:
            # getattr will fetch the config object, e.g., mtq.FP8_DEFAULT_CFG
            quant_cfg = getattr(mtq, quant_cfg_name)
        except AttributeError:
            raise AttributeError(
                f"ModelOpt quantization config '{quant_cfg_name}' not found. "
                "Please verify the ModelOpt library installation."
            )

        logger.info(
            f"Quantizing model with ModelOpt using config: mtq.{quant_cfg_name}"
        )

        # Get ModelOpt configuration from LoadConfig
        modelopt_config = self.load_config.modelopt_config
        quantized_ckpt_restore_path = (
            modelopt_config.checkpoint_restore_path if modelopt_config else None
        )
        quantized_ckpt_save_path = (
            modelopt_config.checkpoint_save_path if modelopt_config else None
        )
        export_path = modelopt_config.export_path if modelopt_config else None
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path, use_fast=True
        )

        try:
            self._setup_modelopt_quantization(
                model,
                tokenizer,
                quant_cfg,
                quantized_ckpt_restore_path=quantized_ckpt_restore_path,
                quantized_ckpt_save_path=quantized_ckpt_save_path,
                export_path=export_path,
            )
        except Exception as e:
            logger.warning(f"ModelOpt quantization failed: {e}")
            rank0_log("Proceeding without quantization...")

        return model.eval()


def get_model_loader(
    load_config: LoadConfig, model_config: Optional[ModelConfig] = None
) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if model_config and (
        (hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant)
        or model_config.quantization in ["modelopt_fp8", "modelopt_fp4", "modelopt"]
    ):
        logger.info("Using ModelOptModelLoader due to ModelOpt quantization config.")
        return ModelOptModelLoader(load_config)

    # Use ModelOptModelLoader for unified quantization flags
    if (
        model_config
        and hasattr(model_config, "quantization")
        and model_config.quantization in ["modelopt_fp8", "modelopt_fp4"]
    ):
        if model_config._is_already_quantized():
            logger.info(
                f"Using ModelOptModelLoader for pre-quantized model: {model_config.quantization}"
            )
        else:
            logger.info(
                f"Using ModelOptModelLoader for quantization: {model_config.quantization}"
            )
        return ModelOptModelLoader(load_config)

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return BitsAndBytesModelLoader(load_config)

    if load_config.load_format == LoadFormat.GGUF:
        return GGUFModelLoader(load_config)

    if load_config.load_format == LoadFormat.LAYERED:
        return LayeredModelLoader(load_config)

    # Check for FLASH_RL format early
    # FP8 approach: BF16/FP16 model with native FP8 quantization
    if load_config.load_format == LoadFormat.FLASH_RL:
        logger.info(
            "Using QuantizedRLModelLoader for RL training with native FP8 quantization."
        )
        logger.info(
            "FP8 approach: Model loads with native SGLang FP8 quantization. "
            "Same model path for both training and inference."
        )

        # Set quantization to FP8 for native SGLang support
        if model_config and not model_config.quantization:
            logger.info(
                "QuantizedRL: Setting quantization to fp8 (native SGLang support). "
                "Model will be loaded with FP8 infrastructure"
            )
            model_config.quantization = "fp8"

        return QuantizedRLModelLoader(load_config)

    if load_config.load_format == LoadFormat.REMOTE:
        return RemoteModelLoader(load_config)

    if load_config.load_format == LoadFormat.REMOTE_INSTANCE:
        return RemoteInstanceModelLoader(load_config)

    if load_config.load_format == LoadFormat.PRIVATE:
        import importlib

        try:
            module = importlib.import_module("sglang.private.private_model_loader")
            return module.PrivateModelLoader(load_config)
        except ImportError:
            raise ValueError("Failed to import sglang.private.private_model_loader")

    return DefaultModelLoader(load_config)

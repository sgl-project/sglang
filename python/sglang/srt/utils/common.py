# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common utilities."""

from __future__ import annotations

import argparse
import asyncio
import builtins
import ctypes
import functools
import gc
import importlib
import inspect
import io
import itertools
import json
import logging
import math
import os
import pickle
import platform
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache, partial
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from io import BytesIO
from json import JSONDecodeError
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from unittest import SkipTest
from urllib.parse import unquote, urlparse

import numpy as np
import orjson
import psutil
import pybase64
import requests
import torch
import torch.distributed as dist
import triton
from packaging import version as pkg_version
from PIL import Image
from starlette.routing import Mount
from torch import nn
from torch.library import Library
from torch.utils._contextlib import _DecoratorContextManager
from torchvision.io import decode_jpeg
from typing_extensions import Literal

from sglang.srt.environ import envs
from sglang.srt.observability.func_timer import enable_func_timer
from sglang.srt.utils.video_decoder import _BACKEND, VideoDecoderWrapper

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
torch_release = pkg_version.parse(torch.__version__).release


# https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
@lru_cache(maxsize=1)
def is_hip() -> bool:
    return torch.version.hip is not None


if is_hip():
    HIP_FP8_E4M3_FNUZ_MAX = 224.0
    FP8_E4M3_MAX = HIP_FP8_E4M3_FNUZ_MAX
else:
    FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

FP8_E4M3_MIN = -FP8_E4M3_MAX

builtins.FP8_E4M3_MAX = FP8_E4M3_MAX
builtins.FP8_E4M3_MIN = FP8_E4M3_MIN

# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


@lru_cache(maxsize=1)
def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


@lru_cache(maxsize=1)
def is_cuda_alike():
    return is_cuda() or is_hip()


@lru_cache(maxsize=1)
def is_hpu() -> bool:
    return hasattr(torch, "hpu") and torch.hpu.is_available()


@lru_cache(maxsize=1)
def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache(maxsize=1)
def is_npu() -> bool:
    if not hasattr(torch, "npu"):
        return False

    if not torch.npu.is_available():
        raise RuntimeError(
            "torch_npu detected, but NPU device is not available or visible."
        )

    return True


@lru_cache(maxsize=1)
def is_host_cpu_x86() -> bool:
    machine = platform.machine().lower()
    return (
        machine in ("x86_64", "amd64", "i386", "i686")
        and hasattr(torch, "cpu")
        and torch.cpu.is_available()
    )


def is_host_cpu_arm64() -> bool:
    machine = platform.machine().lower()
    return (
        machine in ("aarch64", "arm64")
        and hasattr(torch, "cpu")
        and torch.cpu.is_available()
    )


@lru_cache(maxsize=1)
def is_cpu() -> bool:
    is_host_cpu_supported = is_host_cpu_x86() or is_host_cpu_arm64()
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1" and is_host_cpu_supported


@lru_cache(maxsize=1)
def is_musa() -> bool:
    try:
        import torchada  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.version, "musa") and torch.version.musa is not None


@lru_cache(maxsize=1)
def is_mps() -> bool:
    return torch.backends.mps.is_available()


def is_float4_e2m1fn_x2(dtype) -> bool:
    """Check if dtype is float4_e2m1fn_x2 and CUDA is available."""
    target_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return is_cuda() and dtype == target_dtype


def get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


@contextmanager
def device_context(device: torch.device):
    if device.type == "cpu" and is_cpu():
        with torch.device("cpu"):
            yield
    else:
        module = torch.get_device_module(device)
        if module is not None:
            with module.device(device.index):
                yield
        else:
            raise ValueError(f"Unknown device module: {device}")


def _check_cuda_device_version(
    device_capability_majors: List[int], cuda_version: Tuple[int, int]
):
    if not is_cuda():
        return False
    return (
        torch.cuda.get_device_capability()[0] in device_capability_majors
        and tuple(map(int, torch.version.cuda.split(".")[:2])) >= cuda_version
    )


is_ampere_with_cuda_12_3 = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[8], cuda_version=(12, 3)
    )
)
is_hopper_with_cuda_12_3 = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[9], cuda_version=(12, 3)
    )
)
is_blackwell_supported = is_blackwell = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version,
        device_capability_majors=[10, 11, 12],
        cuda_version=(12, 8),
    )
)
is_sm120_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[12], cuda_version=(12, 8)
    )
)
is_sm100_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[10], cuda_version=(12, 8)
    )
)
is_sm90_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[9], cuda_version=(12, 3)
    )
)


try:
    import sgl_kernel  # noqa: F401

    is_intel_amx_backend_available = hasattr(
        torch.ops.sgl_kernel, "convert_weight_packed"
    )
except:
    is_intel_amx_backend_available = False

try:
    # move torch._C._cpu._is_amx_tile_supported() from cpu_has_amx_support
    # to support torch compile
    is_amx_tile_supported = torch._C._cpu._is_amx_tile_supported()
except:
    is_amx_tile_supported = False


def cpu_has_amx_support():
    return is_amx_tile_supported and is_intel_amx_backend_available


def use_intel_amx_backend(layer):
    return getattr(layer, "use_intel_amx_backend", False)


def xpu_has_xmx_support():
    # TODO: update with XPU capability query
    if is_xpu():
        # currently only PVC/LNL/BMG supports F64, so we only support these now
        return torch.xpu.get_device_properties().has_fp64
    return False


def use_intel_xpu_backend():
    return get_bool_env_var("SGLANG_USE_SGL_XPU") and is_xpu()


@lru_cache(maxsize=1)
def is_flashinfer_available():
    """
    Check whether flashinfer is available.
    As of Oct. 6, 2024, it is only available on NVIDIA GPUs.
    """
    if not get_bool_env_var("SGLANG_IS_FLASHINFER_AVAILABLE", default="true"):
        return False
    return importlib.util.find_spec("flashinfer") is not None and is_cuda()


def is_nvidia_cublas_version_ge_12_9():
    """
    temporary fix for issue #11272 (cublas 12.9+)
    """
    for pkg in ("nvidia-cublas", "nvidia-cublas-cu12"):
        if check_pkg_version_at_least(pkg, "12.9"):
            return True
    return False


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        # Warn once per env var key (not per value), otherwise different keys that share the
        # same invalid value may suppress warnings incorrectly.
        if name not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) encountered unrecognized value={value} and will treat as false"
            )
        _warned_bool_env_var_keys.add(name)

    return value in truthy_values


def get_int_env_var(name: str, default: int = 0) -> int:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def support_triton(backend: str) -> bool:
    return backend not in ["torch_native", "intel_amx", "ascend"]


_ENABLE_TORCH_INFERENCE_MODE = get_bool_env_var(
    "SGLANG_ENABLE_TORCH_INFERENCE_MODE", "false"
)


class DynamicGradMode(_DecoratorContextManager):
    """
    A combination of torch.no_grad and torch.inference_mode,
    with their behavior controlled by an environment variable. Just refer to them.
    """

    @staticmethod
    def set_inference_mode(mode: bool):
        if isinstance(mode, bool):
            global _ENABLE_TORCH_INFERENCE_MODE

            _ENABLE_TORCH_INFERENCE_MODE = mode
        else:
            logger.warning("mode is not a boolean object")

    def __init__(self, mode=True):
        if not torch._jit_internal.is_scripting():
            super().__init__()
        if _ENABLE_TORCH_INFERENCE_MODE:
            self.mode = mode
        else:
            self.prev = False

    def __new__(cls, mode_or_orig_func=True if _ENABLE_TORCH_INFERENCE_MODE else None):
        if mode_or_orig_func is None or isinstance(mode_or_orig_func, bool):
            return super().__new__(cls)
        return cls()(mode_or_orig_func)

    def __enter__(self) -> None:
        if _ENABLE_TORCH_INFERENCE_MODE:
            self._inference_mode_context = torch._C._InferenceMode(self.mode)
            self._inference_mode_context.__enter__()
        else:
            self.prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if _ENABLE_TORCH_INFERENCE_MODE:
            self._inference_mode_context.__exit__(exc_type, exc_value, traceback)
        else:
            torch.set_grad_enabled(self.prev)

    def clone(self) -> "DynamicGradMode":
        r"""
        Create a copy of this class
        """
        if _ENABLE_TORCH_INFERENCE_MODE:
            return self.__class__(self.mode)
        else:
            return self.__class__()


show_time_cost = False
time_infos = {}


def enable_show_time_cost():
    global show_time_cost
    show_time_cost = True


class TimeInfo:
    def __init__(self, name, interval=0.1, color=0, indent=0):
        self.name = name
        self.interval = interval
        self.color = color
        self.indent = indent

        self.acc_time = 0
        self.last_acc_time = 0

    def check(self):
        if self.acc_time - self.last_acc_time > self.interval:
            self.last_acc_time = self.acc_time
            return True
        return False

    def pretty_print(self):
        print(f"\x1b[{self.color}m", end="")
        print("-" * self.indent * 2, end="")
        print(f"{self.name}: {self.acc_time:.3f}s\x1b[0m")


def mark_start(name, interval=0.1, color=0, indent=0):
    global time_infos, show_time_cost
    if not show_time_cost:
        return
    torch.cuda.synchronize()
    if time_infos.get(name, None) is None:
        time_infos[name] = TimeInfo(name, interval, color, indent)
    time_infos[name].acc_time -= time.perf_counter()


def mark_end(name):
    global time_infos, show_time_cost
    if not show_time_cost:
        return
    torch.cuda.synchronize()
    time_infos[name].acc_time += time.perf_counter()
    if time_infos[name].check():
        time_infos[name].pretty_print()


def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.perf_counter()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.perf_counter() - start_time) * 1000
                if cost_time > min_cost_ms:
                    print(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


def get_available_gpu_memory(
    device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        assert gpu_id < num_gpus

        if torch.cuda.current_device() != gpu_id:
            print(
                f"WARNING: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
                "which may cause useless memory allocation for torch CUDA context.",
            )

        if empty_cache:
            torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(gpu_id)
        if props.is_integrated:
            # On these devices, which use sysmem as device mem, torch.cuda.mem_get_info()
            # only reports "free" memory, which can be lower than what is actually
            # available due to not including cache memory. So we use the system available
            # memory metric instead.
            free_gpu_memory = psutil.virtual_memory().available
        else:
            free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    elif device == "xpu":
        num_gpus = torch.xpu.device_count()
        assert gpu_id < num_gpus

        if torch.xpu.current_device() != gpu_id:
            print(
                f"WARNING: current device is not {gpu_id}, but {torch.xpu.current_device()}, ",
                "which may cause useless memory allocation for torch XPU context.",
            )

        if empty_cache:
            torch.xpu.empty_cache()
        used_memory = torch.xpu.memory_allocated()
        total_gpu_memory = torch.xpu.get_device_properties(gpu_id).total_memory
        free_gpu_memory = total_gpu_memory - used_memory

    elif device == "hpu":
        num_gpus = torch.hpu.device_count()
        assert gpu_id < num_gpus

        if torch.hpu.current_device() != gpu_id:
            print(
                f"WARNING: current device is not {gpu_id}, but {torch.hpu.current_device()}, ",
                "which may cause useless memory allocation for torch HPU context.",
            )

        free_gpu_memory, total_gpu_memory = torch.hpu.mem_get_info()

    elif device == "cpu":
        # TODO: rename the variables in the current function to be not GPU specific
        total_free_memory = psutil.virtual_memory().available
        n_numa_node: int = len(get_cpu_ids_by_node())
        free_gpu_memory = round(total_free_memory / n_numa_node, 3)
    elif device == "npu":
        num_gpus = torch.npu.device_count()
        assert gpu_id < num_gpus

        if torch.npu.current_device() != gpu_id:
            print(
                f"WARNING: current device is not {gpu_id}, but {torch.npu.current_device()}, ",
                "which may cause useless memory allocation for torch NPU context.",
            )
        if empty_cache:
            torch.npu.empty_cache()
        free_gpu_memory, total_gpu_memory = torch.npu.mem_get_info()
    elif device == "musa":
        num_gpus = torch.musa.device_count()
        assert gpu_id < num_gpus

        if torch.musa.current_device() != gpu_id:
            print(
                f"WARNING: current device is not {gpu_id}, but {torch.musa.current_device()}, ",
                "which may cause useless memory allocation for torch MUSA context.",
            )
        if empty_cache:
            torch.musa.empty_cache()
        props = torch.musa.get_device_properties(gpu_id)
        if props.is_integrated:
            # On these devices, which use sysmem as device mem, torch.musa.mem_get_info()
            # only reports "free" memory, which can be lower than what is actually
            # available due to not including cache memory. So we use the system available
            # memory metric instead.
            free_gpu_memory = psutil.virtual_memory().available
        free_gpu_memory, total_gpu_memory = torch.musa.mem_get_info()
    elif device == "mps":
        free_gpu_memory = psutil.virtual_memory().available

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def is_pin_memory_available(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device is not None and str(device) == "cpu":
        return False
    return True


class LayerFn(Protocol):

    def __call__(self, idx: int, prefix: str) -> torch.nn.Module: ...


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    prefix: str = "",
    return_tuple: bool = False,
    offloader_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.Module, int, int]:
    """Make a list of layers with the given layer function"""
    # circular imports
    from sglang.srt.distributed import get_pp_indices
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.utils.offloader import get_offloader

    assert not pp_size or num_hidden_layers >= pp_size
    start_layer, end_layer = (
        get_pp_indices(
            num_hidden_layers,
            pp_rank,
            pp_size,
        )
        if pp_rank is not None and pp_size is not None
        else (0, num_hidden_layers)
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer(return_tuple=return_tuple) for _ in range(start_layer)]
        + get_offloader().wrap_modules(
            (
                layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
                for idx in range(start_layer, end_layer)
            ),
            **(offloader_kwargs or {}),
        )
        + [
            PPMissingLayer(return_tuple=return_tuple)
            for _ in range(end_layer, num_hidden_layers)
        ]
    )
    if pp_rank is None or pp_size is None:
        return modules
    return modules, start_layer, end_layer


def make_layers_non_pp(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str = "",
) -> torch.nn.ModuleList:
    from sglang.srt.utils.offloader import get_offloader

    layers = torch.nn.ModuleList(
        get_offloader().wrap_modules(
            (
                layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
                for idx in range(num_hidden_layers)
            )
        )
    )
    return layers


@lru_cache(maxsize=1)
def get_device_module():
    return torch.get_device_module()


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(
    audio_file: str, sr: Optional[int] = None, mono: bool = True
) -> np.ndarray:
    if sr is None:
        sr = 16000

    # Normalize input: resolve URL / base64 / file:// to bytes or path
    if isinstance(audio_file, bytes):
        source = audio_file
    elif isinstance(audio_file, str) and audio_file.startswith("data:"):
        source = pybase64.b64decode(audio_file.split(",")[1], validate=True)
    elif isinstance(audio_file, str) and (
        audio_file.startswith("http://") or audio_file.startswith("https://")
    ):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
        with requests.get(audio_file, timeout=timeout) as response:
            response.raise_for_status()
            source = response.content
    elif isinstance(audio_file, str) and audio_file.startswith("file://"):
        source = unquote(urlparse(audio_file).path)
    elif isinstance(audio_file, str):
        source = audio_file
    else:
        raise ValueError(f"Invalid audio format: {audio_file}")

    if _BACKEND == "torchcodec":
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(
            source,
            sample_rate=sr,
            num_channels=1 if mono else None,
        )
        samples = decoder.get_all_samples()
        if mono:
            return samples.data.squeeze(0).numpy()
        return samples.data.T.numpy()

    # Fallback: soundfile + torchaudio (ARM / no FFmpeg)
    import soundfile as sf
    import torch
    import torchaudio

    if isinstance(source, bytes):
        audio, original_sr = sf.read(BytesIO(source))
    else:
        audio, original_sr = sf.read(source)

    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if original_sr != sr:
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.T
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=original_sr, new_freq=sr
        )
        if audio_tensor.shape[0] == 1:
            audio = audio_tensor.squeeze(0).numpy()
        else:
            audio = audio_tensor.T.numpy()

    return audio


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    max_dynamic_patch: Optional[int] = None


image_extension_names = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def is_jpeg_with_cuda(image_bytes: bytes = b"", gpu_image_decode: bool = True) -> bool:
    """
    Check three conditions:
    1. whether CUDA is available.
    2. whether input is recognized as JPEG.
    3. whether GPU image decode is enabled (some models such as CPM forcibly disable this).
    """
    if not is_cuda() or not gpu_image_decode:
        return False
    if image_bytes != b"":
        return image_bytes.startswith(b"\xff\xd8") and image_bytes.endswith(b"\xff\xd9")
    return False


def _load_image(
    image_bytes: bytes = b"",
    image_file: str = "",
    gpu_image_decode: bool = True,
) -> Union[torch.Tensor, Image.Image]:
    """
    Try to decode JPEG with nvJPEG on GPU and return a torch device tensor,
    otherwise fallback to decode with PIL on CPU and return a PIL Image.
    Keep the fallback path since nvJPEG may fail on some JPEG images that are not strictly compliant with the standard, while PIL is more tolerant.
    """
    if image_file != "":
        image_bytes = get_image_bytes(image_file)
    if is_jpeg_with_cuda(image_bytes, gpu_image_decode):
        try:
            encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)
            image_tensor = decode_jpeg(encoded_image, device="cuda")
            return image_tensor
        except Exception as e:
            logger.warning(
                f"Failed to decode JPEG on GPU, falling back to CPU. Error: {e}"
            )
    return Image.open(BytesIO(image_bytes))


def load_image(
    image_file: Union[Image.Image, str, ImageData, bytes],
    gpu_image_decode: bool = True,
) -> tuple[Union[torch.Tensor, Image.Image], Optional[tuple[int, int]]]:
    """
    Load image from multiple input formats, including:
    ImageData, PIL Image, bytes, URL, file path, or base64 string.
    """
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image = None
    image_size: Optional[tuple[int, int]] = None
    if isinstance(image_file, Image.Image):
        image = image_file
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = _load_image(image_bytes=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith(("http://", "https://")):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith("file://"):
        image = _load_image(
            image_file=unquote(urlparse(image_file).path),
            gpu_image_decode=gpu_image_decode,
        )
    elif isinstance(image_file, str) and image_file.lower().endswith(
        image_extension_names
    ):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith("data:"):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(
        image_file, str
    ):  # Other formats, try to decode as base64 by default
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    else:
        raise ValueError(f"Invalid image: {image_file}")
    return image, image_size


def get_image_bytes(image_file: Union[str, bytes]) -> bytes:
    """Normalize various image inputs into raw bytes."""
    if isinstance(image_file, bytes):
        return image_file
    if image_file.startswith(("http://", "https://")):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        try:
            response.raise_for_status()
            result = response.content
        finally:
            response.close()
        return result
    if image_file.startswith(("file://", "/")):
        with open(image_file, "rb") as f:
            return f.read()
    if isinstance(image_file, str) and image_file.startswith("data:"):
        _, encoded = image_file.split(",", 1)
        return pybase64.b64decode(encoded, validate=True)
    if isinstance(image_file, str):
        return pybase64.b64decode(image_file, validate=True)
    raise NotImplementedError(f"Invalid image: {image_file}")


def _normalize_video_input(
    video_file: Union[str, bytes],
) -> Union[str, bytes, None]:
    """Normalize video input (URL, base64, file://, etc.) to a file path or bytes.

    Returns a file path or bytes suitable for a decoder, or None on failure.
    URLs and base64 are returned as bytes (no temp files needed since both
    torchcodec and VideoDecoderWrapper accept bytes natively).
    """
    if isinstance(video_file, bytes):
        return video_file
    elif isinstance(video_file, str):
        if video_file.startswith(("http://", "https://")):
            timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
            response = requests.get(video_file, stream=True, timeout=timeout)
            response.raise_for_status()
            return response.content
        elif video_file.startswith("data:"):
            _, encoded = video_file.split(",", 1)
            return pybase64.b64decode(encoded, validate=True)
        elif video_file.startswith("file://"):
            return unquote(urlparse(video_file).path)
        elif os.path.isfile(unquote(urlparse(video_file).path)):
            return video_file
        else:
            return pybase64.b64decode(video_file, validate=True)
    else:
        return None


def load_video(video_file: Union[str, bytes], use_gpu: bool = True):
    if isinstance(video_file, (list, tuple, torch.Tensor, np.ndarray)):
        return video_file

    source = _normalize_video_input(video_file)
    if source is None:
        raise ValueError(f"Unsupported video input type: {type(video_file)}")

    device = "cuda" if use_gpu else "cpu"
    return VideoDecoderWrapper(source, device=device)


def sample_video_frames(video, *, desired_fps: int, max_frames: int) -> list[int]:
    total_frames = len(video)
    assert total_frames > 0, "Video must have at least one frame"

    avg_fps = video.avg_fps
    duration = total_frames / avg_fps if avg_fps > 0 else 0
    fps = min(desired_fps, avg_fps)

    num_frames = math.floor(duration * fps)
    num_frames = min(max_frames, num_frames, total_frames)
    num_frames = max(1, num_frames)  # At least one frame
    if num_frames == total_frames:
        return list(range(total_frames))
    else:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def encode_video(video_path, frame_count_limit=None):
    if not os.path.exists(video_path):
        logger.error(f"Video {video_path} does not exist")
        return []

    if frame_count_limit == 0:
        return []

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    decoder = VideoDecoderWrapper(video_path)
    avg_fps = decoder.avg_fps
    total_frames = len(decoder)

    sample_fps = round(avg_fps / 1)
    if sample_fps == 0:
        sample_fps = 1
    frame_indices = [i for i in range(0, total_frames, sample_fps)]
    if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
        frame_indices = uniform_sample(frame_indices, frame_count_limit)

    if not frame_indices:
        return []

    frames_data = decoder.get_frames_at(frame_indices)
    frames = [Image.fromarray(v.astype("uint8")) for v in frames_data]

    return frames


def suppress_noisy_warnings():
    """Suppress known noisy warnings from third-party libraries."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )
    warnings.filterwarnings(
        "ignore",
        message="The cuda.cudart module is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The cuda.nvrtc module is deprecated",
        category=FutureWarning,
    )

    # Suppress noisy third-party HTTP loggers.
    # huggingface_hub uses httpx which logs every HTTP request at INFO level.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def suppress_other_loggers():
    suppress_noisy_warnings()

    try:
        from vllm.logger import logger as vllm_default_logger
    except ImportError:
        return

    vllm_default_logger.setLevel(logging.WARN)
    logging.getLogger("vllm.distributed.device_communicators.pynccl").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.distributed.device_communicators.shm_broadcast").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.config").setLevel(logging.ERROR)


def assert_pkg_version(pkg: str, min_version: str, message: str):
    try:
        installed_version = version(pkg)
        if pkg_version.parse(installed_version) < pkg_version.parse(min_version):
            raise Exception(
                f"{pkg} is installed with version {installed_version}, which "
                f"is less than the minimum required version {min_version}. " + message
            )
    except PackageNotFoundError:
        raise Exception(
            f"{pkg} with minimum required version {min_version} is not installed. "
            + message
        )


def check_pkg_version_at_least(pkg: str, min_version: str) -> bool:
    """
    Check if a package is installed and meets the minimum version requirement.

    Args:
        pkg: Package name (distribution name, e.g., "flashinfer-python")
        min_version: Minimum version required (e.g., "0.6.7")

    Returns:
        True if package is installed and version >= min_version, False otherwise
    """
    try:
        installed_version = version(pkg)
        return pkg_version.parse(installed_version) >= pkg_version.parse(min_version)
    except PackageNotFoundError:
        return False


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def monkey_patch_p2p_access_check():
    """
    Monkey patch the slow p2p access check.
    NOTE: We assume the p2p access is always allowed, which can be wrong for some setups.
    """

    import sglang.srt.distributed.device_communicators.custom_all_reduce_utils as tgt

    setattr(tgt, "gpu_p2p_access_check", lambda *arg, **kwargs: True)

    # Suppress the warnings from this delete function when using sglang.bench_one_batch
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    setattr(CustomAllreduce, "__del__", lambda *args, **kwargs: None)


def set_ulimit(target_soft_limit=65535):
    # number of open files
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(f"Fail to set RLIMIT_NOFILE: {e}")

    # stack size
    resource_type = resource.RLIMIT_STACK
    current_soft, current_hard = resource.getrlimit(resource_type)
    target_soft_limit_stack_size = 1024 * target_soft_limit
    if current_soft < target_soft_limit_stack_size:
        try:
            resource.setrlimit(
                resource_type, (target_soft_limit_stack_size, current_hard)
            )
        except ValueError as e:
            logger.warning(f"Fail to set RLIMIT_STACK: {e}")


def rank0_log(msg: str):
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized() or get_tensor_model_parallel_rank() == 0:
        logger.info(msg)


def configure_logger(server_args, prefix: str = ""):
    if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
            raise Exception(
                "Setting SGLANG_LOGGING_CONFIG_PATH from env with "
                f"{SGLANG_LOGGING_CONFIG_PATH} but it does not exist!"
            )
        with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = orjson.loads(file.read())
        logging.config.dictConfig(custom_config)
        return
    maybe_ms = ".%(msecs)03d" if envs.SGLANG_LOG_MS.get() else ""
    format = f"[%(asctime)s{maybe_ms}{prefix}] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Suppress noisy httpx/httpcore loggers in every process that calls
    # configure_logger (main, scheduler, detokenizer). Spawned subprocesses
    # don't inherit the parent's logger state, so this must run here too.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# source: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/lora/utils.py#L9
def replace_submodule(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = True,
):
    """Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not force_cpu_device
        else "musa" if is_musa() and not force_cpu_device else "cpu"
    )

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


def point_to_point_pyobj(
    data: List[Any],
    rank: int,
    group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    dst: int = 1,
    async_send: bool = False,
):
    """Send data from src to dst in group."""
    from sglang.srt.distributed.parallel_state import P2PWork

    if async_send:
        send_func = dist.isend
    else:
        send_func = dist.send
    if rank == src:
        p2p_works = []
        if len(data) == 0:
            tensor_size = torch.tensor(
                [0],
                dtype=torch.long,
            )
            work = send_func(tensor_size, dst, group=group)
            if async_send:
                p2p_works.append(P2PWork(work, tensor_size))
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            )
            tensor_size = torch.tensor([size], dtype=torch.long)

            work = send_func(tensor_size, dst, group=group)
            if async_send:
                p2p_works.append(P2PWork(work, tensor_size))
            work = send_func(tensor_data, dst, group=group)
            if async_send:
                p2p_works.append(P2PWork(work, tensor_data))
        return p2p_works

    elif rank == dst:
        tensor_size = torch.tensor(
            [0],
            dtype=torch.long,
        )
        work = dist.irecv(tensor_size, src=src, group=group)
        work.wait()
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(
            size,
            dtype=torch.uint8,
        )
        work = dist.irecv(tensor_data, src=src, group=group)
        work.wait()

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data

    # Other ranks in pp_group do nothing
    return []


def delete_directory(dirpath):
    try:
        # This will remove the directory and all its contents
        shutil.rmtree(dirpath)
    except OSError as e:
        print(f"Warning: {dirpath} : {e.strerror}")


# Temporary directory for prometheus multiprocess mode
# Cleaned up automatically when this object is garbage collected
prometheus_multiproc_dir: tempfile.TemporaryDirectory


def set_prometheus_multiproc_dir():
    # Set prometheus multiprocess directory
    # sglang uses prometheus multiprocess mode
    # we need to set this before importing prometheus_client
    # https://prometheus.github.io/client_python/multiprocess/
    global prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.debug("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.debug(f"PROMETHEUS_MULTIPROC_DIR: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")


def add_prometheus_middleware(app):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


class RefCountedGauge:
    def __init__(self, gauge):
        self._gauge = gauge
        self._refcount: Dict[str, int] = {}

    def inc(self, key: str):
        if key in self._refcount:
            self._refcount[key] += 1
        else:
            self._refcount[key] = 1
            self._gauge.inc()

    def dec(self, key: str):
        if key in self._refcount:
            self._refcount[key] -= 1
            if self._refcount[key] == 0:
                del self._refcount[key]
                self._gauge.dec()


def add_prometheus_track_response_middleware(app):
    from prometheus_client import Counter, Gauge

    http_request_counter = Counter(
        name="sglang:http_requests_total",
        documentation="Total number of HTTP requests by endpoint and method",
        labelnames=["endpoint", "method"],
    )

    http_response_counter = Counter(
        name="sglang:http_responses_total",
        documentation="Total number of HTTP responses by endpoint and status code",
        labelnames=["endpoint", "status_code", "method"],
    )

    http_requests_active = Gauge(
        name="sglang:http_requests_active",
        documentation="Number of currently active HTTP requests",
        labelnames=["endpoint", "method"],
        multiprocess_mode="livesum",
    )

    routing_keys_active = RefCountedGauge(
        Gauge(
            name="sglang:routing_keys_active",
            documentation="Number of unique routing keys with active requests",
            multiprocess_mode="livesum",
        )
    )

    # Fix: replace BaseHTTPMiddleware's call_next with a pure ASGI version
    # that passes `receive` through, so request.is_disconnected() keeps working.
    from sglang.srt.utils.http_middleware_patch import patch_app_http_middleware

    patch_app_http_middleware(app)

    @app.middleware("http")
    async def track_http_status_code(request, call_next):
        # With recording all requests, we have the risk of high cardinality if requests have arbitrary unhandled paths.
        # But given that SGLang engines with metrics enabled are usually behind routers this looks safe.
        path, is_handled_path = _get_fastapi_request_path(request)
        method = request.method
        routing_key = request.headers.get("x-smg-routing-key")

        http_request_counter.labels(endpoint=path, method=method).inc()
        http_requests_active.labels(endpoint=path, method=method).inc()
        if routing_key:
            routing_keys_active.inc(routing_key)

        try:
            response = await call_next(request)

            http_response_counter.labels(
                endpoint=path,
                method=method,
                status_code=str(response.status_code),
            ).inc()

            return response
        finally:
            http_requests_active.labels(endpoint=path, method=method).dec()
            if routing_key:
                routing_keys_active.dec(routing_key)


# https://github.com/blueswen/fastapi-observability/blob/132a3c576f8b09e5311c68bd553215013bc75685/fastapi_app/utils.py#L98
def _get_fastapi_request_path(request) -> Tuple[str, bool]:
    from starlette.routing import Match

    for route in request.app.routes:
        match, child_scope = route.matches(request.scope)
        if match == Match.FULL:
            return route.path, True

    return request.url.path, False


def get_amdgpu_memory_capacity():
    try:
        # Run rocm-smi and capture the output
        result = subprocess.run(
            [
                "rocminfo | grep 'gfx' -A 100 | grep 'Pool 1' -A 5 | grep 'Size:' | awk '{print $2}'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"rocm-smi error: {result.stderr.strip()}")

        # Parse the output to extract memory values in MiB
        memory_values = [
            float(mem.split("(")[0].strip()) / 1024
            for mem in result.stdout.strip().split("\n")
        ]

        if not memory_values:
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "rocm-smi not found. Ensure AMD ROCm drivers are installed and accessible."
        )


def get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def _cuda_mem_fallback(reason: str) -> int:
    """Fallback to torch.cuda.mem_get_info() and return total GPU memory in MiB.

    Queries all visible CUDA devices and returns the minimum total memory,
    consistent with the nvidia-smi path that takes min(memory_values).

    Returns the total memory in MiB, or raises RuntimeError if CUDA is
    unavailable or mem_get_info() fails.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(reason)
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            # Include the original failure reason for diagnostics
            raise RuntimeError(f"{reason} No CUDA devices found via torch.cuda.")
        memory_values = []
        for i in range(device_count):
            total = torch.cuda.mem_get_info(i)[1] // 1024 // 1024  # unit: MiB
            memory_values.append(total)
        result = min(memory_values)
        logger.warning(
            f"{reason} Falling back to torch.cuda.mem_get_info(). "
            f"Reported total GPU memory per device (MiB): {memory_values}, "
            f"using min: {result} MiB."
        )
        return result
    except (RuntimeError, ValueError, OSError) as e:
        raise RuntimeError(
            f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}"
        ) from e


def get_nvgpu_memory_capacity():
    try:
        # Run nvidia-smi and capture the output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return _cuda_mem_fallback(
                f"nvidia-smi failed (exit code {result.returncode}: {result.stderr.strip()})."
            )

        # Parse the output to extract memory values
        memory_values = [
            float(mem)
            for mem in result.stdout.strip().split("\n")
            if re.match(r"^\d+(\.\d+)?$", mem.strip())
        ]

        if not memory_values:
            # Fallback when nvidia-smi returns no parseable values,
            # typically in NVIDIA MIG mode.
            return _cuda_mem_fallback(
                "Failed to get GPU memory capacity from nvidia-smi."
            )

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        return _cuda_mem_fallback(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible."
        )


def get_hpu_memory_capacity():
    try:
        # Run hl-smi and capture the output
        result = subprocess.run(
            ["hl-smi --query | grep 'Total'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"hl-smi error: {result.stderr.strip()}")

        # Parse the output to extract memory values in MiB
        memory_values = [
            float(mem.split(" ")[-2]) for mem in result.stdout.strip().split("\n")
        ]

        if not memory_values:
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "hl-smi not found. Ensure Habana drivers are installed and accessible."
        )


def get_npu_memory_capacity():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.mem_get_info()[1] // 1024 // 1024  # unit: MB
    except ImportError as e:
        raise ImportError("torch_npu is required when run on npu device.")


def get_cpu_memory_capacity():
    # Per-rank memory capacity cannot be determined for customized core settings
    if os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", ""):
        return None
    n_numa_node: int = len(get_cpu_ids_by_node())
    if n_numa_node == 0:
        # Cannot determine NUMA config, fallback to total memory and avoid ZeroDivisionError.
        return float(psutil.virtual_memory().total // (1 << 20))
    try:
        numa_mem_list = list()
        file_prefix = "/sys/devices/system/node/"
        for numa_id in range(n_numa_node):
            file_meminfo = f"node{numa_id}/meminfo"
            with open(os.path.join(file_prefix, file_meminfo), "r") as f:
                # MemTotal info is at the 1st line
                line = f.readline()
                # Expected format: "Node 0 MemTotal:       100000000 kB"
                parts = line.split()
                if len(parts) >= 4 and parts[2] == "MemTotal:":
                    numa_mem_list.append(int(parts[3]))
                else:
                    raise ValueError(f"Unexpected format in {file_meminfo}: {line}")
        # Retrieved value in KB, need MB
        numa_mem = float(min(numa_mem_list) // 1024)
        return numa_mem
    except (FileNotFoundError, ValueError, IndexError):
        numa_mem = psutil.virtual_memory().total / n_numa_node
        # Retrieved value in Byte, need MB
        return float(numa_mem // (1 << 20))


def get_xpu_memory_capacity():
    try:
        if torch.xpu.is_available():
            return torch.xpu.mem_get_info()[1] // 1024 // 1024  # unit: MB
        raise ValueError("No GPU memory values found.")
    except AttributeError:
        raise RuntimeError("torch.xpu is not available.")


def get_mtgpu_memory_capacity():
    try:
        # Run mthreads-gmi and capture the output
        result = subprocess.run(
            [
                "mthreads-gmi --query | grep 'FB Memory Usage' -A 2 | grep 'Total' | awk -F':' '{print $2}' | awk '{print $1}' | sed 's/MiB//'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"mthreads-gmi error: {result.stderr.strip()}")

        # Parse the output to extract memory values
        memory_values = [
            float(mem)
            for mem in result.stdout.strip().split("\n")
            if re.match(r"^\d+(\.\d+)?$", mem.strip())
        ]

        if not memory_values:
            # Fallback to torch.musa.mem_get_info() when failed to get memory capacity from mthreads-gmi.
            if hasattr(torch, "musa") and torch.musa.is_available():
                logger.warning(
                    "Failed to get GPU memory capacity from mthreads-gmi, falling back to torch.musa.mem_get_info()."
                )
                return torch.musa.mem_get_info()[1] // 1024 // 1024  # unit: MB
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "mthreads-gmi not found. Ensure Moore Threads drivers are installed and accessible."
        )


def get_device_memory_capacity(device: str = None):
    if is_cuda():
        gpu_mem = get_nvgpu_memory_capacity()
    elif is_hip():
        gpu_mem = get_amdgpu_memory_capacity()
    elif device == "hpu":
        gpu_mem = get_hpu_memory_capacity()
    elif device == "npu":
        gpu_mem = get_npu_memory_capacity()
    elif device == "cpu":
        gpu_mem = get_cpu_memory_capacity()
    elif device == "xpu":
        gpu_mem = get_xpu_memory_capacity()
    elif device == "musa":
        gpu_mem = get_mtgpu_memory_capacity()
    else:
        # GPU memory is not known yet or no GPU is available.
        gpu_mem = None

    return gpu_mem


# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
    device_id=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if torch_release >= (2, 6) else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
        device_id=device_id,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def crash_on_warnings():
    # Crash on warning if we are running CI tests
    return get_bool_env_var("SGLANG_IS_IN_CI")


@functools.lru_cache(None)
def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg)


@functools.lru_cache(None)
def print_info_once(msg: str) -> None:
    logger.info(msg)


def get_device_name(device_id: int = 0) -> str:
    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        return torch.cuda.get_device_name(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(device_id)

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu.get_device_name(device_id)

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.get_device_name(device_id)


@lru_cache(maxsize=1)
def is_habana_available() -> bool:
    return find_spec("habana_frameworks") is not None


@lru_cache(maxsize=8)
def get_device(device_id: Optional[int] = None) -> str:
    if is_cpu():
        if cpu_has_amx_support():
            logger.info("Intel AMX is detected, using CPU with Intel AMX support.")
        else:
            logger.warning(
                "CPU device enabled, using torch native backend, low performance expected."
            )
        return "cpu"

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if device_id is None:
            return "cuda"
        return "cuda:{}".format(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if device_id is None:
            return "xpu"
        return "xpu:{}".format(device_id)

    if is_npu():
        if device_id is None:
            return "npu"
        return "npu:{}".format(device_id)

    if is_habana_available():
        try:
            import habana_frameworks.torch.hpu  # noqa: F401

            if torch.hpu.is_available():
                if device_id is None:
                    return "hpu"
                return "hpu:{}".format(device_id)
        except ImportError:
            raise ImportError(
                "Habana frameworks detected, but failed to import 'habana_frameworks.torch.hpu'."
            )

    if is_musa():
        if device_id is None:
            return "musa"
        return "musa:{}".format(device_id)

    if is_mps():
        if device_id is None:
            return "mps"
        return "mps:{}".format(device_id)

    raise RuntimeError("No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) is available.")


@lru_cache(maxsize=1)
def get_device_count() -> int:
    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        try:
            return torch.cuda.device_count()
        except RuntimeError:
            return 0

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            return torch.xpu.device_count()
        except RuntimeError:
            return 0

    if is_habana_available():
        try:
            import habana_frameworks.torch.hpu  # noqa: F401

            if torch.hpu.is_available():
                return torch.hpu.device_count()
        except (ImportError, RuntimeError):
            return 0

    return 0  # No accelerators available


def get_device_core_count(device_id: int = 0) -> int:
    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    return 0


def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
    major, minor = None, None
    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        major, minor = torch.cuda.get_device_capability(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        major, minor, *_ = torch.xpu.get_device_capability(device_id)["version"].split(
            "."
        )
        # Currently XPU version does not contain capability information.
        major, minor = None, None

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        try:
            # TODO(HandH1998): `get_device_capability` is not supported by `torch.hpu` for now.
            # Update this once the support is available.
            # major, minor = torch.hpu.get_device_capability(device_id)
            major, minor = None, None
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while getting device capability of hpu: {e}."
            ) from e

    return major, minor


def get_compiler_backend(mode=None) -> str:
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return "hpu_backend"

    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            import torchair
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
            from torchair.configs.compiler_config import CompilerConfig
        except ImportError as e:
            raise ImportError(
                "NPU detected, but torchair package is not installed. "
                "Please install torchair for torch.compile support on NPU."
            )
        compiler_config = CompilerConfig()
        compiler_config.mode = "max-autotune"
        if mode == "npugraph_ex":
            compiler_config.mode = "reduce-overhead"
            compiler_config.debug.run_eagerly = True
        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        return npu_backend

    return "inductor"


sglang_lib = Library("sglang", "FRAGMENT")  # noqa


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
) -> None:
    """
    NOTE: Please try to use `register_custom_op` instead of this function.
    See `python/sglang/srt/utils/custom_op.py` for details.

    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.

    Note: This function will silently skip registration if the operator
    with the same name is already registered to avoid RuntimeError in
    multi-engine scenarios (e.g., VERL framework).
    """
    import torch.library

    my_lib = target_lib or sglang_lib

    # Check if operator is already registered to avoid duplicate registration
    # This is important for scenarios where multiple SGLang engines run in the same process
    try:
        # Try to access the operator to see if it's already registered
        lib_name = my_lib.m.name if hasattr(my_lib.m, "name") else "sglang"
        if hasattr(torch.ops, lib_name) and hasattr(
            getattr(torch.ops, lib_name), op_name
        ):
            # Operator already exists, skip registration
            return
    except (AttributeError, RuntimeError):
        # Operator doesn't exist, proceed with registration
        pass

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

    try:
        my_lib.define(op_name + schema_str)
        if is_npu():
            # https://github.com/sgl-project/sglang/pull/12287/files#r2499583982
            my_lib.impl(op_name, op_func, "PrivateUse1")
        elif is_xpu():
            my_lib.impl(op_name, op_func, "XPU")
        else:
            my_lib.impl(op_name, op_func, "CUDA")
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    except RuntimeError as error:
        if "Tried to register an operator" in str(error) and "multiple times" in str(
            error
        ):
            # Silently ignore duplicate registration errors
            # This can happen in multi-engine scenarios
            pass
        else:
            # Re-raise other RuntimeErrors
            raise error
    except AttributeError as error:
        # Always re-raise AttributeError as it indicates missing dependencies
        raise error


def set_gpu_proc_affinity(
    pp_size: int,
    tp_size: int,
    nnodes: int,
    gpu_id: int,
):
    # current process
    pid = os.getpid()
    p = psutil.Process(pid)

    nnodes_per_tp_group = max(nnodes // pp_size, 1)
    tp_size_per_node = tp_size // nnodes_per_tp_group

    # total physical cores
    total_pcores = psutil.cpu_count(logical=False)
    # physical cores per TP (N.B. more Cores than GPUs on node)
    num_cores_bind = total_pcores // tp_size_per_node

    # able to handle multiple DP per node
    start_cpu_id = (gpu_id * num_cores_bind) % total_pcores
    end_cpu_id = start_cpu_id + num_cores_bind

    if psutil.cpu_count() != psutil.cpu_count(logical=False):
        # HT on
        lower_cpu_ids = [id for id in range(start_cpu_id, end_cpu_id)]
        upper_cpu_ids = [id + total_pcores for id in range(start_cpu_id, end_cpu_id)]
        bind_cpu_ids = list(itertools.chain(lower_cpu_ids, upper_cpu_ids))
    else:
        # HT off
        bind_cpu_ids = [id for id in range(start_cpu_id, end_cpu_id)]

    # set cpu_affinity to current process
    p.cpu_affinity(bind_cpu_ids)
    logger.info(f"Process {pid} gpu_id {gpu_id} is running on CPUs: {p.cpu_affinity()}")


def permute_weight(x: torch.Tensor) -> torch.Tensor:
    b_ = x.shape[0]
    n_ = x.shape[1]
    k_ = x.shape[2]

    x_ = x
    if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
        x_ = x_.view(int(b_), int(n_ / 16), 16, int(k_ / 32), 4, 8)
    elif x.dtype == torch.float8_e4m3fnuz or x.dtype == torch.int8:
        x_ = x_.view(int(b_), int(n_ / 16), 16, int(k_ / 64), 4, 16)
    else:
        # return x_
        x_ = x_.view(int(b_), int(n_ / 16), 16, int(k_ / 8), 2, 4)

    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return SafeUnpickler(io.BytesIO(data)).load()


class SafeUnpickler(pickle.Unpickler):
    ALLOWED_MODULE_PREFIXES = {
        # --- Python types ---
        "builtins.",
        "collections.",
        "copyreg.",
        "functools.",
        "itertools.",
        "operator.",
        "types.",
        "weakref.",
        # --- PyTorch types ---
        "torch.",
        "torch._tensor.",
        "torch.storage.",
        "torch.nn.parameter.",
        "torch.autograd.function.",
        # --- torch distributed ---
        "torch.distributed.",
        "torch.distributed._shard.",
        "torch.distributed._composable.",
        "torch._C._distributed_c10d.",
        "torch._C._distributed_fsdp.",
        "torch.distributed.optim.",
        # --- multiprocessing ---
        "multiprocessing.resource_sharer.",
        "multiprocessing.reduction.",
        "pickletools.",
        # --- PEFT / LoRA ---
        "peft.",
        "transformers.",
        "huggingface_hub.",
        # --- SGLang & Unitest ---
        "sglang.srt.weight_sync.tensor_bucket.",
        "sglang.srt.model_executor.model_runner.",
        "sglang.srt.layers.",
        "sglang.srt.utils.",
        "torch_npu.",
    }

    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("os", "system"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("codecs", "decode"),
        ("types", "CodeType"),
        ("types", "FunctionType"),
    }

    def find_class(self, module, name):
        # Block deterministic attacks
        if (module, name) in self.DENY_CLASSES:
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name}), "
                f"to prevent exploitation of CVE-2025-10164"
            )
        # Allowlist of safe-to-load modules.
        if any(
            (module + ".").startswith(prefix) for prefix in self.ALLOWED_MODULE_PREFIXES
        ):
            return super().find_class(module, name)

        # Block everything else. (Potential attack surface)
        raise RuntimeError(
            f"Blocked unsafe class loading ({module}.{name}), "
            f"to prevent exploitation of CVE-2025-10164"
        )


def safe_pickle_load(fp):
    """Drop-in replacement for pickle.load() that blocks unsafe class loading."""
    return SafeUnpickler(fp).load()


def debug_timing(func):
    # todo: replace with a more organized instrumentation
    def wrapper(*args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            tic.record()
            result = func(*args, **kwargs)
            toc.record()
            toc.synchronize()  # Wait for the function to complete without synchronizing all ops on the GPU
            elapsed = tic.elapsed_time(toc)
            indices = kwargs.get("indices", args[1] if len(args) > 1 else None)
            num_tokens = len(indices) if indices is not None else 0
            throughput = num_tokens / elapsed * 1000 if elapsed > 0 else 0
            logger.debug(
                f"Transfer time: {elapsed} ms, throughput: {throughput} tokens/s"
            )
            return result
        else:
            return func(*args, **kwargs)

    return wrapper


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def human_readable_int(value: str) -> int:
    """Supports standard SI suffixes (k, M, G, T) and IEC suffixes
    (Ki, Mi, Gi, Ti). Suffixes are case-sensitive.

    Decimals are allowed for SI suffixes only.

    Examples:
        '1k' -> 1000      '1M' -> 1000000    '25.6k' -> 25600
        '1Ki' -> 1024     '1Mi' -> 1048576
    """
    value = value.strip()

    si_multiplier = {"k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
    iec_multiplier = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}

    match = re.fullmatch(r"(\d+(?:\.\d+)?)(Ki|Mi|Gi|Ti|k|M|G|T)", value)
    if match:
        number, suffix = match.groups()
        if suffix in iec_multiplier:
            if "." in number:
                raise argparse.ArgumentTypeError(
                    f"Decimals are not allowed with IEC suffixes like '{suffix}'. "
                    f"Use an integer IEC value such as '{int(Decimal(number))}{suffix}', "
                    f"or an SI value such as '{number}{suffix[0]}'."
                )
            return int(number) * iec_multiplier[suffix]
        return int(Decimal(number) * si_multiplier[suffix])

    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid integer value: '{value}'. "
            "Use a plain integer, SI suffixes (1k, 1M), or IEC suffixes (1Ki, 1Mi). "
            "Suffixes are case-sensitive."
        )


def pyspy_dump_schedulers():
    """py-spy dump on all scheduler in a local node."""
    try:
        pid = psutil.Process().pid
        # Command to run py-spy with the PID
        cmd = f"py-spy dump --native --pid {pid}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        logger.error(f"Pyspy dump for PID {pid}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Pyspy failed to dump PID {pid}. Error: {e.stderr}")


def kill_itself_when_parent_died():
    if sys.platform == "linux":
        # sigkill this process when parent worker manager dies
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    else:
        logger.warning("kill_itself_when_parent_died is only supported in linux.")


class UvicornAccessLogFilter(logging.Filter):
    """Filter uvicorn access logs by request path.

    Notes:
    - Uvicorn access records usually provide `request_line` like: "GET /metrics HTTP/1.1".
    - We defensively fall back to parsing `record.getMessage()` if needed.
    """

    def __init__(self, excluded_path_prefixes=None):
        super().__init__()
        excluded_path_prefixes = excluded_path_prefixes or []
        # Normalize once: drop empty prefixes, stringify, keep as tuple (fast iteration, immutable).
        self.excluded_path_prefixes = tuple(str(p) for p in excluded_path_prefixes if p)

    def filter(self, record: logging.LogRecord) -> bool:
        path = None

        request_line = getattr(record, "request_line", None)
        if request_line:
            parts = str(request_line).split()
            if len(parts) >= 2:
                path = parts[1]

        if not path:
            # Fallback for non-standard formatters/records
            try:
                msg = record.getMessage()
            except Exception:
                msg = None
            if msg:
                q1 = msg.find('"')
                q2 = msg.find('"', q1 + 1) if q1 != -1 else -1
                if q1 != -1 and q2 != -1:
                    rl = msg[q1 + 1 : q2]
                    parts = rl.split()
                    if len(parts) >= 2:
                        path = parts[1]

        if not path:
            return True

        # Strip query string for matching
        path = str(path)
        # Some proxies/clients may emit absolute-form request-target in logs:
        # e.g. "GET https://example.com/metrics HTTP/1.1" -> extract "/metrics".
        if "://" in path:
            try:
                path = urlparse(path).path or path
            except Exception:
                # If parsing fails, fall back to the raw value.
                pass
        path = path.split("?", 1)[0]
        return not any(
            path.startswith(prefix) for prefix in self.excluded_path_prefixes
        )


def set_uvicorn_logging_configs(server_args=None):
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    _configure_uvicorn_access_log_filter(LOGGING_CONFIG, server_args)


def _configure_uvicorn_access_log_filter(
    uvicorn_logging_config: dict, server_args=None
):
    """Configure uvicorn access log path filter into uvicorn LOGGING_CONFIG.

    This optionally filters uvicorn access logs (e.g., suppress noisy /metrics polling).

    Args:
        uvicorn_logging_config: The dict-like LOGGING_CONFIG from uvicorn.
        server_args: Parsed server args object that may contain:
            - uvicorn_access_log_exclude_prefixes (list[str] | tuple[str] | None)
    """
    # Optionally filter uvicorn access logs (e.g., suppress noisy /metrics polling).
    if server_args is None:
        return

    filter_name = "sglang_uvicorn_access_path_filter"

    excluded_prefixes = getattr(
        server_args, "uvicorn_access_log_exclude_prefixes", None
    )
    if not excluded_prefixes:
        return

    # Normalize: accept list/tuple; treat a single string as one prefix (not an iterable of chars).
    if isinstance(excluded_prefixes, str):
        excluded_prefixes = [excluded_prefixes]

    # De-duplicate while keeping order; drop empty prefixes.
    excluded_prefixes = [p for p in excluded_prefixes if p]
    excluded_prefixes = list(dict.fromkeys(excluded_prefixes))
    if not excluded_prefixes:
        return

    uvicorn_logging_config.setdefault("filters", {})
    uvicorn_logging_config["filters"][filter_name] = {
        "()": "sglang.srt.utils.common.UvicornAccessLogFilter",
        "excluded_path_prefixes": excluded_prefixes,
    }

    # Attach filter to access handler and/or uvicorn.access logger (best-effort across uvicorn versions).
    handlers = uvicorn_logging_config.get("handlers", {})
    if "access" in handlers:
        filters_list = handlers["access"].setdefault("filters", [])
        if not isinstance(filters_list, list):
            filters_list = list(filters_list)
            handlers["access"]["filters"] = filters_list
        if filter_name not in filters_list:
            filters_list.append(filter_name)

    loggers_cfg = uvicorn_logging_config.get("loggers", {})
    if "uvicorn.access" in loggers_cfg:
        filters_list = loggers_cfg["uvicorn.access"].setdefault("filters", [])
        if not isinstance(filters_list, list):
            filters_list = list(filters_list)
            loggers_cfg["uvicorn.access"]["filters"] = filters_list
        if filter_name not in filters_list:
            filters_list.append(filter_name)


def launch_dummy_health_check_server(host, port, enable_metrics):
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Response

    from sglang.srt.utils.network import NetworkAddress

    app = FastAPI()

    @app.get("/ping")
    async def ping():
        """Could be used by the checkpoint-engine update script to confirm the server is up."""
        return Response(status_code=200)

    @app.get("/health")
    async def health():
        """Check the health of the http server."""
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        """Check the health of the http server."""
        return Response(status_code=200)

    # Add prometheus middleware
    if enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_keep_alive=5,
        loop="auto",
        log_config=None,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)

    # Run server in a background daemon thread with its own event loop
    # This prevents blocking the main thread while still serving health checks
    def run_server():
        try:
            asyncio.run(server.serve())
        except Exception as e:
            logger.error(f"Dummy health check server failed to start: {e}")
            raise
        finally:
            logger.info(
                f"Dummy health check server stopped at {NetworkAddress(host, port).to_host_port_str()}"
            )

    thread = threading.Thread(
        target=run_server, daemon=True, name="health-check-server"
    )
    thread.start()
    logger.info(
        f"Dummy health check server started in background thread at {NetworkAddress(host, port).to_host_port_str()}"
    )


def set_cuda_arch():
    if is_flashinfer_available():
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = (
            f"{arch}{'a' if capability[0] >= 9 else ''}"
        )


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def round_up(x: int, y: int) -> int:
    return ((x - 1) // y + 1) * y


setattr(triton, "next_power_of_2", next_power_of_2)


class EmptyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def empty_context(*args, **kwargs):
    return EmptyContextManager()


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"


def is_remote_url(url: Union[str, Path]) -> bool:
    """
    Check if the URL is a remote URL of the format:
    <connector_type>://<host>:<port>/<model_name>
    """
    if isinstance(url, Path):
        return False

    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    return m is not None


def parse_connector_type(url: str) -> str:
    """
    Parse the connector type from the URL of the format:
    <connector_type>://<path>
    """
    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    if m is None:
        return ""

    return m.group(1)


def retry(
    fn,
    max_retry: int,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    should_retry: Callable[[Any], bool] = lambda e: True,
):
    for try_index in itertools.count():
        try:
            return fn()
        except SkipTest:
            # Do NOT retry skipped tests - used in CI and unittest
            raise
        except Exception as e:
            traceback.print_exc()

            if try_index >= max_retry:
                raise Exception(f"retry() exceed maximum number of retries.")

            if not should_retry(e):
                raise Exception(f"retry() observe errors that should not be retried.")

            delay = min(initial_delay * (2**try_index), max_delay) * (
                0.75 + 0.25 * random.random()
            )

            logger.warning(
                f"retry() failed once ({try_index}th try, maximum {max_retry} retries). Will delay {delay:.2f}s and retry. Error: {e}"
            )

            time.sleep(delay)


def has_hf_quant_config(model_path: str) -> bool:
    """Check if the model path contains hf_quant_config.json file.

    Args:
        model_path: Path to the model, can be local path or remote URL.

    Returns:
        True if hf_quant_config.json exists, False otherwise.
    """
    # Check if the model_path is a local path
    if os.path.exists(os.path.join(model_path, "hf_quant_config.json")):
        return True

    from huggingface_hub import try_to_load_from_cache

    # Check if the model_path is a HuggingFace model ID and exists locally
    result = try_to_load_from_cache(model_path, "hf_quant_config.json")
    if isinstance(result, str):
        return True

    # Check if the model_path is a remote URL and exists on the HuggingFace Hub
    try:
        from huggingface_hub import HfApi

        hf_api = HfApi()
        return hf_api.file_exists(model_path, "hf_quant_config.json")
    except Exception:
        return False


def get_quantization_config(hf_config) -> str | None:
    """Extract quantization method from HuggingFace config."""
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is not None:
        return quantization_config.get("quant_method")
    return None


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


def is_non_idle_and_non_empty(forward_mode, hidden_states):
    return (
        (forward_mode is not None)
        and not forward_mode.is_idle()
        and hidden_states.shape[0] > 0
    )


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


def bind_or_assign(target, source):
    if target is not None:
        target.copy_(source)
        return target
    else:
        return source


# TODO(hebiao064): Accelerate FA3 Spec Decode with topk > 1.
# TODO(hebiao064): Improve the acc rate for FA3 Spec Decode with topk == 1 and page_size > 1.
def is_no_spec_infer_or_topk_one(server_args):
    return server_args.speculative_eagle_topk is None or (
        server_args.speculative_eagle_topk == 1
        and (server_args.page_size == 1 or server_args.page_size is None)
    )


def is_fa3_default_architecture(hf_config):
    architectures = getattr(hf_config, "architectures", None)
    if not isinstance(architectures, list) or not architectures:
        return False
    default_archs = {
        "Llama4ForConditionalGeneration",
        "LlamaForCausalLM",
        "Olmo2ForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForConditionalGeneration",
        "MixtralForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Glm4MoeForCausalLM",
        "Glm4vForConditionalGeneration",
        "Glm4vMoeForConditionalGeneration",
        "GlmOcrForConditionalGeneration",
        "Step3VLForConditionalGeneration",
        "StepVLForConditionalGeneration",
        "MiMoV2FlashForCausalLM",
    }
    return architectures[0] in default_archs


# Can be more general if it is used in multiple places (keep it simple and thus not general now)
class BumpAllocator:
    def __init__(self, buffer_size: int, dtype, device):
        self._buffer = torch.zeros((buffer_size,), dtype=dtype, device=device)
        self._pointer = 0

    def allocate(self, size: int):
        assert self._pointer + size <= len(self._buffer)
        output = self._buffer[self._pointer : self._pointer + size]
        self._pointer += size
        return output


def log_info_on_rank0(logger, msg):
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    try:
        if torch.distributed.is_initialized() and get_tensor_model_parallel_rank() == 0:
            logger.info(msg)
    except:
        logger.info(msg)


def load_json_config(data: str):
    try:
        return orjson.loads(data)
    except JSONDecodeError:
        return orjson.loads(Path(data).read_text())


def dispose_tensor(x: torch.Tensor):
    """
    Dispose a tensor by freeing its memory.
    During piecewise CUDA graph capture/replay, we skip disposal to avoid
    interfering with torch.compile's memory tracking and graph recording.
    """

    # Skip disposal during piecewise CUDA graph to avoid torch.compile issues
    # we do local import to avoid circular import
    from sglang.srt.compilation.piecewise_context_manager import (
        is_in_piecewise_cuda_graph,
    )

    if is_in_piecewise_cuda_graph():
        return

    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))


T = TypeVar("T")


class Withable(Generic[T]):
    def __init__(self):
        self._value: Optional[T] = None

    @property
    def value(self) -> T:
        return self._value

    @contextmanager
    def with_value(self, new_value: T):
        assert self._value is None
        self._value = new_value
        try:
            yield
        finally:
            assert self._value is new_value
            self._value = None


def require_mlp_tp_gather(server_args: ServerArgs):
    """
    Check if the input of MLP is obtained by all-gather rather than all-reduce. This only happens when each MLP TP group contains multiple attention DP groups.
    """
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    if server_args.enable_dp_attention:
        assert server_args.dp_size > 1, "dp_size must be greater than 1"
        if (
            server_args.moe_dense_tp_size is None
        ):  # TODO(ch-wan): some MoE models do not have dense layers
            return True
        elif not server_args.enable_dp_lm_head:
            return True
        elif get_moe_a2a_backend().is_none():
            return True
        else:
            return (
                server_args.moe_dense_tp_size
                > server_args.tp_size // server_args.dp_size
            )
    else:
        return False


def require_attn_tp_gather(server_args: ServerArgs):
    """
    Check if the input of attention is scattered.
    """
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    assert server_args.moe_dense_tp_size in [1, None]
    if not get_moe_a2a_backend().is_none() or server_args.moe_dense_tp_size == 1:
        if server_args.enable_dp_attention:
            return server_args.dp_size < server_args.tp_size
        else:
            return True
    else:
        return False


def require_gathered_buffer(server_args: ServerArgs):
    return require_mlp_tp_gather(server_args) or require_attn_tp_gather(server_args)


def require_mlp_sync(server_args: ServerArgs):
    return server_args.enable_dp_attention or require_gathered_buffer(server_args)


def find_local_repo_dir(repo_id: str, revision: Optional[str] = None) -> Optional[str]:
    import huggingface_hub as hf

    # Build cache path
    cache_path = os.path.join(
        hf.constants.HF_HUB_CACHE,
        hf.constants.REPO_ID_SEPARATOR.join(["models", *repo_id.split("/")]),
    )

    # Get revision from main ref if not specified
    if not revision:
        ref_path = os.path.join(cache_path, "refs", "main")
        if os.path.isfile(ref_path):
            with open(ref_path) as f:
                revision = f.read().strip()

    # List files from revision directory
    if revision:
        rev_dir = os.path.join(cache_path, "snapshots", revision)
        if os.path.isdir(rev_dir):
            return rev_dir

    return None


def read_system_prompt_from_file(model_name: str) -> str:
    """Read system prompt from a file in the HuggingFace cache directory.

    Args:
        model_name: The model name to construct the file path

    Returns:
        The system prompt content from the file, or empty string if file not found
    """
    try:
        local_repo_dir = find_local_repo_dir(model_name)
        if local_repo_dir:
            system_prompt_file = os.path.join(local_repo_dir, "SYSTEM_PROMPT.txt")
            if os.path.exists(system_prompt_file):
                with open(system_prompt_file, "r", encoding="utf-8") as f:
                    return f.read()

        return ""
    except Exception:
        # If anything fails, return empty string
        return ""


def prepack_weight_if_needed(weight):
    if weight.device != torch.device("cpu"):
        return weight
    if not cpu_has_amx_support():
        return weight

    return torch.ops.sgl_kernel.convert_weight_packed(weight)


# TODO: currently gemm kernel has the below requirements:
# OC % TILE_N == 0, where TILE_N = 16
# IC % TILE_K == 0, where TILE_K = 32
def dim_is_supported(weight):
    return weight.size(0) % 16 == 0 and weight.size(1) % 32 == 0


def _process_weight_after_loading(module, weight_names, transpose_dims=None) -> None:
    # Pack weight for get better performance on CPU
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, f"Expects all weights to be on the same device"
    device = devices.pop()

    if transpose_dims:
        assert len(weight_names) == len(
            transpose_dims
        ), "len(weight_names) should be equal to len(transpose_dims)"

    for i, weight_name in enumerate(weight_names):
        weight_tensor = getattr(module, weight_name)

        # We don't pack weight or use intel amx backend if any weight of this module has unsupported dim.
        if not dim_is_supported(weight_tensor):
            logger.warning(
                f"Expects weight.size(0) % 16 == 0 and weight.size(1) % 32 == 0 "
                f"but {weight_tensor.size(0)=} and {weight_tensor.size(1)=} in {module}. "
                f"{module} won't use intel amx backend."
            )
            module.use_intel_amx_backend = False
            return

        if transpose_dims and transpose_dims[i]:
            weight_tensor = weight_tensor.transpose(*transpose_dims[i])

        packed_weight = torch.nn.Parameter(
            prepack_weight_if_needed(weight_tensor),
            requires_grad=False,
        )
        packed_weight.__dict__ = weight_tensor.__dict__
        setattr(module, weight_name, packed_weight)

    module.use_intel_amx_backend = (
        device == torch.device("cpu") and cpu_has_amx_support()
    )

    if (
        module.use_intel_amx_backend
        and hasattr(module, "bias")
        and module.bias is not None
    ):
        module.bias = torch.nn.Parameter(module.bias.data.float(), requires_grad=False)


class PackWeightMethod:
    def __init__(self, weight_names, transpose_dims=None):
        self.weight_names = weight_names
        self.transpose_dims = transpose_dims

    def process_weights_after_loading(self, module) -> None:
        _process_weight_after_loading(module, self.weight_names, self.transpose_dims)


class LazyValue:
    def __init__(self, creator: Callable):
        self._creator = creator
        self._value = None

    @property
    def value(self):
        if self._creator is not None:
            self._value = self._creator()
            self._creator = None
        return self._value


def dynamic_import(func_path: str):
    parts = func_path.split(".")
    if len(parts) < 2:
        raise ValueError(
            "func_path should contain both module name and func name (such as 'module.func')"
        )
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func


def gc_object_counts():
    import gc

    g0 = len(gc.get_objects(0))
    g1 = len(gc.get_objects(1))
    g2 = len(gc.get_objects(2))
    return g0, g1, g2


def configure_gc_warning(warn_threshold_secs):
    import gc

    gc_start_time = {}

    def gc_callback(phase, info):
        gen = info.get("generation", "?")
        if phase == "start":
            gc_start_time[gen] = time.time()
        elif phase == "stop":
            duration = time.time() - gc_start_time.get(gen, time.time())
            if duration > warn_threshold_secs:
                g0, g1, g2 = gc_object_counts()
                logger.warn(
                    f"LONG GARBAGE COLLECTION DETECTED | Generation {gen} | Duration: {duration:.4f}s | # Objects: gen0={g0}, gen1={g1}, gen2={g2} | "
                    f"This may cause latency jitter. Consider calling the freeze_gc API after sending a few warmup requests."
                )

    gc.callbacks.append(gc_callback)


def freeze_gc(context: str):
    g0_before, g1_before, g2_before = gc_object_counts()
    gc.freeze()
    g0_after, g1_after, g2_after = gc_object_counts()
    logger.info(
        f"Freezing GC in {context} process. "
        f"gen0: {g0_before}->{g0_after}, "
        f"gen1: {g1_before}->{g1_after}, "
        f"gen2: {g2_before}->{g2_after}"
    )


def configure_gc_logger():
    logger.info("Enable GC Logger")

    gc_start_time = {}

    def gc_callback(phase, info):
        gen = info.get("generation", "?")
        if phase == "start":
            gc_start_time[gen] = time.time()
            logger.info(f"GC start: Time {time.time()} | Generation {gen}")
        elif phase == "stop":
            duration = time.time() - gc_start_time.get(gen, time.time())
            collected = info.get("collected", "?")
            uncollectable = info.get("uncollectable", "?")
            logger.info(
                f"GC end: Time {time.time()} | Generation {gen} | "
                f"Duration: {duration:.4f}s | Collected: {collected} | Uncollectable: {uncollectable} "
                f'{"(LONG GC)" if duration > 0.1 else ""}'
            )

    gc.callbacks.append(gc_callback)


# COPIED FROM DeepGEMM
def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def parse_lscpu_topology():
    try:
        # Get CPU topology: CPU,Core,Socket,Node
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,Core,Socket,Node"], text=True
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error running 'lscpu': {e}")

    # Parse only data lines (skip comments)
    cpu_info = []
    for line in output.splitlines():
        if not line.startswith("#"):
            parts = line.strip().split(",")
            if len(parts) != 4:
                logger.warning("Skipping malformed lscpu line: %s", line.strip())
                continue
            cpu = int(parts[0])  # CPU id must always be present
            core, socket, node = [int(p) if p else 0 for p in parts[1:]]
            cpu_info.append((cpu, core, socket, node))

    # [(0,0,0,0),(1,1,0,0),...,(43,43,0,1),...,(256,0,0,0),...]
    return cpu_info


def get_physical_cpus_by_numa():
    cpu_info = parse_lscpu_topology()

    # Map NUMA node -> set of (core_id, socket) to avoid duplicates
    # 0: {(0,0): 0, (1, 0): 1,...}
    # ...
    # 5: {(214,1): 214, (215,1): 215}
    physical_by_node = defaultdict(dict)  # node -> core_id -> cpu_id

    for cpu, core, socket, node in cpu_info:
        key = (core, socket)
        if key not in physical_by_node[node]:
            physical_by_node[node][
                key
            ] = cpu  # pick first CPU seen for that physical core

    # Retrieves CPUs that the current process is allowed to run on
    cpus_allowed_list = psutil.Process().cpu_affinity()

    # Convert to list of physical CPUs per node
    # 0: [0,1,2,...,42]
    # ...
    # 2: [86,87,...,127]
    # ...
    # 5: [214,215,...,255]
    node_to_cpus = {}
    for node, core_to_cpu in physical_by_node.items():
        cpus = sorted(core_to_cpu.values())
        allowed_cpus = set(cpus).intersection(cpus_allowed_list)
        node_to_cpus[node] = allowed_cpus

    return node_to_cpus


# Only physical cores are used. Logical cores are excluded.
def get_cpu_ids_by_node():
    node_to_cpus = get_physical_cpus_by_numa()
    # Sort by NUMA node index
    cpu_ids = [
        ",".join(map(str, sorted(node_to_cpus[node]))) for node in sorted(node_to_cpus)
    ]

    # ['0,1,2,3', '4,5,6,7', '8,9,10,11', '12,13,14,15', '16,17,18,19', '20,21,22,23']
    return cpu_ids


def is_shm_available(dtype, world_size, local_size):
    return (
        (cpu_has_amx_support() or is_host_cpu_arm64())
        and dtype in [torch.bfloat16, torch.float16, torch.float]
        and world_size >= 1
        and world_size == local_size
    )


def lru_cache_frozenset(maxsize=128):
    def _to_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            # Not hashable; convert based on type
            if isinstance(o, (dict)):
                return frozenset(
                    (_to_hashable(k), _to_hashable(v)) for k, v in o.items()
                )
            elif isinstance(o, set):
                return frozenset(_to_hashable(v) for v in o)
            elif isinstance(o, (list, tuple)) or (
                isinstance(o, Sequence) and not isinstance(o, (str, bytes))
            ):
                return tuple(_to_hashable(v) for v in o)
            else:
                raise TypeError(f"Cannot make hashable: {type(o)}")

    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            h_args = tuple(_to_hashable(a) for a in args)
            h_kwargs = frozenset(
                (_to_hashable(k), _to_hashable(v)) for k, v in kwargs.items()
            )
            key = (h_args, h_kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        wrapper.cache_clear = cache.clear  # For manual cache clearing
        return wrapper

    return decorator


def apply_module_patch(target_module, target_function, wrappers):
    original_module, original_function = parse_module_path(
        target_module, target_function, False
    )

    original_function_id = id(original_function)

    candidate = original_function
    for wrapper in wrappers:
        candidate = wrapper(candidate)
    if target_function is not None:
        setattr(original_module, target_function, candidate)

    for key, value in sys.modules.copy().items():
        try:
            if (
                target_function is not None
                and hasattr(value, target_function)
                and id(getattr(value, target_function)) == original_function_id
            ):
                setattr(value, target_function, candidate)
        except ImportError as e:
            # Ignore some modules reporting ImportError when calling hasattr
            logger.warning(f"Ignore {value} reports ImportError with:\n{str(e)}")


def parse_module_path(module_path, function_name, create_dummy):
    from importlib.machinery import ModuleSpec

    def create_dummy_module(full_path, parent=None):
        """Create and register a placeholder module"""
        dummy = types.ModuleType(full_path)
        dummy.__file__ = "vllm_ascend.dummy_module.py"
        dummy.__spec__ = ModuleSpec(full_path, None)
        sys.modules[full_path] = dummy
        if parent:
            setattr(parent, full_path.split(".")[-1], dummy)
        return dummy

    def create_placeholder_function(func_name):
        """Create dummy function that raises when called"""

        def placeholder(*args, **kwargs):
            raise NotImplementedError(f"Function {func_name} is a placeholder")

        placeholder.__name__ = func_name
        return placeholder

    modules = module_path.split(".")
    current_module = None
    processed_path = []

    for idx, part in enumerate(modules):
        current_path = ".".join(modules[: idx + 1])
        parent_path = ".".join(modules[:idx]) if idx > 0 else None

        try:
            current_module = importlib.import_module(current_path)
        except ModuleNotFoundError:
            # Handle missing module
            parent = importlib.import_module(parent_path) if parent_path else None
            if parent and hasattr(parent, part):
                # Use existing attribute from parent
                current_module = getattr(parent, part)
                # Check for early function resolution
                if function_name and hasattr(current_module, function_name):
                    return current_module, getattr(current_module, function_name)
                if function_name and create_dummy:
                    ph_func = create_placeholder_function(function_name)
                    setattr(current_module, function_name, ph_func)
                    return current_module, ph_func
                if function_name:
                    raise AttributeError(
                        f"Function {function_name} missing in {current_path}"
                    )
            else:
                if not create_dummy:
                    raise
                # Create and register dummy module
                current_module = create_dummy_module(
                    current_path,
                    parent=(
                        importlib.import_module(parent_path) if parent_path else None
                    ),
                )

        processed_path.append(part)

    # Final function handling
    final_module = sys.modules[module_path]
    if function_name is not None:
        if not hasattr(final_module, function_name):
            if create_dummy:
                ph_func = create_placeholder_function(function_name)
                setattr(final_module, function_name, ph_func)
            else:
                setattr(final_module, function_name, None)
        return final_module, getattr(final_module, function_name)

    return final_module, None


def mxfp_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    else:
        return False


@lru_cache(maxsize=1)
def is_gfx95_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    else:
        return False


# LoRA-related constants and utilities
SUPPORTED_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
    "embed_tokens",
    "lm_head",
]

LORA_TARGET_ALL_MODULES = "all"


class ConcurrentCounter:
    """
    An asynchronous counter for managing concurrent tasks that need
    coordinated increments, decrements, and waiting until the count reaches zero.

    This class is useful for scenarios like tracking the number of in-flight tasks
    and waiting for them to complete.
    """

    def __init__(self, initial: int = 0):
        """
        Initialize the counter with an optional initial value.

        Args:
            initial (int): The initial value of the counter. Default is 0.
        """
        self._count = initial
        self._condition = asyncio.Condition()

    def value(self) -> int:
        """
        Return the current value of the counter.

        Note:
            This method is not synchronized. It may return a stale value
            if other coroutines are concurrently modifying the counter.

        Returns:
            int: The current counter value.
        """
        return self._count

    def __repr__(self) -> str:
        """Return an informative string representation of the counter."""
        return f"<ConcurrentCounter value={self.value()}>"

    async def increment(self, n: int = 1, notify_all: bool = True):
        """
        Atomically increment the counter by a given amount and notify all waiters.

        Args:
            n (int): The amount to increment the counter by. Default is 1.
            notify_all (bool): Whether to notify all waiters after incrementing. Default is True.
        """
        async with self._condition:
            self._count += n
            if notify_all:
                self._condition.notify_all()

    async def decrement(self, n: int = 1, notify_all: bool = True):
        """
        Atomically decrement the counter by a given amount and notify all waiters.

        Args:
            n (int): The amount to decrement the counter by. Default is 1.
            notify_all (bool): Whether to notify all waiters after decrementing. Default is True.
        """
        async with self._condition:
            self._count -= n
            if notify_all:
                self._condition.notify_all()

    async def wait_for(self, condition: Callable[[int], bool]):
        """
        Asynchronously wait until the counter satisfies a given condition.

        This suspends the calling coroutine without blocking the thread, allowing
        other tasks to run while waiting. When the condition is met, the coroutine resumes.

        Args:
            condition (Callable[[int], bool]): A function that takes the current counter value
                and returns True when the condition is satisfied.
        """
        async with self._condition:
            await self._condition.wait_for(lambda: condition(self._count))

    async def wait_for_zero(self):
        """
        Asynchronously wait until the counter reaches zero.

        This suspends the calling coroutine without blocking the thread, allowing
        other tasks to run while waiting. When the counter becomes zero, the coroutine resumes.
        """
        await self.wait_for(lambda count: count == 0)


@lru_cache(maxsize=1)
def is_triton_kernels_available() -> bool:
    return importlib.util.find_spec("triton_kernels") is not None


@lru_cache(maxsize=1)
def get_nvidia_driver_version() -> tuple:
    """Return the NVIDIA driver version as a tuple of ints, e.g. (595, 58, 3).
    Returns (0,) on failure."""
    version_str = get_nvidia_driver_version_str()
    if version_str is None:
        return (0,)
    try:
        return tuple(int(x) for x in version_str.split("."))
    except ValueError:
        return (0,)


@lru_cache(maxsize=1)
def get_nvidia_driver_version_str() -> str:
    """Return the NVIDIA driver version string, e.g. '595.58.03'.
    Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        version_str = result.stdout.strip().split("\n")[0].strip()
        return version_str if version_str else None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def check_cuda_result(raw_output):
    import cuda.bindings.runtime as cuda_rt

    err, *results = raw_output
    if err != cuda_rt.cudaError_t.cudaSuccess:
        raise Exception(f"CUDA error: {err}")

    return results


def get_physical_device_id(pytorch_device_id: int) -> int:
    """
    Convert PyTorch logical device ID to physical device ID.

    When CUDA_VISIBLE_DEVICES is set, maps the logical device ID (as seen by PyTorch)
    to the actual physical device ID. If CUDA_VISIBLE_DEVICES is not set, returns
    the device ID unchanged.

    Args:
        pytorch_device_id: The logical device ID from PyTorch (e.g., torch.cuda.current_device())

    Returns:
        The physical device ID
    """
    device_idx = int(pytorch_device_id)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices:
        device_list = cuda_visible_devices.split(",")
        return int(device_list[device_idx])
    else:
        return device_idx


def get_device_sm_nvidia_smi():
    try:
        # Run nvidia-smi command and capture output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Get the first line of output (assuming at least one GPU exists)
        compute_cap_str = result.stdout.strip().split("\n")[0]

        # Convert string (e.g., "9.0") to tuple of integers (9, 0)
        major, minor = map(int, compute_cap_str.split("."))
        return (major, minor)

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        # Handle cases where nvidia-smi isn't available or output is unexpected
        print(f"Error getting compute capability: {e}")
        return (0, 0)  # Default/fallback value


def json_list_type(value):
    try:
        return orjson.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON list: {value}. Please provide a valid JSON list."
        )


@contextmanager
def maybe_reindex_device_id(gpu_id: int):

    if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() is False or not is_cuda_alike():
        yield gpu_id
        return

    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if original_cuda_visible_devices:
        cuda_visible_devices = original_cuda_visible_devices.split(",")
    else:
        cuda_visible_devices = []

    str_gpu_id = cuda_visible_devices[gpu_id] if cuda_visible_devices else str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_id

    logger.debug(f"Set CUDA_VISIBLE_DEVICES to {str_gpu_id}")

    yield 0

    if original_cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    else:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def get_extend_input_len_swa_limit(
    sliding_window_size: int, chunked_prefill_size: int, page_size: int
) -> int:
    # 1. a factor of 2x is because each prefill contains chunked_prefill_size tokens,
    #    and between prefills, we run swa_radix_cache.cache_unfinished_req(),
    #    so we unlock the previously locked nodes.
    # 2. max is to handle the case that chunked_prefill_size is larger than sliding_window_size.
    #    in that case, each prefill contains chunked_prefill_size tokens,
    #    and we can only free out-of-sliding-window kv indices after each prefill.
    # 3. page_size is because we want to have 1 token extra for generated tokens.
    return page_size + 2 * max(sliding_window_size, chunked_prefill_size)


def get_num_new_pages(
    seq_lens: torch.Tensor,
    page_size: int,
    prefix_lens: Optional[torch.Tensor] = None,
    decode: bool = False,
) -> torch.Tensor:
    """
    Get the number of new pages for the given prefix and sequence lengths.
    We use cpu tensors to avoid blocking kernel launch.
    """
    cpu_device = torch.device("cpu")
    assert seq_lens.device == cpu_device

    if prefix_lens is None or decode:
        # NOTE: Special case for handling decode, which prefix lens is `seq_lens - 1`.
        assert decode
        return (seq_lens % page_size == 1).int().sum().item()

    assert prefix_lens.device == cpu_device
    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (prefix_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before
    sum_num_new_pages = torch.sum(num_new_pages).to(torch.int64)
    return sum_num_new_pages.item()


class CachedKernel:
    """
    Wrapper that allows kernel[grid](...) syntax with caching based on a key function.

    This wrapper caches compiled Triton kernels based on keys extracted by a
    user-provided key function to avoid redundant compilations.
    """

    def __init__(self, fn, key_fn=None):
        self.fn = fn
        assert isinstance(fn, triton.runtime.jit.JITFunction)

        original_fn = fn.fn
        self.signature = inspect.signature(original_fn)
        self.param_names = tuple(self.signature.parameters.keys())
        self.num_args = len(self.param_names)

        # Check that no parameters have default values
        for name, param in self.signature.parameters.items():
            assert (
                param.default is inspect.Parameter.empty
            ), f"Parameter '{name}' has a default value. Default parameters are not supported in cached kernels."

        functools.update_wrapper(self, original_fn)
        self.kernel_cache = {}

        # Store the key function
        self.key_fn = key_fn

    def __getitem__(self, grid):
        """
        Index with grid to get a launcher function.
        Returns a launcher that will handle caching based on the key function.
        """
        assert (
            isinstance(grid, tuple) and len(grid) <= 3
        ), "Grid must be a tuple with at most 3 dimensions."

        # Normalize grid once
        if len(grid) < 3:
            grid = grid + (1,) * (3 - len(grid))

        def launcher(*args, **kwargs):
            cache_key = self.key_fn(args, kwargs)

            cached_kernel = self.kernel_cache.get(cache_key)

            if cached_kernel is None:
                # First time: compile and cache the kernel
                cached_kernel = self.fn[grid](*args, **kwargs)
                self.kernel_cache[cache_key] = cached_kernel
                return cached_kernel
            else:
                # Use cached kernel
                all_args = self._build_args(args, kwargs)
                cached_kernel[grid](*all_args)
                return cached_kernel

        return launcher

    def _build_args(self, args, kwargs):
        """
        Build the complete argument list for kernel invocation.
        """
        complete_args = list(args)

        for i in range(len(args), self.num_args):
            name = self.param_names[i]
            value = kwargs.get(name, inspect.Parameter.empty)
            if value is not inspect.Parameter.empty:
                complete_args.append(value)
            else:
                raise ValueError(f"Missing argument: {name}")

        return complete_args

    def _clear_cache(self):
        """
        Clear the kernel cache for testing purposes.
        """
        self.kernel_cache.clear()


def cached_triton_kernel(key_fn=None):
    """
    Decorator that enables key-based caching for Triton kernels using a key function.

    It essentially bypasses Triton's built-in caching mechanism, allowing users to
    define their own caching strategy based on kernel parameters. This helps reduce
    the heavy overheads of Triton kernel launch when the kernel specialization dispatch
    is simple.

    Usage:
        @cached_triton_kernel(key_fn=lambda args, kwargs: kwargs.get('BLOCK_SIZE', 1024))
        @triton.jit
        def my_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
            ...

        # Invoke normally
        my_kernel[grid](x, y, BLOCK_SIZE=1024)

    Args:
        key_fn: A function that takes (args, kwargs) and returns the cache key(s).
                The key can be a single value or a tuple of values.

    Returns:
        A decorator that wraps the kernel with caching functionality.

    Note: Kernels with default parameter values are not supported and will raise an assertion error.
    """

    def decorator(fn):
        # Auto-enable the custom kernel cache for CUDA, where it is
        # known to be compatible.
        if is_cuda() and not envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.is_set():
            logger.debug("Detected platform CUDA, using custom triton kernel cache.")
            return CachedKernel(fn, key_fn)

        if envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.get():
            logger.debug(
                f"{envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.name} = True. Using custom triton kernel cache."
            )
            return CachedKernel(fn, key_fn)
        else:
            # Fallback to the native triton cache.
            logger.debug(
                f"{envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.name} = False. Using native triton kernel cache."
            )
            return fn

    return decorator


def reserve_rope_cache_for_long_sequences(
    model, server_args, model_config, logger=None
):
    """Pre-expand RoPE cache for long sequences and speculative decoding."""
    from sglang.srt.environ import envs

    SAFETY_FACTOR = envs.SGLANG_SPEC_EXPANSION_SAFETY_FACTOR.get()
    MARGIN = envs.SGLANG_ROPE_CACHE_SAFETY_MARGIN.get()
    ALIGN = envs.SGLANG_ROPE_CACHE_ALIGN.get()

    # 1) Estimate base context upper bound
    base_ctx = (
        getattr(server_args, "context_length", None)
        or getattr(model_config, "context_len", None)
        or getattr(model_config, "max_model_len", None)
        or getattr(model_config.hf_text_config, "max_position_embeddings", None)
        or 2048
    )

    # 2) Speculative decoding expansion
    steps = int(getattr(server_args, "speculative_num_steps", 0) or 0)
    draft = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
    reserve = base_ctx + steps * draft * SAFETY_FACTOR + MARGIN

    # 3) Align to reduce reallocation frequency
    reserve = (reserve + ALIGN - 1) // ALIGN * ALIGN

    # Recursively expand all RoPE layers
    def reserve_rope_cache_recursive(module):
        for child in module.children():
            if hasattr(child, "_ensure_cos_sin_cache_length") and hasattr(
                child, "cos_sin_cache"
            ):
                child._ensure_cos_sin_cache_length(reserve - 1)
            else:
                reserve_rope_cache_recursive(child)

    reserve_rope_cache_recursive(model)


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@contextmanager
def temp_attr_context(obj, attr, value):
    if obj is None:
        yield
        return

    original_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original_value)


cached_device_index = -1


def get_current_device_stream_fast():
    global cached_device_index
    if cached_device_index == -1:
        cached_device_index = torch.get_device_module().current_device()
    return torch.get_device_module().current_stream(cached_device_index)


def raise_error_or_warn(obj, strict, counter_name, message, log_interval=1000):
    if strict:
        raise ValueError(message)
    else:
        count = getattr(obj, counter_name, 0)
        if count % log_interval == 0:
            logger.warning(message)
        setattr(obj, counter_name, count + 1)


def get_or_create_event_loop():
    """Gets the running event loop or creates a new one if it doesn't exist."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

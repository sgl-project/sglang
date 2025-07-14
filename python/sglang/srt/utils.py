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

import builtins
import ctypes
import dataclasses
import functools
import importlib
import io
import ipaddress
import itertools
import json
import logging
import os
import pickle
import platform
import random
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from io import BytesIO
from json import JSONDecodeError
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import psutil
import pybase64
import requests
import torch
import torch.distributed
import torch.distributed as dist
import triton
import zmq
from fastapi.responses import ORJSONResponse
from packaging import version as pkg_version
from PIL import Image
from starlette.routing import Mount
from torch import nn
from torch.func import functional_call
from torch.library import Library
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils._contextlib import _DecoratorContextManager
from triton.runtime.cache import FileCacheManager

logger = logging.getLogger(__name__)

show_time_cost = False
time_infos = {}

HIP_FP8_E4M3_FNUZ_MAX = 224.0


# https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
def is_hip() -> bool:
    return torch.version.hip is not None


if is_hip():
    FP8_E4M3_MAX = HIP_FP8_E4M3_FNUZ_MAX
else:
    FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

FP8_E4M3_MIN = -FP8_E4M3_MAX

builtins.FP8_E4M3_MAX = FP8_E4M3_MAX
builtins.FP8_E4M3_MIN = FP8_E4M3_MIN


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def is_cuda_alike():
    return is_cuda() or is_hip()


def is_hpu() -> bool:
    return hasattr(torch, "hpu") and torch.hpu.is_available()


def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_host_cpu_x86() -> bool:
    machine = platform.machine().lower()
    return (
        machine in ("x86_64", "amd64", "i386", "i686")
        and hasattr(torch, "cpu")
        and torch.cpu.is_available()
    )


def is_cpu() -> bool:
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1" and is_host_cpu_x86()


def get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


def _check(cc_major):
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] == cc_major and tuple(
        map(int, torch.version.cuda.split(".")[:2])
    ) >= (12, 3)


is_ampere_with_cuda_12_3 = lambda: _check(8)
is_hopper_with_cuda_12_3 = lambda: _check(9)


def is_blackwell():
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] == 10


_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def get_int_env_var(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def support_triton(backend: str) -> bool:
    return backend not in ["torch_native", "intel_amx", "ascend"]


try:
    import sgl_kernel

    is_intel_amx_backend_available = hasattr(
        torch.ops.sgl_kernel, "convert_weight_packed"
    )
except:
    is_intel_amx_backend_available = False


def cpu_has_amx_support():
    return torch._C._cpu._is_amx_tile_supported() and is_intel_amx_backend_available


def use_intel_amx_backend(layer):
    return getattr(layer, "use_intel_amx_backend", False)


def is_flashinfer_available():
    """
    Check whether flashinfer is available.
    As of Oct. 6, 2024, it is only available on NVIDIA GPUs.
    """
    if not get_bool_env_var("SGLANG_IS_FLASHINFER_AVAILABLE", default="true"):
        return False
    return importlib.util.find_spec("flashinfer") is not None and is_cuda()


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
        free_gpu_memory = psutil.virtual_memory().available
    elif device == "npu":
        num_gpus = torch.npu.device_count()
        assert gpu_id < num_gpus

        if torch.npu.current_device() != gpu_id:
            print(
                f"WARNING: current device is not {gpu_id}, but {torch.npu.current_device()}, ",
                "which may cause useless memory allocation for torch NPU context.",
            )
        free_gpu_memory, total_gpu_memory = torch.npu.mem_get_info()

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def is_pin_memory_available() -> bool:
    return torch.cuda.is_available()


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes


def maybe_offload_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    device = next(module.parameters()).device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()
    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break

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
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
        offloaded_parameters = True

    if offloaded_parameters:
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }
            output = functional_call(module, device_state, args=args, kwargs=kwargs)
            module.forward = forward
            return output

        module.forward = forward

    return module


class LayerFn(Protocol):

    def __call__(self, layer_id: int, prefix: str) -> torch.nn.Module: ...


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    prefix: str = "",
    return_tuple: bool = False,
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function"""
    # circula imports
    from sglang.srt.distributed import get_pp_indices
    from sglang.srt.layers.utils import PPMissingLayer

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
        + [
            maybe_offload_to_cpu(layer_fn(idx=idx, prefix=add_prefix(idx, prefix)))
            for idx in range(start_layer, end_layer)
        ]
        + [
            PPMissingLayer(return_tuple=return_tuple)
            for _ in range(end_layer, num_hidden_layers)
        ]
    )
    if pp_rank is None or pp_size is None:
        return modules
    return modules, start_layer, end_layer


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                # It could happen by race condition (the proc dies when psutil.Process is called).
                pass

    return None


def wait_port_available(
    port: int, port_name: str, timeout_s: int = 30, raise_exception: bool = True
) -> bool:
    for i in range(timeout_s):
        if is_port_available(port):
            return True

        if i > 10 and i % 5 == 0:
            process = find_process_using_port(port)
            if process is None:
                logger.warning(
                    f"The port {port} is in use, but we could not find the process that uses it."
                )

            pid = process.pid
            error_message = f"{port_name} is used by a process already. {process.name()=}' {process.cmdline()=} {process.status()=} {pid=}"
            logger.info(
                f"port {port} is in use. Waiting for {i} seconds for {port_name} to be available. {error_message}"
            )
        time.sleep(0.1)

    if raise_exception:
        raise ValueError(
            f"{port_name} at {port} is not available in {timeout_s} seconds. {error_message}"
        )
    return False


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False


def get_free_port():
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def decode_video_base64(video_base64):
    from PIL import Image

    # Decode the base64 string
    video_bytes = pybase64.b64decode(video_base64, validate=True)

    # Placeholder for the start indices of each PNG image
    img_starts = []

    frame_format = "PNG"  # str(os.getenv('FRAME_FORMAT', "JPEG"))

    assert frame_format in [
        "PNG",
        "JPEG",
    ], "FRAME_FORMAT must be either 'PNG' or 'JPEG'"

    if frame_format == "PNG":
        # Find each PNG start signature to isolate images
        i = 0
        while i < len(video_bytes) - 7:  # Adjusted for the length of the PNG signature
            # Check if we found the start of a PNG file
            if (
                video_bytes[i] == 0x89
                and video_bytes[i + 1] == 0x50
                and video_bytes[i + 2] == 0x4E
                and video_bytes[i + 3] == 0x47
                and video_bytes[i + 4] == 0x0D
                and video_bytes[i + 5] == 0x0A
                and video_bytes[i + 6] == 0x1A
                and video_bytes[i + 7] == 0x0A
            ):
                img_starts.append(i)
                i += 8  # Skip the PNG signature
            else:
                i += 1
    else:
        # Find each JPEG start (0xFFD8) to isolate images
        i = 0
        while (
            i < len(video_bytes) - 1
        ):  # Adjusted for the length of the JPEG SOI signature
            # Check if we found the start of a JPEG file
            if video_bytes[i] == 0xFF and video_bytes[i + 1] == 0xD8:
                img_starts.append(i)
                # Move to the next byte to continue searching for the next image start
                i += 2
            else:
                i += 1

    frames = []
    for start_idx in img_starts:
        # Assuming each image is back-to-back, the end of one image is the start of another
        # The last image goes until the end of the byte string
        end_idx = (
            img_starts[img_starts.index(start_idx) + 1]
            if img_starts.index(start_idx) + 1 < len(img_starts)
            else len(video_bytes)
        )
        img_bytes = video_bytes[start_idx:end_idx]

        # Convert bytes to a PIL Image
        img = Image.open(BytesIO(img_bytes))

        # Convert PIL Image to a NumPy array
        frame = np.array(img)

        # Append the frame to the list of frames
        frames.append(frame)

    # Ensure there's at least one frame to avoid errors with np.stack
    if frames:
        return np.stack(frames, axis=0), img.size
    else:
        return np.array([]), (
            0,
            0,
        )  # Return an empty array and size tuple if no frames were found


def load_audio(audio_file: str, sr: int = 16000, mono: bool = True) -> np.ndarray:
    # Use soundfile here, since librosa use it under the hood,
    # and librosa will not support audio loading in the future
    import soundfile as sf
    from scipy.signal import resample

    # Load audio data
    if isinstance(audio_file, bytes):
        audio, original_sr = sf.read(BytesIO(audio_file))
    elif audio_file.startswith("data:"):
        audio_file = audio_file.split(",")[1]
        audio, original_sr = sf.read(
            BytesIO(pybase64.b64decode(audio_file, validate=True))
        )
    elif audio_file.startswith("http://") or audio_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
        response = requests.get(audio_file, stream=True, timeout=timeout)
        audio_file = BytesIO(response.content)
        response.close()
        audio, original_sr = sf.read(audio_file)
    elif isinstance(audio_file, str):
        audio, original_sr = sf.read(audio_file)
    else:
        raise ValueError(f"Invalid audio format: {audio_file}")

    # Resample audio if the original sample rate is different from the desired sample rate
    if original_sr != sr:
        num_samples = int(len(audio) * float(sr) / original_sr)
        audio = resample(audio, num_samples)

    # Convert to mono if requested and audio is stereo
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio


def load_image(
    image_file: Union[Image.Image, str, bytes],
) -> tuple[Image.Image, tuple[int, int]]:
    image = image_size = None
    if isinstance(image_file, Image.Image):
        image = image_file
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, stream=True, timeout=timeout).raw
        image = Image.open(response)
        response.close()
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    else:
        raise ValueError(f"Invalid image: {image}")

    return image, image_size


def load_video(video_file: Union[str, bytes], use_gpu: bool = True):
    # We import decord here to avoid a strange Segmentation fault (core dumped) issue.
    from decord import VideoReader, cpu, gpu

    try:
        from decord.bridge import decord_bridge

        ctx = gpu(0)
        _ = decord_bridge.get_ctx_device(ctx)
    except Exception:
        ctx = cpu(0)

    tmp_file = None
    vr = None
    try:
        if isinstance(video_file, bytes):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(video_file)
            tmp_file.close()
            vr = VideoReader(tmp_file.name, ctx=ctx)
        elif isinstance(video_file, str):
            if video_file.startswith(("http://", "https://")):
                timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
                response = requests.get(video_file, stream=True, timeout=timeout)
                response.raise_for_status()
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file.close()
                vr = VideoReader(tmp_file.name, ctx=ctx)
            elif video_file.startswith("data:"):
                _, encoded = video_file.split(",", 1)
                video_bytes = base64.b64decode(encoded)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_file.write(video_bytes)
                tmp_file.close()
                vr = VideoReader(tmp_file.name, ctx=ctx)
            elif os.path.isfile(video_file):
                vr = VideoReader(video_file, ctx=ctx)
            else:
                video_bytes = base64.b64decode(video_file)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_file.write(video_bytes)
                tmp_file.close()
                vr = VideoReader(tmp_file.name, ctx=ctx)
        else:
            raise ValueError(f"Unsupported video input type: {type(video_file)}")

        return vr

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


def suppress_other_loggers():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )

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


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

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


def monkey_patch_vllm_gguf_config():
    try:
        from vllm.model_executor.layers.quantization.gguf import (
            GGUFConfig,
            GGUFEmbeddingMethod,
            GGUFLinearMethod,
        )
    except ImportError:
        return

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    def get_quant_method_with_embedding_replaced(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            # patch to own VocabParallelEmbedding
            return GGUFEmbeddingMethod(self)
        return None

    setattr(GGUFConfig, "get_quant_method", get_quant_method_with_embedding_replaced)


def maybe_set_triton_cache_manager() -> None:
    """Set environment variable to tell Triton to use a
    custom cache manager"""
    cache_manger = os.environ.get("TRITON_CACHE_MANAGER", None)
    if cache_manger is None:
        manager = "sglang.srt.utils:CustomCacheManager"
        logger.debug("Setting Triton cache manager to: %s", manager)
        os.environ["TRITON_CACHE_MANAGER"] = manager


class CustomCacheManager(FileCacheManager):
    # Adapted from: https://github.com/tdoublep/vllm/blob/3307522289fdfefe323b6c00d0db696651989a2f/vllm/triton_utils/custom_cache_manager.py
    def __init__(self, key, override=False, dump=False):
        from sglang.srt.distributed.parallel_state import get_tp_group

        self.key = key
        self.lock_path = None

        try:
            module_path = "triton.runtime.cache"
            cache_module = importlib.import_module(module_path)

            default_cache_dir = getattr(cache_module, "default_cache_dir", None)
            default_dump_dir = getattr(cache_module, "default_dump_dir", None)
            default_override_dir = getattr(cache_module, "default_override_dir", None)
        except (ModuleNotFoundError, AttributeError) as e:
            default_cache_dir = None
            default_dump_dir = None
            default_override_dir = None

        if dump:
            self.cache_dir = (
                default_dump_dir()
                if default_dump_dir is not None
                else os.path.join(Path.home(), ".triton", "dump")
            )
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = (
                default_override_dir()
                if default_override_dir is not None
                else os.path.join(Path.home(), ".triton", "override")
            )
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or (
                default_cache_dir()
                if default_cache_dir is not None
                else os.path.join(Path.home(), ".triton", "cache")
            )
            if self.cache_dir:
                try:
                    self.cache_dir = f"{self.cache_dir}_{get_tp_group().local_rank}"
                except:
                    self.cache_dir = f"{self.cache_dir}_{os.getpid()}"
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")


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


def add_api_key_middleware(app, api_key: str):
    @app.middleware("http")
    async def authentication(request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/health"):
            return await call_next(request)
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
        if request.headers.get("Authorization") != "Bearer " + api_key:
            return ORJSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


def prepare_model_and_tokenizer(model_path: str, tokenizer_path: str):
    if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
        if not os.path.exists(model_path):
            from modelscope import snapshot_download

            model_path = snapshot_download(model_path)
            tokenizer_path = snapshot_download(
                tokenizer_path, ignore_patterns=["*.bin", "*.safetensors"]
            )
    return model_path, tokenizer_path


def configure_logger(server_args, prefix: str = ""):
    if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
            raise Exception(
                "Setting SGLANG_LOGGING_CONFIG_PATH from env with "
                f"{SGLANG_LOGGING_CONFIG_PATH} but it does not exist!"
            )
        with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())
        logging.config.dictConfig(custom_config)
        return
    format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


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
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
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
):
    """Send data from src to dst in group using DeviceToDevice communication."""

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor(
                [0], dtype=torch.long, device=torch.cuda.current_device()
            )
            dist.send(tensor_size, dst=dst, group=group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).cuda(
                device=torch.cuda.current_device()
            )  # Move to GPU
            tensor_size = torch.tensor(
                [size], dtype=torch.long, device=torch.cuda.current_device()
            )

            dist.send(tensor_size, dst=dst, group=group)
            dist.send(tensor_data, dst=dst, group=group)
        return data

    elif rank == dst:
        tensor_size = torch.tensor(
            [0], dtype=torch.long, device=torch.cuda.current_device()
        )
        dist.recv(tensor_size, src=src, group=group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(
            size, dtype=torch.uint8, device=torch.cuda.current_device()
        )
        dist.recv(tensor_data, src=src, group=group)

        serialized_data = bytes(
            tensor_data.cpu().numpy()
        )  # Move back to host for deserialization
        data = pickle.loads(serialized_data)
        return data

    # Other ranks in pp_group do nothing
    return []


step_counter = 0


def pytorch_profile(name, func, *args, data_size=-1):
    """
    Args:
        name (string): the name of recorded function.
        func: the function to be profiled.
        args: the arguments of the profiled function.
        data_size (int): some measurement of the computation complexity.
            Usually, it could be the batch size.
    """
    global step_counter
    os.makedirs("trace", exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        # on_trace_ready=tensorboard_trace_handler('./log_dir'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function(name):
            with open(f"trace/size_{step_counter}.json", "w") as f:
                json.dump({"size": data_size}, f)
            result = func(*args)
    prof.export_chrome_trace(f"trace/{name}_{step_counter}.json")
    step_counter += 1
    return result


def get_zmq_socket(
    context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool
):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


def dump_to_file(dirpath, name, value):
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return

    os.makedirs(dirpath, exist_ok=True)
    if value.dtype is torch.bfloat16:
        value = value.float()
    value = value.cpu().numpy()
    output_filename = os.path.join(dirpath, f"pytorch_dump_{name}.npy")
    logger.info(f"Dump a tensor to {output_filename}. Shape = {value.shape}")
    np.save(output_filename, value)


def is_triton_3():
    return triton.__version__.startswith("3.")


def maybe_torch_compile(*args, **kwargs):
    """
    torch.compile does not work for triton 2.2.0, which is needed in xlm1's jax.
    Therefore, we disable it here.
    """

    def decorator(func):
        if is_triton_3():
            return torch.compile(*args, **kwargs)(func)
        return func

    return decorator


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


def bind_port(port):
    """Bind to a specific port, assuming it's available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allows address reuse
    sock.bind(("", port))
    sock.listen(1)
    return sock


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
            raise RuntimeError(f"nvidia-smi error: {result.stderr.strip()}")

        # Parse the output to extract memory values
        memory_values = [
            float(mem)
            for mem in result.stdout.strip().split("\n")
            if re.match(r"^\d+(\.\d+)?$", mem.strip())
        ]

        if not memory_values:
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
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
        import torch_npu

        return torch.npu.mem_get_info()[1] // 1024 // 1024  # unit: MB
    except ImportError as e:
        raise ImportError("torch_npu is required when run on npu device.")


def get_device_memory_capacity(device: str = None):
    if is_cuda():
        gpu_mem = get_nvgpu_memory_capacity()
    elif is_hip():
        gpu_mem = get_amdgpu_memory_capacity()
    elif device == "hpu":
        gpu_mem = get_hpu_memory_capacity()
    elif device == "npu":
        gpu_mem = get_npu_memory_capacity()
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
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
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
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def crash_on_warnings():
    # Crash on warning if we are running CI tests
    return get_bool_env_var("SGLANG_IS_IN_CI")


def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg, stacklevel=2)


@functools.lru_cache(None)
def print_info_once(msg: str) -> None:
    logger.info(msg)


def get_device_name(device_id: int = 0) -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
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
        if device_id == None:
            return "xpu"
        return "xpu:{}".format(device_id)

    if hasattr(torch, "npu") and torch.npu.is_available():
        if device_id == None:
            return "npu"
        return "npu:{}".format(device_id)

    if is_habana_available():
        try:
            import habana_frameworks.torch.hpu

            if torch.hpu.is_available():
                if device_id == None:
                    return "hpu"
                return "hpu:{}".format(device_id)
        except ImportError as e:
            raise ImportError(
                "Habana frameworks detected, but failed to import 'habana_frameworks.torch.hpu'."
            )

    raise RuntimeError("No accelerator (CUDA, XPU, HPU) is available.")


@lru_cache(maxsize=1)
def get_device_count() -> int:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
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
            import habana_frameworks.torch.hpu

            if torch.hpu.is_available():
                return torch.hpu.device_count()
        except (ImportError, RuntimeError):
            return 0

    return 0  # No accelerators available


def get_device_core_count(device_id: int = 0) -> int:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    return 0


def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
    major, minor = None, None
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        major, minor, *_ = torch.xpu.get_device_capability(device_id)["version"].split(
            "."
        )
        major, minor = int(major), int(minor)

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


def get_npu_compiler_config():
    config = {
        "frozen_parameter": True,
        "tiling_schedule_optimize": True,
        "topology_sorting_strategy": "StableRDFS",
    }
    return config


def get_compiler_backend() -> str:
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
        predefined_config = get_npu_compiler_config()
        for k, v in predefined_config.items():
            setattr(compiler_config.experimental_config, k, v)

        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        return npu_backend

    return "inductor"


sglang_lib = Library("sglang", "FRAGMENT")  # noqa


# Some backends use pytorch version < 2.4.0 which doesn't
# support `torch.library.custom_op`.
def supports_custom_op() -> bool:
    return hasattr(torch.library, "custom_op")


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
):
    """
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
    """
    import torch.library

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

    my_lib = target_lib or sglang_lib
    my_lib.define(op_name + schema_str)
    my_lib.impl(op_name, op_func, "CUDA")
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


def set_gpu_proc_affinity(
    tp_size: int,
    nnodes: int,
    gpu_id: int,
):
    # current process
    pid = os.getpid()
    p = psutil.Process(pid)

    tp_size_per_node = tp_size // nnodes

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


@lru_cache(maxsize=2)
def disable_request_logging() -> bool:
    return get_bool_env_var("SGLANG_DISABLE_REQUEST_LOGGING")


def dataclass_to_string_truncated(
    data, max_length=2048, skip_names: Optional[Set[str]] = None
):
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return str(data[:half_length]) + " ... " + str(data[-half_length:])
        else:
            return str(data)
    elif isinstance(data, dict):
        return (
            "{"
            + ", ".join(
                f"'{k}': {dataclass_to_string_truncated(v, max_length)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={dataclass_to_string_truncated(getattr(data, f.name), max_length)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


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

        return ForkingPickler.loads(data)


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


def pyspy_dump_schedulers():
    """py-spy dump on all scheduler in a local node."""
    try:
        pid = psutil.Process().pid
        # Command to run py-spy with the PID
        cmd = f"py-spy dump --pid {pid}"
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


def set_uvicorn_logging_configs():
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def get_ip() -> str:
    # SGLANG_HOST_IP env can be ignore
    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " SGLANG_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def get_open_port() -> int:
    port = os.getenv("SGLANG_PORT")
    if port is not None:
        port = int(port)
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def configure_ipv6(dist_init_addr):
    addr = dist_init_addr
    end = addr.find("]")
    if end == -1:
        raise ValueError("invalid IPv6 address format: missing ']'")

    host = addr[: end + 1]

    # this only validates the address without brackets: we still need the below checks.
    # if it's invalid, immediately raise an error so we know it's not formatting issues.
    if not is_valid_ipv6_address(host[1:end]):
        raise ValueError(f"invalid IPv6 address: {host}")

    port_str = None
    if len(addr) > end + 1:
        if addr[end + 1] == ":":
            port_str = addr[end + 2 :]
        else:
            raise ValueError("received IPv6 address format: expected ':' after ']'")

    if not port_str:
        raise ValueError(
            "a port must be specified in IPv6 address (format: [ipv6]:port)"
        )

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"invalid port in IPv6 address: '{port_str}'")
    return port, host


def rank0_log(msg: str):
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() == 0:
        logger.info(msg)


def launch_dummy_health_check_server(host, port):
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Response

    app = FastAPI()

    @app.get("/health")
    async def health():
        """Check the health of the http server."""
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        """Check the health of the http server."""
        return Response(status_code=200)

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

    try:
        loop = asyncio.get_running_loop()
        logger.info(
            f"Dummy health check server scheduled on existing loop at {host}:{port}"
        )
        loop.create_task(server.serve())

    except RuntimeError:
        logger.info(f"Starting dummy health check server at {host}:{port}")
        server.run()


def create_checksum(directory: str):
    raise NotImplementedError()


def set_cuda_arch():
    if is_flashinfer_available():
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


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
        except Exception as e:
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
            traceback.print_exc()

            time.sleep(delay)


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


class DeepEPMode(Enum):
    normal = "normal"
    low_latency = "low_latency"
    auto = "auto"

    def enable_normal(self):
        return self in [DeepEPMode.normal, DeepEPMode.auto]

    def enable_low_latency(self):
        return self in [DeepEPMode.low_latency, DeepEPMode.auto]

    def resolve(self, is_extend_in_batch: bool):
        if self != DeepEPMode.auto:
            return self

        if is_extend_in_batch:
            return DeepEPMode.normal
        else:
            return DeepEPMode.low_latency


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


def get_local_ip_auto() -> str:
    interface = os.environ.get("SGLANG_LOCAL_IP_NIC", None)
    return (
        get_local_ip_by_nic(interface)
        if interface is not None
        else get_local_ip_by_remote()
    )


def get_local_ip_by_nic(interface: str) -> str:
    try:
        import netifaces
    except ImportError as e:
        raise ImportError(
            "Environment variable SGLANG_LOCAL_IP_NIC requires package netifaces, please install it through 'pip install netifaces'"
        ) from e

    try:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                ip = addr_info.get("addr")
                if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
                    return ip
        if netifaces.AF_INET6 in addresses:
            for addr_info in addresses[netifaces.AF_INET6]:
                ip = addr_info.get("addr")
                if ip and not ip.startswith("fe80::") and ip != "::1":
                    return ip.split("%")[0]
    except (ValueError, OSError) as e:
        raise ValueError(
            "Can not get local ip from NIC. Please verify whether SGLANG_LOCAL_IP_NIC is set correctly."
        )

    # Fallback
    return get_local_ip_by_remote()


def get_local_ip_by_remote() -> str:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        raise ValueError("Can not get local ip")


def is_page_size_one(server_args):
    return server_args.page_size == 1


# TODO(hebiao064): Accelerate FA3 Spec Decode with topk > 1.
# TODO(hebiao064): Improve the acc rate for FA3 Spec Decode with topk == 1 and page_size > 1.
def is_no_spec_infer_or_topk_one(server_args):
    return server_args.speculative_eagle_topk is None or (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk == 1
        and is_page_size_one(server_args)
    )


def is_fa3_default_architecture(hf_config):
    architectures = getattr(hf_config, "architectures", None)
    if not isinstance(architectures, list) or not architectures:
        return False
    default_archs = {
        "Qwen2ForCausalLM",
        "Llama4ForConditionalGeneration",
        "LlamaForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForConditionalGeneration",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
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

    if get_tensor_model_parallel_rank() == 0:
        logger.info(msg)


def load_json_config(data: str):
    try:
        return json.loads(data)
    except JSONDecodeError:
        return json.loads(Path(data).read_text())


def dispose_tensor(x: torch.Tensor):
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


def require_mlp_tp_gather(server_args):
    """
    Check if the input of MLP is obtained by all-gather rather than all-reduce. This only happens when each MLP TP group contains multiple attention DP groups.
    """
    if server_args.enable_dp_attention:
        assert server_args.dp_size > 1, "dp_size must be greater than 1"
        if (
            server_args.moe_dense_tp_size is None
        ):  # TODO(ch-wan): some MoE models do not have dense layers
            return True
        elif not server_args.enable_dp_lm_head:
            return True
        elif not server_args.enable_deepep_moe:
            return True
        else:
            return (
                server_args.moe_dense_tp_size
                > server_args.tp_size // server_args.dp_size
            )
    else:
        return False


def require_attn_tp_gather(server_args):
    """
    Check if the input of attention is scattered.
    """
    assert server_args.moe_dense_tp_size in [1, None]
    if server_args.enable_deepep_moe or server_args.moe_dense_tp_size == 1:
        if server_args.enable_dp_attention:
            return server_args.dp_size < server_args.tp_size
        else:
            return True
    else:
        return False


def require_gathered_buffer(server_args):
    return require_mlp_tp_gather(server_args) or require_attn_tp_gather(server_args)


def require_mlp_sync(server_args):
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


def bind_or_assign(target, source):
    if target is not None:
        target.copy_(source)
        return target
    else:
        return source


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


def configure_gc_logger():
    logger.info("Enable GC Logger")

    import gc

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
def align(x: int, y: int) -> int:
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
            cpu, core, socket, node = map(int, line.strip().split(","))
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
        cpu_has_amx_support()
        and dtype in [torch.bfloat16, torch.float]
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
        if (
            target_function is not None
            and hasattr(value, target_function)
            and id(getattr(value, target_function)) == original_function_id
        ):
            setattr(value, target_function, candidate)


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

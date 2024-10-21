"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Common utilities."""

import base64
import ipaddress
import json
import logging
import os
import pickle
import random
import resource
import socket
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil
import requests
import torch
import torch.distributed as dist
from fastapi.responses import ORJSONResponse
from packaging import version as pkg_version
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from triton.runtime.cache import (
    FileCacheManager,
    default_cache_dir,
    default_dump_dir,
    default_override_dir,
)

logger = logging.getLogger(__name__)


show_time_cost = False
time_infos = {}


def is_hip() -> bool:
    """Return whether it is HIP on the AMD ROCm platform."""
    return torch.version.hip is not None


def is_flashinfer_available():
    """
    Check whether flashinfer is available.
    As of Oct. 6, 2024, it is only available on NVIDIA GPUs.
    """
    return torch.cuda.is_available() and not is_hip()


def is_ipv6(address):
    try:
        ipaddress.IPv6Address(address)
        return True
    except ipaddress.AddressValueError:
        return False


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
    time_infos[name].acc_time -= time.time()


def mark_end(name):
    global time_infos, show_time_cost
    if not show_time_cost:
        return
    torch.cuda.synchronize()
    time_infos[name].acc_time += time.time()
    if time_infos[name].check():
        time_infos[name].pretty_print()


def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.time() - start_time) * 1000
                if cost_time > min_cost_ms:
                    print(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


def get_available_gpu_memory(device, gpu_id, distributed=False):
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
        torch.xpu.empty_cache()
        used_memory = torch.xpu.memory_allocated()
        total_gpu_memory = torch.xpu.get_device_properties(gpu_id).total_memory
        free_gpu_memory = total_gpu_memory - used_memory

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device(device, gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def is_multimodal_model(model_architectures):
    if (
        "LlavaLlamaForCausalLM" in model_architectures
        or "LlavaQwenForCausalLM" in model_architectures
        or "LlavaMistralForCausalLM" in model_architectures
        or "LlavaVidForCausalLM" in model_architectures
        or "MllamaForConditionalGeneration" in model_architectures
        or "Qwen2VLForConditionalGeneration" in model_architectures
    ):
        return True
    else:
        return False


def is_attention_free_model(model_architectures):
    return False


def model_has_inner_state(model_architectures):
    return False


def is_embedding_model(model_architectures):
    if (
        "LlamaEmbeddingModel" in model_architectures
        or "MistralModel" in model_architectures
        or "LlamaForSequenceClassification" in model_architectures
        or "LlamaForSequenceClassificationWithNormal_Weights" in model_architectures
    ):
        return True
    else:
        return False


def is_generation_model(model_architectures, is_embedding: bool = False):
    # We have two ways to determine whether a model is a generative model.
    # 1. Check the model architectue
    # 2. check the `is_embedding` server args

    if (
        "LlamaEmbeddingModel" in model_architectures
        or "MistralModel" in model_architectures
        or "LlamaForSequenceClassification" in model_architectures
        or "LlamaForSequenceClassificationWithNormal_Weights" in model_architectures
    ):
        return False
    else:
        return not is_embedding


def decode_video_base64(video_base64):
    from PIL import Image

    # Decode the base64 string
    video_bytes = base64.b64decode(video_base64)

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


def load_image(image_file: Union[str, bytes]):
    from PIL import Image

    image = image_size = None

    if isinstance(image_file, bytes):
        image = Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content))
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    elif image_file.startswith("video:"):
        image_file = image_file.replace("video:", "")
        image, image_size = decode_video_base64(image_file)
    elif isinstance(image_file, str):
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    else:
        raise ValueError(f"Invalid image: {image}")

    return image, image_size


def suppress_other_loggers():
    from vllm.logger import logger as vllm_default_logger

    vllm_default_logger.setLevel(logging.WARN)
    logging.getLogger("vllm.config").setLevel(logging.ERROR)
    logging.getLogger("vllm.distributed.device_communicators.pynccl").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.distributed.device_communicators.shm_broadcast").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.selector").setLevel(logging.WARN)
    logging.getLogger("vllm.utils").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )


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


def kill_parent_process():
    """Kill the parent process and all children of the parent process."""
    current_process = psutil.Process()
    parent_process = current_process.parent()
    kill_child_process(parent_process.pid, skip_pid=current_process.pid)


def kill_child_process(pid, including_parent=True, skip_pid=None):
    """Kill the process and all its children process."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if including_parent:
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass


def monkey_patch_vllm_p2p_access_check(gpu_id: int):
    """
    Monkey patch the slow p2p access check in vllm.
    NOTE: We assume the p2p access is always allowed, which can be wrong for some setups.
    """

    import vllm.distributed.device_communicators.custom_all_reduce_utils as tgt

    setattr(tgt, "gpu_p2p_access_check", lambda *arg, **kwargs: True)


def monkey_patch_vllm_dummy_weight_loader():
    """
    Monkey patch the dummy weight loader in vllm to call process_weights_after_loading.
    """

    from vllm.model_executor.model_loader.loader import (
        CacheConfig,
        DeviceConfig,
        DummyModelLoader,
        LoRAConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        _initialize_model,
        initialize_dummy_weights,
        nn,
        set_default_torch_dtype,
    )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    lora_config,
                    cache_config,
                )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)

            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        return model.eval()

    setattr(DummyModelLoader, "load_model", load_model)


vllm_all_gather_backup = None


def monkey_patch_vllm_all_gather(reverse: bool = False):
    """Monkey patch all-gather to remove in-place operations."""
    from torch.distributed import _functional_collectives as funcol
    from vllm.distributed.parallel_state import GroupCoordinator

    global vllm_all_gather_backup
    if vllm_all_gather_backup is None:
        vllm_all_gather_backup = GroupCoordinator.all_gather

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )

        output_tensor = funcol.all_gather_tensor(
            input_, gather_dim=0, group=self.device_group
        ).view((world_size,) + input_size)

        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    if reverse:
        setattr(GroupCoordinator, "all_gather", vllm_all_gather_backup)
    else:
        setattr(GroupCoordinator, "all_gather", all_gather)


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

        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = (
                os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
            )
            if self.cache_dir:
                self.cache_dir = f"{self.cache_dir}_{os.getpid()}"
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(f"Fail to set RLIMIT_NOFILE: {e}")


def add_api_key_middleware(app, api_key: str):
    @app.middleware("http")
    async def authentication(request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/health"):
            return await call_next(request)
        if request.headers.get("Authorization") != "Bearer " + api_key:
            return ORJSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


def prepare_model_and_tokenizer(model_path: str, tokenizer_path: str):
    if "SGLANG_USE_MODELSCOPE" in os.environ:
        if not os.path.exists(model_path):
            from modelscope import snapshot_download

            model_path = snapshot_download(model_path)
            tokenizer_path = snapshot_download(
                tokenizer_path, ignore_patterns=["*.bin", "*.safetensors"]
            )
    return model_path, tokenizer_path


def configure_logger(server_args, prefix: str = ""):
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
):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            )
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


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


def first_rank_print(*args, **kwargs):
    if torch.cuda.current_device() == 0:
        print(*args, **kwargs)
    else:
        pass

"""Common utilities."""

import base64
import os
import random
import socket
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import List, Optional

import numpy as np
import pydantic
import requests
import torch
from fastapi.responses import JSONResponse
from packaging import version as pkg_version
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from sglang.utils import get_exception_traceback

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


def get_available_gpu_memory(gpu_id, distributed=True):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARNING: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def set_random_seed(seed: int) -> None:
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alloc_usable_network_port(num, used_list=()):
    port_list = []
    for port in range(10000, 65536):
        if port in used_list:
            continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                s.listen(1)  # Attempt to listen on the port
                port_list.append(port)
            except socket.error:
                pass  # If any error occurs, this port is not usable

            if len(port_list) == num:
                return port_list
    return None


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            return True
        except socket.error:
            return False


def allocate_init_ports(
    port: Optional[int] = None,
    additional_ports: Optional[List[int]] = None,
    tp_size: int = 1,
):
    port = 30000 if port is None else port
    additional_ports = [] if additional_ports is None else additional_ports
    additional_ports = (
        [additional_ports] if isinstance(additional_ports, int) else additional_ports
    )
    # first check on server port
    if not check_port(port):
        new_port = alloc_usable_network_port(1, used_list=[port])[0]
        print(f"WARNING: Port {port} is not available. Use {new_port} instead.")
        port = new_port

    # then we check on additional ports
    additional_unique_ports = set(additional_ports) - {port}
    # filter out ports that are already in use
    can_use_ports = [port for port in additional_unique_ports if check_port(port)]

    num_specified_ports = len(can_use_ports)
    if num_specified_ports < 4 + tp_size:
        addtional_can_use_ports = alloc_usable_network_port(
            num=4 + tp_size - num_specified_ports, used_list=can_use_ports + [port]
        )
        can_use_ports.extend(addtional_can_use_ports)

    additional_ports = can_use_ports[: 4 + tp_size]
    return port, additional_ports


def get_int_token_logit_bias(tokenizer, vocab_size):
    # a bug when model's vocab size > tokenizer.vocab_size
    vocab_size = tokenizer.vocab_size
    logit_bias = np.zeros(vocab_size, dtype=np.float32)
    for t_id in range(vocab_size):
        ss = tokenizer.decode([t_id]).strip()
        if not (ss.isdigit() or len(ss) == 0 or t_id == tokenizer.eos_token_id):
            logit_bias[t_id] = -1e5

    return logit_bias


def wrap_kernel_launcher(kernel):
    """A faster launcher for triton kernels."""
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    kernels = kernel.cache[rank].values()
    kernel = next(iter(kernels))

    # Different trition versions use different low-level names
    if hasattr(kernel, "cu_function"):
        kfunction = kernel.cu_function
    else:
        kfunction = kernel.function

    if hasattr(kernel, "c_wrapper"):
        run = kernel.c_wrapper
    else:
        run = kernel.run

    add_cluster_dim = True

    def ret_func(grid, num_warps, *args):
        nonlocal add_cluster_dim

        try:
            if add_cluster_dim:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    1,
                    1,
                    1,
                    1,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
            else:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
        except TypeError:
            add_cluster_dim = not add_cluster_dim
            ret_func(grid, num_warps, *args)

    return ret_func


def is_multimodal_model(model):
    from sglang.srt.model_config import ModelConfig

    if isinstance(model, str):
        model = model.lower()
        return "llava" in model or "yi-vl" in model or "llava-next" in model

    if isinstance(model, ModelConfig):
        model_path = model.path.lower()
        return "llava" in model_path or "yi-vl" in model_path or "llava-next" in model_path

    raise ValueError("unrecognized type")


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


def load_image(image_file):
    from PIL import Image

    image = image_size = None

    if image_file.startswith("http://") or image_file.startswith("https://"):
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
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file)))

    return image, image_size


def assert_pkg_version(pkg: str, min_version: str):
    try:
        installed_version = version(pkg)
        if pkg_version.parse(installed_version) < pkg_version.parse(min_version):
            raise Exception(
                f"{pkg} is installed with version {installed_version} which "
                f"is less than the minimum required version {min_version}"
            )
    except PackageNotFoundError:
        raise Exception(
            f"{pkg} with minimum required version {min_version} is not installed"
        )


API_KEY_HEADER_NAME = "X-API-Key"


class APIKeyValidatorMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        # extract API key from the request headers
        api_key_header = request.headers.get(API_KEY_HEADER_NAME)
        if not api_key_header or api_key_header != self.api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API Key"},
            )
        response = await call_next(request)
        return response


# FIXME: Remove this once we drop support for pydantic 1.x
IS_PYDANTIC_1 = int(pydantic.VERSION.split(".")[0]) == 1


def jsonify_pydantic_model(obj: BaseModel):
    if IS_PYDANTIC_1:
        return obj.json(ensure_ascii=False)
    return obj.model_dump_json()

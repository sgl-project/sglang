"""Common utilities."""

import base64
import fcntl
import logging
import multiprocessing
import os
import random
import socket
import struct
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import List, Optional

import numpy as np
import psutil
import requests
import rpyc
import torch
import triton
from fastapi.responses import JSONResponse
from packaging import version as pkg_version
from rpyc.utils.server import ThreadedServer
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


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


def get_available_gpu_memory(gpu_id, distributed=False):
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

    torch.cuda.empty_cache()
    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
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


def allocate_init_ports(
    port: Optional[int] = None,
    additional_ports: Optional[List[int]] = None,
    tp_size: int = 1,
    dp_size: int = 1,
):
    """Allocate ports for all connections."""
    if additional_ports:
        ret_ports = [port] + additional_ports
    else:
        ret_ports = [port]

    ret_ports = list(set(x for x in ret_ports if is_port_available(x)))
    cur_port = ret_ports[-1] + 1 if len(ret_ports) > 0 else 10000

    # HTTP + Tokenizer + Controller + Detokenizer + dp_size * (nccl + tp_size)
    num_ports_needed = 4 + dp_size * (1 + tp_size)
    while len(ret_ports) < num_ports_needed:
        if cur_port not in ret_ports and is_port_available(cur_port):
            ret_ports.append(cur_port)
        cur_port += 1

    if port is not None and ret_ports[0] != port:
        logger.warn(
            f"WARNING: Port {port} is not available. Use port {ret_ports[0]} instead."
        )

    return ret_ports[0], ret_ports[1:num_ports_needed]


def get_int_token_logit_bias(tokenizer, vocab_size):
    """Get the logit bias for integer-only tokens."""
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
    if int(triton.__version__.split(".")[0]) >= 3:
        return None

    gpu_id = torch.cuda.current_device()
    kernels = kernel.cache[gpu_id].values()
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
        return (
            "llava" in model_path or "yi-vl" in model_path or "llava-next" in model_path
        )

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


def connect_rpyc_service(host, port):
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect(
                host,
                port,
                config={
                    "allow_public_attrs": True,
                    "allow_pickle": True,
                    "sync_request_timeout": 3600,
                },
            )
            break
        except ConnectionRefusedError as e:
            time.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise RuntimeError(f"Connect rpyc error: {e}")

    return con.root


def start_rpyc_service(service: rpyc.Service, port: int):
    t = ThreadedServer(
        service=service,
        port=port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 3600,
        },
    )
    t.logger.setLevel(logging.WARN)
    t.start()


def start_rpyc_service_process(service: rpyc.Service, port: int):
    proc = multiprocessing.Process(target=start_rpyc_service, args=(service, port))
    proc.start()
    return proc


def suppress_other_loggers():
    from vllm.logger import logger as vllm_default_logger

    vllm_default_logger.setLevel(logging.WARN)
    logging.getLogger("vllm.config").setLevel(logging.ERROR)
    logging.getLogger("vllm.distributed.device_communicators.pynccl").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.selector").setLevel(logging.WARN)
    logging.getLogger("vllm.utils").setLevel(logging.WARN)


def assert_pkg_version(pkg: str, min_version: str, message: str):
    try:
        installed_version = version(pkg)
        if pkg_version.parse(installed_version) < pkg_version.parse(min_version):
            raise Exception(
                f"{pkg} is installed with version {installed_version}, which "
                f"is less than the minimum required version {min_version}. " +
                message
            )
    except PackageNotFoundError:
        raise Exception(
            f"{pkg} with minimum required version {min_version} is not installed. " +
            message
        )


def kill_parent_process():
    """Kill the parent process and all children of the parent process."""
    current_process = psutil.Process()
    parent_process = current_process.parent()
    children = current_process.children(recursive=True)
    for child in children:
        if child.pid != current_process.pid:
            os.kill(child.pid, 9)
    os.kill(parent_process.pid, 9)


def monkey_patch_vllm_p2p_access_check(gpu_id: int):
    """
    Monkey patch the slow p2p access check in vllm.
    NOTE: We assume the p2p access is always allowed, which can be wrong for some setups.
    """

    # TODO: need a better check than just dev str name match
    # compat: skip RTX 40 series as they do not have P2P feature and even checking for them may cause errors
    device_name = torch.cuda.get_device_name(gpu_id)
    if "RTX 40" not in device_name:
        import vllm.distributed.device_communicators.custom_all_reduce_utils as tgt

        setattr(tgt, "gpu_p2p_access_check", lambda *arg, **kwargs: True)


def monkey_patch_vllm_dummy_weight_loader():
    """
    Monkey patch the dummy weight loader in vllm to call process_weights_after_loading.
    """

    from vllm.model_executor.model_loader.loader import (
        ModelConfig, DeviceConfig, LoRAConfig, VisionLanguageConfig,
        ParallelConfig, SchedulerConfig, CacheConfig, nn,
        set_default_torch_dtype, _initialize_model, initialize_dummy_weights,
        DummyModelLoader
    )

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()

            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        return model.eval()

    setattr(DummyModelLoader, "load_model", load_model)


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


def get_ip_address(ifname):
    """
    Get the IP address of a network interface.

    :param ifname: Name of the network interface (e.g., 'eth0')
    :return: IP address of the network interface
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip_address = fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', bytes(ifname[:15], 'utf-8'))
    )[20:24]
    return socket.inet_ntoa(ip_address)


def send_addrs_to_rank_0(model_port_args, server_args):
    assert server_args.node_rank != 0 and server_args.dp_size == 1
    import torch.distributed as dist

    ifname = os.environ.get("SGLANG_SOCKET_IFNAME", os.environ.get("NCCL_SOCKET_IFNAME", "eth0"))
    ip_addr = get_ip_address(ifname)

    num_tp_ports = server_args.tp_size // server_args.nnodes
    model_port_args.model_tp_ips[:num_tp_ports] = [ip_addr] * num_tp_ports
    ip_addr = [int(x) for x in ip_addr.split(".")]
    addrs_tensor = torch.tensor(ip_addr + model_port_args.model_tp_ports, dtype=torch.int)

    init_method = f"tcp://{server_args.nccl_init_addr}"
    dist.init_process_group(backend="gloo", init_method=init_method, rank=server_args.node_rank, world_size=server_args.nnodes)
    dist.send(addrs_tensor, dst=0)
    print(f"Node {server_args.node_rank} sent: ip_address {ip_addr} and ports {model_port_args.model_tp_ports}")

    dist.barrier()
    dist.destroy_process_group() 


def receive_addrs(model_port_args, server_args):
    assert server_args.node_rank == 0 and server_args.dp_size == 1
    import torch.distributed as dist

    ifname = os.environ.get("SGLANG_SOCKET_IFNAME", os.environ.get("NCCL_SOCKET_IFNAME", "eth0"))
    ip_addr = get_ip_address(ifname)

    num_tp_ports = server_args.tp_size // server_args.nnodes
    model_port_args.model_tp_ips[:num_tp_ports] = [ip_addr] * num_tp_ports

    init_method = f"tcp://{server_args.nccl_init_addr}"
    dist.init_process_group(backend="gloo", init_method=init_method, rank=server_args.node_rank, world_size=server_args.nnodes)

    for src_rank in range(1, server_args.nnodes):
        tensor = torch.zeros(4 + num_tp_ports, dtype=torch.int)
        dist.recv(tensor, src=src_rank)
        ip = ".".join([str(x) for x in tensor[:4].tolist()])
        ports = tensor[4:].tolist()
        model_port_args.model_tp_ips[num_tp_ports * src_rank: num_tp_ports * (src_rank + 1)] = [ip] * num_tp_ports
        model_port_args.model_tp_ports[num_tp_ports * src_rank: num_tp_ports * (src_rank + 1)] = ports
        print(f"Node 0 received from rank {src_rank}: {tensor.tolist()}")

    dist.barrier()
    dist.destroy_process_group() 

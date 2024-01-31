import base64
import os
import random
import socket
import sys
import time
import traceback
from io import BytesIO
from typing import List, Optional

import numpy as np
import requests
import torch
import torch.distributed as dist

is_show_cost_time = False


def mark_cost_time(func_name):
    def inner_func(func):
        def time_func(*args, **kwargs):
            if dist.get_rank() in [0, 1] and is_show_cost_time:
                torch.cuda.synchronize()
                start_time = time.time()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                print(func_name, "cost time:", (time.time() - start_time) * 1000)
                return ans
            else:
                torch.cuda.synchronize()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                return ans

        return time_func

    return inner_func


time_mark = {}


def mark_start(key):
    torch.cuda.synchronize()
    global time_mark
    time_mark[key] = time.time()
    return


def mark_end(key, print_min_cost=0.0):
    torch.cuda.synchronize()
    global time_mark
    cost_time = (time.time() - time_mark[key]) * 1000
    if cost_time > print_min_cost:
        print(f"cost {key}:", cost_time)


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
            try:
                s.bind(("", port))
                port_list.append(port)
            except socket.error:
                pass

            if len(port_list) == num:
                return port_list
    return None


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except socket.error:
            return False


def handle_port_init(
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
        print(f"Port {port} is not available, using {new_port} instead.")
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


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def get_int_token_logit_bias(tokenizer, vocab_size):
    from transformers import LlamaTokenizer, LlamaTokenizerFast

    # a bug when model's vocab size > tokenizer.vocab_size
    vocab_size = tokenizer.vocab_size
    logit_bias = np.zeros(vocab_size, dtype=np.float32)
    for t_id in range(vocab_size):
        ss = tokenizer.decode([t_id]).strip()
        if not (ss.isdigit() or len(ss) == 0 or t_id == tokenizer.eos_token_id):
            logit_bias[t_id] = -1e5
        # else:
        #    print(ss, t_id)

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
    if isinstance(model, str):
        return "llava" in model
    from sglang.srt.model_config import ModelConfig

    if isinstance(model, ModelConfig):
        return "llava" in model.path.lower()
    raise Exception("unrecognized type")


def load_image(image_file):
    from PIL import Image

    image = None

    if image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content))
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file)))

    return image

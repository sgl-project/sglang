import functools
import os

colors = {
    "RED_FG": "\033[31m",
    "GREEN_FG": "\033[32m",
    "CYAN_FG": "\033[36m",
    "GRAY_FG": "\033[90m",
    "YELLOW_FG": "\033[33m",
    "RED_BG": "\033[41m",
    "GREEN_BG": "\033[42m",
    "CYAN_BG": "\033[46m",
    "YELLOW_BG": "\033[43m",
    "GRAY_BG": "\033[100m",
    "CLEAR": "\033[0m",
}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@functools.lru_cache()
def is_using_profiling_tools() -> bool:
    """
    Return whether we are running under profiling tools like nsys or ncu

    NOTE cuda-gdb will also cause conflict with CUPTI (bench_kineto) but currently we lack ways to detect it
    """
    is_using_nsys = os.environ.get("NSYS_PROFILING_SESSION_ID") is not None
    is_using_ncu = os.environ.get("NV_COMPUTE_PROFILER_PERFWORKS_DIR") is not None
    is_using_compute_sanitizer = (
        os.environ.get("NV_SANITIZER_INJECTION_PORT_RANGE_BEGIN") is not None
    )
    return is_using_nsys or is_using_ncu or is_using_compute_sanitizer


def set_random_seed(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Counter:
    def __init__(self):
        self.count = 0

    def next(self) -> int:
        self.count += 1
        return self.count - 1

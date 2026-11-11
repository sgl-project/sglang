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
import functools
import importlib
import itertools
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import traceback
import types
import uuid
from array import array
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from json import JSONDecodeError
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest import SkipTest
from unittest.case import _ShouldStop

import numpy as np
import orjson
import torch
from packaging import version as pkg_version
from torch.utils._contextlib import _DecoratorContextManager

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_flags,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
torch_release = pkg_version.parse(torch.__version__).release


@lru_cache(maxsize=1)
def is_mlu() -> bool:
    return hasattr(torch, "mlu") and torch.mlu.is_available()


def assert_int64_array(values: array, name: str) -> None:
    """Require a signed int64 array suitable for zero-copy tensor views."""
    assert (
        isinstance(values, array) and values.typecode == "q" and values.itemsize == 8
    ), f"{name} must be array('q') with 8-byte items"


def flatten_arrays_to_pinned_cpu(parts: List[array[int]], pin: bool) -> torch.Tensor:
    """Flatten array.array('q') buffers into one int64 CPU tensor.

    NumPy memcpy instead of a per-element PyLong-to-int64 walk. Stays on
    (optionally pinned) CPU; H2D is the caller's job.
    """
    combined = np.concatenate([np.frombuffer(p, dtype=np.int64) for p in parts])
    cpu_t = torch.from_numpy(combined)
    if pin:
        cpu_t = cpu_t.pin_memory()
    return cpu_t


def flatten_arrays_to_int64_tensor(
    parts: List[array[int]], device, pin: bool
) -> torch.Tensor:
    """Flatten a list of array.array('q') buffers into one int64 tensor on `device`."""
    return flatten_arrays_to_pinned_cpu(parts, pin).to(device, non_blocking=True)


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


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


@contextmanager
def temp_set_env(*, allow_sglang: bool = False, **env_vars: Any):
    """Temporarily set environment variables, restoring originals on exit.

    By default, SGLANG_*/SGL_* keys are rejected — use ``Envs`` descriptors
    for those.  Pass ``allow_sglang=True`` only for special env vars that
    intentionally bypass ``environ.py``.
    """
    if not allow_sglang:
        for key in env_vars:
            if key.startswith("SGLANG_") or key.startswith("SGL_"):
                raise ValueError("temp_set_env should not be used for sglang env vars")

    backup = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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

    def clone(self) -> DynamicGradMode:
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


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if is_mlu():
        torch.mlu.manual_seed_all(seed)


_KERNEL_VERSION_CHECK_PACKAGES = frozenset(
    {
        "flashinfer-python",
        "flashinfer_python",
        "sglang-kernel",
        "sglang_kernel",
    }
)


def _should_skip_kernel_pkg_version_check(pkg: str) -> bool:
    return (
        pkg in _KERNEL_VERSION_CHECK_PACKAGES
        and envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.get()
    )


def assert_pkg_version(pkg: str, min_version: str, message: str):
    if _should_skip_kernel_pkg_version_check(pkg):
        return

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
    """Check if a package is installed and meets the minimum version requirement."""
    if _should_skip_kernel_pkg_version_check(pkg):
        return True

    try:
        installed_version = version(pkg)
        return pkg_version.parse(installed_version) >= pkg_version.parse(min_version)
    except PackageNotFoundError:
        return False


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


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def round_up(x: int, y: int) -> int:
    return ((x - 1) // y + 1) * y


class EmptyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def empty_context(*args, **kwargs):
    return EmptyContextManager()


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


def run_with_deadline(fn: Callable[[], Any], *, timeout_s: float, what: str) -> Any:
    result: list = []
    error: list = []

    def _target():
        try:
            result.append(fn())
        except BaseException as e:
            error.append(e)

    # An overrunning fn cannot be cancelled; only process exit reaps the daemon thread.
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise RuntimeError(
            f"{what} did not return within {timeout_s}s on {socket.gethostname()}"
        )
    if error:
        raise error[0]
    return result[0]


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
        except _ShouldStop:
            # `unittest.case._ShouldStop` is raised by `subTest.__exit__`
            # when a subtest fails/skips and `result.failfast` is True
            # (CI invokes `python3 file.py -f`). It signals the outer
            # `testPartExecutor` to stop the test method cleanly; do
            # NOT retry, just propagate so unittest handles it.
            raise
        except Exception as e:
            traceback.print_exc()

            if try_index >= max_retry:
                raise Exception("retry() exceed maximum number of retries.")

            if not should_retry(e):
                raise Exception("retry() observe errors that should not be retried.")

            delay = min(initial_delay * (2**try_index), max_delay) * (
                0.75 + 0.25 * random.random()
            )

            logger.warning(
                f"retry() failed once ({try_index}th try, maximum {max_retry} retries). Will delay {delay:.2f}s and retry. Error: {e}"
            )

            time.sleep(delay)


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


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

    # Skip disposal under a captured prefill graph (piecewise or breakable):
    # freeing the backing storage would invalidate addresses recorded in the
    # graph. Local imports avoid a circular dependency.
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    if is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph():
        return

    if get_flags().capture.disable_dispose_tensor:
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


def find_local_repo_dir(repo_id: str, revision: Optional[str] = None) -> Optional[str]:
    import huggingface_hub as hf

    # Build cache path
    cache_path = os.path.join(
        hf.constants.HF_HUB_CACHE,
        hf.constants.REPO_ID_SEPARATOR.join(["models", *repo_id.split("/")]),
    )

    # A branch or tag name (default "main") maps to a commit through refs/;
    # snapshots/ is keyed by commit only.
    ref_path = os.path.join(cache_path, "refs", revision or "main")
    if os.path.isfile(ref_path):
        with open(ref_path) as f:
            revision = f.read().strip()

    # List files from revision directory
    if revision:
        rev_dir = os.path.join(cache_path, "snapshots", revision)
        if os.path.isdir(rev_dir):
            return rev_dir

    return None


class LazyValue:
    def __init__(self, creator: Callable):
        self._creator = creator
        self._value = None

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

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


# COPIED FROM DeepGEMM
def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


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


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def get_or_create_event_loop():
    """Gets the running event loop or creates a new one if it doesn't exist."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        # Use dictionary for fast exact type matching, using OrderedDict(mapping)
        # to maintains registration order
        self._mapping = OrderedDict(mapping)
        # MRO cache for inheritance-based matching
        self._mro_cache = {}
        self._fallback_fn = None

    def add_fallback_fn(self, fallback_fn: Callable):
        self._fallback_fn = fallback_fn

    def __iadd__(self, other: TypeBasedDispatcher):
        for ty, fn in other._mapping.items():
            if ty not in self._mapping:
                self._mapping[ty] = fn

        self._mro_cache.clear()
        return self

    def __call__(self, obj: Any):
        obj_type = type(obj)
        # 1. First try exact match(o(1))
        fn = self._mapping.get(obj_type)
        if fn is not None:
            return fn(obj)

        # 2. If exact match fails, check MRO cache
        cached_fn = self._mro_cache.get(obj_type)
        if cached_fn is not None:
            return cached_fn(obj)

        # 3.search in registration order for compatible type(maintains origin behavior)
        for ty, fn in self._mapping.items():
            if isinstance(obj, ty):
                self._mro_cache[obj_type] = fn
                return fn(obj)

        # 4. if no matching type found, cache this result
        self._mro_cache[obj_type] = None

        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")

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

import builtins
import contextlib
import functools
import inspect
import keyword
import operator
import os
import re
import threading
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import torch
import torch._prims_common as utils
import triton
from triton.runtime.cache import default_cache_dir


def type_promotion(*args, type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND):
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    return computation_dtype, result_dtype


def broadcast(s1: Tuple[int], s2: Tuple[int]) -> Tuple[int]:
    _s1, _s2 = s1, s2
    r1 = len(s1)
    if r1 == 0:
        return s2
    r2 = len(s2)
    if r2 == 0:
        return s1

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    s = list(s1)

    for i in range(r2):
        if s1[d + i] == 1:
            s[d + i] = s2[i]
        elif s2[i] == 1:
            s[d + i] = s1[d + i]
        elif s2[i] == s1[d + i]:
            s[d + i] = s2[i]
        else:
            raise ValueError(f"Unbroadcastable {_s1} and {_s2}")
    s = tuple(s)
    return s


def broadcastable_to(s1: Tuple[int], s2: Tuple[int]) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:  # r1 > 0
        return False

    if r1 > r2:
        return False

    d = r2 - r1
    for i in range(r1):
        if s1[i] == 1 or s1[i] == s2[d + i]:
            continue
        return False
    return True


def broadcast_shapes(shapes: Iterable[Tuple[int]]) -> Tuple[int]:
    if len(shapes) == 0:
        return ()
    shape = shapes[0]
    for s in shapes[1:]:
        shape = broadcast(shape, s)
    return shape


def broadcasted_stride(
    shape: Tuple[int], stride: Tuple[int], new_shape: Tuple[int]
) -> Tuple[int]:
    assert broadcastable_to(shape, new_shape)
    r1 = len(shape)
    r2 = len(new_shape)
    d = r2 - r1
    new_stride = [0 for _ in range(r2)]
    for i in range(r1):
        new_stride[d + i] = 0 if (shape[i] == 1 and new_shape[d + i] > 1) else stride[i]
    return tuple(new_stride)


def volume(shape: Tuple[int]) -> int:
    return functools.reduce(operator.mul, shape, 1)


def c_contiguous_stride(shape: Tuple[int]) -> Tuple[int]:
    strides = []
    s = 1
    for size in reversed(shape):
        strides.append(s)
        s *= size

    return tuple(reversed(strides))


def size_in_bytes(a):
    return a.numel() * a.element_size()


class LibEntry(triton.KernelInterface):
    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.kernel_cache = tuple(dict() for _ in range(torch.cuda.device_count()))

        fn = self.fn
        while not isinstance(fn, triton.runtime.JITFunction):
            fn = fn.fn
        self.jit_function: triton.runtime.JITFunction = fn
        self.specialize_indices = [
            p.num
            for p in self.jit_function.params
            if not p.is_constexpr and not p.do_not_specialize
        ]
        self.do_not_specialize_indices = [
            p.num
            for p in self.jit_function.params
            if not p.is_constexpr and p.do_not_specialize
        ]
        self.lock = threading.Lock()

    def key(self, spec_args, dns_args, const_args):
        spec_key = [
            (
                (arg.dtype, arg.data_ptr() % self.divisibility == 0)
                if hasattr(arg, "data_ptr")
                else (type(arg), arg)
            )
            for arg in spec_args
        ]
        dns_key = [
            (
                arg.dtype
                if hasattr(arg, "data_ptr")
                else (
                    type(arg)
                    if not isinstance(arg, int)
                    else (
                        "i32"
                        if -(2**31) <= arg and arg <= 2**31 - 1
                        else "u64" if 2**63 <= arg and arg <= 2**64 - 1 else "i64"
                    )
                )
            )
            for arg in dns_args
        ]
        # const args passed by position
        return tuple(spec_key + dns_key + const_args)

    def run(self, *args, **kwargs):
        grid = kwargs["grid"]

        # collect all the arguments
        spec_args = []  # specialize arguments
        dns_args = []  # do not specialize arguments
        const_args = []  # constexpr arguments
        k_args = []  # kernel arguments
        for i, arg in enumerate(args):
            if i in self.specialize_indices:
                k_args.append(arg)
                spec_args.append(arg)
            elif i in self.do_not_specialize_indices:
                k_args.append(arg)
                dns_args.append(arg)
            else:
                const_args.append(arg)
        for p in self.jit_function.params[len(args) :]:
            if p.name in kwargs:
                val = kwargs[p.name]
            elif p.default is inspect._empty:
                continue
            else:
                val = p.default

            if p.is_constexpr:
                const_args.append(val)
            elif p.do_not_specialize:
                dns_args.append(val)
                k_args.append(val)
            else:
                spec_args.append(val)
                k_args.append(val)

        entry_key = self.key(spec_args, dns_args, const_args)
        device = torch.cuda.current_device()
        cache = self.kernel_cache[device]
        while entry_key not in cache:
            # NOTE: we serialize the first run of a jit function regardless of which device to run on
            # because Triton runtime is currently not threadsafe.
            with self.lock:
                if entry_key in cache:
                    break
                kernel = self.fn.run(*args, **kwargs)
                fn = self.fn
                # collect constexpr arguments for grid computation
                constexprs = {}
                while not isinstance(fn, triton.runtime.JITFunction):
                    if isinstance(fn, triton.runtime.Autotuner):
                        config = fn.best_config
                        constexprs["num_warps"] = config.num_warps
                        constexprs["num_stages"] = config.num_stages
                        constexprs["num_ctas"] = config.num_ctas
                        constexprs = {**constexprs, **config.kwargs}
                    elif isinstance(fn, triton.runtime.Heuristics):
                        for v, heur in fn.values.items():
                            constexprs[v] = heur(
                                {
                                    **dict(zip(fn.arg_names, args)),
                                    **kwargs,
                                    **constexprs,
                                }
                            )
                    else:
                        raise RuntimeError("Invalid Runtime Function")
                    fn = fn.fn
                for p in self.jit_function.params:
                    if p.is_constexpr and p.name not in constexprs:
                        constexprs[p.name] = p.default
                cache[entry_key] = (kernel, constexprs)
            return kernel, constexprs

        kernel, constexprs = cache[entry_key]

        if callable(grid):
            # collect all arguments to the grid fn，ie:
            # 1. args,
            # 2. kwargs,
            # 3. all all other captured arguments in CompiledKernel from Autotunner & Heuristics
            # when kwargs & captured args conflict, captured args have higher priority
            meta = {**dict(zip(self.arg_names, args)), **kwargs, **constexprs}
            grid = grid(meta)
        grid = grid + (1, 1)

        kernel[grid[0:3]](*k_args)
        return kernel, constexprs


def libentry():
    """
    Decorator for triton library entries.
    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        self.writeline("\n")

    def writeline(self, line):
        if line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()


class NameSpace:
    def __init__(self):
        self._used_names: Set[str] = set()
        self._base_count: Dict[str, int] = defaultdict(int)

        self._illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
        self._name_suffix_regex = re.compile(r"(.*)_(\d+)$")

    def create_name(self, candidate: str) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
        """
        # delete all characters that are illegal in a Python identifier
        candidate = self._illegal_char_regex.sub("_", candidate)

        if not candidate:
            candidate = "_unnamed"

        if candidate[0].isdigit():
            candidate = f"_{candidate}"

        match = self._name_suffix_regex.match(candidate)
        if match is None:
            base = candidate
            num = None
        else:
            base, num_str = match.group(1, 2)
            num = int(num_str)

        candidate = base if num is None else f"{base}_{num}"
        if not num:
            num = self._base_count[base]

        while candidate in self._used_names or self._is_illegal_name(candidate):
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        return candidate

    def _is_illegal_name(self, name: str) -> bool:
        # 1. keywords are never allowed as names.
        if name in keyword.kwlist:
            return True

        # 2. Can't shadow a builtin name, unless you *are* that builtin.
        if name in builtins.__dict__:
            return True

        return False


@functools.lru_cache(maxsize=None)
def cache_dir_path() -> Path:
    return default_cache_dir()


def cache_dir() -> Path:
    _cache_dir = cache_dir_path()
    os.makedirs(_cache_dir, exist_ok=True)
    return _cache_dir

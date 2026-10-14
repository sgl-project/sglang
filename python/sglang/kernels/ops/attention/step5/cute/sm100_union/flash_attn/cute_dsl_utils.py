# Copyright (c) 2025, Tri Dao.

import os
import itertools
import pathlib
import shutil
from typing import Tuple
from functools import partial, lru_cache
from dataclasses import dataclass, fields

import torch

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.cutlass_dsl import NumericMeta


StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
load_cubin_module_og = cutlass.base_dsl.runtime.cuda.load_cubin_module
cute_compile_og = cute.compile
_cubin_dump_counter = itertools.count()


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        types, self._values_pos = [], []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                obj_types = obj.__get_mlir_types__()
                types.extend(obj_types)
                self._values_pos.append(len(obj_types))
            else:
                self._values_pos.append(0)
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def load_cubin_module_data_patched(cubin_data, filepath):
    pathlib.Path(filepath).write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def load_cubin_module_patched(cubin_file, filepath):
    shutil.copyfile(cubin_file, filepath)
    return load_cubin_module_og(cubin_file)


def cute_compile_patched(*args, **kwargs):
    """A patched version of cute.compile that dump the SASS to a file if CUTE_CUBIN_PATH is set."""
    cubin_path = os.getenv("CUTE_CUBIN_PATH", None)
    cubin_dir = os.getenv("CUTE_CUBIN_DIR", None)
    if cubin_path is None and cubin_dir is not None:
        pathlib.Path(cubin_dir).mkdir(parents=True, exist_ok=True)
        func_name = getattr(args[0], "__class__", type(args[0])).__name__
        cubin_path = os.path.join(
            cubin_dir,
            f"cute_{os.getpid()}_{next(_cubin_dump_counter):04d}_{func_name}.cubin",
        )
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=cubin_path
        )
        cutlass.base_dsl.runtime.cuda.load_cubin_module = partial(
            load_cubin_module_patched, filepath=cubin_path
        )
    try:
        output = cute_compile_og(*args, **kwargs)
    finally:
        if cubin_path is not None:
            cutlass.base_dsl.runtime.cuda.load_cubin_module_data = load_cubin_module_data_og
            cutlass.base_dsl.runtime.cuda.load_cubin_module = load_cubin_module_og
    if cubin_path is not None:
        if pathlib.Path(cubin_path).exists() and extract is not None:
            try:
                sass = extract(cubin_path, None)
                pathlib.Path(cubin_path).with_suffix(".annotated.sass").write_text(sass)
            except Exception as exc:
                # Note(wangbojun/codex): Some runtime images ship a cuobjdump that
                # cannot disassemble Blackwell cubins. Keep the cubin for later
                # CUDA 13.2 tooling instead of failing the model startup path.
                print(
                    f"[cute_compile_patched] cubin dumped to {cubin_path}; "
                    f"SASS extraction skipped: {exc}",
                    flush=True,
                )
        elif not pathlib.Path(cubin_path).exists():
            print(
                f"[cute_compile_patched] CUTE_CUBIN_PATH was set but no cubin was "
                f"loaded through the patched runtime hooks: {cubin_path}",
                flush=True,
            )
    return output

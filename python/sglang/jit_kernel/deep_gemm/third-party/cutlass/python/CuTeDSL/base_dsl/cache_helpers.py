# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
This module provides jit cache load/dump helper functions
"""

import os
import uuid
import random
import tempfile
import pwd
import time
from pathlib import Path
import hashlib

from .utils.logger import log
from .jit_executor import JitExecutor

from .._mlir import ir

# =============================================================================
# Jit Cache Helper functions
# =============================================================================


def get_current_user():
    # Try to get the user from the environment variable first
    user = os.getenv("USER") or os.getenv("USERNAME")
    if not user:
        # Fallback for Unix-like systems
        user = pwd.getpwuid(os.getuid()).pw_name
    return user


try:
    default_generated_ir_path = f"/tmp/{get_current_user()}/cutlass_python_cache/"
except Exception as e:
    # If all else fails, provide a default fallback path
    default_generated_ir_path = "/tmp/cutlass_python_cache/"
    print(f"Could not determine user, using default path. Error: {e}")


def load_ir(file, asBytecode=False):
    """Load generated IR from a file."""
    assert "mlir" in file
    func_name = file.split(".mlir")[0].split("dsl_")[-1]
    with ir.Context() as ctx:
        with open(file, "rb" if asBytecode else "r") as f:
            module = ir.Module.parse(f.read())

    return func_name, module


def make_unique_filename(fpath: Path, new_ext: str = None) -> Path:
    """Generate a unique filename with an optional new extension."""
    random_part = random.randint(0, 999999)
    timestamp = time.time()
    hash_input = f"{fpath}_{timestamp}_{random_part}".encode()
    hash_code = hashlib.md5(hash_input).hexdigest()[:16]  # Shorter hash for readability
    stem_with_hash = f"{fpath.stem}_{hash_code}"
    return fpath.with_name(stem_with_hash).with_suffix(new_ext or fpath.suffix)


def save_ir(
    dsl_name: str,
    module: object,
    fname: str,
    isTemp: bool = False,
    asBytecode: bool = False,
) -> str:
    """Save generated IR to a file."""
    initial_name = f"{dsl_name.lower()}_{fname}.mlir"
    save_path = Path(tempfile.gettempdir() if isTemp else os.getcwd())
    save_fname = save_path / initial_name
    # Random ID to avoid any collisions
    rnd_id = str(uuid.uuid4())
    pid = os.getpid()
    # use temp dir to be robust against program interruptions
    temp_dir = os.path.join(save_path, f"tmp.pid_{pid}_{rnd_id}")
    # If the process exits abnormally, may leave a temporary folder. Needs to be removed manually.
    os.makedirs(temp_dir, exist_ok=False)
    temp_fname = os.path.join(temp_dir, initial_name)

    if asBytecode:
        with open(temp_fname, "wb") as f:
            module.operation.write_bytecode(f)
    else:
        with open(temp_fname, "w") as f:
            print(module, file=f)
    # os.replace is guaranteed to be atomic on POSIX systems if it succeeds
    # so filepath cannot see a partial write
    os.replace(temp_fname, save_fname)
    os.removedirs(temp_dir)
    log().debug("Generated IR saved into %s", save_fname)
    return save_fname


def check_func_name(jit_cache, func_name):
    if not func_name in jit_cache:
        jit_cache[func_name] = JitExecutor(None, None, None, None, None, None)
    return jit_cache


def load_cache_from_path(dsl_name, cache_limit, path=default_generated_ir_path):
    """Load cache from a directory path."""
    if not os.path.exists(path):
        return dict()
    files = os.listdir(path)
    jit_cache = dict()
    try:
        for idx, file in enumerate(files):
            if idx >= int(cache_limit):
                break
            # identify dsl prefix
            if not file.startswith(f"{dsl_name.lower()}"):
                continue
            if ".mlir" in file:
                func_name, ir_module = load_ir(
                    os.path.join(path, file), asBytecode=True
                )
                jit_cache = check_func_name(jit_cache, func_name)
                jit_cache[func_name].ir_module = ir_module
    except Exception as e:
        print(f"{dsl_name} failed with loading generated IR cache.", e)
        jit_cache = dict()
    return jit_cache


def dump_cache_to_path(
    dsl_name, jit_cache, cache_limit, path=default_generated_ir_path
):
    log().info("JIT cache : dumping [%s] items=[%s]", dsl_name, len(jit_cache))
    os.makedirs(path, exist_ok=True)
    original_path = os.getcwd()
    try:
        os.chdir(path)
        for idx, [key, value] in enumerate(jit_cache.items()):
            if idx >= int(cache_limit):
                break
            save_ir(dsl_name, value.ir_module, key, asBytecode=True)
    except Exception as e:
        print(f"{dsl_name} failed with caching generated IR", e)
    finally:
        os.chdir(original_path)

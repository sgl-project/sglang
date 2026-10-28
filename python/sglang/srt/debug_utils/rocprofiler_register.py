"""rocprofiler-register helpers.

rocprofiler-register records the library that claimed registration in an
environment variable, which exec then passes to every child. This module drops
such an inherited claim. It runs before torch is imported, so nothing it
imports may pull torch in.
"""

import importlib.util
import os
import pathlib

from sglang.srt.environ import envs


def torch_built_with_hip() -> bool:
    """Whether torch is a ROCm build, without importing torch.

    Same answer as srt.utils.common.is_hip, but that one imports torch and this
    has to run before torch is imported. torch/version.py is nothing but a few
    constant assignments, so reading only that file is enough.
    """
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        return False
    version_path = pathlib.Path(torch_spec.origin).with_name("version.py")
    version_spec = importlib.util.spec_from_file_location("torch_version", version_path)
    if version_spec is None or version_spec.loader is None:
        return False
    version = importlib.util.module_from_spec(version_spec)
    try:
        version_spec.loader.exec_module(version)
    except OSError:
        return False
    return getattr(version, "hip", None) is not None


def _rocprofiler_sdk_already_loaded() -> bool:
    """Check whether this process has already loaded a rocprofiler-sdk library.

    A profiler that is really running has loaded its library into this process,
    so the library shows up in the process memory map. If this function cannot
    read that map, it returns True, and the caller then treats the process as
    one where a profiler may be running and changes nothing.
    """
    try:
        with open("/proc/self/maps") as f:
            return "librocprofiler-sdk" in f.read()
    except OSError:
        return True


def clear_inherited_rocprofiler_registration():
    """Delete ROCPROFILER_REGISTER_LIBRARY if this process inherited it.

    When a profiler registers, HIP writes the library path into the environment
    variable ROCPROFILER_REGISTER_LIBRARY. A child process inherits that
    variable, so the child thinks a profiler has already registered and refuses
    to register its own. torch.profiler in the child then records CPU rows
    only: it reports no GPU events and no error.

    Only ROCm sets this variable, so this function returns right away on a CUDA
    build. Only PyTorch 2.12 and newer hit the problem, because of
    rocprofiler-sdk; PyTorch 2.11 and older use roctracer, so deleting the
    variable changes nothing for them.

    This function deletes the variable only when no profiler is running in this
    process, so it never breaks a profiler that is really collecting data.
    The process must call this function before it imports torch.
    """
    if os.environ.get("ROCPROFILER_REGISTER_LIBRARY") is None:
        return
    if not torch_built_with_hip():
        return
    if envs.SGLANG_ROCM_KEEP_INHERITED_ROCPROFILER_REGISTER.get():
        return
    if _rocprofiler_sdk_already_loaded():
        return
    os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)

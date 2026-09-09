"""Read-only server checks; keep driver, Toolkit and Torch CUDA versions separate."""

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


class Unavailable(RuntimeError):
    """A missing hardware capability is SKIP (exit 77), never PASS."""


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def snapshot():
    nvcc = shutil.which("nvcc")
    return {
        "python": sys.executable,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "os_release": platform.freedesktop_os_release(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_path": torch.__file__,
        "torch_arch_list": (
            torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        ),
        "nvidia_smi": command(["nvidia-smi"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
        "ffmpeg": command(["ffmpeg", "-version"]),
        "nvcc": command([nvcc, "--version"]) if nvcc else {"error": "nvcc not in PATH"},
        "pip_check": command([sys.executable, "-m", "pip", "check"]),
        "packages": sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
        ),
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
            }
            for i in range(torch.cuda.device_count())
        ],
    }


def binding_check():
    import nccl.core as core
    import nccl.ep as ep

    expected = {
        "nccl4py": "0.4.1",
        "nccl-extensions": "0.1.0",
        "nvidia-nccl-cu13": "2.30.7",
    }
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, got {actual}")
    if torch.__version__ != "2.11.0+cu130":
        raise RuntimeError(
            f"Expected official Torch 2.11.0+cu130, got {torch.__version__}"
        )
    if torch.cuda.nccl.version() != (2, 28, 9):
        raise RuntimeError("Official Torch NCCL headers differ from 2.28.9")
    if not callable(getattr(ep.Handle, "update", None)):
        raise RuntimeError("The selected EP binding does not expose Handle.update")
    version = core.get_version()
    nccl_path = Path(version.libnccl.path).resolve()
    loaded = ctypes.CDLL(str(nccl_path))
    value = ctypes.c_int()
    if loaded.ncclGetVersion(ctypes.byref(value)) != 0 or value.value != 23007:
        raise RuntimeError(f"Unexpected NCCL runtime version {value.value}")
    ep_version = str(ep.get_lib_version())  # Force native library loading.
    ep_path = Path(ep.get_lib_path()).resolve()
    prefix = Path(sys.prefix).resolve()
    if not nccl_path.is_relative_to(prefix) or not ep_path.is_relative_to(prefix):
        raise RuntimeError("NCCL/EP loaded outside the selected runtime environment")
    mapped_nccl = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and "/libnccl.so" in line
    }
    if mapped_nccl != {nccl_path}:
        raise RuntimeError(
            f"Multiple or unexpected NCCL libraries loaded: {mapped_nccl}"
        )
    return {
        "torch_nccl_compile_time": torch.cuda.nccl.version(),
        "nccl_runtime": value.value,
        "nccl_cuda_variant": str(version.libnccl.cuda_variant),
        "nccl_path": str(nccl_path),
        "nccl_sha256": hashlib.file_digest(nccl_path.open("rb"), "sha256").hexdigest(),
        "ep_version": ep_version,
        "ep_path": str(ep_path),
        "ep_sha256": hashlib.file_digest(ep_path.open("rb"), "sha256").hexdigest(),
        "ep_handle_update": callable(getattr(ep.Handle, "update", None)),
    }


def prepare_jit():
    """Select the installed wheel's headers and record the actual JIT compiler."""
    import nccl.ep as ep
    import nvidia.nccl as nccl_package

    expected = {
        "NCCL_EP_JIT_SOURCE_DIR": Path(ep.__file__).parent / "include/nccl_ep",
        "NCCL_EP_JIT_BUILD_INCLUDE_DIR": Path(nccl_package.__path__[0]) / "include",
    }
    for variable, path in expected.items():
        configured = Path(os.environ.get(variable, path)).resolve()
        if configured != path.resolve():
            raise RuntimeError(
                f"{variable} must use the selected wheel headers: {path}"
            )
        if not path.is_dir():
            raise RuntimeError(f"Missing installed headers: {path}")
        os.environ[variable] = str(configured)
    toolkit = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH"))
    candidate = os.environ.get("NCCL_EP_JIT_NVCC") or os.environ.get("NVCC")
    if not candidate:
        candidate = str(Path(toolkit) / "bin/nvcc") if toolkit else "nvcc"
    nvcc = shutil.which(candidate)
    if not nvcc:
        raise RuntimeError("EP JIT needs nvcc; set CUDA_HOME or NCCL_EP_JIT_NVCC")
    version = command([nvcc, "--version"])
    if version.get("returncode") != 0 or not re.search(
        r"release 13\.", version["stdout"]
    ):
        raise RuntimeError(f"Expected CUDA 13.x nvcc for the cu13 EP wheel: {version}")
    includes = Path(nvcc).resolve().parents[1] / "include"
    if not (includes / "cuda_runtime.h").is_file():
        raise RuntimeError(f"Missing matching CUDA headers: {includes}")
    configured = Path(
        os.environ.get("NCCL_EP_JIT_CUDA_INCLUDE_DIR", includes)
    ).resolve()
    if configured != includes.resolve():
        raise RuntimeError("The JIT CUDA include path must match the selected nvcc")
    os.environ["NCCL_EP_JIT_NVCC"] = str(Path(nvcc).resolve())
    os.environ["NCCL_EP_JIT_CUDA_INCLUDE_DIR"] = str(includes)
    return {
        "nvcc": version,
        "paths": {
            key: os.environ.get(key)
            for key in (
                *expected,
                "NCCL_EP_JIT_NVCC",
                "NCCL_EP_JIT_CUDA_INCLUDE_DIR",
                "NCCL_EP_JIT_CACHE_DIR",
            )
        },
    }


def require_pair(*, ep=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or torch.cuda.device_count() != 2:
        raise Unavailable("Use torchrun with exactly two ranks and two visible GPUs")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not torch.cuda.can_device_access_peer(rank, 1 - rank):
        raise Unavailable(f"GPU {rank} cannot directly access its peer")
    if ep and torch.cuda.get_device_capability(rank)[0] < 9:
        raise Unavailable(
            "NCCL EP requires SM90+; SM89 can run only non-EP experiments"
        )
    return int(os.environ["RANK"]), rank


def report(stage, operation):
    """Every entrypoint writes an unambiguous machine-readable terminal status."""
    status, code = "PASS", 0
    try:
        result = operation()
    except Unavailable as error:
        status, code, result = "SKIP", 77, {"reason": str(error)}
    except Exception as error:
        import traceback

        traceback.print_exc()
        status, code, result = "FAIL", 1, {"error": repr(error)}
    payload = {
        "stage": stage,
        "rank": int(os.environ.get("RANK", "0")),
        "status": status,
        "result": result,
    }
    print(json.dumps(payload, indent=2), flush=True)
    report_dir = os.environ.get("NCCL_EP_REPORT_DIR")
    if report_dir:
        path = Path(report_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{stage}-rank{payload['rank']}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    return code

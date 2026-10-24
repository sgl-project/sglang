"""hipcc build and HIP module-launch helpers for the hand-written gfx950 kernels."""

from __future__ import annotations

import ctypes
import os

import torch

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    find_loaded_library,
)

_hip = None


def hip_lib():
    global _hip
    if _hip is None:
        # Bind to the HIP runtime torch already mapped. ROCm 10 ships a second
        # libamdhip64 in _rocm_sdk_devel; an unversioned CDLL("libamdhip64.so")
        # picks that one, and every launch on a torch stream then fails with
        # hipErrorContextIsDestroyed (709).
        torch.cuda.current_device()
        _hip = ctypes.CDLL(find_loaded_library("libamdhip64") or "libamdhip64.so")
        _hip.hipModuleLoad.restype = ctypes.c_int
        _hip.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _hip.hipModuleGetFunction.restype = ctypes.c_int
        _hip.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _hip.hipModuleLaunchKernel.restype = ctypes.c_int
        _hip.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def check_hip(rc, what):
    if rc != 0:
        raise RuntimeError(
            f"smallm_moe: {what} failed: {rc} ({hip_lib().hipGetErrorString(rc).decode()})"
        )


class HipKernel:
    def __init__(self, mod, name):
        self.fn = ctypes.c_void_p()
        check_hip(
            hip_lib().hipModuleGetFunction(ctypes.byref(self.fn), mod, name.encode()),
            f"hipModuleGetFunction {name}",
        )

    def launch(self, grid, args, stream, block=256, shared=0):
        size = ctypes.c_size_t(ctypes.sizeof(args))
        extra = (ctypes.c_void_p * 5)(
            1,
            ctypes.cast(ctypes.byref(args), ctypes.c_void_p).value,
            2,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value,
            3,
        )
        check_hip(
            hip_lib().hipModuleLaunchKernel(
                self.fn,
                grid,
                1,
                1,
                block,
                1,
                1,
                shared,
                ctypes.c_void_p(stream),
                None,
                ctypes.cast(extra, ctypes.c_void_p),
            ),
            "hipModuleLaunchKernel",
        )


def hipcc_path():
    return os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")


def hip_version():
    if torch.version.hip:
        return tuple(int(x) for x in torch.version.hip.split("-")[0].split(".")[:3])
    return (0, 0, 0)

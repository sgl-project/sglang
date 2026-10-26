"""Muse Spark decode MoE finalize fused with the TP4 all-reduce (MI355X, ``SGLANG_MUSE_FINALIZE_AR_PUSH``).

``finalize_allreduce.hip`` replaces the FlyDSL finalize + aiter two-stage all-reduce pair of the compact A4
target-verify MoE with one launch; see its header for the protocol. The code object is built at first use with
``hipcc --genco`` and launched through ``hipModuleLaunchKernel``, so it is graph-capturable without a torch
extension. Construction is collective over the TP CPU group and must happen outside graph capture.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import tempfile

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "finalize_allreduce.hip"
)
_NR = 4
_HIDDEN = 4096
_TOPK = 8
_ROW_BYTES = _HIDDEN * 2
_IPC_LAZY_ENABLE_PEER_ACCESS = 0x1


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _Args(ctypes.Structure):
    _fields_ = [
        ("y", ctypes.c_void_p),
        ("gamma", ctypes.c_void_p),
        ("scores", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("peers", ctypes.c_void_p * _NR),
        ("row_state", ctypes.c_void_p),
        ("local_start", ctypes.c_int32),
        ("num_local", ctypes.c_int32),
        ("eps", ctypes.c_float),
        ("gain", ctypes.c_float),
        ("rank", ctypes.c_int32),
        ("rows", ctypes.c_int32),
        ("max_rows", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class _FillArgs(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p), ("n16", ctypes.c_uint64)]


assert ctypes.sizeof(_Args) == 112 and ctypes.sizeof(_FillArgs) == 16

_hip = None


def _hip_lib():
    global _hip
    if _hip is None:
        h = ctypes.CDLL("libamdhip64.so")
        h.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        h.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        h.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        h.hipMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        h.hipFree.argtypes = [ctypes.c_void_p]
        h.hipIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        h.hipIpcOpenMemHandle.argtypes = [ctypes.c_void_p, _IpcHandle, ctypes.c_uint]
        h.hipGetErrorString.restype = ctypes.c_char_p
        _hip = h
    return _hip


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(
            f"finalize_allreduce: {what} failed: {rc} ({_hip_lib().hipGetErrorString(rc).decode()})"
        )


def _build(arch: str, sqrt: str) -> str:
    out_dir = tempfile.mkdtemp(prefix="sglang_muse_finalize_ar_")
    co = os.path.join(out_dir, "finalize_allreduce.co")
    hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
    # No FMA contraction: the FlyDSL finalize rounds every multiply and add separately.
    cmd = [hipcc, "--genco", f"--offload-arch={arch}", "-O3", "-ffp-contract=off",
           f"-DMUSE_SQRT={sqrt}", "-o", co, _SRC]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return co


class _Kernel:
    def __init__(self, module, name: str):
        self.fn = ctypes.c_void_p()
        _check(
            _hip_lib().hipModuleGetFunction(
                ctypes.byref(self.fn), module, name.encode()
            ),
            name,
        )

    def launch(self, grid: int, args, stream: int) -> None:
        size = ctypes.c_size_t(ctypes.sizeof(args))
        extra = (ctypes.c_void_p * 5)(
            1, ctypes.addressof(args), 2, ctypes.addressof(size), 3
        )  # HIP_LAUNCH_PARAM_BUFFER_POINTER / _SIZE / _END
        _check(
            _hip_lib().hipModuleLaunchKernel(
                self.fn,
                grid,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                None,
                ctypes.cast(extra, ctypes.c_void_p),
            ),
            "hipModuleLaunchKernel",
        )


class MuseFinalizeAllreduce:
    """Per-process TP4 finalize + all-reduce; every rank must issue the same sequence of calls."""

    def __init__(
        self,
        cpu_group,
        device: torch.device,
        max_rows: int = 512,
        sqrt: str = "__builtin_amdgcn_sqrtf",
    ):
        self.rank = dist.get_rank(group=cpu_group)
        if dist.get_world_size(group=cpu_group) != _NR:
            raise ValueError("finalize_allreduce supports TP4 only")
        self.device = device
        self.max_rows = int(max_rows)
        h = _hip_lib()
        with torch.cuda.device(device):
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            co = _build(arch, sqrt)
            module = ctypes.c_void_p()
            try:
                _check(
                    h.hipModuleLoad(ctypes.byref(module), co.encode()), "hipModuleLoad"
                )
            finally:
                shutil.rmtree(os.path.dirname(co), ignore_errors=True)
            self._module = module
            self._k = _Kernel(module, "muse_finalize_allreduce_tp4")
            fill = _Kernel(module, "muse_finalize_allreduce_fill")
            nbytes = 3 * (_NR + 1) * self.max_rows * _ROW_BYTES
            ptr = ctypes.c_void_p()
            _check(h.hipMalloc(ctypes.byref(ptr), nbytes), "hipMalloc")
            handle = _IpcHandle()
            _check(
                h.hipIpcGetMemHandle(ctypes.byref(handle), ptr), "hipIpcGetMemHandle"
            )
            handles = [None] * _NR
            dist.all_gather_object(
                handles, ctypes.string_at(ctypes.addressof(handle), 64), group=cpu_group
            )
            self._peers = (ctypes.c_void_p * _NR)()
            for i, raw in enumerate(handles):
                if i == self.rank:
                    self._peers[i] = ptr.value
                    continue
                peer, ph = ctypes.c_void_p(), _IpcHandle()
                ctypes.memmove(ctypes.addressof(ph), raw, 64)
                _check(
                    h.hipIpcOpenMemHandle(
                        ctypes.byref(peer), ph, _IPC_LAZY_ENABLE_PEER_ACCESS
                    ),
                    "hipIpcOpenMemHandle",
                )
                self._peers[i] = peer.value
            self.row_state = torch.tensor(
                [0, -1], dtype=torch.int32, device=device
            ).repeat(self.max_rows)
            fill.launch(
                1024,
                _FillArgs(ptr.value, nbytes // 16),
                torch.cuda.current_stream(device).cuda_stream,
            )
            torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def applies(self, rows: int) -> bool:
        return 0 < rows <= self.max_rows and rows % _NR == 0

    def __call__(
        self,
        y,
        gamma,
        scores,
        ids,
        local_start: int,
        num_local: int,
        eps: float,
        gain: float,
    ):
        """bf16 [M, 4096]: the all-reduced finalize of per-slot rows ``y`` [M * 8, 4096] (FlyDSL finalize numerics)."""
        m = ids.shape[0]
        assert (
            self.applies(m)
            and ids.shape == (m, _TOPK)
            and y.shape == (m * _TOPK, _HIDDEN)
        )
        assert (
            y.dtype == gamma.dtype == torch.bfloat16
            and scores.dtype == torch.float32
            and ids.dtype == torch.int32
        )
        assert (
            y.is_contiguous()
            and scores.is_contiguous()
            and ids.is_contiguous()
            and gamma.is_contiguous()
        )
        out = torch.empty((m, _HIDDEN), dtype=torch.bfloat16, device=y.device)
        args = _Args(
            y.data_ptr(), gamma.data_ptr(), scores.data_ptr(), ids.data_ptr(), out.data_ptr(),
            self._peers, self.row_state.data_ptr(), int(local_start), int(num_local), float(eps), float(gain),
            self.rank, m, self.max_rows, 0,
        )  # fmt: skip
        self._k.launch(m, args, torch.cuda.current_stream(y.device).cuda_stream)
        return out


_instance: MuseFinalizeAllreduce | None = None


def ensure_finalize_allreduce(cpu_group, device: torch.device) -> MuseFinalizeAllreduce:
    """The process-wide instance, built on first use (collective; call outside graph capture)."""
    global _instance
    if _instance is None:
        _instance = MuseFinalizeAllreduce(cpu_group, device)
        logger.info(
            "Muse fused finalize + TP4 all-reduce enabled (rank %d)", _instance.rank
        )
    return _instance


def run_compact_a4_allreduce(
    op: MuseFinalizeAllreduce,
    x,
    scores,
    ids,
    experts,
    local_start,
    norm_weight,
    eps,
    gain,
    quantized=None,
):
    """Compact A4 experts of this rank, then the fused finalize + TP all-reduce: the reduced MoE output."""
    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import compact_a4_experts
    from sglang.srt.rocm_muse_spark.mxfp4_moe import _compact_a4_operands

    y = compact_a4_experts(
        x, ids, *_compact_a4_operands(experts), local_start=local_start,
        shuffled=experts.native_dropped, quantized=quantized,
    )  # fmt: skip
    return op(
        y, norm_weight, scores, ids, local_start, experts.num_local_experts, eps, gain
    )

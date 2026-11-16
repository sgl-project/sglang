"""gfx1250 bf16 push all-reduce (push_ar.hip): one-shot below ONESHOT_BELOW bytes, two-shot above; graph-capturable."""

import ctypes
import os
import shutil
import subprocess
import tempfile

import torch
import torch.distributed as dist

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "push_ar.hip")
_MAX_NR = 8


class _H(ctypes.Structure):
    _fields_ = [("r", ctypes.c_char * 64)]


class _Args(ctypes.Structure):
    _fields_ = [
        ("inp", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("inbox", ctypes.c_void_p * _MAX_NR),
        ("state", ctypes.c_void_p),
        ("n16", ctypes.c_uint32),
        ("max16", ctypes.c_uint32),
        ("rank", ctypes.c_int32),
        ("oneshot", ctypes.c_uint32),
    ]


class _Fill(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p), ("n16", ctypes.c_uint64)]


assert ctypes.sizeof(_Args) == 104
_hip = None


def _lib():
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
        h.hipExtMallocWithFlags.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        h.hipIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        h.hipIpcOpenMemHandle.argtypes = [ctypes.c_void_p, _H, ctypes.c_uint]
        h.hipGetErrorString.restype = ctypes.c_char_p
        _hip = h
    return _hip


def _ck(rc, what):
    if rc:
        raise RuntimeError(
            f"push_ar {what}: {rc} {_lib().hipGetErrorString(rc).decode()}"
        )


class _K:
    def __init__(self, mod, name):
        self.fn = ctypes.c_void_p()
        _ck(
            _lib().hipModuleGetFunction(ctypes.byref(self.fn), mod, name.encode()), name
        )

    def launch(self, grid, args, stream):
        sz = ctypes.c_size_t(ctypes.sizeof(args))
        extra = (ctypes.c_void_p * 5)(
            1, ctypes.addressof(args), 2, ctypes.addressof(sz), 3
        )
        _ck(
            _lib().hipModuleLaunchKernel(
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
            "launch",
        )


class PushAllreduce:
    def __init__(
        self, cpu_group, device, max_bytes=8 << 20, grid=None, oneshot_below=None
    ):
        self.rank = dist.get_rank(cpu_group)
        self.nr = dist.get_world_size(cpu_group)
        self.grid = int(
            grid
            or os.environ.get("PUSH_AR_GRID", 0)
            or 2 * torch.cuda.get_device_properties(device).multi_processor_count
        )
        self.oneshot_below = int(
            oneshot_below
            if oneshot_below is not None
            else os.environ.get("PUSH_AR_ONESHOT_BELOW", 512 << 10)
        )
        self.max16 = max_bytes // 16
        h = _lib()
        with torch.cuda.device(device):
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            d = tempfile.mkdtemp(prefix="push_ar_")
            co = os.path.join(d, "k.co")
            hipcc = os.path.join(
                os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc"
            )
            subprocess.run(
                [
                    hipcc,
                    "--genco",
                    f"--offload-arch={arch}",
                    "-O3",
                    f"-DNR={self.nr}",
                    "-o",
                    co,
                    _SRC,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            mod = ctypes.c_void_p()
            _ck(h.hipModuleLoad(ctypes.byref(mod), co.encode()), "load")
            shutil.rmtree(d, True)
            self._mod = mod
            self._k = _K(mod, "push_allreduce")
            fill = _K(mod, "push_allreduce_fill")
            nbytes = (3 * self.nr + 3) * max_bytes
            ptr = ctypes.c_void_p()
            _ck(h.hipMalloc(ctypes.byref(ptr), nbytes), "malloc")
            hd = _H()
            _ck(h.hipIpcGetMemHandle(ctypes.byref(hd), ptr), "ipcget")
            hs = [None] * self.nr
            dist.all_gather_object(
                hs, ctypes.string_at(ctypes.addressof(hd), 64), group=cpu_group
            )
            self._peers = (ctypes.c_void_p * _MAX_NR)()
            for i, raw in enumerate(hs):
                if i == self.rank:
                    self._peers[i] = ptr.value
                    continue
                p, ph = ctypes.c_void_p(), _H()
                ctypes.memmove(ctypes.addressof(ph), raw, 64)
                _ck(h.hipIpcOpenMemHandle(ctypes.byref(p), ph, 1), "ipcopen")
                self._peers[i] = p.value
            self.state = torch.zeros(self.grid * 4, dtype=torch.int32, device=device)
            fill.launch(
                1024,
                _Fill(ptr.value, nbytes // 16),
                torch.cuda.current_stream(device).cuda_stream,
            )
            torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def applies(self, x):
        nb = x.numel() * x.element_size()
        return (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and nb <= self.max16 * 16
            and nb % (16 * self.nr) == 0
        )

    def __call__(self, x, out=None):
        out = torch.empty_like(x) if out is None else out
        nb = x.numel() * 2
        a = _Args(
            x.data_ptr(),
            out.data_ptr(),
            self._peers,
            self.state.data_ptr(),
            nb // 16,
            self.max16,
            self.rank,
            int(nb < self.oneshot_below),
        )
        self._k.launch(self.grid, a, torch.cuda.current_stream(x.device).cuda_stream)
        return out

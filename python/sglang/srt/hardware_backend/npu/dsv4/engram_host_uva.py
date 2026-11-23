# SPDX-License-Identifier: Apache-2.0
# Host registration / chunked addresses adapted from vllm-ascend Engram HOST_UVA.
# INT8 + E8M0 is intentionally distinct from its INT8 + FP32 scale path.
"""Experimental NPU host-resident INT8/E8M0 Engram rows; no CUDA dependency."""

import ctypes
import logging
import mmap
import weakref

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _lookup_int8_e8m0(
    weight_bases,
    scale_bases,
    ids,
    output,
    row_start,
    row_end,
    count,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    if row < count:
        index = tl.load(ids + row).to(tl.int64)
        owned = (index >= row_start) & (index < row_end)
        local = tl.where(owned, index - row_start, 0)
        chunk, offset = local // CHUNK, local % CHUNK
        w = tl.load(weight_bases + chunk).to(tl.pointer_type(tl.int8))
        s = tl.load(scale_bases + chunk).to(tl.pointer_type(tl.uint8))
        col = tl.arange(0, WIDTH)
        value = tl.load(w + offset * WIDTH + col).to(tl.float32)
        groups = tl.arange(0, WIDTH // GROUP)
        exponent = tl.load(s + offset * (WIDTH // GROUP) + groups).to(tl.int32) & 255
        # E8M0 byte 0 denotes 2^-127, byte 255 denotes NaN.
        bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
        bits = tl.where(exponent == 255, 0x7FC00000, bits)
        scale = bits.to(tl.float32, bitcast=True)
        value = tl.reshape(value, (WIDTH // GROUP, GROUP)) * scale[:, None]
        value = tl.reshape(value, (WIDTH,)).to(tl.bfloat16)
        tl.store(output + row * WIDTH + col, tl.where(owned, value, 0))


class _RegisteredRows:
    """Own one CPU allocation and its mapped device addresses for a worker lifetime."""

    def __init__(self, rows, width, dtype, device, chunk_rows):
        self.lib = ctypes.CDLL("libascendcl.so")
        self.lib.aclrtHostRegisterV2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self.lib.aclrtHostRegisterV2.restype = ctypes.c_int
        self.lib.aclrtHostGetDevicePointer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
        ]
        self.lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
        self.lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.lib.aclrtHostUnregister.restype = ctypes.c_int
        self.registered = False
        self.tensor = None
        self.bases = None
        self.mm = mmap.mmap(
            -1, rows * width, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        self.mm.madvise(mmap.MADV_HUGEPAGE)
        self.pointer = ctypes.c_void_p(
            ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        )
        try:
            # Populate our own anonymous pages before the driver pins them. This
            # avoids faulting an entire multi-GiB shard inside registration and
            # does not drop the host's page cache or alter global NUMA policy.
            logger.info("Engram CPU prefault begin bytes=%s", rows * width)
            pages = np.frombuffer(self.mm, dtype=np.uint8)
            pages[:: mmap.PAGESIZE] = 0
            del pages
            # Standard A2/A3 CANN ABI: pinned + mapped, not the A5-specific flags.
            rc = self.lib.aclrtHostRegisterV2(self.pointer, rows * width, 0x10000002)
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 rc={rc}, bytes={rows * width}")
            self.registered = True
            # Parameter views retain the mapping even if the module releases
            # its owner. Unregister before mmap deallocation on that path.
            self._unregister = weakref.finalize(
                self.mm, self.lib.aclrtHostUnregister, self.pointer
            )
            self._unregister.atexit = False
            ptr = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(ptr), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer rc={rc}")
            self.tensor = torch.frombuffer(self.mm, dtype=dtype).reshape(rows, width)
            self.bases = torch.tensor(
                [ptr.value + i * width for i in range(0, rows, chunk_rows)],
                dtype=torch.int64,
                device=device,
            )
        except BaseException:
            self.close(synchronize=False)
            raise

    def close(self, *, synchronize=True):
        if self.registered:
            if synchronize:
                torch.npu.synchronize()
            rc = self.lib.aclrtHostUnregister(self.pointer)
            if rc:
                # Keep backing alive if the driver has not released registration.
                raise RuntimeError(f"aclrtHostUnregister rc={rc}")
            self._unregister.detach()
            self.registered = False
        self.tensor = None
        self.bases = None
        if self.mm is not None:
            self.mm.close()
            self.mm = None


class NpuHostInt8Engram:
    """Per-TP-rank CPU shards; caller retains existing TP/DP lookup collectives.

    Weight/scale tensors are CPU views, not re-quantized copies. Keep this owner
    alive as long as its parameters or any captured graph can use the pointers.
    Explicit close requires all graph/parameter views to have been released.
    """

    def __init__(self, rows, width, device, *, group=32, chunk_rows=None):
        if (
            group != 32
            or rows < 1
            or width < group
            or width % group
            or width & (width - 1)
        ):
            raise ValueError(
                "Require positive rows and power-of-two, group-aligned width"
            )
        self.rows, self.width, self.group = rows, width, group
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("NpuHostInt8Engram requires an NPU device")
        self.chunk_rows = chunk_rows or (64 * 1024 * 1024 // width)
        if self.chunk_rows < 1 or self.chunk_rows * width > 2**30:
            raise ValueError("Pointer chunks must keep row offsets below 1 GiB")
        self.weight_buffer = _RegisteredRows(
            rows, width, torch.int8, self.device, self.chunk_rows
        )
        try:
            self.scale_buffer = _RegisteredRows(
                rows, width // group, torch.uint8, self.device, self.chunk_rows
            )
        except BaseException:
            self.weight_buffer.close()
            raise

    @property
    def weight(self):
        return self.weight_buffer.tensor

    @property
    def scale(self):
        return self.scale_buffer.tensor

    def lookup(self, ids, row_start):
        if ids.device != self.device or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lookup requires int32/int64 IDs on the table's NPU")
        flat = ids.contiguous().view(-1)
        out = torch.empty(
            (*ids.shape, self.width), dtype=torch.bfloat16, device=ids.device
        )
        if flat.numel():
            _lookup_int8_e8m0[(flat.numel(),)](
                self.weight_buffer.bases,
                self.scale_buffer.bases,
                flat,
                out,
                row_start,
                row_start + self.rows,
                flat.numel(),
                WIDTH=self.width,
                GROUP=self.group,
                CHUNK=self.chunk_rows,
                num_warps=4,
            )
        return out

    def close(self):
        self.scale_buffer.close()
        self.weight_buffer.close()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sglang's pynccl_wrapper.py for Intel oneCCL.

# This file is a pure Python (ctypes) wrapper for Intel's oneCCL library,
# binding its NCCL-compatible C API (the ``oneccl*`` symbols exported by
# ``libccl.so``). It is the XPU counterpart of ``pynccl_wrapper.py`` and is used
# by ``PyXcclCommunicator`` (``pyxccl.py``) to call oneCCL directly instead of
# routing collectives through ``torch.distributed``'s XCCL backend.
#
# A pure Python wrapper (rather than a compiled C/C++ binding) keeps oneCCL
# version-agnostic: the library is selected at runtime via the
# ``SGLANG_PYXCCL_SO_PATH`` environment variable or the dynamic-linker default,
# with no recompilation needed to switch oneCCL builds.
#
# For the oneCCL C API definitions, see
# ``<oneapi>/include/oneapi/ccl.h`` and ``ccl/v2/types.h``.

import ctypes
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


def find_ccl_library() -> str:
    """
    Locate the oneCCL shared library. Uses ``SGLANG_PYXCCL_SO_PATH`` when set,
    otherwise falls back to the dynamic-linker default ``libccl.so.1``.
    """
    so_file = os.environ.get("SGLANG_PYXCCL_SO_PATH", None)
    if so_file:
        logger.info(
            "Found oneCCL from environment variable SGLANG_PYXCCL_SO_PATH=%s", so_file
        )
    else:
        so_file = "libccl.so.1"
        logger.debug("Using default oneCCL library %s", so_file)
    return so_file


# === export types and functions from oneCCL to Python ===
# for the original oneCCL definitions, see include/oneapi/ccl.h and
# include/oneapi/ccl/v2/types.h.

onecclResult_t = ctypes.c_int
onecclComm_t = ctypes.c_void_p

# oneCCL's unique id is 4096 bytes (ONECCL_UNIQUE_ID_BYTES), unlike NCCL's 128.
ONECCL_UNIQUE_ID_BYTES = 4096


class onecclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * ONECCL_UNIQUE_ID_BYTES)]


# oneCCL takes a ``void *stream`` that is a pointer to a SYCL queue on XPU.
xpuStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

onecclDataType_t = ctypes.c_int


class onecclDataTypeEnum:
    # Values from onecclDataType_t in ccl/v2/types.h.
    onecclInt8 = 0
    onecclChar = 0
    onecclUint8 = 1
    onecclInt32 = 2
    onecclInt = 2
    onecclUint32 = 3
    onecclInt64 = 4
    onecclUint64 = 5
    onecclFloat16 = 6
    onecclHalf = 6
    onecclFloat32 = 7
    onecclFloat = 7
    onecclFloat64 = 8
    onecclDouble = 8
    onecclBfloat16 = 9

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.onecclInt8
        if dtype == torch.uint8:
            return cls.onecclUint8
        if dtype == torch.int32:
            return cls.onecclInt32
        if dtype == torch.int64:
            return cls.onecclInt64
        if dtype == torch.float16:
            return cls.onecclFloat16
        if dtype == torch.float32:
            return cls.onecclFloat32
        if dtype == torch.float64:
            return cls.onecclFloat64
        if dtype == torch.bfloat16:
            return cls.onecclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


onecclRedOp_t = ctypes.c_int


class onecclRedOpTypeEnum:
    # Values from onecclRedOp_t in ccl/v2/types.h.
    onecclSum = 0
    onecclProd = 1
    onecclMax = 2
    onecclMin = 3
    onecclAvg = 4

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.onecclSum
        if op == ReduceOp.PRODUCT:
            return cls.onecclProd
        if op == ReduceOp.MAX:
            return cls.onecclMax
        if op == ReduceOp.MIN:
            return cls.onecclMin
        if op == ReduceOp.AVG:
            return cls.onecclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class ONECCLLibrary:
    exported_functions = [
        # const char* onecclGetErrorString(onecclResult_t result);
        Function("onecclGetErrorString", ctypes.c_char_p, [onecclResult_t]),
        # onecclResult_t onecclGetVersion(int *version);
        Function("onecclGetVersion", onecclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # onecclResult_t onecclGetUniqueId(onecclUniqueId *uniqueId);
        Function("onecclGetUniqueId", onecclResult_t, [ctypes.POINTER(onecclUniqueId)]),
        # onecclResult_t onecclSetDevice(uint32_t index);
        Function("onecclSetDevice", onecclResult_t, [ctypes.c_uint32]),
        # onecclResult_t onecclCommInitRank(
        #   onecclComm_t *comm, size_t nranks, onecclUniqueId commId, int rank);
        # note that onecclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer, and nranks is size_t (not int).
        Function(
            "onecclCommInitRank",
            onecclResult_t,
            [
                ctypes.POINTER(onecclComm_t),
                ctypes.c_size_t,
                onecclUniqueId,
                ctypes.c_int,
            ],
        ),
        # onecclResult_t onecclAllReduce(
        #   void *sendbuff, void *recvbuff, size_t count,
        #   onecclDataType_t datatype, onecclRedOp_t reduction_op,
        #   onecclComm_t comm, void *stream);
        Function(
            "onecclAllReduce",
            onecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                onecclDataType_t,
                onecclRedOp_t,
                onecclComm_t,
                xpuStream_t,
            ],
        ),
        # onecclResult_t onecclAllGather(
        #   const void *sendbuff, void *recvbuff, size_t sendcount,
        #   onecclDataType_t datatype, onecclComm_t comm, void *stream);
        Function(
            "onecclAllGather",
            onecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                onecclDataType_t,
                onecclComm_t,
                xpuStream_t,
            ],
        ),
        # onecclResult_t onecclReduceScatter(
        #   const void *sendbuff, void *recvbuff, size_t recvcount,
        #   onecclDataType_t datatype, onecclRedOp_t redop,
        #   onecclComm_t comm, void *stream);
        Function(
            "onecclReduceScatter",
            onecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                onecclDataType_t,
                onecclRedOp_t,
                onecclComm_t,
                xpuStream_t,
            ],
        ),
        # onecclResult_t onecclBroadcast(
        #   const void *sendbuff, void *recvbuff, size_t count,
        #   onecclDataType_t datatype, int root, onecclComm_t comm,
        #   void *stream);
        Function(
            "onecclBroadcast",
            onecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                onecclDataType_t,
                ctypes.c_int,
                onecclComm_t,
                xpuStream_t,
            ],
        ),
        # onecclResult_t onecclSend(
        #   const void *sendbuff, size_t count, onecclDataType_t datatype,
        #   int peer, onecclComm_t comm, void *stream);
        Function(
            "onecclSend",
            onecclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                onecclDataType_t,
                ctypes.c_int,
                onecclComm_t,
                xpuStream_t,
            ],
        ),
        # onecclResult_t onecclRecv(
        #   void *recvbuff, size_t count, onecclDataType_t datatype,
        #   int peer, onecclComm_t comm, void *stream);
        Function(
            "onecclRecv",
            onecclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                onecclDataType_t,
                ctypes.c_int,
                onecclComm_t,
                xpuStream_t,
            ],
        ),
        # be cautious! onecclCommDestroy is a collective call; because Python
        # object destruction can happen in random order, it is better not to
        # rely on __del__ to call it.
        # onecclResult_t onecclCommDestroy(onecclComm_t comm);
        Function("onecclCommDestroy", onecclResult_t, [onecclComm_t]),
        # onecclResult_t onecclGroupStart();
        Function("onecclGroupStart", onecclResult_t, []),
        # onecclResult_t onecclGroupEnd();
        Function("onecclGroupEnd", onecclResult_t, []),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or find_ccl_library()

        try:
            if so_file not in ONECCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                ONECCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = ONECCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load oneCCL library from %s . "
                "It is expected if you are not running on Intel XPUs. "
                "Otherwise, the oneCCL library might not exist, be corrupted, "
                "or it does not support the current platform %s. "
                "If you already have the library, please set the "
                "environment variable SGLANG_PYXCCL_SO_PATH"
                " to point to the correct oneCCL library path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in ONECCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in ONECCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            ONECCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = ONECCLLibrary.path_to_dict_mapping[so_file]

    def onecclGetErrorString(self, result: onecclResult_t) -> str:
        return self._funcs["onecclGetErrorString"](result).decode("utf-8")

    def ONECCL_CHECK(self, result: onecclResult_t) -> None:
        if result != 0:
            error_str = self.onecclGetErrorString(result)
            raise RuntimeError(f"oneCCL error: {error_str}")

    def onecclGetRawVersion(self) -> int:
        version = ctypes.c_int()
        self.ONECCL_CHECK(self._funcs["onecclGetVersion"](ctypes.byref(version)))
        # e.g. 20211702 for oneCCL 2021.17.2
        return version.value

    def onecclGetVersion(self) -> str:
        # ONECCL_VERSION_CODE = major*10000 + minor*100 + suffix
        code = self.onecclGetRawVersion()
        major = code // 10000
        minor = (code % 10000) // 100
        patch = code % 100
        return f"{major}.{minor}.{patch}"

    def onecclGetUniqueId(self) -> onecclUniqueId:
        unique_id = onecclUniqueId()
        self.ONECCL_CHECK(self._funcs["onecclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def onecclSetDevice(self, index: int) -> None:
        self.ONECCL_CHECK(self._funcs["onecclSetDevice"](index))

    def onecclCommInitRank(
        self, world_size: int, unique_id: onecclUniqueId, rank: int
    ) -> onecclComm_t:
        comm = onecclComm_t()
        self.ONECCL_CHECK(
            self._funcs["onecclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def onecclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: onecclComm_t,
        stream: xpuStream_t,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def onecclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: onecclComm_t,
        stream: xpuStream_t,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def onecclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: onecclComm_t,
        stream: xpuStream_t,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def onecclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: onecclComm_t,
        stream: xpuStream_t,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def onecclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: onecclComm_t,
        stream: xpuStream_t,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def onecclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: onecclComm_t,
        stream: xpuStream_t,
    ) -> None:
        self.ONECCL_CHECK(
            self._funcs["onecclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def onecclCommDestroy(self, comm: onecclComm_t) -> None:
        self.ONECCL_CHECK(self._funcs["onecclCommDestroy"](comm))

    def onecclGroupStart(self) -> None:
        self.ONECCL_CHECK(self._funcs["onecclGroupStart"]())

    def onecclGroupEnd(self) -> None:
        self.ONECCL_CHECK(self._funcs["onecclGroupEnd"]())


__all__ = [
    "ONECCLLibrary",
    "onecclDataTypeEnum",
    "onecclRedOpTypeEnum",
    "onecclUniqueId",
    "onecclComm_t",
    "xpuStream_t",
    "buffer_type",
]

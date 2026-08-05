# Adapted from https://github.com/vllm-project/vllm-ascend/blob/v0.11.0-dev/vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py

import ctypes
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

# export types and functions from hccl to Python ===
# for the original hccl definition, please check
# https://github.com/EternalLied/cann-hccl-new/blob/64ec6ce2923319caa5df8c3c531e06bdc148ce9c/inc/hccl/hccl.h#L90
# https://github.com/EternalLied/cann-hccl-new/blob/64ec6ce2923319caa5df8c3c531e06bdc148ce9c/inc/hccl/hccl_types.h#L48

hcclResult_t = ctypes.c_int
hcclComm_t = ctypes.c_void_p


logger = logging.getLogger(__name__)


class hcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 4108)]


aclrtStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

hcclDataType_t = ctypes.c_int

_CURRENT_STREAM = None


def find_hccl_library() -> str:
    """
    We either use the library file specified by the `HCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libhccl.so` can be
    found by `ctypes` automatically.
    """
    so_file = os.environ.get("HCCL_SO_PATH", None)

    # manually load the hccl library
    if so_file:
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s", so_file)
    else:
        if torch.version.cann is not None:
            so_file = "libhccl.so"
        else:
            raise ValueError("HCCL only supports Ascend NPU backends.")
        logger.info("Found hccl from library %s", so_file)
    return so_file


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `pyhccl_wrapper.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _CURRENT_STREAM
    if _CURRENT_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _CURRENT_STREAM = torch.npu.current_stream()
    return _CURRENT_STREAM


class HcclCommConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic_word", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
        ("hccl_buffer_size", ctypes.c_uint32),
        ("hccl_deterministic", ctypes.c_uint32),
        ("hccl_comm_name", ctypes.c_char * 128),
        ("hccl_udi", ctypes.c_char * 128),
        ("hccl_op_expansion_mode", ctypes.c_uint32),
        ("hccl_rdma_traffic_class", ctypes.c_uint32),
        ("hccl_rdma_service_level", ctypes.c_uint32),
        ("hcll_world_rank_id", ctypes.c_uint32),
        ("hccl_job_id", ctypes.c_uint64),
        ("comm_engine", ctypes.c_int32),
        ("thread_num", ctypes.c_uint32),
        ("notify_num_per_thread", ctypes.c_uint32),
        ("acl_graph_zero_copy_enable", ctypes.c_uint8),
    ]


class hcclDataTypeEnum:
    hcclInt8 = 0
    hcclInt16 = 1
    hcclInt32 = 2
    hcclFloat16 = 3
    hcclFloat32 = 4
    hcclInt64 = 5
    hcclUint64 = 6
    hcclUint8 = 7
    hcclUint16 = 8
    hcclUint32 = 9
    hcclFloat64 = 10
    hcclBfloat16 = 11
    hcclInt128 = 12

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.hcclInt8
        if dtype == torch.uint8:
            return cls.hcclUint8
        if dtype == torch.int32:
            return cls.hcclInt32
        if dtype == torch.int64:
            return cls.hcclInt64
        if dtype == torch.float16:
            return cls.hcclFloat16
        if dtype == torch.float32:
            return cls.hcclFloat32
        if dtype == torch.float64:
            return cls.hcclFloat64
        if dtype == torch.bfloat16:
            return cls.hcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


hcclRedOp_t = ctypes.c_int


class hcclRedOpTypeEnum:
    hcclSum = 0
    hcclProd = 1
    hcclMax = 2
    hcclMin = 3

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.hcclSum
        if op == ReduceOp.PRODUCT:
            return cls.hcclProd
        if op == ReduceOp.MAX:
            return cls.hcclMax
        if op == ReduceOp.MIN:
            return cls.hcclMin
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class HCCLLibrary:
    exported_functions = [
        # const char* HcclGetErrorString(HcclResult code);
        Function("HcclGetErrorString", ctypes.c_char_p, [hcclResult_t]),
        # HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
        Function("HcclGetRootInfo", hcclResult_t, [ctypes.POINTER(hcclUniqueId)]),
        # HcclResult HcclCommInitRootInfo(
        #   uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);
        # note that HcclComm is a pointer type, so the last argument is a pointer to a pointer
        Function(
            "HcclCommInitRootInfo",
            hcclResult_t,
            [
                ctypes.c_int,
                ctypes.POINTER(hcclUniqueId),
                ctypes.c_int,
                ctypes.POINTER(hcclComm_t),
            ],
        ),
        # HcclResult HcclAllReduce(
        #   void *sendBuf, void *recvBuf, uint64_t count,
        #   HcclDataType dataType, HcclReduceOp op, HcclComm comm,
        #   aclrtStream stream);
        Function(
            "HcclAllReduce",
            hcclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclRedOp_t,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclBroadcast(
        #   void *buf, uint64_t count,
        #   HcclDataType dataType, uint32_t root,
        #   HcclComm comm, aclrtStream stream);
        Function(
            "HcclBroadcast",
            hcclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclAllGather(
        #   void *sendBuf, void *recvBuf,
        #   uint64_t sendCount, HcclDataType dataType,
        #   HcclComm comm, aclrtStream stream);
        Function(
            "HcclAllGather",
            hcclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclReduceScatter(
        #   void *sendBuf, void *recvBuf,
        #   uint64_t recvCount, HcclDataType dataType,
        #   HcclReduceOp op, HcclComm comm,
        #   aclrtStream stream);
        Function(
            "HcclReduceScatter",
            hcclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclRedOp_t,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclBarrier(
        #   HcclComm comm, aclrtStream stream);
        Function(
            "HcclBarrier",
            hcclResult_t,
            [
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclCreateSubCommConfig(
        #   HcclComm *comm, uin32_t rankNum, uint32_t *rankIds, uint64_t subCommId,
        #   uint32_t subCommRankId, HcclCommConfig *config, HcclComm *subComm
        # )
        Function(
            "HcclCreateSubCommConfig",
            hcclResult_t,
            [
                ctypes.POINTER(hcclComm_t),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.POINTER(HcclCommConfig),
                ctypes.POINTER(hcclComm_t),
            ],
        ),
        # HcclResult HcclCommDestroy(HcclComm comm);
        Function("HcclCommDestroy", hcclResult_t, [hcclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    # to the correspongding directory
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or find_hccl_library()

        try:
            if so_file not in HCCLLibrary.path_to_dict_mapping:

                lib = ctypes.CDLL(so_file)
                HCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = HCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load HCCL library from %s. "
                "It is expected if you are not running on Ascend NPUs."
                "Otherwise, the hccl library might not exist, be corrupted "
                "or it does not support the current platform %s. "
                "If you already have the library, please set the "
                "environment variable HCCL_SO_PATH"
                " to point to the correct hccl library path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in HCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in HCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            HCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = HCCLLibrary.path_to_dict_mapping[so_file]

    def hcclGetErrorString(self, result: hcclResult_t) -> str:
        return self._funcs["HcclGetErrorString"](result).decode("utf-8")

    def HCCL_CHECK(self, result: hcclResult_t) -> None:
        if result != 0:
            error_str = self.hcclGetErrorString(result)
            raise RuntimeError(f"HCCL error: {error_str}")

    def hcclGetUniqueId(self) -> hcclUniqueId:
        unique_id = hcclUniqueId()
        self.HCCL_CHECK(self._funcs["HcclGetRootInfo"](ctypes.byref(unique_id)))
        return unique_id

    def hcclCommInitRank(
        self, world_size: int, unique_id: hcclUniqueId, rank: int
    ) -> hcclComm_t:
        comm = hcclComm_t()
        self.HCCL_CHECK(
            self._funcs["HcclCommInitRootInfo"](
                world_size, ctypes.byref(unique_id), rank, ctypes.byref(comm)
            )
        )
        return comm

    def hcclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: hcclComm_t,
        stream: aclrtStream_t,
    ) -> None:
        # `datatype` actually should be `hcclDataType_t`
        # and `op` should be `hcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.HCCL_CHECK(
            self._funcs["HcclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def hcclBroadcast(
        self,
        buf: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: hcclComm_t,
        stream: aclrtStream_t,
    ) -> None:
        self.HCCL_CHECK(
            self._funcs["HcclBroadcast"](buf, count, datatype, root, comm, stream)
        )

    def hcclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: hcclComm_t,
        stream: aclrtStream_t,
    ) -> None:
        self.HCCL_CHECK(
            self._funcs["HcclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def hcclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: hcclComm_t,
        stream: aclrtStream_t,
    ) -> None:
        self.HCCL_CHECK(
            self._funcs["HcclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def hcclBarrier(self, comm: hcclComm_t, stream: aclrtStream_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclBarrier"](comm, stream))

    def hcclCreateSubcommConfig(
        self, comm, ranks_size, c_rank_ids, subcomm_id, subcomm_rank, comm_config
    ):
        subcomm = hcclComm_t()
        self.HCCL_CHECK(
            self._funcs["HcclCreateSubCommConfig"](
                ctypes.byref(comm),
                ranks_size,
                c_rank_ids,
                subcomm_id,
                subcomm_rank,
                ctypes.byref(comm_config),
                ctypes.byref(subcomm),
            )
        )
        return subcomm

    def hcclCommDestroy(self, comm: hcclComm_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclCommDestroy"](comm))


__all__ = [
    "HCCLLibrary",
    "hcclDataTypeEnum",
    "hcclRedOpTypeEnum",
    "hcclUniqueId",
    "hcclComm_t",
    "aclrtStream_t",
    "buffer_type",
]

import logging
from typing import Union

import msgspec
from zmq import Socket
from zmq.asyncio import Socket as AsyncSocket

from sglang.srt.environ import envs

from .pickle_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BackupDramReq,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    BlockReqType,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    EmbeddingReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    Function,
    GenerateReqInput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    LazyDumpTensorsReqInput,
    LazyDumpTensorsReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    MultimodalDataInputFormat,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ParseFunctionCallReq,
    PauseGenerationReqInput,
    ProfileReq,
    ProfileReqInput,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SeparateReasoningReqInput,
    SessionParams,
    SetInjectDumpMetadataReqInput,
    SetInjectDumpMetadataReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    SpeculativeDecodingMetricsMixin,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    Tool,
    UnloadLoRAAdapterReqInput,
    UpdateExpertBackupReq,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
    WatchLoadUpdateReq,
)

if envs.SGLANG_IPC_USE_MSGPACK.get():
    from .msgpack_struct import (
        BatchEmbeddingOutput,
        BatchStrOutput,
        BatchTokenIDOutput,
        DisaggregationMetrics,
        FlushCacheReqInput,
        FlushCacheReqOutput,
        FreezeGCReq,
        GetLoadsReqOutput,
        LoRAMetrics,
        MemoryMetrics,
        QueueMetrics,
        SpeculativeMetrics,
    )
else:
    from .pickle_struct import (
        BatchEmbeddingOutput,
        BatchStrOutput,
        BatchTokenIDOutput,
        DisaggregationMetrics,
        FlushCacheReqInput,
        FlushCacheReqOutput,
        FreezeGCReq,
        GetLoadsReqOutput,
        LoRAMetrics,
        MemoryMetrics,
        QueueMetrics,
        SpeculativeMetrics,
    )


logger = logging.getLogger(__name__)

LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput

PICKLE_MAGIC_NUMBER = b"0xSG01"
MSGPACK_MAGIC_NUMBER = b"0xSG02"


def sock_send(
    socket: Socket, obj: Union[BaseReq, BaseBatchReq, msgspec.Struct], flags=0
):
    # if the msgpack magic number is not used, fallback to pickle
    if not envs.SGLANG_IPC_USE_MSGPACK.get():
        logger.debug(
            f"Sending pickle object of type {type(obj)} since SGLANG_IPC_USE_MSGPACK is disabled"
        )
        socket.send_pyobj(obj, flags=flags)
        return

    if isinstance(obj, msgspec.Struct):
        from .msgpack_struct import serialize

        magic_number = MSGPACK_MAGIC_NUMBER
        logger.debug(
            f"Sending msgpack object of type {type(obj)} with magic number {magic_number}"
        )
        socket.send_multipart([magic_number, serialize(obj)], flags=flags)
    else:
        from .pickle_struct import serialize

        magic_number = PICKLE_MAGIC_NUMBER
        logger.debug(
            f"Sending pickle object of type {type(obj)} with magic number {magic_number}"
        )
        socket.send_multipart([magic_number, serialize(obj)], flags=flags)


def sock_recv(socket: Socket, flags=0) -> Union[BaseReq, BaseBatchReq, msgspec.Struct]:
    if not envs.SGLANG_IPC_USE_MSGPACK.get():
        obj = socket.recv_pyobj(flags=flags)
        logger.debug(
            f"Receiving pickle object of type {type(obj)} since SGLANG_IPC_USE_MSGPACK is disabled"
        )
        return obj

    magic_number, data = socket.recv_multipart(flags=flags)
    if magic_number == MSGPACK_MAGIC_NUMBER:
        from .msgpack_struct import deserialize

        obj = deserialize(data)
        logger.debug(
            f"Received msgpack object of type {type(obj)} with magic number {magic_number}"
        )
        return obj
    else:
        from .pickle_struct import deserialize

        obj = deserialize(data)
        logger.debug(f"Received pickle object of type {type(obj)}")
        return obj


async def sock_send_async(
    socket: AsyncSocket, obj: Union[BaseReq, BaseBatchReq, msgspec.Struct], flags=0
):
    if not envs.SGLANG_IPC_USE_MSGPACK.get():
        logger.debug(
            f"Async sending pickle object of type {type(obj)} since SGLANG_IPC_USE_MSGPACK is disabled"
        )
        await socket.send_pyobj(obj, flags=flags)
        return

    if isinstance(obj, msgspec.Struct):
        from .msgpack_struct import serialize

        magic_number = MSGPACK_MAGIC_NUMBER
        logger.debug(
            f"Async sending msgpack object of type {type(obj)} with magic number {magic_number}"
        )
        await socket.send_multipart([magic_number, serialize(obj)], flags=flags)
    else:
        from .pickle_struct import serialize

        magic_number = PICKLE_MAGIC_NUMBER
        logger.debug(
            f"Async sending pickle object of type {type(obj)} with magic number {magic_number}"
        )
        await socket.send_multipart([magic_number, serialize(obj)], flags=flags)


async def sock_recv_async(
    socket: AsyncSocket, flags=0
) -> Union[BaseReq, BaseBatchReq, msgspec.Struct]:
    if not envs.SGLANG_IPC_USE_MSGPACK.get():
        obj = await socket.recv_pyobj(flags=flags)
        logger.debug(
            f"Async receiving pickle object of type {type(obj)} since SGLANG_IPC_USE_MSGPACK is disabled"
        )
        return obj

    magic_number, data = await socket.recv_multipart(flags=flags)
    if magic_number == MSGPACK_MAGIC_NUMBER:
        from .msgpack_struct import deserialize

        obj = deserialize(data)
        logger.debug(
            f"Async receiving msgpack object of type {type(obj)} with magic number {magic_number}"
        )
        return obj
    else:
        from .pickle_struct import deserialize

        obj = deserialize(data)
        logger.debug(
            f"Async receiving pickle object of type {type(obj)} with magic number {magic_number}"
        )
        return obj


__all__ = [
    "BaseReq",
    "BaseBatchReq",
    "SpeculativeDecodingMetricsMixin",
    "SessionParams",
    "GenerateReqInput",
    "TokenizedGenerateReqInput",
    "BatchTokenizedGenerateReqInput",
    "EmbeddingReqInput",
    "TokenizedEmbeddingReqInput",
    "BatchTokenizedEmbeddingReqInput",
    "BatchTokenIDOutput",
    "BatchStrOutput",
    "BatchEmbeddingOutput",
    "ClearHiCacheReqInput",
    "ClearHiCacheReqOutput",
    "FlushCacheReqInput",
    "FlushCacheReqOutput",
    "AddExternalCorpusReqInput",
    "AddExternalCorpusReqOutput",
    "RemoveExternalCorpusReqInput",
    "RemoveExternalCorpusReqOutput",
    "ListExternalCorporaReqInput",
    "ListExternalCorporaReqOutput",
    "AttachHiCacheStorageReqInput",
    "AttachHiCacheStorageReqOutput",
    "DetachHiCacheStorageReqInput",
    "DetachHiCacheStorageReqOutput",
    "PauseGenerationReqInput",
    "ContinueGenerationReqInput",
    "UpdateWeightFromDiskReqInput",
    "UpdateWeightFromDiskReqOutput",
    "UpdateWeightsFromDistributedReqInput",
    "UpdateWeightsFromDistributedReqOutput",
    "UpdateWeightsFromTensorReqInput",
    "UpdateWeightsFromTensorReqOutput",
    "InitWeightsSendGroupForRemoteInstanceReqInput",
    "UpdateWeightsFromIPCReqInput",
    "UpdateWeightsFromIPCReqOutput",
    "InitWeightsSendGroupForRemoteInstanceReqOutput",
    "SendWeightsToRemoteInstanceReqInput",
    "SendWeightsToRemoteInstanceReqOutput",
    "UpdateExpertBackupReq",
    "BackupDramReq",
    "InitWeightsUpdateGroupReqInput",
    "InitWeightsUpdateGroupReqOutput",
    "DestroyWeightsUpdateGroupReqInput",
    "DestroyWeightsUpdateGroupReqOutput",
    "UpdateWeightVersionReqInput",
    "GetWeightsByNameReqInput",
    "GetWeightsByNameReqOutput",
    "ReleaseMemoryOccupationReqInput",
    "ReleaseMemoryOccupationReqOutput",
    "ResumeMemoryOccupationReqInput",
    "ResumeMemoryOccupationReqOutput",
    "CheckWeightsReqInput",
    "CheckWeightsReqOutput",
    "SlowDownReqInput",
    "SlowDownReqOutput",
    "AbortReq",
    "ActiveRanksOutput",
    "GetInternalStateReq",
    "GetInternalStateReqOutput",
    "SetInternalStateReq",
    "SetInternalStateReqOutput",
    "ProfileReqInput",
    "ProfileReqType",
    "ProfileReq",
    "ProfileReqOutput",
    "FreezeGCReq",
    "ConfigureLoggingReq",
    "OpenSessionReqInput",
    "CloseSessionReqInput",
    "OpenSessionReqOutput",
    "HealthCheckOutput",
    "ExpertDistributionReqType",
    "ExpertDistributionReq",
    "ExpertDistributionReqOutput",
    "Function",
    "Tool",
    "ParseFunctionCallReq",
    "SeparateReasoningReqInput",
    "VertexGenerateReqInput",
    "RpcReqInput",
    "RpcReqOutput",
    "LoadLoRAAdapterReqInput",
    "UnloadLoRAAdapterReqInput",
    "LoadLoRAAdapterFromTensorsReqInput",
    "LoRAUpdateOutput",
    "BlockReqType",
    "BlockReqInput",
    "MemoryMetrics",
    "SpeculativeMetrics",
    "LoRAMetrics",
    "DisaggregationMetrics",
    "QueueMetrics",
    "GetLoadsReqInput",
    "GetLoadsReqOutput",
    "WatchLoadUpdateReq",
    "SetInjectDumpMetadataReqInput",
    "SetInjectDumpMetadataReqOutput",
    "LazyDumpTensorsReqInput",
    "LazyDumpTensorsReqOutput",
    "DumperControlReqInput",
    "DumperControlReqOutput",
    "LoadLoRAAdapterReqOutput",
    "UnloadLoRAAdapterReqOutput",
    "LoadLoRAAdapterFromTensorsReqOutput",
    "MultimodalDataInputFormat",
    "sock_send",
    "sock_recv",
]

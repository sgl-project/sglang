
from dataclasses import dataclass, field
import msgspec
from typing import Union, Optional, List

from zmq import Socket
from zmq.asyncio import Socket as AsyncSocket

from sglang.srt.environ import envs

from .pickle_struct import (
    # Below objects are not used for IPC
    BaseBatchReq,
    BaseReq,
    EmbeddingReqInput,
    GenerateReqInput,
    ConfigureLoggingReq,
    ProfileReqInput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
    MultimodalDataInputFormat,

    # Below objects are not in used.
    SeparateReasoningReqInput,
    SetInjectDumpMetadataReqInput,
    SetInjectDumpMetadataReqOutput,
    LazyDumpTensorsReqInput,
    LazyDumpTensorsReqOutput,
    # SpeculativeDecodingMetricsMixin,
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
        GetLoadsReqInput,
        GetLoadsReqOutput,
        LoRAMetrics,
        MemoryMetrics,
        QueueMetrics,
        SpeculativeMetrics,
        SessionParams,
        TokenizedGenerateReqInput,
        BatchTokenizedGenerateReqInput,
        TokenizedEmbeddingReqInput,
        BatchTokenizedEmbeddingReqInput,
        AbortReq,
        OpenSessionReqInput,
        OpenSessionReqOutput,
        CloseSessionReqInput,
        UpdateWeightFromDiskReqInput,
        UpdateWeightFromDiskReqOutput,
        HealthCheckOutput,
        ActiveRanksOutput,
        InitWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqOutput,
        DestroyWeightsUpdateGroupReqInput,
        DestroyWeightsUpdateGroupReqOutput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromDistributedReqOutput,
        InitWeightsSendGroupForRemoteInstanceReqInput,
        InitWeightsSendGroupForRemoteInstanceReqOutput,
        SendWeightsToRemoteInstanceReqInput,
        SendWeightsToRemoteInstanceReqOutput,
        UpdateWeightsFromTensorReqInput,
        UpdateWeightsFromTensorReqOutput,
        UpdateWeightsFromIPCReqInput,
        UpdateWeightsFromIPCReqOutput,
        GetWeightsByNameReqInput,
        GetWeightsByNameReqOutput,
        ReleaseMemoryOccupationReqInput,
        ReleaseMemoryOccupationReqOutput,
        ResumeMemoryOccupationReqInput,
        ResumeMemoryOccupationReqOutput,
        CheckWeightsReqInput,
        CheckWeightsReqOutput,
        SlowDownReqInput,
        SlowDownReqOutput,
        AddExternalCorpusReqInput,
        AddExternalCorpusReqOutput,
        RemoveExternalCorpusReqInput,
        RemoveExternalCorpusReqOutput,
        ListExternalCorporaReqInput,
        ListExternalCorporaReqOutput,
        ClearHiCacheReqInput,
        ClearHiCacheReqOutput,
        AttachHiCacheStorageReqInput,
        AttachHiCacheStorageReqOutput,
        DetachHiCacheStorageReqInput,
        DetachHiCacheStorageReqOutput,
        ProfileReqType,
        ProfileReq,
        ProfileReqOutput,
        GetInternalStateReq,
        GetInternalStateReqOutput,
        SetInternalStateReq,
        SetInternalStateReqOutput,
        ExpertDistributionReqType,
        ExpertDistributionReq,
        ExpertDistributionReqOutput,
        LoadLoRAAdapterReqInput,
        UnloadLoRAAdapterReqInput,
        LoadLoRAAdapterFromTensorsReqInput,
        LoRAUpdateOutput,
        DumperControlReqInput,
        DumperControlReqOutput,
        WatchLoadUpdateReq,
        ContinueGenerationReqInput,
        PauseGenerationReqInput,
        RpcReqInput,
        RpcReqOutput,
        BlockReqType,
        BlockReqInput,
        BackupDramReq,
        UpdateExpertBackupReq,
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
        GetLoadsReqInput,
        GetLoadsReqOutput,
        LoRAMetrics,
        MemoryMetrics,
        QueueMetrics,
        SpeculativeMetrics,
        SessionParams,
        TokenizedGenerateReqInput,
        BatchTokenizedGenerateReqInput,
        TokenizedEmbeddingReqInput,
        BatchTokenizedEmbeddingReqInput,
        AbortReq,
        OpenSessionReqInput,
        OpenSessionReqOutput,
        CloseSessionReqInput,
        UpdateWeightFromDiskReqInput,
        UpdateWeightFromDiskReqOutput,
        HealthCheckOutput,
        ActiveRanksOutput,
        InitWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqOutput,
        DestroyWeightsUpdateGroupReqInput,
        DestroyWeightsUpdateGroupReqOutput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromDistributedReqOutput,
        InitWeightsSendGroupForRemoteInstanceReqInput,
        InitWeightsSendGroupForRemoteInstanceReqOutput,
        SendWeightsToRemoteInstanceReqInput,
        SendWeightsToRemoteInstanceReqOutput,
        UpdateWeightsFromTensorReqInput,
        UpdateWeightsFromTensorReqOutput,
        UpdateWeightsFromIPCReqInput,
        UpdateWeightsFromIPCReqOutput,
        GetWeightsByNameReqInput,
        GetWeightsByNameReqOutput,
        ReleaseMemoryOccupationReqInput,
        ReleaseMemoryOccupationReqOutput,
        ResumeMemoryOccupationReqInput,
        ResumeMemoryOccupationReqOutput,
        CheckWeightsReqInput,
        CheckWeightsReqOutput,
        SlowDownReqInput,
        SlowDownReqOutput,
        AddExternalCorpusReqInput,
        AddExternalCorpusReqOutput,
        RemoveExternalCorpusReqInput,
        RemoveExternalCorpusReqOutput,
        ListExternalCorporaReqInput,
        ListExternalCorporaReqOutput,
        ClearHiCacheReqInput,
        ClearHiCacheReqOutput,
        AttachHiCacheStorageReqInput,
        AttachHiCacheStorageReqOutput,
        DetachHiCacheStorageReqInput,
        DetachHiCacheStorageReqOutput,
        ProfileReqType,
        ProfileReq,
        ProfileReqOutput,
        GetInternalStateReq,
        GetInternalStateReqOutput,
        SetInternalStateReq,
        SetInternalStateReqOutput,
        ExpertDistributionReqType,
        ExpertDistributionReq,
        ExpertDistributionReqOutput,
        LoadLoRAAdapterReqInput,
        UnloadLoRAAdapterReqInput,
        LoadLoRAAdapterFromTensorsReqInput,
        LoRAUpdateOutput,
        DumperControlReqInput,
        DumperControlReqOutput,
        WatchLoadUpdateReq,
        ContinueGenerationReqInput,
        PauseGenerationReqInput,
        RpcReqInput,
        RpcReqOutput,
        BlockReqType,
        BlockReqInput,
        BackupDramReq,
        UpdateExpertBackupReq,
    )

import logging

logger = logging.getLogger(__name__)

LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput

PICKLE_MAGIC_NUMBER = b"0xSG01"
MSGPACK_MAGIC_NUMBER = b"0xSG02"

@dataclass
class Function:
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[object] = None


@dataclass
class Tool:
    function: Function
    type: Optional[str] = "function"


@dataclass
class ParseFunctionCallReq(BaseReq):
    text: str  # The text to parse.
    tools: List[Tool] = field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


def sock_send(
    socket: Socket, obj: Union[BaseReq, BaseBatchReq, msgspec.Struct], flags=0
):
    # if the msgpack magic number is not used, fallback to pickle
    if not envs.SGLANG_IPC_USE_MSGPACK.get():
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Sending pickle object of type %s since SGLANG_IPC_USE_MSGPACK is disabled",
                type(obj),
            )
        socket.send_pyobj(obj, flags=flags)
        return

    if isinstance(obj, msgspec.Struct):
        from .msgpack_struct import serialize

        magic_number = MSGPACK_MAGIC_NUMBER
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Sending msgpack object of type %s with magic number %s",
                type(obj),
                magic_number,
            )
        socket.send_multipart([magic_number, serialize(obj)], flags=flags)
    else:
        from .pickle_struct import serialize

        magic_number = PICKLE_MAGIC_NUMBER
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Sending pickle object of type %s with magic number %s",
                type(obj),
                magic_number,
            )
        socket.send_multipart([magic_number, serialize(obj)], flags=flags)


def sock_recv(socket: Socket, flags=0) -> Union[BaseReq, BaseBatchReq, msgspec.Struct]:
    if not envs.SGLANG_IPC_USE_MSGPACK.get():
        obj = socket.recv_pyobj(flags=flags)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Receiving pickle object of type %s since SGLANG_IPC_USE_MSGPACK is disabled",
                type(obj),
            )
        return obj

    magic_number, data = socket.recv_multipart(flags=flags)
    if magic_number == MSGPACK_MAGIC_NUMBER:
        from .msgpack_struct import deserialize

        obj = deserialize(data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Received msgpack object of type %s with magic number %s",
                type(obj),
                magic_number,
            )
        return obj
    else:
        from .pickle_struct import deserialize

        obj = deserialize(data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Received pickle object of type %s", type(obj))
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

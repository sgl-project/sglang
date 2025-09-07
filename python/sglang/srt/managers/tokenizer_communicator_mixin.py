from __future__ import annotations

import asyncio
import os
from ast import List
from collections import deque
from typing import TYPE_CHECKING, Deque, Generic, Optional, TypeVar

from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GenerateReqInput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HealthCheckOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    LoRAUpdateResult,
    MultiTokenizerWrapper,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SessionParams,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.tokenizer_manager import TypeBasedDispatcher
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

T = TypeVar("T")


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    enable_multi_tokenizer = False

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Future] = deque()

    async def __call__(self, obj):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj:
            if _Communicator.enable_multi_tokenizer:
                obj = MultiTokenizerWrapper(worker_id=os.getpid(), obj=obj)
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()


class TokenizerCommunicatorMixin:
    """Mixin class for TokenizerManager to handle communication with the scheduler."""

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        # Communicators
        self.init_weights_update_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_tensor_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_weights_by_name_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.slow_down_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.flush_cache_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.clear_hicache_storage_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.profile_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.expert_distribution_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_lora_adapter_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        self._result_dispatcher += self._get_communicator_dispatcher()

    def _get_communicator_dispatcher(self):
        return TypeBasedDispatcher(
            [
                (
                    InitWeightsUpdateGroupReqOutput,
                    self.init_weights_update_group_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromTensorReqOutput,
                    self.update_weights_from_tensor_communicator.handle_recv,
                ),
                (
                    GetWeightsByNameReqOutput,
                    self.get_weights_by_name_communicator.handle_recv,
                ),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    SlowDownReqOutput,
                    self.slow_down_communicator.handle_recv,
                ),
                (
                    ClearHiCacheReqOutput,
                    self.clear_hicache_storage_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    SetInternalStateReqOutput,
                    self.set_internal_state_communicator.handle_recv,
                ),
                (
                    ExpertDistributionReqOutput,
                    self.expert_distribution_communicator.handle_recv,
                ),
                (
                    LoRAUpdateResult,
                    self.update_lora_adapter_communicator.handle_recv,
                ),
            ]
        )

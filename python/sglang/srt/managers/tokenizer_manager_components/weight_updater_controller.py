from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import fastapi

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.aio_rwlock import RWLock

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class WeightUpdaterController:
    send_to_scheduler: Any
    abort_request: Callable[..., None]
    update_model_path_info: Callable[[str, str], None]
    is_pause_getter: Callable[[], bool]
    is_pause_cond: asyncio.Condition
    model_update_lock: RWLock
    server_args: ServerArgs
    auto_create_handle_loop: Callable[[], None]
    initial_weights_loaded: bool = True
    model_update_result: Optional[Awaitable[Any]] = None
    model_update_tmp: List[Any] = field(default_factory=list)
    init_weights_update_group_communicator: Any = (
        None  # set after facade.init_communicators
    )
    destroy_weights_update_group_communicator: Any = (
        None  # set after facade.init_communicators
    )
    update_weights_from_distributed_communicator: Any = (
        None  # set after facade.init_communicators
    )
    update_weights_from_tensor_communicator: Any = (
        None  # set after facade.init_communicators
    )
    update_weights_from_ipc_communicator: Any = (
        None  # set after facade.init_communicators
    )
    get_weights_by_name_communicator: Any = None  # set after facade.init_communicators
    release_memory_occupation_communicator: Any = (
        None  # set after facade.init_communicators
    )
    resume_memory_occupation_communicator: Any = (
        None  # set after facade.init_communicators
    )
    check_weights_communicator: Any = None  # set after facade.init_communicators

    def __post_init__(self) -> None:
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Immediately update the weights if the engine is in paused state
        async with self.is_pause_cond:
            is_paused = self.is_pause_getter()

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            success, message, num_paused_requests = (
                await self._wait_for_model_update_from_disk(obj)
            )

        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message, num_paused_requests

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        self.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if self.server_args.dp_size == 1:
            result = await self.model_update_result
            if result.success:
                self.update_model_path_info(obj.model_path, obj.load_format)
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self.update_model_path_info(obj.model_path, obj.load_format)
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    def handle_update_weights_from_disk_req_output(self, recv_obj):
        if self.server_args.dp_size == 1:
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are received
            if len(self.model_update_tmp) == self.server_args.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)

    def _update_weight_version_if_provided(self, weight_version: Optional[str]) -> None:
        """Update weight version if provided."""
        if weight_version is not None:
            self.server_args.weight_version = weight_version

    async def init_weights_update_group(
        self,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        results = await self.init_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def destroy_weights_update_group(
        self,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"

        results = await self.destroy_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def update_weights_from_distributed(
        self,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            is_paused = self.is_pause_getter()
            if is_paused:
                results = await self.update_weights_from_distributed_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_distributed_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def update_weights_from_tensor(
        self,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        async with self.is_pause_cond:
            is_paused = self.is_pause_getter()
            if is_paused:
                results = await self.update_weights_from_tensor_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_tensor_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def update_weights_from_ipc(
        self,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Update weights via IPC for checkpoint-engine integration."""
        self.auto_create_handle_loop()
        try:
            # For now, we only support single data parallel instance
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            logger.info("Starting IPC weight update")

            async with self.is_pause_cond:
                is_paused = self.is_pause_getter()
                if is_paused:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message

            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg

        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def get_weights_by_name(
        self,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def check_weights(
        self,
        obj: CheckWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str, Optional[List[Dict]], Optional[str]]:
        self.auto_create_handle_loop()
        results = await self.check_weights_communicator(obj)
        success, message = FanOutCommunicator.merge_results(results)
        ranks: Optional[List[Dict]] = None
        per_engine_checksum: Optional[str] = None
        if any(r.payload is not None for r in results):
            ranks = []
            for r in results:
                if isinstance(r.payload, list):
                    ranks.extend(r.payload)
                else:
                    ranks.append(r.payload)
            h = hashlib.sha256()
            for rank in ranks:
                h.update(rank["per_gpu_checksum"].encode())
            per_engine_checksum = h.hexdigest()
        return success, message, ranks, per_engine_checksum

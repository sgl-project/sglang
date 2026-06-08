from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import fastapi

from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
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

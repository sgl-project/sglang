from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import fastapi

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    LoRAUpdateOutput,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizerControlAckReq,
    TokenizerControlBackendResultReq,
    TokenizerControlBroadcastReq,
    TokenizerControlResultReq,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    async_sock_send,
    msgpack_decode,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.managers.tokenizer_control import (
    CONTROL_TIMEOUT_SECONDS,
    ControlCoordinator,
    ControlRpc,
    make_control_request,
)
from sglang.srt.runtime_context import (
    get_lora,
    get_parallel,
    get_serving,
    get_spec,
)
from sglang.srt.server_args import LoRARef
from sglang.srt.utils import (
    get_bool_env_var,
    normalize_serialized_named_tensor_payloads,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Declarative spec: (attr_name_prefix, response_type[, mode])
# Each entry creates self.{prefix}_communicator and registers
# response_type -> communicator.handle_recv in the dispatch table.
_COMMUNICATOR_SPECS = [
    ("init_weights_update_group", InitWeightsUpdateGroupReqOutput),
    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),
    (
        "init_weights_send_group_for_remote_instance",
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    ("send_weights_to_remote_instance", SendWeightsToRemoteInstanceReqOutput),
    ("get_weights_by_name", GetWeightsByNameReqOutput),
    ("check_weights", CheckWeightsReqOutput),
    ("slow_down", SlowDownReqOutput),
    ("flush_cache", FlushCacheReqOutput),
    ("add_external_corpus", AddExternalCorpusReqOutput),
    ("remove_external_corpus", RemoveExternalCorpusReqOutput),
    ("list_external_corpora", ListExternalCorporaReqOutput),
    ("clear_hicache_storage", ClearHiCacheReqOutput),
    ("attach_hicache_storage", AttachHiCacheStorageReqOutput),
    ("detach_hicache_storage", DetachHiCacheStorageReqOutput),
    ("profile", ProfileReqOutput),
    ("get_internal_state", GetInternalStateReqOutput),
    ("set_internal_state", SetInternalStateReqOutput),
    ("expert_distribution", ExpertDistributionReqOutput),
    ("update_lora_adapter", LoRAUpdateOutput),
    ("dumper_control", DumperControlReqOutput),
    ("scale_elastic_ep", ScaleElasticEPReqOutput),
]


def _merge_lora_update_results(results: List[LoRAUpdateOutput]) -> LoRAUpdateOutput:
    """Merge the per-rank replies of a LoRA load/unload fan-out into one result.

    The operation succeeded only if every rank succeeded. Reporting a partial
    failure as success would let the tokenizer-side LoRA registry drift from
    the ranks that failed, so failures win: their deduplicated error messages
    are joined, and loaded_adapters reflects the first failed rank.
    """
    failed = [r for r in results if not r.success]
    if not failed:
        return results[0]
    error_messages = list(
        dict.fromkeys(r.error_message for r in failed if r.error_message)
    )
    return LoRAUpdateOutput(
        success=False,
        error_message=" | ".join(error_messages),
        loaded_adapters=failed[0].loaded_adapters,
    )


class TokenizerControlMixin:
    """Mixin for TokenizerManager's control-plane operations (weights, cache, lora,
    profile, internal state, etc.) -- everything that talks to the scheduler via
    FanOutCommunicator, as opposed to data-plane inference requests multiplexed by rid.
    """

    def init_communicators(self: TokenizerManager):
        dispatch_pairs = []
        for spec in _COMMUNICATOR_SPECS:
            name, resp_type = spec[0], spec[1]
            mode = spec[2] if len(spec) > 2 else "queueing"
            comm = FanOutCommunicator(
                self._dispatch_to_scheduler,
                get_parallel().dp_size,
                mode,
            )
            setattr(self, f"{name}_communicator", comm)
            dispatch_pairs.append((resp_type, comm.handle_recv))
        self._control_futures = {}
        self._control_tasks = set()
        self._control_apply_lock = asyncio.Lock()
        self._control_error = None
        self._control_revision = 0
        self._control_rpc = ControlRpc(self._send_control_to_scheduler)
        self._control = ControlCoordinator(
            self._control_rpc.call, self._apply_control_broadcast, self._fail_control
        )
        dispatch_pairs.extend(
            [
                (TokenizerControlBackendResultReq, self._control_rpc.handle_recv),
                (TokenizerControlBroadcastReq, self._handle_control_broadcast),
                (TokenizerControlResultReq, self._handle_control_result),
            ]
        )
        self._result_dispatcher += TypeBasedDispatcher(dispatch_pairs)

    async def _send_control_to_scheduler(self, obj):
        # Preserve the coordinator return route instead of stamping an HTTP worker.
        await async_sock_send(self.send_to_scheduler, obj)

    def _own_control_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._control_tasks.add(task)

        def completed(task):
            self._control_tasks.discard(task)
            if not task.cancelled():
                task.exception()  # A disconnected HTTP caller no longer observes it.

        task.add_done_callback(completed)
        return task

    async def _call_control(self, obj, kind):
        self.auto_create_handle_loop()
        if self._control_error is not None:
            raise RuntimeError(
                f"Previous control operation failed; restart the server: {self._control_error}"
            )
        fan_out = self.elastic_worker_count
        if kind in ("pause", "continue"):
            fan_out = self.get_internal_state_communicator._fan_out
        request = make_control_request(
            obj, kind, fan_out, self.get_internal_state_communicator._fan_out
        )
        return await asyncio.shield(self._own_control_task(self._wait_control(request)))

    async def _wait_control(self, request):
        if get_serving().tokenizer_worker_num == 1:
            return await self._control.run(request)
        future = asyncio.get_running_loop().create_future()
        self._control_futures[request.operation_id] = future
        try:
            await self._async_dispatch_to_scheduler(request)
            return await asyncio.wait_for(future, CONTROL_TIMEOUT_SECONDS + 5)
        except asyncio.TimeoutError:
            self._fail_control(request.operation_id, "Control coordinator timed out")
            raise
        finally:
            self._control_futures.pop(request.operation_id, None)

    def _handle_control_result(self, obj):
        future = self._control_futures.get(obj.operation_id)
        if future is None or future.done():
            return
        if obj.error is not None:
            future.set_exception(RuntimeError(obj.error))
        else:
            future.set_result([msgpack_decode(r) for r in obj.results])

    def _fail_control(self, operation_id, error):
        self._control_error = error
        self.is_pause = True

    def _handle_control_broadcast(self, obj):
        if obj.action == "abort":
            # Cancellation must bypass an action waiting for readers to drain.
            self._mark_abort_requests(obj.rid, obj.abort_all)
            self._own_control_task(self._notify_control_waiters())
            self._ack_control_action(obj)
        else:
            self._own_control_task(self._apply_and_ack_control(obj))

    def _ack_control_action(self, obj, error=None):
        self._dispatch_to_scheduler(
            TokenizerControlAckReq(
                broadcast_id=obj.broadcast_id,
                worker_ipc_name=self.tokenizer_ipc_name,
                error=error,
            )
        )

    async def _notify_control_waiters(self):
        async with self.is_pause_cond:
            self.is_pause_cond.notify_all()

    async def _apply_and_ack_control(self, obj):
        error = None
        try:
            if obj.action == "fail":
                await self._apply_control_broadcast(obj)
            else:
                # A duplicate action must not ACK while the first copy is draining.
                async with self._control_apply_lock:
                    await self._apply_control_broadcast(obj)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        if obj.action != "fail":
            self._ack_control_action(obj, error)

    async def _apply_control_broadcast(self, obj):
        if obj.action == "fail":
            self._fail_control(obj.broadcast_id, obj.error)
            return
        if self._control_error is not None:
            raise RuntimeError(self._control_error)
        if obj.revision <= self._control_revision:
            return
        self._control_revision = obj.revision
        async with self.is_pause_cond:
            if obj.action == "pause":
                self.is_pause = True
            elif obj.is_pause is not None:
                self.is_pause = obj.is_pause
            if obj.abort_all:
                self._mark_abort_requests("", True)
            if obj.updates:
                if "model_path" in obj.updates:
                    self._update_model_path_info(
                        obj.updates["model_path"], obj.updates["load_format"]
                    )
                self.record_config_updates("tokenizer.control", **obj.updates)
            if obj.clear_mm_cache and self.mm_processor is not None:
                self.mm_processor.clear_preprocess_cache()
            if obj.weights_ready:
                self.initial_weights_loaded = True
            if not self.is_pause or obj.abort_all:
                self.is_pause_cond.notify_all()
        if obj.wait_for_requests or obj.action == "drain":
            while await self.model_update_lock.is_locked() or any(
                state.abort_requested and not state.finished
                for state in self.rid_to_state.values()
            ):
                if self._control_error is not None:
                    raise RuntimeError(self._control_error)
                await asyncio.sleep(0.01)

    async def _update_weights(self, obj):
        try:
            results = await self._call_control(obj, "weights")
        except (RuntimeError, TimeoutError) as exc:
            return False, str(exc) or type(exc).__name__, []
        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            message += f" Weight version updated to {obj.weight_version}."
        return success, message, results

    def update_control_communicator_fan_out(self: TokenizerManager, worker_count: int):
        primary_group_control = (
            get_parallel().enable_dp_attention
            and not get_parallel().enable_dp_attention_local_control_broadcast
        )
        if primary_group_control:
            control_fan_out = (
                worker_count + get_parallel().tp_size - 1
            ) // get_parallel().tp_size
        else:
            control_fan_out = worker_count

        for spec in _COMMUNICATOR_SPECS:
            getattr(self, f"{spec[0]}_communicator").set_fan_out(worker_count)

        self.get_internal_state_communicator.set_fan_out(control_fan_out)

    async def add_external_corpus(
        self: TokenizerManager, obj: AddExternalCorpusReqInput
    ) -> AddExternalCorpusReqOutput:
        self.auto_create_handle_loop()
        if get_spec().speculative_algorithm != "NGRAM":
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        truncated = False
        try:
            if not obj.corpus_id:
                import uuid

                obj.corpus_id = uuid.uuid4().hex
            if obj.file_path is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    iter_external_corpus_chunks,
                )

                max_tokens = get_spec().speculative_ngram_external_corpus_max_tokens
                obj.token_chunks = list(
                    iter_external_corpus_chunks(
                        obj.file_path, self.tokenizer, max_tokens
                    )
                )
            elif obj.documents is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    SEPARATOR_TOKEN,
                )

                max_tokens = get_spec().speculative_ngram_external_corpus_max_tokens
                token_chunks = []
                total_tokens = 0
                has_prev = False
                for doc in obj.documents:
                    if not doc:
                        continue
                    token_ids = list(
                        self.tokenizer.encode(doc, add_special_tokens=False)
                    )
                    if not token_ids:
                        continue
                    if has_prev:
                        token_ids = [SEPARATOR_TOKEN] + token_ids
                    if total_tokens + len(token_ids) > max_tokens:
                        truncated = True
                        break
                    token_chunks.append(token_ids)
                    total_tokens += len(token_ids)
                    has_prev = True
                obj.token_chunks = token_chunks
            else:
                return AddExternalCorpusReqOutput(
                    success=False,
                    message="Either file_path or documents must be provided.",
                )
            obj.file_path = None
            obj.documents = None
            results = await self.add_external_corpus_communicator(obj)
            all_success, all_message = FanOutCommunicator.merge_results(results)
            if truncated and all_success:
                all_message += f" (truncated: exceeded {max_tokens} token limit)"
            return AddExternalCorpusReqOutput(
                success=all_success,
                corpus_id=results[0].corpus_id if all_success else "",
                message=all_message,
                loaded_token_count=results[0].loaded_token_count if all_success else 0,
            )
        except Exception as e:
            return AddExternalCorpusReqOutput(success=False, message=str(e))

    async def remove_external_corpus(
        self: TokenizerManager, corpus_id: str
    ) -> RemoveExternalCorpusReqOutput:
        self.auto_create_handle_loop()
        if get_spec().speculative_algorithm != "NGRAM":
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.remove_external_corpus_communicator(
            RemoveExternalCorpusReqInput(corpus_id=corpus_id)
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        return RemoveExternalCorpusReqOutput(success=all_success, message=all_message)

    async def list_external_corpora(
        self: TokenizerManager,
    ) -> ListExternalCorporaReqOutput:
        self.auto_create_handle_loop()
        if get_spec().speculative_algorithm != "NGRAM":
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.list_external_corpora_communicator(
            ListExternalCorporaReqInput()
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        # Merge corpus token counts from all DP ranks (each rank loads the same set).
        corpus_token_counts = results[0].corpus_token_counts if all_success else {}
        return ListExternalCorporaReqOutput(
            success=all_success,
            corpus_token_counts=corpus_token_counts,
            message=all_message,
        )

    async def flush_cache(
        self: TokenizerManager, timeout_s: Optional[float] = None
    ) -> FlushCacheReqOutput:
        self.auto_create_handle_loop()
        result = (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
        )[0]
        if result.success and self.mm_processor is not None:
            self.mm_processor.clear_preprocess_cache()
        return result

    async def clear_hicache_storage(self: TokenizerManager) -> ClearHiCacheReqOutput:
        """Clear the hierarchical cache storage."""
        self.auto_create_handle_loop()
        # Delegate to the scheduler to handle HiCacheStorage clearing
        return (await self.clear_hicache_storage_communicator(ClearHiCacheReqInput()))[
            0
        ]

    async def attach_hicache_storage(
        self: TokenizerManager,
        hicache_storage_backend: str,
        hicache_storage_backend_extra_config_json: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> AttachHiCacheStorageReqOutput:
        """Attach (enable) HiCache storage backend at runtime."""
        self.auto_create_handle_loop()
        results = await self.attach_hicache_storage_communicator(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend=hicache_storage_backend,
                hicache_storage_backend_extra_config_json=hicache_storage_backend_extra_config_json,
                hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
                hicache_write_policy=hicache_write_policy,
            )
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = AttachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            # Keep tokenizer side server_info consistent with scheduler side.
            hicache_fields = {"hicache_storage_backend": hicache_storage_backend}
            if hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    hicache_storage_backend_extra_config_json
                )
            if hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    hicache_storage_prefetch_policy
                )
            if hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = hicache_write_policy
            self.record_config_updates("tokenizer.attach_hicache", **hicache_fields)
        return out

    async def detach_hicache_storage(
        self: TokenizerManager,
    ) -> DetachHiCacheStorageReqOutput:
        """Detach (disable) HiCache storage backend at runtime."""
        self.auto_create_handle_loop()
        results = await self.detach_hicache_storage_communicator(
            DetachHiCacheStorageReqInput()
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = DetachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            self.record_config_updates(
                "tokenizer.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
        return out

    async def start_profile(
        self: TokenizerManager,
        req: Optional[ProfileReq] = None,
    ):
        self.auto_create_handle_loop()
        req = req or ProfileReq()
        req.req_type = ProfileReqType.START_PROFILE
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        req.with_stack = (
            False if req.with_stack is False or env_with_stack is False else True
        )
        env_record_shapes: bool = get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )
        req.record_shapes = (req.record_shapes is not False) and env_record_shapes
        req.profile_id = req.profile_id or str(time.time())
        return await self._execute_profile(req)

    async def stop_profile(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self: TokenizerManager, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def start_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.START_RECORD)
        await self.expert_distribution_communicator(req)

    async def stop_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.STOP_RECORD)
        await self.expert_distribution_communicator(req)

    async def dump_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.DUMP_RECORD)
        await self.expert_distribution_communicator(req)

    async def init_weights_update_group(
        self: TokenizerManager,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for update weights from distributed"
        )

        results = await self.init_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def destroy_weights_update_group(
        self: TokenizerManager,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"
        )

        results = await self.destroy_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for update weights from distributed"
        )

        success, message, _ = await self._update_weights(obj)
        return success, message

    async def init_weights_send_group_for_remote_instance(
        self: TokenizerManager,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert get_parallel().dp_size == 1, (
            "dp_size must be 1 for init_weights_send_group_for_remote_instance"
        )
        result = (
            await self.init_weights_send_group_for_remote_instance_communicator(obj)
        )[0]
        return result.success, result.message

    async def send_weights_to_remote_instance(
        self: TokenizerManager,
        obj: SendWeightsToRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert get_parallel().dp_size == 1, (
            "dp_size must be 1 for send_weights_to_remote_instance"
        )
        result = (await self.send_weights_to_remote_instance_communicator(obj))[0]
        return result.success, result.message

    async def update_weights_from_tensor(
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for update weights from tensor"
        )

        obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
            obj.serialized_named_tensors
        )
        success, message, _ = await self._update_weights(obj)
        return success, message

    async def update_weights_from_ipc(
        self: TokenizerManager,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
        success, message, _ = await self._update_weights(obj)
        return success, message

    async def _unload_lora_adapter_locked(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
    ) -> UnloadLoRAAdapterReqOutput:
        assert self.lora_update_lock.locked(), (
            "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"
        )

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        await self.lora_registry.wait_for_unload(lora_id)
        result = _merge_lora_update_results(
            await self.update_lora_adapter_communicator(obj)
        )

        return result

    async def load_lora_adapter(
        self: TokenizerManager,
        obj: LoadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not get_lora().enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
                "dp_size must be 1 or dp attention must be enabled for dynamic lora loading"
            )
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            async with self.lora_update_lock:
                # Generate new uniquely identifiable LoRARef object.
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                obj.lora_id = new_adapter.lora_id
                result = _merge_lora_update_results(
                    await self.update_lora_adapter_communicator(obj)
                )

                # Register the LoRA adapter only after loading is successful.
                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter

                if get_lora().max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > get_lora().max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {get_lora().max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_lora_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()

        try:
            if not get_lora().enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
                "dp_size must be 1 or dp attention must be enabled for dynamic lora loading"
            )
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",
                obj.lora_name,
            )

            obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
                obj.serialized_named_tensors
            )

            async with self.lora_update_lock:
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.lora_id = new_adapter.lora_id
                result = _merge_lora_update_results(
                    await self.update_lora_adapter_communicator(obj)
                )

                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter
                if get_lora().max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > get_lora().max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {get_lora().max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_lora_adapter(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not get_lora().enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert obj.lora_name is not None, (
                "lora_name must be provided to unload LoRA adapter"
            )

            assert get_parallel().dp_size == 1 or get_parallel().enable_dp_attention, (
                "dp_size must be 1 or dp attention must be enabled for dynamic lora loading"
            )
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                return await self._unload_lora_adapter_locked(obj)
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def get_weights_by_name(
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if get_parallel().dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self: TokenizerManager,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self._call_control(obj, "release")

    async def resume_memory_occupation(
        self: TokenizerManager,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self._call_control(obj, "resume")

    async def check_weights(
        self: TokenizerManager,
        obj: CheckWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str, Optional[List[Dict]], Optional[str]]:
        self.auto_create_handle_loop()
        results = await self.check_weights_communicator(obj)
        success, message = FanOutCommunicator.merge_results(results)
        ranks: Optional[List[Dict]] = None
        per_engine_checksum: Optional[str] = None
        if any(r.payload is not None for r in results):
            rank_infos: List[ChecksumInfo] = []
            for r in results:
                if r.payload is not None:
                    rank_infos.extend(r.payload)
            h = hashlib.sha256()
            for info in rank_infos:
                h.update(info.per_gpu_checksum.encode())
            per_engine_checksum = h.hexdigest()
            ranks = [msgspec_to_builtins(info) for info in rank_infos]
        return success, message, ranks, per_engine_checksum

    async def slow_down(
        self: TokenizerManager,
        obj: SlowDownReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.slow_down_communicator(obj)

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
        self.auto_create_handle_loop()
        req = GetInternalStateReq()
        responses: List[
            GetInternalStateReqOutput
        ] = await self.get_internal_state_communicator(req)
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        self.auto_create_handle_loop()
        responses: List[
            SetInternalStateReqOutput
        ] = await self.set_internal_state_communicator(obj)
        return [res.updated for res in responses]

    async def dumper_control(
        self: TokenizerManager, obj: DumperControlReqInput
    ) -> List[DumperControlReqOutput]:
        self.auto_create_handle_loop()
        return await self.dumper_control_communicator(obj)

    async def get_loads(
        self: TokenizerManager,
        include: Optional[List[str]] = None,
        dp_rank: Optional[int] = None,
    ) -> List[LoadSnapshot]:
        """
        Get load snapshots for /v1/loads endpoint.

        Args:
            include: List of sections to include. Options: core, memory, spec, lora, disagg, queues, all
            dp_rank: Optional filter for specific DP rank

        Returns:
            List of LoadSnapshot, one per scheduler (filtered by dp_rank if specified)
        """
        self.auto_create_handle_loop()
        if dp_rank is not None and (
            dp_rank < 0 or dp_rank >= self.elastic_worker_count
        ):
            return []

        reader = self.load_snapshot_reader
        if dp_rank is not None:
            load = reader.read(dp_rank)
            results = [load] if load is not None else []
        else:
            results = reader.read_all()

        return results

    async def open_session(
        self: TokenizerManager,
        obj: OpenSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        if obj.streaming:
            if not get_serving().enable_streaming_session:
                raise ValueError(
                    "Streaming sessions are disabled. "
                    "Please relaunch with --enable-streaming-session."
                )

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        future = asyncio.Future()
        self.session_futures[obj.session_id] = future
        self._dispatch_to_scheduler(obj)

        try:
            return await future
        finally:
            self.session_futures.pop(obj.session_id, None)

    async def close_session(
        self: TokenizerManager,
        obj: CloseSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        await self._async_dispatch_to_scheduler(obj)

    async def update_weight_version(self, obj: UpdateWeightVersionReqInput):
        await self._call_control(obj, "version")

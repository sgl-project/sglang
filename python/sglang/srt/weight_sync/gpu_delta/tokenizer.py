"""Control plane for engines exclusively owned by the GPU-delta coordinator."""

import asyncio
import time
import uuid
from collections.abc import Callable

from sglang.srt.runtime_context import get_serving
from sglang.srt.weight_sync.gpu_delta.io import (
    AbortGpuDeltaReqInput,
    ApplyGpuDeltaReqInput,
    ClearGpuDeltaStateReqInput,
    GetGpuDeltaStatusReqInput,
    GpuDeltaReqOutput,
    PrepareGpuDeltaReqInput,
    ReleaseGpuDeltaCacheReqInput,
    ResumeGpuDeltaReqInput,
    UpdateWeightsFromGpuDeltaReqInput,
)


class GpuDeltaCommunicator:
    """Serialize delta controls and correlate replies without blocking generation."""

    def __init__(self, send: Callable, fan_out: int):
        self._send = send
        self._fan_out = fan_out
        self._lock = asyncio.Lock()
        self._rid = self._event = self._results = self._expected = None

    async def __call__(self, request):
        async with self._lock:
            request.rid = self._rid = uuid.uuid4().hex
            self._event = asyncio.Event()
            self._results = []
            self._expected = self._fan_out
            try:
                self._send(request)
                await self._event.wait()
                return self._results
            finally:
                self._rid = self._event = self._results = self._expected = None

    def handle_recv(self, reply):
        # Canceled requests can still reply after the next control was sent.
        if self._event is None or reply.rid != self._rid:
            return
        self._results.append(reply)
        if len(self._results) == self._expected:
            self._event.set()

    def set_fan_out(self, fan_out: int):
        self._fan_out = fan_out


class GpuDeltaTokenizerControl:
    def __init__(self, manager, fan_out: int):
        from sglang.utils import TypeBasedDispatcher

        self.manager = manager
        self.communicator = GpuDeltaCommunicator(
            manager._dispatch_to_scheduler, fan_out
        )
        manager._result_dispatcher += TypeBasedDispatcher(
            [(GpuDeltaReqOutput, self.communicator.handle_recv)]
        )

    async def request(self, obj):
        try:
            self.manager.auto_create_handle_loop()
            if get_serving().tokenizer_worker_num != 1:
                return {
                    "success": False,
                    "message": "GPU delta requires one Python tokenizer worker",
                    "participants": [],
                }
            if isinstance(obj, UpdateWeightsFromGpuDeltaReqInput):
                return await self._update(obj)
            if isinstance(obj, ClearGpuDeltaStateReqInput):
                return await self._clear()
            # Admission/status do not gate generation or acquire its writer lock.
            return await self._request(obj)
        except Exception as exc:
            return {"success": False, "message": str(exc), "participants": []}

    async def _resume(self, obj):
        async with self.manager.is_pause_cond:
            result = await self._request(obj)
            if result["success"]:
                self.manager._update_weight_version_if_provided(
                    str(result["participants"][0]["target_version"])
                    if obj.weight_version is None
                    else obj.weight_version
                )
                if not obj.keep_pause:
                    self.manager.is_pause = False
                    self.manager.is_pause_cond.notify_all()
            return result

    async def _apply(self, obj):
        async with self.manager.is_pause_cond:
            was_paused = self.manager.is_pause
            self.manager.is_pause = True
            if obj.abort_all_requests:
                self.manager.abort_request(abort_all=True)
            result = await self._request(obj)
            if (
                result["success"]
                and obj.flush_cache
                and self.manager.mm_processor is not None
            ):
                self.manager.mm_processor.clear_preprocess_cache()
            return result, was_paused

    async def _update(self, obj):
        started = time.monotonic()
        prepare = PrepareGpuDeltaReqInput(
            session_id=obj.session_id,
            manifest_path=obj.manifest_path,
            manifest_sha256=obj.manifest_sha256,
            stream_id=obj.stream_id,
            base_version=obj.base_version,
            target_version=obj.target_version,
            plan_digest=obj.plan_digest,
            participants=obj.participants,
        )
        # Preparation can fail safely while the engine still serves base weights.
        # After apply dispatch, neither cancellation nor failure may replay XOR
        # or reopen admission: a participant may already have modified weights.
        try:
            result = await asyncio.wait_for(self._prepare(prepare), timeout=1800)
        except BaseException:
            await self._request(AbortGpuDeltaReqInput(session_id=obj.session_id))
            raise
        if not result["success"]:
            await self._request(AbortGpuDeltaReqInput(session_id=obj.session_id))
            return result
        prepared_at = time.monotonic()
        applied, was_paused = await self._apply(
            ApplyGpuDeltaReqInput(
                session_id=obj.session_id,
                flush_cache=obj.flush_cache,
                abort_all_requests=obj.abort_all_requests,
            ),
        )
        if not applied["success"]:
            return applied
        applied_at = time.monotonic()
        result = await self._resume(
            ResumeGpuDeltaReqInput(
                session_id=obj.session_id,
                keep_pause=was_paused,
                weight_version=obj.weight_version,
            )
        )
        if not result["success"]:
            return result
        resumed_at = time.monotonic()
        result["applied_participants"] = applied["participants"]
        result["timings"] = {
            "prepare_s": prepared_at - started,
            "apply_s": applied_at - prepared_at,
            "resume_s": resumed_at - applied_at,
            "activation_s": resumed_at - started,
        }
        if obj.release_state:
            cleared = await self._clear()
            if not cleared["success"]:
                return cleared
        return result

    async def _prepare(self, request):
        result = await self._request(request)
        while result["success"] and any(
            rank["state"] != "PREPARED" for rank in result["participants"]
        ):
            await asyncio.sleep(0.05)
            result = await self._request(
                GetGpuDeltaStatusReqInput(session_id=request.session_id)
            )
        return result

    async def _clear(self):
        result = await self._request(ClearGpuDeltaStateReqInput())
        if not result["success"]:
            return result
        owners = {}
        for rank in result["participants"]:
            identity = rank["identity"]
            if identity is not None:
                owners.setdefault(
                    (identity["hostname"], identity["host_cache_id"]),
                    identity["rank_id"],
                )
        return await self._request(
            ReleaseGpuDeltaCacheReqInput(owner_rank_ids=list(owners.values()))
        )

    async def _request(self, obj):
        results = await self.communicator(obj)
        participants = [result.participant for result in results]
        success = all(result.success for result in results)
        return {
            "success": success,
            "message": " | ".join(
                result.message for result in results if result.message
            ),
            "participants": participants,
        }

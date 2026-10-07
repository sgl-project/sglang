"""Control plane for engines exclusively owned by the GPU-delta coordinator."""

import asyncio
import hashlib
import uuid
from pathlib import Path

import orjson

from sglang.srt.runtime_context import get_serving
from sglang.srt.weight_sync.gpu_delta.io import (
    AbortWeightsFromDeltaReqInput,
    ClearWeightsDeltaStateReqInput,
    DeltaWeightsReqOutput,
    GetWeightsDeltaInfoReqInput,
    GetWeightsDeltaStatusReqInput,
    LoadWeightsFromDeltaReqInput,
    PrepareWeightsFromDeltaReqInput,
    ReleaseWeightsDeltaCacheReqInput,
    ResumeWeightsFromDeltaReqInput,
    UpdateWeightsFromDeltaReqInput,
)
from sglang.srt.weight_sync.gpu_delta.session import GpuDeltaCommunicator


class GpuDeltaTokenizerControl:
    def __init__(self, manager, fan_out: int):
        from sglang.utils import TypeBasedDispatcher

        self.manager = manager
        self.communicator = GpuDeltaCommunicator(
            manager._dispatch_to_scheduler, fan_out
        )
        manager._result_dispatcher += TypeBasedDispatcher(
            [(DeltaWeightsReqOutput, self.communicator.handle_recv)]
        )

    async def request(self, obj):
        if isinstance(
            obj, (LoadWeightsFromDeltaReqInput, ClearWeightsDeltaStateReqInput)
        ):
            try:
                if isinstance(obj, LoadWeightsFromDeltaReqInput):
                    return await self._load(obj.manifest_path, obj.release_state)
                return await self._clear()
            except Exception as exc:
                return {"success": False, "message": str(exc), "participants": []}
        if isinstance(
            obj, (UpdateWeightsFromDeltaReqInput, ResumeWeightsFromDeltaReqInput)
        ):
            async with self.manager.is_pause_cond:
                if isinstance(obj, UpdateWeightsFromDeltaReqInput):
                    self.manager.is_pause = True
                result = await self._request(obj)
                if (
                    isinstance(obj, ResumeWeightsFromDeltaReqInput)
                    and result["success"]
                ):
                    self.manager._update_weight_version_if_provided(
                        str(result["participants"][0]["target_version"])
                    )
                    self.manager.is_pause = False
                    self.manager.is_pause_cond.notify_all()
                return result
        # Preparation/status do not gate generation or acquire its writer lock.
        return await self._request(obj)

    async def _load(self, manifest_path, release_state):
        path = Path(manifest_path).resolve(strict=True)
        content = await asyncio.to_thread(path.read_bytes)
        manifest = orjson.loads(content)
        if manifest["base_version"] != 0:
            raise ValueError("standalone delta loading requires an HF-base publication")
        described = await self._request(
            GetWeightsDeltaInfoReqInput(engine_id="standalone-" + uuid.uuid4().hex)
        )
        if not described["success"]:
            return described
        if any(
            rank["version"] != 0 or rank["state"] not in {"IDLE", "CLEARED"}
            for rank in described["participants"]
        ):
            raise ValueError(
                "standalone delta loading requires freshly loaded base weights"
            )
        session_id = uuid.uuid4().hex
        prepare = PrepareWeightsFromDeltaReqInput(
            session_id=session_id,
            manifest_path=str(path),
            manifest_sha256=hashlib.sha256(content).hexdigest(),
            stream_id=manifest["stream_id"],
            base_version=manifest["base_version"],
            target_version=manifest["target_version"],
            plan_digest=manifest["plan_digest"],
            participants=[rank["identity"] for rank in described["participants"]],
        )
        del manifest, content
        # Preparation can fail safely while the engine still serves base weights.
        # After update dispatch, neither cancellation nor failure may replay XOR
        # or reopen admission: a participant may already have modified weights.
        try:
            result = await asyncio.wait_for(self._prepare(prepare), timeout=1800)
        except BaseException:
            await self._request(AbortWeightsFromDeltaReqInput(session_id=session_id))
            raise
        if not result["success"]:
            await self._request(AbortWeightsFromDeltaReqInput(session_id=session_id))
            return result
        result = await self.request(
            UpdateWeightsFromDeltaReqInput(session_id=session_id)
        )
        if not result["success"]:
            return result
        result = await self.request(
            ResumeWeightsFromDeltaReqInput(session_id=session_id)
        )
        if not result["success"]:
            return result
        return await self._clear() if release_state else result

    async def _prepare(self, request):
        result = await self._request(request)
        while result["success"] and any(
            rank["state"] != "PREPARED" for rank in result["participants"]
        ):
            await asyncio.sleep(0.05)
            result = await self._request(
                GetWeightsDeltaStatusReqInput(session_id=request.session_id)
            )
        return result

    async def _clear(self):
        result = await self._request(ClearWeightsDeltaStateReqInput())
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
            ReleaseWeightsDeltaCacheReqInput(owner_rank_ids=list(owners.values()))
        )

    async def _request(self, obj):
        self.manager.auto_create_handle_loop()
        if get_serving().tokenizer_worker_num != 1:
            return {
                "success": False,
                "message": "GPU delta requires one Python tokenizer worker",
                "participants": [],
            }
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

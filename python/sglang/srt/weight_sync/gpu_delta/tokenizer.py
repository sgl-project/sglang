"""Control plane for engines exclusively owned by the GPU-delta coordinator."""

from sglang.srt.runtime_context import get_serving
from sglang.srt.weight_sync.gpu_delta.io import (
    DeltaWeightsReqOutput,
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

"""Tokenizer-side delta coordination, separate from ordinary control APIs."""

from sglang.srt.runtime_context import get_serving
from sglang.srt.weight_sync.gpu_delta_session import (
    GpuDeltaCommunicator,
    GpuDeltaConflict,
    guard_tokenizer_dispatch,
)


class GpuDeltaTokenizerControl:
    def __init__(self, manager, fan_out: int):
        from sglang.srt.managers.io_struct import DeltaWeightsReqOutput
        from sglang.utils import TypeBasedDispatcher

        self.manager = manager
        self.session_id = None
        self.participants = None
        self.communicator = GpuDeltaCommunicator(
            manager._dispatch_to_scheduler, fan_out
        )
        manager._result_dispatcher += TypeBasedDispatcher(
            [(DeltaWeightsReqOutput, self.communicator.handle_recv)]
        )

    def guard_dispatch(self, obj):
        guard_tokenizer_dispatch(self, obj)

    async def request(self, obj, request=None):
        """No model-update writer lock: preparation must overlap generation."""
        self.manager.auto_create_handle_loop()
        import json

        from sglang.srt.managers.io_struct import (
            AbortWeightsFromDeltaReqInput,
            GetWeightsDeltaInfoReqInput,
            PrepareWeightsFromDeltaReqInput,
        )

        if get_serving().tokenizer_worker_num != 1:
            return {
                "success": False,
                "message": "GPU delta requires one Python tokenizer worker",
                "participants": [],
            }
        if isinstance(obj, PrepareWeightsFromDeltaReqInput):
            identities = self.participants
            if identities is None or {
                json.dumps(item, sort_keys=True) for item in identities
            } != {json.dumps(item, sort_keys=True) for item in obj.participants}:
                return {
                    "success": False,
                    "message": "prepare must bind the original described participants",
                    "participants": [],
                }
        results = await self.communicator(obj)
        participants = [result.participant for result in results]
        identities = [item.get("identity") for item in participants]

        keys = [json.dumps(identity, sort_keys=True) for identity in identities]
        success = all(result.success for result in results)
        if len(set(keys)) != len(keys) or any(
            identity is None for identity in identities
        ):
            success = False
        expected_identities = self.participants
        if (
            not isinstance(obj, GetWeightsDeltaInfoReqInput)
            and expected_identities is not None
        ):
            success &= set(keys) == {
                json.dumps(item, sort_keys=True) for item in expected_identities
            }
        if hasattr(obj, "session_id"):
            success &= all(
                item.get("session_id") == obj.session_id for item in participants
            )
        phase_states = {
            "GetWeightsDeltaInfoReqInput": {"IDLE"},
            "UpdateWeightsFromDeltaReqInput": {"APPLIED", "COMMITTED", "RESUMED"},
            "CommitWeightsFromDeltaReqInput": {"COMMITTED", "RESUMED"},
            "ContinueWeightsFromDeltaReqInput": {"RESUMED"},
            "AbortWeightsFromDeltaReqInput": {"ABORTED"},
        }
        allowed = phase_states.get(type(obj).__name__)
        if allowed is not None:
            success &= all(item.get("state") in allowed for item in participants)
        if isinstance(obj, GetWeightsDeltaInfoReqInput) and success:
            self.participants = identities
        if (
            isinstance(obj, AbortWeightsFromDeltaReqInput)
            and success
            and all(item["state"] == "ABORTED" for item in participants)
            and self.session_id == obj.session_id
        ):
            self.session_id = None
        return {
            "success": success,
            "message": " | ".join(
                result.message for result in results if result.message
            ),
            "participants": participants,
        }

    async def before_pause(self, obj):
        if self.session_id:
            from sglang.srt.managers.io_struct import GetWeightsDeltaStatusReqInput

            if obj.mode != "retract":
                raise GpuDeltaConflict("GPU delta requires retract pause")
            result = await self.request(
                GetWeightsDeltaStatusReqInput(
                    session_id=self.session_id,
                )
            )
            if not result["success"] or any(
                item["state"] not in {"PREPARED", "QUIESCED"}
                for item in result["participants"]
            ):
                raise GpuDeltaConflict(
                    "all original ranks must be prepared before pause"
                )

    async def resume(self, obj):
        if self.session_id and obj.delta_session_id is None:
            raise GpuDeltaConflict(
                "GPU delta session requires a global commit certificate before resume"
            )
        if obj.delta_session_id is not None:
            from sglang.srt.managers.io_struct import ContinueWeightsFromDeltaReqInput

            if not obj.delta_commit_receipts:
                raise GpuDeltaConflict(
                    "GPU delta resume requires every engine's committed receipts"
                )
            result = await self.request(
                ContinueWeightsFromDeltaReqInput(
                    session_id=obj.delta_session_id,
                    receipts=obj.delta_commit_receipts,
                )
            )
            if not result["success"]:
                raise GpuDeltaConflict(result["message"] or "GPU delta resume rejected")
            if self.session_id == obj.delta_session_id:
                self.session_id = None
            self.manager._update_weight_version_if_provided(
                str(result["participants"][0]["target_version"])
            )
            return {"success": True, "participants": result["participants"]}

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.orchestration.spmd.entrypoint import Entrypoint
from sglang.srt.server.engine_base import EngineBase


class EngineFragment(EngineBase):
    def __init__(
            self,
            tp_rank: int,
            gpu_id: int,
    ):
        self._entrypoint = Entrypoint()

    def _generate_impl(self, obj: GenerateReqInput):
        return self._entrypoint.generate(obj)

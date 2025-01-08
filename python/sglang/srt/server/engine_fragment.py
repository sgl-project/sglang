from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.orchestration.spmd.entrypoint import Entrypoint
from sglang.srt.server.engine_base import EngineBase
from sglang.srt.server_args import ServerArgs


class EngineFragment(EngineBase):
    def __init__(
            self,
            nccl_port: int,
            gpu_id: int,
            tp_rank: int,
            log_level: str = "error",
            *args,
            **kwargs,
    ):
        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        self._entrypoint = Entrypoint(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )

    def _generate_impl(self, obj: GenerateReqInput):
        return self._entrypoint.generate(obj)

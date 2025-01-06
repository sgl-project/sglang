from typing import Union

from fastapi import Request
from sglang.srt.managers.io_struct import GenerateReqInput, EmbeddingReqInput
from sglang.srt.managers.scheduler.core import SchedulerCore
from sglang.srt.server.engine_base import EngineBase
from sglang.srt.server_args import ServerArgs


# TODO rename this class
class EngineFragment(EngineBase):
    """
    Similar to `Engine`. The difference is that, `Engine` handles TP internally, thus users only need
    to have one single `Engine`. Contrary to that, users need to have one `EngineFragment` per TP rank.
    """

    def __init__(
        self,
        log_level: str = "error",
        *args,
        nccl_port: int,  # TODO maybe hide this into an opaque struct etc from API
        gpu_id: int, # TODO do we need ALL these several args?
        tp_rank: int,
        dp_rank: int,
        **kwargs,
    ):
        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        self._scheduler_core = SchedulerCore(
            server_args=server_args, nccl_port=nccl_port,
            gpu_id=gpu_id, tp_rank=tp_rank, dp_rank=dp_rank,
        )
        self._scheduler_core.callback = TODO

    async def _generate_request_impl(self, obj: Union[GenerateReqInput, EmbeddingReqInput], request: Request):
        self._scheduler_core.handle_generate_request(TODO)
        self._scheduler_core.handle_embedding_request(TODO)
        return TODO

    def _create_abort_task_impl(self, obj: GenerateReqInput):
        return None  # not supported yet

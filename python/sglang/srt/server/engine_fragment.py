from fastapi import Request
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server.engine_base import EngineBase


# TODO rename this class
class EngineFragment(EngineBase):
    """
    Similar to `Engine`. The difference is that, `Engine` handles TP internally, thus users only need
    to have one single `Engine`. Contrary to that, users need to have one `EngineFragment` per TP rank.
    """

    def __init__(self, log_level: str = "error", *args, **kwargs):
        TODO

    async def _generate_request_impl(self, obj: GenerateReqInput, request: Request):
        return TODO

    def _create_abort_task_impl(self, obj: GenerateReqInput):
        return None  # not supported yet

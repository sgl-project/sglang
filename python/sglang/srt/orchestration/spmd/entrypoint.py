from sglang.srt.managers.io_struct import GenerateReqInput, BatchTokenIDOut
from sglang.srt.managers.scheduler import Scheduler, SchedulerCallback
from sglang.srt.server_args import ServerArgs


class Entrypoint:
    def __init__(
        self,
        server_args: ServerArgs,
        nccl_port: int,
        gpu_id: int,
        tp_rank: int,
    ):
        self._scheduler = Scheduler(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=None,
        )

    def generate(self, obj: GenerateReqInput):
        def _handle_scheduler_output(out: BatchTokenIDOut):
            TODO

        self._scheduler.callback = SchedulerCallback(on_generation_output=_handle_scheduler_output)

        tokenized_requests = TODO
        for tokenized_request in tokenized_requests:
            self._scheduler.handle_generate_request(tokenized_request)

        while TODO:
            self._scheduler.process_batch()

        return TODO

from typing import List

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.generation_manager import GenerationConverter
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
        self._detokenizer = DetokenizerManager(server_args)
        self._generation_converter = GenerationConverter(server_args, model_config=TODO)

    def generate(self, obj: GenerateReqInput):
        outputs: List[TODO] = []

        def _handle_scheduler_output(batch_token_id_out: BatchTokenIDOut):
            batch_str_out = self._detokenizer.handle_batch_token_id_out(batch_token_id_out)
            self._generation_converter.postprocess_response(batch_str_out, index=TODO, rid=TODO, req_obj=TODO)
            TODO

        self._scheduler.callback = SchedulerCallback(on_generation_output=_handle_scheduler_output)

        obj.normalize_batch_and_arguments()
        for i in range(obj.batch_size):
            TODO_await
            tokenized_request = self._generation_converter.tokenize_request(obj[i])
            self._scheduler.handle_generate_request(tokenized_request)

        while True:
            has_batch = self._scheduler.process_batch()
            if not has_batch:
                break

        return outputs

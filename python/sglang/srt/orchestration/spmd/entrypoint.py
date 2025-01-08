from typing import List, Any, Dict

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
        self._generation_converter = GenerationConverter(server_args)
        self._detokenizer = DetokenizerManager(server_args)

    def generate(self, obj: GenerateReqInput):
        obj.normalize_batch_and_arguments()
        tokenized_requests = self._generation_converter.tokenize_requests(obj)
        rid_to_req_index = {r.rid: i for i, r in enumerate(tokenized_requests)}

        outputs: List[Dict[str, Any]] = [None] * obj.batch_size

        def _handle_scheduler_output(batch_token_id_out: BatchTokenIDOut):
            batch_str_out = self._detokenizer.handle_batch_token_id_out(batch_token_id_out)
            for output_index in range(len(batch_str_out)):
                req_index = rid_to_req_index[batch_str_out.rids[output_index]]
                outputs[req_index] = self._generation_converter.postprocess_response(
                    batch_str_out, index=output_index, req_obj=obj[req_index])

        self._scheduler.callback = SchedulerCallback(on_generation_output=_handle_scheduler_output)

        for tokenized_request in tokenized_requests:
            self._scheduler.handle_generate_request(tokenized_request)

        while self._scheduler.process_batch():
            pass

        return outputs

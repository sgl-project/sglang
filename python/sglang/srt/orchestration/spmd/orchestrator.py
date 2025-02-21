from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.generation_manager import GenerationConverter
from sglang.srt.managers.io_struct import BatchTokenIDOut, GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs


class SpmdOrchestrator:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: Optional[int],
        tp_rank: int,
        parallel_process_groups: Optional[ParallelProcessGroups],
    ):
        if parallel_process_groups is None:
            assert nccl_port is not None and tp_rank is not None
        else:
            assert (
                nccl_port is None and tp_rank is None
            ), "When passing `parallel_process_groups`, there is no need to pass `nccl_port` and `tp_rank`"
            tp_rank = parallel_process_groups.tp.device_mesh_device.get_local_rank()

        self._scheduler = Scheduler(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=None,
            parallel_process_groups=parallel_process_groups,
        )
        self._generation_converter = GenerationConverter(server_args)
        self._detokenizer = DetokenizerManager(server_args)

    def generate(self, obj: GenerateReqInput):
        obj.normalize_batch_and_arguments()
        tokenized_requests = self._generation_converter.tokenize_requests(obj)
        rid_to_req_index = {r.rid: i for i, r in enumerate(tokenized_requests)}

        outputs: List[Dict[str, Any]] = [None] * obj.batch_size

        def _handle_scheduler_output(batch_token_id_out: BatchTokenIDOut):
            batch_str_out = self._detokenizer.handle_batch_token_id_out(
                batch_token_id_out
            )
            for output_index in range(len(batch_str_out.rids)):
                req_index = rid_to_req_index[batch_str_out.rids[output_index]]
                outputs[req_index] = self._generation_converter.postprocess_response(
                    batch_str_out, index=output_index, req_obj=obj[req_index]
                )

        self._scheduler.on_generation_output = _handle_scheduler_output

        for tokenized_request in tokenized_requests:
            self._scheduler.handle_generate_request(tokenized_request)

        while self._scheduler.process_batch():
            pass

        return outputs

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
    ):
        self._scheduler.update_weights_from_tensor_raw(named_tensors, load_format)

    def shutdown(self):
        self._scheduler.shutdown()

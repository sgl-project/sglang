from typing import Type
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.schedule_batch import ScheduleBatch, Req


class SpeculativeWorker(TpModelWorker):
    is_draft_worker = True
    
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        target_worker: TpModelWorker
    ): 
        super().__init__(gpu_id=gpu_id, tp_rank=tp_rank, server_args=server_args, nccl_port=nccl_port)
        self.target_worker = target_worker
    
    def forward_batch_speculative_generate(self, batch: ScheduleBatch):
        raise NotImplementedError()
    
    def finish_request(self, req: Req):
        raise NotImplementedError()
    
class SpecWorkerFactory:
    def __init__(self):
        self.factory = {}

    def register(self, name: str) -> SpeculativeWorker:
        def wrapper(info: Type[SpeculativeWorker]) -> Type[SpeculativeWorker]:
            self.factory[name] = info
            return info

        return wrapper

    def get(self, name):
        return self.factory[name]
    
spec_worker_factory = SpecWorkerFactory()
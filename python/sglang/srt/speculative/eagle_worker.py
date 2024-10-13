import torch
from sglang.srt.speculative.speculative_worker import SpeculativeWorker, spec_worker_factory
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.speculative_utils import DraftInfoFactory

@spec_worker_factory.register('EAGLE')
class EAGLEWorker(SpeculativeWorker):
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        target_worker: TpModelWorker
    ): 
        super().__init__(gpu_id=gpu_id, tp_rank=tp_rank, server_args=server_args, nccl_port=nccl_port, target_worker=target_worker)
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.model_runner.model.set_embed_and_head(embed, head)
        
    
    
    def forward_draft_decode(self, batch: ScheduleBatch):
        print('** start decode **')
        batch.spec_info.prepare_for_decode(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)
        self.model_runner.forward(forward_batch)
    
    def forward_draft_extend(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)  
        forward_batch.spec_info.prepare_for_extend(forward_batch)
        logits_output = self.model_runner.forward(forward_batch)
        next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        model_worker_batch.spec_info.verified_id = next_token_ids

    def forward_batch_speculative_generate(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            for i in range(self.server_args.num_speculative_steps):
                self.forward_draft_decode(batch)
                
            model_worker_batch = batch.get_model_worker_batch()
            self.forward_batch_generation(model_worker_batch)
            return self.draft_worker.verify(model_worker_batch)
        else:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.spec_info = DraftInfoFactory.get(model_worker_batch.spec_algorithm)()
            model_worker_batch.spec_info.init(self.server_args)
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(model_worker_batch)
            model_worker_batch.spec_info.verified_id = next_token_ids
            self.forward_draft_extend(model_worker_batch)
            batch.spec_info = model_worker_batch.spec_info
            return logits_output, next_token_ids, model_worker_batch.spec_info
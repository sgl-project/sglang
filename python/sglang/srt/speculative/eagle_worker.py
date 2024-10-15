import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.speculative_worker import SpeculativeWorker, spec_worker_factory
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.speculative_utils import EAGLEDraftInput, EagleVerifyInput
from sglang.srt.model_executor.model_runner import ModelRunner

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
        batch.spec_info.prepare_for_decode(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.is_draft_batch = True
        self.model_runner.forward(forward_batch)
    
    def forward_draft_extend(self, batch: ScheduleBatch):
        self._swap_mem_pool(batch, self.model_runner)
        batch.spec_info.prepare_for_extend(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)  
        logits_output = self.model_runner.forward(forward_batch)
        self._swap_mem_pool(batch, self.target_worker.model_runner)

    def forward_batch_speculative_generate(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            self._swap_mem_pool(batch, self.model_runner)
            for i in range(self.server_args.num_speculative_steps):
                self.forward_draft_decode(batch)
            batch.spec_info.clear_draft_cache(batch)
            self._swap_mem_pool(batch, self.target_worker.model_runner)
            next_draft_input, logits_output = self.verify(batch)
            verified_id = next_draft_input.verified_id
            next_draft_input.init(self.server_args)
            batch.spec_info = next_draft_input
            self.forward_extend_after_decode(batch)
            return logits_output, verified_id
            
        else:
            batch.spec_info = EAGLEDraftInput()
            batch.spec_info.init(self.server_args)
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(model_worker_batch)
            model_worker_batch.spec_info.verified_id = next_token_ids
            self.forward_draft_extend(batch)
            return logits_output, next_token_ids
        
    def verify(self, batch: ScheduleBatch):
        verify_input = batch.spec_info.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.SPECVERIFY
        verify_input.prepare_for_verify(batch)
        batch.spec_info = verify_input
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, next_token_ids = self.target_worker.forward_batch_generation(model_worker_batch)
        res = verify_input.verify(batch, logits_output)
        batch.forward_mode = ForwardMode.DECODE
        return res
        
    def _swap_mem_pool(self, batch: ScheduleBatch, runner: ModelRunner):
        batch.token_to_kv_pool = runner.token_to_kv_pool
        batch.req_to_token_pool = runner.req_to_token_pool
        
    def forward_extend_after_decode(self, batch: ScheduleBatch):
        self._swap_mem_pool(batch, self.model_runner)
        batch.forward_mode = ForwardMode.SPECEXTEND 
        batch.spec_info.prepare_extend_after_decode(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.is_draft_batch = True
        logits_output = self.model_runner.forward(forward_batch)
        batch.forward_mode = ForwardMode.DECODE 
        self._swap_mem_pool(batch, self.model_runner)
        
    def post_decode_process(self, batch):
        return self.forward_extend_after_decode(batch)

        
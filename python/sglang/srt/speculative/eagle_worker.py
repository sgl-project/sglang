import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.speculative_worker import SpeculativeWorker, spec_worker_factory
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch, Req
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EAGLEDraftInput, EagleVerifyInput
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
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
        disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        super().__init__(gpu_id=gpu_id, tp_rank=tp_rank, server_args=server_args, nccl_port=nccl_port, target_worker=target_worker)
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.model_runner.model.set_embed_and_head(embed, head)
        self.model_runner.server_args.disable_cuda_graph = disable_cuda_graph
        self.model_runner.init_cuda_graphs()
        self.finish_extend_len = None
    
    def forward_draft_decode(self, batch: ScheduleBatch):
        batch.spec_info.prepare_for_decode(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.is_draft_batch = True
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
    
    def forward_draft_extend(self, batch: ScheduleBatch):
        self._swap_mem_pool(batch, self.model_runner)
        batch.spec_info.prepare_for_extend(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)  
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        self._swap_mem_pool(batch, self.target_worker.model_runner)

    def forward_batch_speculative_generate(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            self._swap_mem_pool(batch, self.model_runner)
            for i in range(self.server_args.num_speculative_steps):
                self.forward_draft_decode(batch)
            batch.spec_info.clear_draft_cache(batch)
            self._swap_mem_pool(batch, self.target_worker.model_runner)
            next_draft_input, logits_output, verified_id, self.finish_extend_len = self.verify(batch)
            print('aval', self.model_runner.token_to_kv_pool.available_size())
            next_draft_input.init(self.server_args)
            batch.spec_info = next_draft_input
            # if it is None, means all requsets are finished
            if batch.spec_info.verified_id is not None:
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
        logits_output, _ = self.target_worker.forward_batch_generation(model_worker_batch, need_token_id=False)
        res = verify_input.verify(batch, logits_output)
        batch.forward_mode = ForwardMode.DECODE
        return res
        
    def _swap_mem_pool(self, batch: ScheduleBatch, runner: ModelRunner):
        batch.token_to_kv_pool = runner.token_to_kv_pool
        batch.req_to_token_pool = runner.req_to_token_pool
        
    def forward_extend_after_decode(self, batch: ScheduleBatch):
        self._swap_mem_pool(batch, self.model_runner)
        batch.forward_mode = ForwardMode.SPECEXTEND 
        if batch.spec_info.has_finished:
            index = batch.spec_info.unfinished_index
            seq_lens = batch.seq_lens
            batch.seq_lens = batch.seq_lens[index]
        batch.spec_info.prepare_extend_after_decode(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        
        forward_batch.is_draft_batch = True
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        batch.forward_mode = ForwardMode.DECODE 
        if batch.spec_info.has_finished:
            batch.seq_lens = seq_lens
        self._swap_mem_pool(batch, self.model_runner)

    def capture_for_decode(self, hidden_states, forward_batch):
        # lm head is not support cuda graph currently. But it could be support theoretically.
        # TODO: Support it. @kavioyu
        logits_output = self.model_runner.model.logits_processor(
            None, hidden_states, self.model_runner.model.lm_head.weight, forward_batch
        )
        if isinstance(logits_output, LogitsProcessorOutput):
            logits = logits_output.next_token_logits
        sample_output = torch.softmax(
            logits, dim=-1
        )  # TODO: Support more sampling method @kavioyu
        forward_batch.spec_info.capture_for_decode(
            sample_output, forward_batch.forward_mode
        )
    
    # Don't support prefix share now.
    def finish_request(self, req):
        req_len = len(req.origin_input_ids) + len(req.output_ids) - self.finish_extend_len[req.rid] - 1
        kv_indices = self.model_runner.req_to_token_pool.req_to_token[
            req.req_pool_idx
        ][:req_len]
        self.model_runner.token_to_kv_pool.free(kv_indices)
        self.model_runner.req_to_token_pool.free(req.req_pool_idx)
        
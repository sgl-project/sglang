"""The dLLM model worker: the target worker's model under a dLLM algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.managers.tp_worker import BaseTpWorker
from sglang.srt.managers.utils import GenerationBatchResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs


class DllmWorker(BaseTpWorker):
    """Runs a dLLM step: the algorithm denoises the step's blocks with one or
    more forwards of the target worker's model."""

    def __init__(self, target_worker: TpModelWorker, server_args: ServerArgs):
        self.target_worker = target_worker
        self.dllm_algorithm = DllmAlgorithm.from_server_args(server_args)

    @property
    def model_runner(self) -> ModelRunner:
        return self.target_worker.model_runner

    def get_worker_info(self):
        (
            max_total_num_tokens,
            max_prefill_tokens,
            max_running_requests,
            max_queued_requests,
            max_req_len,
            max_req_input_len,
            *rest,
        ) = self.target_worker.get_worker_info()
        # A request needs room for one block past its input.
        max_req_input_len -= self.dllm_algorithm.block_size
        return (
            max_total_num_tokens,
            max_prefill_tokens,
            max_running_requests,
            max_queued_requests,
            max_req_len,
            max_req_input_len,
            *rest,
        )

    def forward_model(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult:
        batch, forward_batch = self.target_worker.init_forward_batch(batch)

        algo_states = None
        if self.dllm_algorithm.fdfo and batch is not None:
            algo_states = [req.dllm.algo_state for req in batch.reqs]

        (
            logits_output,
            next_token_ids,
            accept_length_per_req_cpu,
            dllm_algo_state,
            can_run_cuda_graph,
        ) = self.dllm_algorithm.run(self.model_runner, forward_batch, algo_states)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            dllm_algo_state=dllm_algo_state,
            can_run_cuda_graph=can_run_cuda_graph,
        )

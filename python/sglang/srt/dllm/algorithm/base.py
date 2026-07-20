from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs

DllmRunOutput = Tuple[
    Union[LogitsProcessorOutput, torch.Tensor],
    List,
    Optional[List[int]],
    Optional[List[Any]],
    bool,
]


class DllmAlgorithm:
    """dLLM algorithm: subclasses implement ``step``; the base owns the
    synchronous and FDFO (``--dllm-fdfo``) execution loops in ``run``.
    """

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.config = config
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        self.fdfo = config.first_done_first_out_mode

    def select_block_size(self, running_bs: int) -> int:
        return self.config.select_block_size(running_bs)

    @staticmethod
    def from_server_args(server_args: "ServerArgs"):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [None] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        return block_size + 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """One denoise step, advancing ``forward_batch.input_ids``/``states`` in
        place. Returns, per block, whether it was already complete *on entry* --
        i.e. this forward persisted its final KV cache and it can be emitted.
        """
        raise NotImplementedError

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        if self.fdfo:
            return self._run_fdfo(model_runner, forward_batch, algo_states)
        return self._run_sync(model_runner, forward_batch)

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        return (input_ids != self.mask_id).sum(dim=1).tolist()

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        start_list = self._block_start_list(forward_batch)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # No mask to denoise: return empty so process_batch_result_dllm skips the
        # stream branch (matches the pre-refactor behavior).
        if all(start == self.block_size for start in start_list):
            return out.logits_output, [], None, None, out.can_run_graph

        states = self.init_step_state(forward_batch)
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            if all(done):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        return out.logits_output, next_token_ids_list, None, None, out.can_run_graph

    def _run_fdfo(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]],
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size

        if algo_states is None:
            algo_states = [None] * batch_size
        fresh: Optional[List[Any]] = None
        states: List[Any] = []
        for i, carried in enumerate(algo_states):
            if carried is None:
                if fresh is None:
                    fresh = self.init_step_state(forward_batch)
                states.append(fresh[i])
            else:
                states.append(carried)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        done = self.step(forward_batch, out.logits_output.full_logits, states)

        accept_length_per_req_cpu = [self.block_size if d else 0 for d in done]
        next_token_ids_list = forward_batch.input_ids.view(
            batch_size, self.block_size
        ).tolist()
        states_out = [None if done[i] else states[i] for i in range(batch_size)]

        return (
            out.logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            states_out,
            out.can_run_graph,
        )

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs

# run() return value:
#   (logits_output, next_token_ids, accept_length_per_req_cpu, algo_states, can_run_cuda_graph)
# accept_length_per_req_cpu and algo_states are only populated in FDFO mode.
DllmRunOutput = Tuple[
    Union[LogitsProcessorOutput, torch.Tensor],
    List,
    Optional[List[int]],
    Optional[List[Any]],
    bool,
]


class DllmAlgorithm:
    """Base class for diffusion LLM (dLLM) denoising algorithms.

    A concrete algorithm only implements the single-step token-selection strategy
    (``step``); the base owns both execution modes: synchronous (denoise a whole
    block per ``run``) and FDFO (``--dllm-fdfo``, one step per ``run``, yielding to
    the scheduler so finished requests leave the batch immediately). Cross-step
    state is built in ``init_step_state`` and carried across FDFO rounds via
    ``Req.dllm_algo_state``.
    """

    def __init__(self, config: DllmConfig):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        self.fdfo = config.first_done_first_out_mode

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)

    # ------------------------------------------------------------------
    # Strategy: implemented by concrete algorithms.
    # ------------------------------------------------------------------
    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        """Per-block initial cross-step state (one ``None`` per block if stateless)."""
        return [None] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        """Max denoise steps per block, including the trailing completion step."""
        return block_size + 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Run one denoise step, advancing ``forward_batch.input_ids``/``states`` in
        place. Returns, per block, whether it was already complete *on entry* --
        i.e. this forward pass persisted its final KV cache and it can be emitted.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Execution: provided by the base class, shared by all algorithms.
    # ------------------------------------------------------------------
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
        """Number of already-filled (non-mask) tokens at the start of each block."""
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        return (input_ids != self.mask_id).sum(dim=1).tolist()

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        start_list = self._block_start_list(forward_batch)
        states = self.init_step_state(forward_batch)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            if all(done):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        # Per-request slice of newly generated tokens (tensors, as the synchronous
        # process_batch_result_dllm path expects).
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

        # Fresh blocks (first decode round) carry no state; initialise them.
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

        # Complete-on-entry blocks have their final KV persisted; accept the whole
        # block. Otherwise carry the in-progress state to the next round.
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

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu

_is_npu = is_npu()


class DllmRunOutput(NamedTuple):
    logits_output: Union[LogitsProcessorOutput, torch.Tensor]
    block_tokens: torch.Tensor  # [batch_size, block_size]
    block_done: Optional[torch.Tensor] # [batch_size] bool; FDFO only, true means block KV is ready to commit.
    algo_states: Optional[List[Any]]
    can_run_cuda_graph: bool


class DllmAlgorithm:
    """dLLM algorithm: subclasses implement ``step``; the base owns the
    synchronous and FDFO (``--dllm-fdfo``) execution loops in ``run``.
    """

    def __init__(self, config: DllmConfig):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        self.fdfo = config.first_done_first_out_mode

    @staticmethod
    def from_server_args(server_args: ServerArgs):
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
    ) -> torch.Tensor:
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

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        block_tokens = forward_batch.input_ids.view(batch_size, self.block_size)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if not bool((block_tokens == self.mask_id).any()):
            return DllmRunOutput(
                out.logits_output, block_tokens.clone(), None, None, out.can_run_graph
            )

        states = self.init_step_state(forward_batch)
        # NPU: attention metadata is stable across a block's denoise steps (the
        # first forward above already planned it), so mark it ready once and let
        # every later forward skip re-planning.
        if _is_npu:
            forward_batch.mark_forward_metadata_ready()
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            if bool(done.all()):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        return DllmRunOutput(
            out.logits_output, block_tokens.clone(), None, None, out.can_run_graph
        )

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
        # Clone so a later in-place step cannot race the async D2H of this result.
        block_tokens = forward_batch.input_ids.view(batch_size, self.block_size).clone()

        return DllmRunOutput(
            out.logits_output, block_tokens, done, states, out.can_run_graph
        )

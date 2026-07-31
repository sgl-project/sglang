from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu

_is_npu = is_npu()

_argmax_softmax_prob_fused = None
if _is_npu:
    try:
        from sgl_kernel_npu.sample.argmax_softmax_prob import (
            argmax_softmax_prob_fused as _argmax_softmax_prob_fused,
        )
    except (ImportError, OSError):
        pass

DllmRunOutput = Tuple[
    Union[LogitsProcessorOutput, torch.Tensor],
    List,
    Optional[List[int]],
    Optional[List[Any]],
    bool,
]


def argmax_and_softmax_prob(
    full_logits_2d: torch.Tensor,
    vocab_chunk: int = 16384,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row argmax and its exact softmax probability.

    Three implementations of the same quantity, picked by what the caller has
    already paid for:

    * a fused single-pass kernel when one is available for this device;
    * ``softmax(logits).gather(argmax)`` when the logits are already an fp32
      ``[tokens, vocab]`` tensor -- this is the upstream formulation and is
      bitwise what dLLM did before this path existed;
    * otherwise a max reduction plus a vocab-chunked exp-sum, whose transients
      are only ``[tokens, vocab_chunk]``. This is the path that lets a caller
      keep the logits in lm_head dtype and never materialize the fp32
      ``[tokens, vocab]`` copy that OOMs large dLLM batches. Chunking matters
      on NPU: even ``sum(dtype=float32)`` on a bf16 operand materializes a full
      fp32 cast internally.

    The argmax is identical across all three; the probability differs only by
    exp's precision in the input dtype (2.8e-4 max relative error on bf16
    LLaDA2-shaped logits at vocab 157k, with no threshold-decision flips
    at 0.5).
    """
    if _argmax_softmax_prob_fused is not None:
        # Single-pass fused kernel: one read of the logits vs the 5-6 below.
        return _argmax_softmax_prob_fused(full_logits_2d)

    if full_logits_2d.dtype == torch.float32:
        # The fp32 [tokens, vocab] tensor already exists, so there is nothing
        # to save by re-reducing it in chunks -- and a fused softmax beats the
        # chunked loop. Keeps non-NPU dLLM bitwise on its pre-existing path.
        # Callers that want the memory back hand over lm_head-dtype logits
        # instead, which lands on the chunked branch below.
        argmax_ids = torch.argmax(full_logits_2d, dim=-1)
        prob = torch.gather(
            torch.softmax(full_logits_2d, dim=-1),
            dim=-1,
            index=argmax_ids.unsqueeze(-1),
        ).squeeze(-1)
        return argmax_ids, prob

    row_max, argmax_ids = torch.max(full_logits_2d, dim=-1)
    row_max_col = row_max.unsqueeze(1)
    denom: Optional[torch.Tensor] = None
    for start in range(0, full_logits_2d.shape[1], vocab_chunk):
        shifted = full_logits_2d[:, start : start + vocab_chunk] - row_max_col
        shifted.exp_()
        part = shifted.sum(dim=-1, dtype=torch.float32)
        denom = part if denom is None else denom + part
    return argmax_ids, 1.0 / denom


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
        # NPU: attention metadata is stable across a block's denoise steps (the
        # first forward above already planned it), so mark it ready once and let
        # every later forward skip re-planning.
        if _is_npu:
            forward_batch.mark_forward_metadata_ready()
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

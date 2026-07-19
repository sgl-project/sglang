from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs


@dataclass
class DllmStepMapState:
    step_maps: torch.Tensor
    diffusion_step: int = 0


DllmRunOutput = Tuple[
    Union[LogitsProcessorOutput, torch.Tensor],
    List,
    Optional[List[int]],
    Optional[List[Any]],
    Optional[List[Optional[List[int]]]],
    Optional[List[Optional[DllmStepMapState]]],
    bool,
]


class DllmAlgorithm:
    """dLLM algorithm: subclasses implement ``step``; the base owns the
    synchronous and FDFO (``--dllm-fdfo``) execution loops in ``run``.
    """

    supports_step_maps = False

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
        return_step_maps: Optional[List[bool]] = None,
        step_map_states: Optional[List[Optional[DllmStepMapState]]] = None,
    ) -> DllmRunOutput:
        return_step_maps = return_step_maps or [False] * forward_batch.batch_size
        if len(return_step_maps) != forward_batch.batch_size:
            raise RuntimeError(
                "dLLM return_step_maps batch mismatch: got "
                f"{len(return_step_maps)} flags for {forward_batch.batch_size} requests"
            )
        if any(return_step_maps) and not self.supports_step_maps:
            raise ValueError(
                f"return_step_maps is not supported by {type(self).__name__}; "
                "use the LowConfidence dLLM algorithm"
            )
        if self.fdfo:
            return self._run_fdfo(
                model_runner,
                forward_batch,
                algo_states,
                return_step_maps,
                step_map_states,
            )
        return self._run_sync(model_runner, forward_batch, return_step_maps)

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        return (input_ids != self.mask_id).sum(dim=1).tolist()

    def _run_sync(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        return_step_maps: List[bool],
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        start_list = self._block_start_list(forward_batch)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # No mask to denoise: return empty so process_batch_result_dllm skips the
        # stream branch (matches the pre-refactor behavior).
        if all(start == self.block_size for start in start_list):
            return out.logits_output, [], None, None, None, None, out.can_run_graph

        states = self.init_step_state(forward_batch)
        step_map_states = self._prepare_step_map_states(
            forward_batch, return_step_maps, None
        )
        for _ in range(self.max_steps(self.block_size)):
            mask_before = self._mask_snapshot(forward_batch, step_map_states)
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            self._record_step_maps(
                forward_batch, mask_before, step_map_states, return_step_maps
            )
            if all(done):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        step_maps = self._format_sync_step_maps(
            step_map_states, return_step_maps, start_list
        )
        return (
            out.logits_output,
            next_token_ids_list,
            None,
            None,
            step_maps,
            None,
            out.can_run_graph,
        )

    def _run_fdfo(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]],
        return_step_maps: List[bool],
        carried_step_map_states: Optional[List[Optional[DllmStepMapState]]],
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
        step_map_states = self._prepare_step_map_states(
            forward_batch, return_step_maps, carried_step_map_states
        )
        mask_before = self._mask_snapshot(forward_batch, step_map_states)
        done = self.step(forward_batch, out.logits_output.full_logits, states)
        self._record_step_maps(
            forward_batch, mask_before, step_map_states, return_step_maps
        )

        accept_length_per_req_cpu = [self.block_size if d else 0 for d in done]
        next_token_ids_list = forward_batch.input_ids.view(
            batch_size, self.block_size
        ).tolist()
        states_out = [None if done[i] else states[i] for i in range(batch_size)]
        step_maps = (
            [
                (
                    step_map_states[i].step_maps.tolist()
                    if return_step_maps[i] and done[i]
                    else None
                )
                for i in range(batch_size)
            ]
            if any(return_step_maps)
            else None
        )
        step_map_states_out = (
            [
                None if done[i] or not return_step_maps[i] else step_map_states[i]
                for i in range(batch_size)
            ]
            if any(return_step_maps)
            else None
        )

        return (
            out.logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            states_out,
            step_maps,
            step_map_states_out,
            out.can_run_graph,
        )

    def _prepare_step_map_states(
        self,
        forward_batch: ForwardBatch,
        return_step_maps: List[bool],
        carried_states: Optional[List[Optional[DllmStepMapState]]],
    ) -> Optional[List[Optional[DllmStepMapState]]]:
        if not any(return_step_maps):
            return None
        if carried_states is not None and len(carried_states) != len(return_step_maps):
            raise RuntimeError(
                "dLLM carried step-map state batch mismatch: got "
                f"{len(carried_states)} states for {len(return_step_maps)} requests"
            )

        states: List[Optional[DllmStepMapState]] = []
        for i, enabled in enumerate(return_step_maps):
            if not enabled:
                states.append(None)
                continue
            carried = carried_states[i] if carried_states is not None else None
            states.append(
                carried
                if carried is not None
                else DllmStepMapState(
                    step_maps=torch.full(
                        (self.block_size,),
                        -1,
                        dtype=torch.int32,
                        device=forward_batch.input_ids.device,
                    )
                )
            )
        return states

    def _mask_snapshot(
        self,
        forward_batch: ForwardBatch,
        step_map_states: Optional[List[Optional[DllmStepMapState]]],
    ) -> Optional[torch.Tensor]:
        if step_map_states is None:
            return None
        return (
            forward_batch.input_ids.view(-1, self.block_size) == self.mask_id
        ).clone()

    def _record_step_maps(
        self,
        forward_batch: ForwardBatch,
        mask_before: Optional[torch.Tensor],
        step_map_states: Optional[List[Optional[DllmStepMapState]]],
        return_step_maps: List[bool],
    ) -> None:
        if mask_before is None or step_map_states is None:
            return
        mask_after = forward_batch.input_ids.view(-1, self.block_size) == self.mask_id
        changed = mask_before & ~mask_after
        for i, enabled in enumerate(return_step_maps):
            if not enabled or not bool(changed[i].any()):
                continue
            state = step_map_states[i]
            assert state is not None
            state.diffusion_step += 1
            state.step_maps[changed[i]] = state.diffusion_step

    def _format_sync_step_maps(
        self,
        step_map_states: Optional[List[Optional[DllmStepMapState]]],
        return_step_maps: List[bool],
        start_list: List[int],
    ) -> Optional[List[Optional[List[int]]]]:
        if step_map_states is None:
            return None
        result: List[Optional[List[int]]] = []
        for i, enabled in enumerate(return_step_maps):
            if not enabled:
                result.append(None)
                continue
            state = step_map_states[i]
            assert state is not None
            step_map = state.step_maps[start_list[i] :].tolist()
            if any(step <= 0 for step in step_map):
                raise RuntimeError(
                    "dLLM produced an incomplete step map for request index "
                    f"{i}: {step_map}"
                )
            result.append(step_map)
        return result

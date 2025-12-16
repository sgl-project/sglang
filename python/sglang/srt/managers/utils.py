from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[torch.Tensor] = None
    num_accepted_tokens: int = 0
    accept_length_per_req_cpu: Optional[List[int]] = None
    can_run_cuda_graph: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[FutureIndices] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    # For confidential compute mode - stores Future objects for async D2H copy
    use_confidential_compute: bool = False
    next_token_ids_future: Optional[Future] = None
    next_token_logprobs_future: Optional[Future] = None
    input_token_logprobs_future: Optional[Future] = None
    hidden_states_future: Optional[Future] = None
    accept_lens_future: Optional[Future] = None

    def copy_to_cpu(self, return_logprob: bool):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = (
                    self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
        if self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        self.copy_done.record()

    def copy_to_cpu_confidential(
        self,
        host_copy_executor: ThreadPoolExecutor,
        return_logprob: bool,
    ):
        """Copy tensors to CPU using worker thread for confidential compute mode.

        In confidential compute environments, blocking D2H copies in a worker thread
        can provide better performance by avoiding CUDA synchronization on the main thread.
        """
        self.use_confidential_compute = True

        # Record an event on the current stream for synchronization
        copy_ready = torch.cuda.Event()
        copy_ready.record()

        def _to_host_tensor(
            tensor: torch.Tensor, event: torch.cuda.Event
        ) -> torch.Tensor:
            # Synchronize with the CUDA event before blocking copy
            # synchronize() is intentionally chosen instead of wait();
            # otherwise, the blocking copy will stall subsequent CUDA API
            # calls on the main thread
            event.synchronize()
            return tensor.to("cpu", non_blocking=False)

        # Submit async copy tasks to worker thread
        self.next_token_ids_future = host_copy_executor.submit(
            _to_host_tensor, self.next_token_ids, copy_ready
        )

        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.next_token_logprobs_future = host_copy_executor.submit(
                    _to_host_tensor, self.logits_output.next_token_logprobs, copy_ready
                )
            if self.logits_output.input_token_logprobs is not None:
                self.input_token_logprobs_future = host_copy_executor.submit(
                    _to_host_tensor, self.logits_output.input_token_logprobs, copy_ready
                )

        if self.logits_output.hidden_states is not None:
            self.hidden_states_future = host_copy_executor.submit(
                _to_host_tensor, self.logits_output.hidden_states, copy_ready
            )

        if self.accept_lens is not None:
            self.accept_lens_future = host_copy_executor.submit(
                _to_host_tensor, self.accept_lens, copy_ready
            )

    def resolve_confidential_futures(self):
        """Resolve all Future objects from confidential compute mode D2H copies."""
        if not self.use_confidential_compute:
            return

        if self.next_token_ids_future is not None:
            self.next_token_ids = self.next_token_ids_future.result()
            self.next_token_ids_future = None

        if self.next_token_logprobs_future is not None:
            self.logits_output.next_token_logprobs = (
                self.next_token_logprobs_future.result()
            )
            self.next_token_logprobs_future = None

        if self.input_token_logprobs_future is not None:
            self.logits_output.input_token_logprobs = (
                self.input_token_logprobs_future.result()
            )
            self.input_token_logprobs_future = None

        if self.hidden_states_future is not None:
            self.logits_output.hidden_states = self.hidden_states_future.result()
            self.hidden_states_future = None

        if self.accept_lens_future is not None:
            self.accept_lens = self.accept_lens_future.result()
            self.accept_lens_future = None

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # TODO(lsyin): refactor PP and avoid using dict
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req

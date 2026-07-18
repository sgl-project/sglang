from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.eplb.expert_distribution import ExpertDistributionMetrics
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.state_capturer.base import TopkCaptureOutput

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


def _async_d2h(t: torch.Tensor) -> torch.Tensor:
    """Async D2H copy for overlap scheduling. On CUDA the dest is pinned (a D2H
    to pageable host memory blocks the caller until done) and record_stream keeps
    the source alive until the copy stream drains, so the caching allocator can't
    recycle it early. Non-CUDA falls back to a plain copy."""
    if not t.is_cuda:
        return t.to("cpu", non_blocking=True)
    cpu_t = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    cpu_t.copy_(t, non_blocking=True)
    t.record_stream(torch.cuda.current_stream(t.device))
    return cpu_t


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[
        Union[torch.Tensor, List[torch.Tensor], List[List[int]]]
    ] = None
    num_correct_drafts: int = 0  # no bonus included
    num_correct_drafts_per_req_cpu: Optional[List[int]] = None
    num_block_accept_tokens: int = 0
    num_cap_tokens: int = 0
    # FDFO dLLM batching: per-request accepted block length and carried algo state.
    accept_length_per_req_cpu: Optional[List[int]] = None
    dllm_algo_state: Optional[List[Any]] = None
    step_maps: Optional[List[Optional[List[int]]]] = None
    dllm_step_map_state: Optional[List[Any]] = None
    can_run_cuda_graph: bool = False

    # PP skip output comm: True when output send/recv was skipped and
    # next_token_ids are placeholder zeros. Used by process_batch_result_prefill
    # to validate that skipped output is never consumed.
    skipped_output_comm: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[torch.Tensor] = None
    speculative_num_draft_tokens: Optional[int] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None

    block_accept_lens: Optional[torch.Tensor] = None

    cap_lens: Optional[torch.Tensor] = None

    # Next-iter seq_lens; published via on_publish.
    new_seq_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    # Refs the worker wants scheduler to keep alive for the same 2-iter window
    # as batch_record_buf. Used for cross-stream tensor lifetime (e.g. a spec
    # V2 verify ForwardBatch whose tensors must outlive mid-iter SB rebinds).
    extra_keep_alive_refs: Optional[List[Any]] = None

    # Routed experts: pending async D2H for overlap scheduling
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None

    # metrics
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None

    # Forward pass metrics (FPM) — GPU-accurate timing via CUDA events
    fpm_start_event: Optional[torch.cuda.Event] = None
    fpm_end_event: Optional[torch.cuda.Event] = None

    @property
    def has_sampled_token_ids(self) -> bool:
        """True when this iter sampled token ids; False when none were produced
        this rank/split (a non-last PP rank or a non-final prefill split)."""
        return isinstance(self.next_token_ids, torch.Tensor)

    @torch.profiler.record_function("copy_result_to_cpu")
    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = _async_d2h(
                    self.logits_output.next_token_logprobs
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = _async_d2h(
                    self.logits_output.input_token_logprobs
                )
            if self.logits_output.next_token_top_logprobs_val is not None:
                self.logits_output.next_token_top_logprobs_val = [
                    _async_d2h(v) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_top_logprobs_val
                ]
            if self.logits_output.next_token_top_logprobs_idx is not None:
                self.logits_output.next_token_top_logprobs_idx = [
                    _async_d2h(x) if torch.is_tensor(x) else x
                    for x in self.logits_output.next_token_top_logprobs_idx
                ]
            if self.logits_output.next_token_token_ids_logprobs_val is not None:
                self.logits_output.next_token_token_ids_logprobs_val = [
                    _async_d2h(v) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_token_ids_logprobs_val
                ]
        if return_hidden_states and self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = _async_d2h(
                self.logits_output.hidden_states
            )
        self.next_token_ids = _async_d2h(self.next_token_ids)

        if self.accept_lens is not None:
            self.accept_lens = _async_d2h(self.accept_lens)

        if self.block_accept_lens is not None:
            self.block_accept_lens = _async_d2h(self.block_accept_lens)

        if self.cap_lens is not None:
            self.cap_lens = _async_d2h(self.cap_lens)

        # Sub-objects only declare their device fields; the single copy+safety
        # primitive (_async_d2h: pinned D2H + record_stream) is injected here so
        # all device->host copying and lifetime safety lives in one place.
        for holder in (
            self.routed_experts_output,
            self.indexer_topk_output,
            self.expert_distribution_metrics,
        ):
            if holder is not None:
                holder.map_device_tensors(_async_d2h)

        self.copy_done.record()

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
        "next_token_sampling_mask_idx": result.logits_output.next_token_sampling_mask_idx,
        "next_token_sampling_logprobs": result.logits_output.next_token_sampling_logprobs,
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
        next_token_sampling_mask_idx=next_pp_outputs["next_token_sampling_mask_idx"],
        next_token_sampling_logprobs=next_pp_outputs["next_token_sampling_logprobs"],
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


@dataclass
class EmbeddingBatchResult:
    """Result from an embedding/classification forward pass.

    Attributes:
        embeddings: Model output — pooled embeddings or classification logits.
        pooled_hidden_states: Raw hidden states before the task head.  Present
            only when the batch contained ``return_pooled_hidden_states=True``
            requests.  Tensor (uniform shapes) or list of tensors (MIS).
        copy_done: CUDA event recorded after the async CPU copy completes.
    """

    embeddings: torch.Tensor
    pooled_hidden_states: Optional[torch.Tensor] = None
    copy_done: Optional[torch.cuda.Event] = None

    @property
    def can_run_cuda_graph(self) -> bool:
        return False

    @torch.profiler.record_function("copy_embedding_to_cpu")
    def copy_to_cpu(self):
        """Copy embeddings and pooled hidden states to CPU for overlap scheduling."""
        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = _async_d2h(self.embeddings)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [_async_d2h(emb) for emb in self.embeddings]

        if self.pooled_hidden_states is not None:
            if isinstance(self.pooled_hidden_states, list):
                self.pooled_hidden_states = [
                    _async_d2h(t) for t in self.pooled_hidden_states
                ]
            else:
                self.pooled_hidden_states = _async_d2h(self.pooled_hidden_states)

        self.copy_done.record()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)

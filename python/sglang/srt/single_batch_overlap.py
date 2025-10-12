from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.utils import is_sbo_enabled
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE


class SboFlags:
    # TODO may have: "enable_dispatch_shared_one_stream_overlap", "enable_dispatch_gateup_gemm_two_stream_overlap", ...

    @classmethod
    def enable_combine_down_gemm_two_stream_overlap(cls):
        return (
            is_sbo_enabled()
            # currently only cutedsl backend supports it
            and get_moe_runner_backend().is_flashinfer_cutedsl()
        )

    @classmethod
    def enable_combine_shared_two_stream_overlap(cls):
        return is_sbo_enabled()

    @classmethod
    def fuse_shared_experts_inside_sbo(cls):
        # TODO after antgroup's PR, should be `... or cls.enable_dispatch_shared_one_stream_overlap()`
        return cls.enable_combine_shared_two_stream_overlap()


@dataclass
class CombineOverlapArgs:
    # this "overlap" flag means overlapping with down gemm, not the general two-stream overlap
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: int
    signal: Optional[torch.Tensor] = None
    threshold: int = -1


@dataclass
class DownGemmOverlapArgs:
    num_sms: int
    signal: torch.Tensor
    start_event: torch.cuda.Event


def execute_sbo(
    forward_shared_experts: Callable[[], Any],
    experts: "DeepEPMoE",
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    forward_batch: ForwardBatch,
    alt_stream: Optional = None,
):
    shared_output = None

    dispatch_output = experts.dispatch(
        hidden_states, topk_idx, topk_weights, forward_batch
    )

    combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
        _compute_overlap_args(dispatch_output, alt_stream)
    )

    hidden_states = experts.moe_impl(
        dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
    )
    if (e := meta_overlap_args.get("record_event_after_down")) is not None:
        e.record()

    if SboFlags.enable_combine_shared_two_stream_overlap():
        # TODO reduce sm for non-deepgemm
        with deep_gemm_wrapper.configure_deep_gemm_num_sms(
            meta_overlap_args["compute_num_sms"]
        ):
            shared_output = forward_shared_experts()

    hidden_states = experts.combine(
        hidden_states,
        dispatch_output.topk_idx,
        dispatch_output.topk_weights,
        forward_batch,
        overlap_args=combine_overlap_args,
    )

    return hidden_states, shared_output


def _compute_overlap_args(dispatch_output, alt_stream):
    if not (
        SboFlags.enable_combine_down_gemm_two_stream_overlap()
        or SboFlags.enable_combine_shared_two_stream_overlap()
    ):
        return None, None, {}

    hidden_states = dispatch_output.hidden_states_fp8
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    num_local_experts, num_tokens_static, hidden_dim = hidden_states.shape

    total_num_sms = torch.cuda.get_device_properties(
        device="cuda"
    ).multi_processor_count
    communicate_num_sms = get_int_env_var("SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS", 32)
    compute_num_sms = total_num_sms - communicate_num_sms

    assert alt_stream is not None
    combine_wait_event = torch.cuda.Event()
    combine_overlap_args = CombineOverlapArgs(
        overlap=False,
        num_sms=communicate_num_sms,
        stream=alt_stream,
        wait_event=combine_wait_event,
    )
    meta_overlap_args = dict(
        compute_num_sms=compute_num_sms,
    )
    down_gemm_overlap_args = None

    if SboFlags.enable_combine_down_gemm_two_stream_overlap():
        # TODO use zero_allocator to remove this `torch.zeros` call
        # NOTE ours v2 use uint32 not int32 currently
        combine_signal = torch.zeros(
            num_local_experts, dtype=torch.uint32, device=hidden_states.device
        )

        down_gemm_overlap_args = DownGemmOverlapArgs(
            signal=combine_signal,
            start_event=combine_wait_event,
            num_sms=compute_num_sms,
        )
        combine_overlap_args.overlap = True
        combine_overlap_args.signal = combine_signal
        combine_overlap_args.threshold = compute_num_sms
    else:
        meta_overlap_args |= dict(
            record_event_after_down=combine_wait_event,
        )

    return combine_overlap_args, down_gemm_overlap_args, meta_overlap_args

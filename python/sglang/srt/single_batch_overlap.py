from typing import Optional

import torch

from dataclasses import dataclass

from sglang.srt.utils import get_int_env_var, ceil_div


@dataclass
class CombineOverlapArgs:
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: int
    signal: Optional[torch.Tensor] = None
    block_m: int = -1
    threshold: int = -1


@dataclass
class DownGemmOverlapArgs:
    num_sms: int
    signal: torch.Tensor
    start_event: torch.cuda.Event


def _compute_overlap_args(dispatch_output, alt_stream):
    if not ENABLE_DEEPEP_COMBINE_OVERLAP:
        return None, None, {}

    hidden_states = dispatch_output.hidden_states_fp8
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    num_local_experts, num_tokens_static, hidden_dim = hidden_states.shape
    # TODO do not hardcode
    block_m, block_n = 128, 128

    total_num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    communicate_num_sms = get_int_env_var("SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS", 32)
    compute_num_sms = total_num_sms - communicate_num_sms

    assert alt_stream is not None
    combine_wait_event = torch.cuda.Event()
    combine_overlap_args = dict(
        # this "overlap" flag means overlapping with down gemm, not the general two-stream overlap
        overlap=False,
        num_sms=communicate_num_sms,
        stream=alt_stream,
        wait_event=combine_wait_event,
    )
    meta_overlap_args = dict(
        compute_num_sms=compute_num_sms,
    )
    down_gemm_overlap_args = None

    if ENABLE_DEEPEP_COMBINE_DOWN_GEMM_OVERLAP:
        # TODO use zero_allocator
        combine_signal = torch.zeros(
            # TODO their deepep requires the size to be this large, temp use theirs to avoid changing their code
            #      but should optimize later
            #      this may be better: (num_local_experts, ceil_div(num_tokens_static, block_m)),
            num_local_experts * ceil_div(num_tokens_static, 64),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        down_gemm_overlap_args = dict(
            # TODO after improving DeepEP's `combine_signal`, simplify this
            down_signals=combine_signal[:num_local_experts * ceil_div(num_tokens_static, block_m)].view(
                num_local_experts, ceil_div(num_tokens_static, block_m)),
            down_start_event=combine_wait_event,
            down_sm_count=compute_num_sms,
        )
        combine_overlap_args |= dict(
            overlap=True,
            signal=combine_signal,
            block_m=block_m,
            threshold=ceil_div(hidden_dim, block_n),
        )
    else:
        meta_overlap_args |= dict(
            record_event_after_down=combine_wait_event,
        )

    return combine_overlap_args, down_gemm_overlap_args, meta_overlap_args

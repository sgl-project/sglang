from __future__ import annotations

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import is_peo_enable
from sglang.srt.layers.moe.ep_moe.layer import PeoDeepEPMoE


def forward_overlap_1(
    experts: PeoDeepEPMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    current_stream: torch.cuda.Stream,
):
    global combine_state, dispatch_output
    gemm_stream = torch.cuda.Stream()
    states = list()
    gemm_done_events = list()
    moe_hidden_states = list()

    # dispatch send
    for round_id in range(experts.num_rounds):
        send_num_sms = experts.num_device_sms
        recv_num_sms = experts.num_device_sms if round_id == 0 else experts.num_deepep_sms
        state = experts.dispatcher.dispatch_a_peo(
            hidden_states=hidden_states,
            topk_output=topk_output,
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=send_num_sms,
            recv_num_sms=recv_num_sms,
            hook_use_comm_stream=False,
        )
        states.append(state)

    # dispatch recv and GEMM
    for round_id in range(experts.num_rounds):
        dispatch_output = experts.dispatcher.dispatch_b_peo(
            forward_batch=states[round_id][0],
            inner_state=states[round_id][1],
        )
        gemm_stream.wait_stream(current_stream)
        with torch.cuda.stream(gemm_stream):
            num_experts_per_round = experts.num_experts // experts.num_ranks // experts.num_rounds
            start_idx = num_experts_per_round * round_id
            end_idx = start_idx + num_experts_per_round
            moe_hidden_state = experts.run_moe_core(dispatch_output, start_idx, end_idx)
            moe_hidden_states.append(moe_hidden_state)
            gemm_done_event = torch.cuda.Event()
            gemm_stream.record_event(gemm_done_event)
            gemm_done_events.append(gemm_done_event)

    # combine send
    for round_id in range(experts.num_rounds):
        send_num_sms = experts.num_device_sms if round_id == (experts.num_rounds - 1) else experts.num_deepep_sms
        recv_num_sms = experts.num_device_sms
        current_stream.wait_event(gemm_done_events[round_id])
        combine_state = experts.dispatcher.combine_a_peo(
            hidden_states=moe_hidden_states[round_id],
            topk_idx=dispatch_output.topk_idx,
            topk_weights=dispatch_output.topk_weights,
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=send_num_sms,
            recv_num_sms=recv_num_sms,
        )

        if round_id == experts.num_rounds - 1:
            del experts.down_output

    # combine recv
    combined_x = experts.dispatcher.combine_b_peo(
        forward_batch=combine_state[0], inner_state=combine_state[1])

    current_stream.wait_stream(gemm_stream)
    return combined_x


def forward_overlap_2_3(
    experts,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    current_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
):
    states = list()
    gemm_done_events = list()
    gemm_stream = torch.cuda.Stream()

    global dispatch_output, combine_state
    hook_use_default_stream = experts.overlap_type == 2
    # dispatch and GEMM
    for round_id in range(experts.num_rounds):
        # dispatch send
        send_num_sms = experts.num_device_sms if round_id == 0 else experts.num_deepep_sms
        recv_num_sms = experts.num_deepep_sms if round_id != 0 else experts.num_device_sms if hook_use_default_stream else experts.num_device_sms // 2
        state = experts.dispatcher.dispatch_a_peo(
            hidden_states=hidden_states,
            topk_output=topk_output,
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=send_num_sms,
            recv_num_sms=recv_num_sms,
            hook_use_comm_stream=False,
        )
        states.append(state)

        # dispatch recv
        if hook_use_default_stream:
            dispatch_output = experts.dispatcher.dispatch_b_peo(
                forward_batch=state[0],
                inner_state=state[1],
            )
        else:
            comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(comm_stream):
                dispatch_output = experts.dispatcher.dispatch_b_peo(
                    forward_batch=state[0],
                    inner_state=state[1],
                )
            gemm_stream.wait_stream(comm_stream)

        # GEMM
        gemm_stream.wait_stream(current_stream)
        num_experts_per_round = experts.num_experts // experts.num_ranks // experts.num_rounds
        start_idx = num_experts_per_round * round_id
        end_idx = start_idx + num_experts_per_round
        with torch.cuda.stream(gemm_stream):
            experts.run_moe_core(dispatch_output, start_idx, end_idx)
            gemm_done_event = torch.cuda.Event()
            gemm_stream.record_event(gemm_done_event)
            gemm_done_events.append(gemm_done_event)

    # combine send
    for round_id in range(experts.num_rounds):
        send_num_sms = experts.num_device_sms if round_id == (experts.num_rounds - 1) else experts.num_deepep_sms
        recv_num_sms = experts.num_device_sms
        combine_state = experts.dispatcher.combine_a_peo(
            hidden_states=hidden_states,
            topk_idx=dispatch_output.topk_idx,
            topk_weights=dispatch_output.topk_weights,
            use_expert_overlap=True,
            num_rounds=experts.num_rounds,
            round_id=round_id,
            send_num_sms=send_num_sms,
            recv_num_sms=recv_num_sms,
        )
        current_stream.wait_event(gemm_done_events[round_id])
        if not hook_use_default_stream:
            current_stream.wait_stream(comm_stream)

    # combine recv
    if hook_use_default_stream:
        combined_x, event, hook = experts.dispatcher.combine_b_peo(inner_state=combine_state)
    else:
        comm_stream.wait_stream(current_stream)
        with torch.cuda.stream(comm_stream):
            combined_x, event, hook = experts.dispatcher.combine_b_peo(inner_state=combine_state)
        current_stream.wait_stream(comm_stream)

    current_stream.wait_stream(gemm_stream)
    return combined_x

def forward_overlap_4(
    experts,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    current_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
):
    gemm_done_events = list()
    # dispatch
    comm_stream.wait_stream(current_stream)
    with torch.cuda.stream(comm_stream):
        dispatch_output = experts.dispatch(hidden_states, topk_output)

        # current_stream.wait_stream(comm_stream)
        for round_id in range(experts.num_rounds):
            num_experts_per_round = experts.num_experts // experts.num_ranks // experts.num_rounds
            start_idx = num_experts_per_round * round_id
            end_idx = start_idx + num_experts_per_round
            experts.run_moe_core(dispatch_output, start_idx, end_idx)

            gemm_done_event = torch.cuda.Event()
            comm_stream.record_event(gemm_done_event)
            gemm_done_events.append(gemm_done_event)

    # combine send
    with torch.cuda.stream(current_stream):
        for round_id in range(experts.num_rounds):
            send_num_sms = experts.num_device_sms if round_id == (experts.num_rounds - 1) else experts.num_deepep_sms
            recv_num_sms = experts.num_device_sms
            combine_state = experts.dispatcher.combine_a_peo(
                hidden_states=hidden_states,
                topk_idx=dispatch_output.topk_idx,
                topk_weights=dispatch_output.topk_weights,
                use_expert_overlap=True,
                num_rounds=experts.num_rounds,
                round_id=round_id,
                send_num_sms=send_num_sms,
                recv_num_sms=recv_num_sms,
            )

        # combine recv
        combined_x, event, hook = experts.dispatcher.combine_b_peo(inner_state=combine_state)

    current_stream.wait_stream(comm_stream)
    return combined_x


def execute_peo(
    experts: PeoDeepEPMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
):
    current_stream = torch.cuda.current_stream()

    with torch.cuda.stream(current_stream):
        if experts.overlap_method == 1:
            return forward_overlap_1(
                experts,
                hidden_states,
                topk_output,
                current_stream,
            )
        elif experts.overlap_method == 2 or experts.overlap_method == 3:
            comm_stream = torch.cuda.Stream()
            return forward_overlap_2_3(
                experts,
                hidden_states,
                topk_output,
                current_stream,
                comm_stream,
            )
        elif experts.overlap_method == 4:
            comm_stream = torch.cuda.Stream()
            return forward_overlap_4(
                experts,
                hidden_states,
                topk_output,
                current_stream,
                comm_stream,
            )
        else:
            raise ValueError(f"Invalid overlap_method: {experts.overlap_method}")


# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.utils import is_sbo_enabled
from sglang.srt.utils import is_blackwell


class SboFlags:
    # TODO may have: "enable_dispatch_gateup_gemm_two_stream_overlap", ...

    @classmethod
    def enable_combine_down_gemm_two_stream_overlap(cls):
        return (
            is_sbo_enabled()
            # currently only cutedsl backend supports it
            and (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                or (get_moe_runner_backend().is_deep_gemm() and not is_blackwell())
            )
        )

    @classmethod
    def enable_combine_shared_two_stream_overlap(cls):
        return is_sbo_enabled() and not cls.enable_dispatch_shared_one_stream_overlap()

    @classmethod
    def enable_dispatch_shared_one_stream_overlap(cls):
        return is_sbo_enabled() and not is_blackwell()

    @classmethod
    def fuse_shared_experts_inside_sbo(cls):
        return (
            cls.enable_combine_shared_two_stream_overlap()
            or cls.enable_dispatch_shared_one_stream_overlap()
        )


@dataclass
class CombineOverlapArgs:
    # this "overlap" flag means overlapping with down gemm, not the general two-stream overlap
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: Optional[int] = None
    signal: Optional[torch.Tensor] = None
    block_m: Optional[int] = 64
    threshold: Optional[int] = 0


@dataclass
class DownGemmOverlapArgs:
    num_sms: int
    signal: torch.Tensor
    start_event: torch.cuda.Event


def compute_overlap_args(dispatch_output, alt_stream):
    if not (
        SboFlags.enable_combine_down_gemm_two_stream_overlap()
        or SboFlags.enable_combine_shared_two_stream_overlap()
    ):
        return None, None, {}

    hidden_states = dispatch_output.hidden_states

    num_local_experts, num_tokens_static, hidden_dim = hidden_states.shape

    total_num_sms = torch.cuda.get_device_properties(
        device="cuda"
    ).multi_processor_count

    if envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.is_set():
        communicate_num_sms = envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.get()
    else:
        communicate_num_sms = 32 if is_blackwell() else 3
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
        if is_blackwell():
            combine_signal = torch.zeros(
                num_local_experts, dtype=torch.uint32, device=hidden_states.device
            )
        else:
            MIN_BLOCK_M = 64
            combine_signal_size = num_local_experts * (
                (num_tokens_static + MIN_BLOCK_M - 1) // MIN_BLOCK_M
            )
            combine_signal = torch.zeros(
                combine_signal_size, dtype=torch.int32, device=hidden_states.device
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

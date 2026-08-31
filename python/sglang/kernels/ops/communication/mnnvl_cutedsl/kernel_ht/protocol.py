# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-throughput MNNVL protocol and its two operation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict, cast

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from ..symmetric_buffer import SymmetricBuffer
from .device_kernel import _MoeFinalizeAllReduceRMSNormHTDeviceKernel


@dataclass(frozen=True, slots=True)
class HTFinalizeTuning:
    persistent_ctas: int | None = None
    consumer_threads: int = 512
    vectors_per_thread: int = 2
    stages: int = 6
    reduction_warps: int = 1
    reduction_cta_groups: int | None = None
    rms_token_groups: int = 2
    rms_pipeline_stages: int = 2
    rms_shard_major: bool = False
    enable_pdl: bool = True


@dataclass(frozen=True, slots=True)
class HTAllReduceTuning:
    persistent_ctas: int | None = None
    consumer_threads: int = 512
    vectors_per_thread: int = 2
    stages: int = 2
    reduction_warps: int = 2
    reduction_cta_groups: int | None = None
    rms_token_groups: int = 2
    rms_pipeline_stages: int = 1
    rms_shard_major: bool = False
    enable_pdl: bool = True


HT_FINALIZE_GB300_TP8_H8192_K10_M_LE_2048 = HTFinalizeTuning(
    stages=7,
    reduction_warps=2,
    rms_pipeline_stages=3,
    rms_shard_major=True,
)
HT_FINALIZE_GB300_TP8_H8192_K10_M_GE_2049 = HTFinalizeTuning()
HT_FINALIZE_GB300_TP8_H8192_K10 = HT_FINALIZE_GB300_TP8_H8192_K10_M_GE_2049
HT_FINALIZE_GB300_TP16_H8192_K10 = HTFinalizeTuning(
    stages=7,
    reduction_warps=2,
    rms_pipeline_stages=3,
    rms_shard_major=True,
)
HT_ALL_REDUCE_GB300_TP8_H8192 = HTAllReduceTuning()
HT_ALL_REDUCE_GB300_TP16_H8192 = HTAllReduceTuning()


@dataclass(slots=True)
class HTProtocolState:
    local_contributions: SymmetricBuffer
    prenorm_mailbox: SymmetricBuffer
    routed_ready_counters: SymmetricBuffer
    routed_processed_counters: torch.Tensor
    all_reduce_ready_counters: SymmetricBuffer
    all_reduce_processed_counters: torch.Tensor


class _PathKwargs(TypedDict):
    hidden_size: int
    top_k: int
    capacity_m: int
    write_residual_output: bool


class _HTPath:
    def __init__(
        self,
        *,
        compiled: Any,
        hidden_size: int,
        top_k: int,
        capacity_m: int,
        write_residual_output: bool,
    ) -> None:
        self._compiled = compiled
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.capacity_m = capacity_m
        self.write_residual_output = write_residual_output

    def _outputs(
        self,
        m: int,
        norm_output: torch.Tensor | None,
        residual_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        shape = (m, self.hidden_size)
        device = torch.device("cuda", torch.cuda.current_device())
        if norm_output is None:
            norm_output = torch.empty(shape, dtype=torch.bfloat16, device=device)
        if self.write_residual_output and residual_output is None:
            residual_output = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return norm_output, residual_output

    def _state_buffers(
        self, state: HTProtocolState
    ) -> tuple[SymmetricBuffer, SymmetricBuffer]:
        return state.local_contributions, state.prenorm_mailbox

    def _validate_m(self, m: int) -> None:
        if not 1 <= m <= self.capacity_m:
            raise ValueError(f"m must be in [1, {self.capacity_m}]")


class FinalizeAllReduceRMSNormHTKernel(_HTPath):
    def __call__(
        self,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        shared_output: torch.Tensor | None,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: HTProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_m(m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        local, prenorm = self._state_buffers(state)
        peers = cast(torch.Tensor, state.routed_ready_counters.peer_addresses)
        shared_arg = shared_output if shared_output is not None else norm_output
        residual_arg = residual_source if residual_source is not None else norm_output
        residual_output_arg = (
            residual_output if residual_output is not None else norm_output
        )
        self._compiled(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=self.top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=self.top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(gamma, 16),
            to_cute(local.tensor.flatten(), 16),
            to_cute(prenorm.tensor.flatten(), 16),
            to_cute_dynamic(
                residual_output_arg.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute(peers, 8),
            to_cute(state.routed_ready_counters.tensor, 4),
            to_cute(state.routed_processed_counters.flatten(), 4),
            Int64(cast(int, local.multicast_address)),
            Int64(cast(int, prenorm.multicast_address)),
            Int32(m),
            current_cu_stream(),
        )
        return norm_output, residual_output


class AllReduceRMSNormHTKernel(_HTPath):
    def __call__(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: HTProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_m(m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        local, prenorm = self._state_buffers(state)
        peers = cast(torch.Tensor, state.all_reduce_ready_counters.peer_addresses)
        residual_arg = residual_source if residual_source is not None else norm_output
        residual_output_arg = (
            residual_output if residual_output is not None else norm_output
        )
        index_arg = state.all_reduce_processed_counters.view(torch.int32)
        # top_k=0 disables metadata reads, so the aliased placeholders stay unused.
        self._compiled(
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(local_contribution.flatten(), 2, divisibility=1),
            to_cute_dynamic(index_arg.flatten(), 4, divisibility=1),
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(gamma, 16),
            to_cute(local.tensor.flatten(), 16),
            to_cute(prenorm.tensor.flatten(), 16),
            to_cute_dynamic(
                residual_output_arg.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute(peers, 8),
            to_cute(state.all_reduce_ready_counters.tensor, 4),
            to_cute(state.all_reduce_processed_counters.flatten(), 4),
            Int64(cast(int, local.multicast_address)),
            Int64(cast(int, prenorm.multicast_address)),
            Int32(m),
            current_cu_stream(),
        )
        return norm_output, residual_output


class HTProtocol:
    """Own tuning-independent HT State and both persistent path variants."""

    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        rms_epsilon: float,
        routed_scaling_factor: float,
        weight_bias: float,
        *,
        include_shared_expert: bool,
        add_residual: bool,
        write_residual_output: bool,
        finalize_tunings: tuple[HTFinalizeTuning, ...],
        all_reduce_tunings: tuple[HTAllReduceTuning, ...],
        group: dist.ProcessGroup,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.rms_epsilon = rms_epsilon
        self.routed_scaling_factor = routed_scaling_factor
        self.weight_bias = weight_bias
        self.include_shared_expert = include_shared_expert
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output

        self.finalize_kernels = {
            tuning: FinalizeAllReduceRMSNormHTKernel(
                compiled=self._compile_finalize(tuning),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(finalize_tunings)
        }
        self.all_reduce_kernels = {
            tuning: AllReduceRMSNormHTKernel(
                compiled=self._compile_all_reduce(tuning),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(all_reduce_tunings)
        }
        self.state = self._create_state(group)

    def _path_kwargs(self) -> _PathKwargs:
        return {
            "hidden_size": self.hidden_size,
            "top_k": self.top_k,
            "capacity_m": self.capacity_m,
            "write_residual_output": self.write_residual_output,
        }

    def _resolve_ctas(self, persistent_ctas: int | None) -> int:
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        # min_blocks_per_mp=1 guarantees one resident CTA per SM for this kernel.
        resident_ctas = (sm_count // self.tp_size) * self.tp_size
        if resident_ctas == 0:
            raise ValueError("tp_size exceeds the available SM count")
        if persistent_ctas is None:
            return resident_ctas
        if persistent_ctas <= 0 or persistent_ctas % self.tp_size:
            raise ValueError(
                "persistent_ctas must be positive and divisible by tp_size"
            )
        return min(persistent_ctas, resident_ctas)

    def _compile_finalize(self, tuning: HTFinalizeTuning):
        active_ctas = self._resolve_ctas(tuning.persistent_ctas)
        groups = tuning.reduction_cta_groups or active_ctas // self.tp_size
        kernel = _MoeFinalizeAllReduceRMSNormHTDeviceKernel(
            hidden=self.hidden_size,
            top_k=self.top_k,
            tp=self.tp_size,
            rank=self.rank,
            active_ctas=active_ctas,
            stages=tuning.stages,
            consumer_threads=tuning.consumer_threads,
            vectors_per_thread=tuning.vectors_per_thread,
            reduction_warps=tuning.reduction_warps,
            reduction_cta_groups=groups,
            rms_token_groups=tuning.rms_token_groups,
            rms_pipeline_stages=tuning.rms_pipeline_stages,
            rms_shard_major=tuning.rms_shard_major,
            rms_epsilon=self.rms_epsilon,
            routed_scaling_factor=self.routed_scaling_factor,
            weight_bias=self.weight_bias,
            include_shared_expert=self.include_shared_expert,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
        )
        return self._compile(kernel, top_k=self.top_k)

    def _compile_all_reduce(self, tuning: HTAllReduceTuning):
        active_ctas = self._resolve_ctas(tuning.persistent_ctas)
        groups = tuning.reduction_cta_groups or active_ctas // self.tp_size
        kernel = _MoeFinalizeAllReduceRMSNormHTDeviceKernel(
            hidden=self.hidden_size,
            top_k=0,
            tp=self.tp_size,
            rank=self.rank,
            active_ctas=active_ctas,
            stages=tuning.stages,
            consumer_threads=tuning.consumer_threads,
            vectors_per_thread=tuning.vectors_per_thread,
            reduction_warps=tuning.reduction_warps,
            reduction_cta_groups=groups,
            rms_token_groups=tuning.rms_token_groups,
            rms_pipeline_stages=tuning.rms_pipeline_stages,
            rms_shard_major=tuning.rms_shard_major,
            rms_epsilon=self.rms_epsilon,
            routed_scaling_factor=1.0,
            weight_bias=self.weight_bias,
            include_shared_expert=True,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
        )
        return self._compile(kernel, top_k=0)

    def _compile(
        self,
        kernel: _MoeFinalizeAllReduceRMSNormHTDeviceKernel,
        *,
        top_k: int,
    ):
        activation = self.capacity_m * self.hidden_size
        token_slots = (self.capacity_m + self.tp_size - 1) // self.tp_size
        args = (
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=2, divisibility=max(top_k, 1)
            ),
            make_fake_dynamic_compact_tensor(
                Int32, alignment=4, divisibility=max(top_k, 1)
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_compact_tensor(BFloat16, (activation,), assumed_align=16),
            make_fake_compact_tensor(BFloat16, (activation,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8),
            make_fake_compact_tensor(Uint32, (token_slots,), assumed_align=4),
            make_fake_compact_tensor(
                Uint32, (token_slots * self.tp_size,), assumed_align=4
            ),
            Int64(0),
            Int64(0),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return cute.compile(kernel, *args)

    def _allocate_large_buffers(
        self, group: dist.ProcessGroup
    ) -> tuple[SymmetricBuffer, SymmetricBuffer]:
        shape = (self.capacity_m, self.hidden_size)
        device = torch.device("cuda", torch.cuda.current_device())
        return (
            SymmetricBuffer.allocate(
                shape,
                torch.bfloat16,
                device,
                group,
                require_multicast=True,
            ),
            SymmetricBuffer.allocate(
                shape,
                torch.bfloat16,
                device,
                group,
                require_multicast=True,
            ),
        )

    def _create_state(self, group: dist.ProcessGroup) -> HTProtocolState:
        device = torch.device("cuda", torch.cuda.current_device())
        token_slots = (self.capacity_m + self.tp_size - 1) // self.tp_size

        def counters() -> tuple[SymmetricBuffer, torch.Tensor]:
            ready = SymmetricBuffer.allocate(
                (token_slots,),
                torch.uint32,
                device,
                group,
                materialize_peer_addresses=True,
            )
            ready.tensor.zero_()
            processed = torch.zeros(
                (token_slots, self.tp_size), dtype=torch.uint32, device=device
            )
            return ready, processed

        routed_ready, routed_processed = counters()
        all_reduce_ready, all_reduce_processed = counters()
        local, prenorm = self._allocate_large_buffers(group)
        return HTProtocolState(
            local_contributions=local,
            prenorm_mailbox=prenorm,
            routed_ready_counters=routed_ready,
            routed_processed_counters=routed_processed,
            all_reduce_ready_counters=all_reduce_ready,
            all_reduce_processed_counters=all_reduce_processed,
        )

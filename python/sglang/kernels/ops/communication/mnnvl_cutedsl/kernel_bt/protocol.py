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

"""Balanced MNNVL protocol and its two operation paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import VEC_BF16
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from ..symmetric_buffer import SymmetricBuffer
from .device_kernels import (
    LAMPORT_GENERATIONS,
    _MaterializeRMSNormDeviceKernel,
    _NarrowVectorFinalizeUnicastDeviceKernel,
    _OwnerReduceMulticastDeviceKernel,
    _ScalarFinalizeUnicastDeviceKernel,
    _SharedOnlyPublishDeviceKernel,
    _VectorFinalizeUnicastDeviceKernel,
)


@dataclass(frozen=True, slots=True)
class BTCollectiveTuning:
    reduction_threads: int = 128
    rms_threads: int = 1024
    enable_pdl: bool = True


@dataclass(frozen=True, slots=True)
class BTFinalizeTuning:
    elements_per_thread: int = VEC_BF16
    threads: int = 128
    prefetch_group: int = 1
    load_shared_expert_before_pdl: bool = False
    collective: BTCollectiveTuning = BTCollectiveTuning()


@dataclass(frozen=True, slots=True)
class BTAllReduceTuning:
    publish_threads: int = 128
    publish_vectors_per_thread: int = 1
    collective: BTCollectiveTuning = BTCollectiveTuning(reduction_threads=32)


BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256
)
BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1 = BTFinalizeTuning()
BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256
)
BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1 = BTFinalizeTuning()
BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1 = BTAllReduceTuning(
    collective=BTCollectiveTuning(reduction_threads=320)
)
BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1 = BTAllReduceTuning(
    collective=BTCollectiveTuning(reduction_threads=320)
)


@dataclass(slots=True)
class BTProtocolState:
    contribution_mailbox: SymmetricBuffer
    prenorm_mailbox: SymmetricBuffer
    stage_state: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CompiledTail:
    reduce: Any
    rms_norm: Any


@dataclass(frozen=True, slots=True)
class _CompiledFinalize:
    publish: Any
    tail: _CompiledTail


@dataclass(frozen=True, slots=True)
class _CompiledAllReduce:
    publish: Any
    tail: _CompiledTail


class _PathKwargs(TypedDict):
    hidden_size: int
    top_k: int
    capacity_m: int
    write_residual_output: bool


class _BTPath:
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        capacity_m: int,
        write_residual_output: bool,
    ) -> None:
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

    def _validate_state(self, state: BTProtocolState, m: int) -> None:
        if not 1 <= m <= self.capacity_m:
            raise ValueError(f"m must be in [1, {self.capacity_m}]")
        if state.contribution_mailbox.peer_addresses is None:
            raise ValueError("BT contribution mailbox requires peer addresses")
        address = state.prenorm_mailbox.multicast_address
        if address is None or address % 16:
            raise ValueError(
                "BT prenorm mailbox requires a 16-byte-aligned multicast address"
            )

    def _launch_tail(
        self,
        tail: _CompiledTail,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        state: BTProtocolState,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor | None,
        m: int,
    ) -> None:
        residual_arg = residual_source if residual_source is not None else norm_output
        residual_output_arg = (
            residual_output if residual_output is not None else norm_output
        )
        stream = current_cu_stream()
        tail.reduce(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            Int64(cast(int, state.prenorm_mailbox.multicast_address)),
            Int32(m),
            stream,
        )
        tail.rms_norm(
            to_cute(state.prenorm_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(
                residual_output_arg.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute(gamma, 16),
            to_cute(state.stage_state, 4),
            Int32(m),
            stream,
        )


class FinalizeAllReduceRMSNormBTKernel(_BTPath):
    def __init__(self, *, compiled: _CompiledFinalize, **kwargs) -> None:
        super().__init__(**kwargs)
        self._compiled = compiled

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
        state: BTProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        shared_arg = shared_output if shared_output is not None else norm_output
        peers = cast(torch.Tensor, state.contribution_mailbox.peer_addresses)
        self._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=self.top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=self.top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            to_cute(peers, 8),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_tail(
            self._compiled.tail,
            residual_source,
            gamma,
            state,
            norm_output,
            residual_output,
            m,
        )
        return norm_output, residual_output


class AllReduceRMSNormBTKernel(_BTPath):
    def __init__(self, *, compiled: _CompiledAllReduce, **kwargs) -> None:
        super().__init__(**kwargs)
        self._compiled = compiled

    def __call__(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: BTProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        peers = cast(torch.Tensor, state.contribution_mailbox.peer_addresses)
        self._compiled.publish(
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute(state.stage_state, 4),
            to_cute(peers, 8),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_tail(
            self._compiled.tail,
            residual_source,
            gamma,
            state,
            norm_output,
            residual_output,
            m,
        )
        return norm_output, residual_output


class BTProtocol:
    """Own BT State and protocol-local compiled variants for both paths."""

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
        finalize_tunings: tuple[BTFinalizeTuning, ...],
        all_reduce_tunings: tuple[BTAllReduceTuning, ...],
        group: dist.ProcessGroup,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.local_capacity = math.ceil(capacity_m / tp_size)
        self.rms_epsilon = rms_epsilon
        self.routed_scaling_factor = routed_scaling_factor
        self.weight_bias = weight_bias
        self.include_shared_expert = include_shared_expert
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output

        tail_cache = {
            tuning: self._compile_tail(tuning)
            for tuning in {
                *(item.collective for item in finalize_tunings),
                *(item.collective for item in all_reduce_tunings),
            }
        }
        self.finalize_kernels = {
            tuning: FinalizeAllReduceRMSNormBTKernel(
                compiled=_CompiledFinalize(
                    publish=self._compile_finalize(tuning),
                    tail=tail_cache[tuning.collective],
                ),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(finalize_tunings)
        }
        self.all_reduce_kernels = {
            tuning: AllReduceRMSNormBTKernel(
                compiled=_CompiledAllReduce(
                    publish=self._compile_all_reduce_publish(tuning),
                    tail=tail_cache[tuning.collective],
                ),
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

    def _compile_finalize(self, tuning: BTFinalizeTuning):
        if tuning.elements_per_thread not in (1, 2, 4, VEC_BF16):
            raise ValueError("BT finalize elements_per_thread must be 1, 2, 4, or 8")
        kwargs: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "top_k": self.top_k,
            "tp_size": self.tp_size,
            "rank": self.rank,
            "local_capacity": self.local_capacity,
            "threads": tuning.threads,
            "routed_scaling_factor": self.routed_scaling_factor,
            "include_shared_expert": self.include_shared_expert,
            "load_shared_expert_before_pdl": tuning.load_shared_expert_before_pdl,
            "enable_pdl": tuning.collective.enable_pdl,
            "prefetch_group": tuning.prefetch_group,
        }
        device_kernel: Any
        if tuning.elements_per_thread == 1:
            device_kernel = _ScalarFinalizeUnicastDeviceKernel(**kwargs)
        elif tuning.elements_per_thread == VEC_BF16:
            device_kernel = _VectorFinalizeUnicastDeviceKernel(**kwargs)
        else:
            device_kernel = _NarrowVectorFinalizeUnicastDeviceKernel(
                **kwargs, elements_per_thread=tuning.elements_per_thread
            )
        return cute.compile(
            device_kernel,
            *self._publish_compile_args(include_routed=True),
        )

    def _compile_all_reduce_publish(self, tuning: BTAllReduceTuning):
        device_kernel = _SharedOnlyPublishDeviceKernel(
            hidden_size=self.hidden_size,
            tp_size=self.tp_size,
            rank=self.rank,
            local_capacity=self.local_capacity,
            threads=tuning.publish_threads,
            vectors_per_thread=tuning.publish_vectors_per_thread,
            enable_pdl=tuning.collective.enable_pdl,
        )
        return cute.compile(
            device_kernel,
            *self._publish_compile_args(include_routed=False),
        )

    def _publish_compile_args(self, *, include_routed: bool) -> tuple:
        activation = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=self.hidden_size
        )
        common = (
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        if not include_routed:
            return (activation, *common)
        return (
            activation,
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=2, divisibility=self.top_k
            ),
            make_fake_dynamic_compact_tensor(
                Int32, alignment=4, divisibility=self.top_k
            ),
            activation,
            *common,
        )

    def _compile_tail(self, tuning: BTCollectiveTuning) -> _CompiledTail:
        reduce_kernel = _OwnerReduceMulticastDeviceKernel(
            hidden_size=self.hidden_size,
            tp_size=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            local_capacity=self.local_capacity,
            threads=tuning.reduction_threads,
            add_residual=self.add_residual,
            enable_pdl=tuning.enable_pdl,
        )
        reduce_elements = (
            LAMPORT_GENERATIONS * self.tp_size * self.local_capacity * self.hidden_size
        )
        reduce = cute.compile(
            reduce_kernel,
            make_fake_compact_tensor(BFloat16, (reduce_elements,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        rms_kernel = _MaterializeRMSNormDeviceKernel(
            hidden_size=self.hidden_size,
            capacity_m=self.capacity_m,
            threads=tuning.rms_threads,
            rms_epsilon=self.rms_epsilon,
            weight_bias=self.weight_bias,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
        )
        prenorm_elements = LAMPORT_GENERATIONS * self.capacity_m * self.hidden_size
        rms_norm = cute.compile(
            rms_kernel,
            make_fake_compact_tensor(BFloat16, (prenorm_elements,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return _CompiledTail(reduce=reduce, rms_norm=rms_norm)

    def _create_state(self, group: dist.ProcessGroup) -> BTProtocolState:
        if dist.get_world_size(group) != self.tp_size:
            raise ValueError("ProcessGroup size does not match tp_size")
        if dist.get_rank(group) != self.rank:
            raise ValueError("ProcessGroup rank does not match rank")
        device = torch.device("cuda", torch.cuda.current_device())
        contribution = SymmetricBuffer.allocate(
            (
                LAMPORT_GENERATIONS,
                self.tp_size,
                self.local_capacity,
                self.hidden_size,
            ),
            torch.bfloat16,
            device,
            group,
            materialize_peer_addresses=True,
        )
        contribution.tensor.view(torch.int16).fill_(-32768)
        prenorm = SymmetricBuffer.allocate(
            (
                LAMPORT_GENERATIONS,
                self.capacity_m,
                self.hidden_size,
            ),
            torch.bfloat16,
            device,
            group,
            require_multicast=True,
        )
        prenorm.tensor.view(torch.int16).fill_(-32768)
        return BTProtocolState(
            contribution_mailbox=contribution,
            prenorm_mailbox=prenorm,
            stage_state=torch.zeros((2,), dtype=torch.int32, device=device),
        )

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

"""Low-latency MNNVL protocol and its two operation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict, cast

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import QUAD_BF16
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from ..symmetric_buffer import SymmetricBuffer
from .device_kernels import (
    LAMPORT_GENERATIONS,
    _LamportResidualRMSNormDeviceKernel,
    _QuadFinalizePublishDeviceKernel,
    _ScalarFinalizePublishDeviceKernel,
    _SharedOnlyPublishDeviceKernel,
)


@dataclass(frozen=True, slots=True)
class LLCollectiveTuning:
    cluster_size: int = 8
    rank_lanes: int = 1
    threads: int = 128
    enable_pdl: bool = True


@dataclass(frozen=True, slots=True)
class LLFinalizeTuning:
    elements_per_thread: int = 4
    threads: int = 128
    prefetch_group: int = 10
    load_shared_expert_before_pdl: bool = False
    collective: LLCollectiveTuning = LLCollectiveTuning()


@dataclass(frozen=True, slots=True)
class LLAllReduceTuning:
    publish_elements_per_thread: int = 8
    publish_threads: int = 128
    publish_release_before_store: bool = False
    collective: LLCollectiveTuning = LLCollectiveTuning()


LL_FINALIZE_GB300_TP8_H8192_K10 = LLFinalizeTuning()
LL_FINALIZE_GB300_TP8_H8192_K10_M_GE_20 = LLFinalizeTuning(
    collective=LLCollectiveTuning(cluster_size=16, rank_lanes=2)
)
LL_FINALIZE_GB300_TP16_H8192_K10 = LLFinalizeTuning(
    collective=LLCollectiveTuning(cluster_size=16, rank_lanes=2)
)
LL_ALL_REDUCE_GB300_TP8_H8192_M_LE_4 = LLAllReduceTuning(
    collective=LLCollectiveTuning(cluster_size=16, threads=64)
)
LL_ALL_REDUCE_GB300_TP8_H8192_M_GE_5 = LLAllReduceTuning()
LL_ALL_REDUCE_GB300_TP16_H8192_M_LE_10 = LLAllReduceTuning(
    collective=LLCollectiveTuning(cluster_size=16, threads=64)
)
LL_ALL_REDUCE_GB300_TP16_H8192_M_11_TO_17 = LLAllReduceTuning()
LL_ALL_REDUCE_GB300_TP16_H8192_M_GE_18 = LLAllReduceTuning(
    collective=LLCollectiveTuning(cluster_size=16, rank_lanes=2)
)
LL_ALL_REDUCE_GB300_TP8_H8192 = LL_ALL_REDUCE_GB300_TP8_H8192_M_GE_5
LL_ALL_REDUCE_GB300_TP16_H8192 = LL_ALL_REDUCE_GB300_TP16_H8192_M_LE_10


@dataclass(slots=True)
class LLProtocolState:
    contribution_mailbox: SymmetricBuffer
    stage_state: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CompiledFinalize:
    publish: Any
    collective: Any


@dataclass(frozen=True, slots=True)
class _CompiledAllReduce:
    publish: Any
    collective: Any


class _PathKwargs(TypedDict):
    hidden_size: int
    top_k: int
    capacity_m: int
    write_residual_output: bool


class _LLPath:
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

    def _validate_state(self, state: LLProtocolState, m: int) -> None:
        if not 1 <= m <= self.capacity_m:
            raise ValueError(f"m must be in [1, {self.capacity_m}]")
        address = state.contribution_mailbox.multicast_address
        if address is None or address % 16:
            raise ValueError(
                "LL contribution mailbox requires a 16-byte-aligned multicast address"
            )

    def _launch_collective(
        self,
        collective,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        state: LLProtocolState,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor | None,
        m: int,
    ) -> None:
        residual_arg = residual_source if residual_source is not None else norm_output
        residual_output_arg = (
            residual_output if residual_output is not None else norm_output
        )
        collective(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(gamma, 16),
            to_cute_dynamic(
                residual_output_arg.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            Int32(m),
            current_cu_stream(),
        )


class FinalizeAllReduceRMSNormLLKernel(_LLPath):
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
        state: LLProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        shared_arg = shared_output if shared_output is not None else norm_output
        self._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=self.top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=self.top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            Int64(cast(int, state.contribution_mailbox.multicast_address)),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_collective(
            self._compiled.collective,
            residual_source,
            gamma,
            state,
            norm_output,
            residual_output,
            m,
        )
        return norm_output, residual_output


class AllReduceRMSNormLLKernel(_LLPath):
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
        state: LLProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        self._compiled.publish(
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute(state.stage_state, 4),
            Int64(cast(int, state.contribution_mailbox.multicast_address)),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_collective(
            self._compiled.collective,
            residual_source,
            gamma,
            state,
            norm_output,
            residual_output,
            m,
        )
        return norm_output, residual_output


class LLProtocol:
    """Own LL State and protocol-local compiled variants for both paths."""

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
        finalize_tunings: tuple[LLFinalizeTuning, ...],
        all_reduce_tunings: tuple[LLAllReduceTuning, ...],
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

        collective_cache = {
            tuning: self._compile_collective(tuning)
            for tuning in {
                *(item.collective for item in finalize_tunings),
                *(item.collective for item in all_reduce_tunings),
            }
        }
        self.finalize_kernels = {
            tuning: FinalizeAllReduceRMSNormLLKernel(
                compiled=_CompiledFinalize(
                    publish=self._compile_finalize(tuning),
                    collective=collective_cache[tuning.collective],
                ),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(finalize_tunings)
        }
        self.all_reduce_kernels = {
            tuning: AllReduceRMSNormLLKernel(
                compiled=_CompiledAllReduce(
                    publish=self._compile_all_reduce_publish(tuning),
                    collective=collective_cache[tuning.collective],
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

    def _compile_finalize(self, tuning: LLFinalizeTuning):
        if tuning.elements_per_thread not in (1, QUAD_BF16):
            raise ValueError("LL finalize elements_per_thread must be 1 or 4")
        kwargs: dict[str, Any] = {
            "hidden": self.hidden_size,
            "top_k": self.top_k,
            "tp": self.tp_size,
            "rank": self.rank,
            "capacity_m": self.capacity_m,
            "threads": tuning.threads,
            "routed_scaling_factor": self.routed_scaling_factor,
            "include_shared_expert": self.include_shared_expert,
            "load_shared_expert_before_pdl": tuning.load_shared_expert_before_pdl,
            "enable_pdl": tuning.collective.enable_pdl,
            "prefetch_group": tuning.prefetch_group,
        }
        device_kernel = (
            _ScalarFinalizePublishDeviceKernel(**kwargs)
            if tuning.elements_per_thread == 1
            else _QuadFinalizePublishDeviceKernel(**kwargs)
        )
        args = (
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=2, divisibility=self.top_k
            ),
            make_fake_dynamic_compact_tensor(
                Int32, alignment=4, divisibility=self.top_k
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return cute.compile(device_kernel, *args)

    def _compile_all_reduce_publish(self, tuning: LLAllReduceTuning):
        device_kernel = _SharedOnlyPublishDeviceKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            elements_per_thread=tuning.publish_elements_per_thread,
            threads=tuning.publish_threads,
            release_before_store=tuning.publish_release_before_store,
            enable_pdl=tuning.collective.enable_pdl,
        )
        args = (
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return cute.compile(device_kernel, *args)

    def _compile_collective(self, tuning: LLCollectiveTuning):
        device_kernel = _LamportResidualRMSNormDeviceKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            capacity_m=self.capacity_m,
            cluster_size=tuning.cluster_size,
            rank_lanes=tuning.rank_lanes,
            threads=tuning.threads,
            rms_epsilon=self.rms_epsilon,
            weight_bias=self.weight_bias,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
        )
        activation = self.capacity_m * self.hidden_size
        args = (
            make_fake_compact_tensor(
                BFloat16,
                (LAMPORT_GENERATIONS * self.tp_size * activation,),
                assumed_align=16,
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return cute.compile(device_kernel, *args)

    def _create_state(self, group: dist.ProcessGroup) -> LLProtocolState:
        if dist.get_world_size(group) != self.tp_size:
            raise ValueError("ProcessGroup size does not match tp_size")
        if dist.get_rank(group) != self.rank:
            raise ValueError("ProcessGroup rank does not match rank")
        device = torch.device("cuda", torch.cuda.current_device())
        mailbox = SymmetricBuffer.allocate(
            (
                LAMPORT_GENERATIONS,
                self.tp_size,
                self.capacity_m,
                self.hidden_size,
            ),
            torch.bfloat16,
            device,
            group,
            require_multicast=True,
        )
        mailbox.tensor.view(torch.int16).fill_(-32768)
        return LLProtocolState(
            contribution_mailbox=mailbox,
            stage_state=torch.zeros((2,), dtype=torch.int32, device=device),
        )

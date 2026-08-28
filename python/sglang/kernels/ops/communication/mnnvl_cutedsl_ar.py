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

"""MNNVL AllReduce fusion backend implemented with CuTe DSL.

Ported from flashinfer-ai/flashinfer#4358 at main commit 906181e (with
sibling package mnnvl_cutedsl/), pending an installable FlashInfer release
that ships flashinfer.comm.mnnvl_cutedsl. The base communication
infrastructure (mnnvl probing, pattern enum, workspace ABC) still comes
from the installed FlashInfer; the pinned release provides it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Keep the copied backend and kernel package self-contained while reusing the
# stable communication infrastructure already supplied by the serving image.
from flashinfer.comm.mnnvl import is_multicast_supported
from flashinfer.comm.trtllm_ar import AllReduceFusionPattern
from flashinfer.comm.workspace_base import AllReduceFusionWorkspace
from torch.distributed import ProcessGroup

from .mnnvl_cutedsl import DEFAULT_CONFIG, MNNVLCuteDSLConfig, ProtocolKind
from .mnnvl_cutedsl.config import StaticProfile
from .mnnvl_cutedsl.kernel_bt import BTAllReduceTuning, BTFinalizeTuning
from .mnnvl_cutedsl.kernel_bt.protocol import BTProtocol
from .mnnvl_cutedsl.kernel_ht import HTAllReduceTuning, HTFinalizeTuning
from .mnnvl_cutedsl.kernel_ht.protocol import HTProtocol
from .mnnvl_cutedsl.kernel_ll import LLAllReduceTuning, LLFinalizeTuning
from .mnnvl_cutedsl.kernel_ll.protocol import LLProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "MNNVLCuteDSLAllReduceFusionWorkspace",
    "mnnvl_cutedsl_allreduce_fusion",
]


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int | None, ...],
    dtype: torch.dtype,
    device: torch.device,
    alignment: int,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if tensor.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(tensor.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has an unsupported shape")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def _warn_pdl_mismatch(
    workspace: MNNVLCuteDSLAllReduceFusionWorkspace,
    pattern: int,
    m: int,
    launch_with_pdl: bool,
) -> None:
    preset_pdl = workspace._uses_pdl(pattern, m)
    if launch_with_pdl != preset_pdl:
        logger.warning(
            "launch_with_pdl does not match the selected MNNVL CuTe DSL "
            "preset; using enable_pdl=%s",
            preset_pdl,
        )


class MNNVLCuteDSLAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    """Compiled LL, BT, and HT protocols for one static problem shape.

    Workspace construction compiles the selected kernels and must finish before
    the first invocation. Calls using the same workspace must not overlap.
    Feature-disabled tensor slots use internal placeholders that are not read.
    """

    _destroyed: bool

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        *,
        group: Optional[ProcessGroup] = None,
        top_k: int = 10,
        rms_eps: float = 1e-6,
        routed_scaling_factor: float = 1.0,
        weight_bias: float = 0.0,
        include_shared_expert: bool = True,
        add_residual: bool = True,
        write_residual_output: bool = True,
        config: MNNVLCuteDSLConfig = DEFAULT_CONFIG,
    ) -> None:
        if tp_size not in (2, 4, 8, 16):
            raise ValueError("tp_size must be 2, 4, 8, or 16")
        if not 0 <= tp_rank < tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if max_token_num <= 0:
            raise ValueError("max_token_num must be positive")
        if dtype != torch.bfloat16:
            raise ValueError("MNNVL CuTe DSL kernels only support torch.bfloat16")
        if not torch.cuda.is_available():
            raise RuntimeError("MNNVL CuTe DSL kernels require CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device)[0] < 10:
            raise RuntimeError("MNNVL CuTe DSL kernels require a Blackwell GPU")
        if symm_mem.get_backend(device) is None:
            raise RuntimeError("PyTorch Symmetric Memory is unavailable")
        if not is_multicast_supported(device.index):
            raise RuntimeError("NVLink multicast is unavailable")
        if group is None:
            if not dist.is_initialized():
                raise ValueError("A ProcessGroup is required before initialization")
            group = dist.group.WORLD
        if dist.get_world_size(group) != tp_size:
            raise ValueError("ProcessGroup size does not match tp_size")
        if dist.get_rank(group) != tp_rank:
            raise ValueError("ProcessGroup rank does not match tp_rank")

        super().__init__(tp_size, tp_rank)
        self._protocols: dict[ProtocolKind, LLProtocol | BTProtocol | HTProtocol] = {}
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.dtype = dtype
        self.group = group
        self.rms_eps = rms_eps
        self.routed_scaling_factor = routed_scaling_factor
        self.weight_bias = weight_bias
        self.include_shared_expert = include_shared_expert
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output
        self.config = config
        self.profile = config.resolve(
            tp_size=tp_size,
            hidden_size=hidden_dim,
            top_k=top_k,
            dtype=dtype,
            capacity_m=max_token_num,
        )

        for protocol in (ProtocolKind.LL, ProtocolKind.BT, ProtocolKind.HT):
            capacity = self.profile.protocol_capacity(
                protocol, capacity_m=max_token_num
            )
            if capacity is None:
                continue
            finalize_tunings = self._tunings(
                self.profile, protocol, finalize=True, capacity_m=capacity
            )
            all_reduce_tunings = self._tunings(
                self.profile, protocol, finalize=False, capacity_m=capacity
            )
            common = dict(
                hidden_size=hidden_dim,
                top_k=top_k,
                tp_size=tp_size,
                rank=tp_rank,
                capacity_m=capacity,
                rms_epsilon=rms_eps,
                routed_scaling_factor=routed_scaling_factor,
                weight_bias=weight_bias,
                include_shared_expert=include_shared_expert,
                add_residual=add_residual,
                write_residual_output=write_residual_output,
                group=group,
            )
            instance: LLProtocol | BTProtocol | HTProtocol
            if protocol is ProtocolKind.LL:
                instance = LLProtocol(
                    **common,
                    finalize_tunings=finalize_tunings,
                    all_reduce_tunings=all_reduce_tunings,
                )
            elif protocol is ProtocolKind.BT:
                instance = BTProtocol(
                    **common,
                    finalize_tunings=finalize_tunings,
                    all_reduce_tunings=all_reduce_tunings,
                )
            else:
                instance = HTProtocol(
                    **common,
                    finalize_tunings=finalize_tunings,
                    all_reduce_tunings=all_reduce_tunings,
                )
            self._protocols[protocol] = instance

        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    @staticmethod
    def _tunings(
        profile: StaticProfile,
        protocol: ProtocolKind,
        *,
        finalize: bool,
        capacity_m: int,
    ) -> tuple:
        routes = profile.finalize_routes if finalize else profile.all_reduce_routes
        tunings = tuple(
            dict.fromkeys(
                target.preset
                for target in routes.targets_for_capacity(capacity_m)
                if target.protocol is protocol
            )
        )
        expected_type = {
            (ProtocolKind.LL, True): LLFinalizeTuning,
            (ProtocolKind.LL, False): LLAllReduceTuning,
            (ProtocolKind.BT, True): BTFinalizeTuning,
            (ProtocolKind.BT, False): BTAllReduceTuning,
            (ProtocolKind.HT, True): HTFinalizeTuning,
            (ProtocolKind.HT, False): HTAllReduceTuning,
        }[(protocol, finalize)]
        if not all(isinstance(tuning, expected_type) for tuning in tunings):
            path = "finalize" if finalize else "all-reduce"
            raise TypeError(f"Invalid {protocol.value} {path} preset")
        return tunings

    def _uses_pdl(self, pattern: int, m: int) -> bool:
        if pattern == AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm:
            target = self.profile.finalize_routes.select(m)
        elif pattern == AllReduceFusionPattern.kARResidualRMSNorm:
            target = self.profile.all_reduce_routes.select(m)
        else:
            raise NotImplementedError("Unsupported MNNVL CuTe DSL fusion pattern")
        preset = cast(Any, target.preset)
        enabled = getattr(preset, "enable_pdl", None)
        if enabled is None:
            enabled = preset.collective.enable_pdl
        return bool(enabled)

    @property
    def backend(self) -> str:
        return "mnnvl-cutedsl"

    def is_buffer_size_sufficient(
        self,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot=None,
    ) -> bool:
        del use_oneshot
        return (
            tp_size == self.world_size
            and num_tokens <= self.max_token_num
            and hidden_dim == self.hidden_dim
            and dtype == self.dtype
        )

    def _finalize_all_reduce_rms_norm(
        self,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        shared_output: torch.Tensor | None,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        norm_output: torch.Tensor | None,
        residual_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target = self.profile.finalize_routes.select(m)
        protocol = cast(Any, self._protocols[target.protocol])
        kernel = protocol.finalize_kernels[target.preset]
        return kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            residual_source,
            gamma,
            m,
            state=protocol.state,
            norm_output=norm_output,
            residual_output=residual_output,
        )

    def _all_reduce_rms_norm(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        norm_output: torch.Tensor | None,
        residual_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target = self.profile.all_reduce_routes.select(m)
        protocol = cast(Any, self._protocols[target.protocol])
        kernel = protocol.all_reduce_kernels[target.preset]
        return kernel(
            local_contribution,
            residual_source,
            gamma,
            m,
            state=protocol.state,
            norm_output=norm_output,
            residual_output=residual_output,
        )

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._protocols.clear()
        self._destroyed = True


def _mnnvl_cutedsl_allreduce_fusion(
    input: torch.Tensor,
    workspace: MNNVLCuteDSLAllReduceFusionWorkspace,
    pattern: int,
    *,
    launch_with_pdl: bool,
    output: Optional[torch.Tensor] = None,
    residual_in: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    rms_gamma: Optional[torch.Tensor] = None,
    rms_eps: float = 1e-6,
    scale_factor: Optional[torch.Tensor | float] = None,
    layout_code: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    fp32_acc: bool = False,
    moe_reduction_device_num_experts: Optional[int] = None,
    moe_reduction_scale_input: Optional[torch.Tensor] = None,
    moe_reduction_active_experts_token_input: Optional[torch.Tensor] = None,
    moe_reduction_token_input: Optional[torch.Tensor] = None,
    weight_bias: float = 0.0,
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    expert_scale_factor: Optional[torch.Tensor] = None,
    shared_expert_output: Optional[torch.Tensor] = None,
    block_quant_group_size: Optional[int] = None,
) -> torch.Tensor:
    if workspace._destroyed:
        raise RuntimeError(
            "The MNNVLCuteDSLAllReduceFusionWorkspace has been destroyed"
        )
    if pattern not in (
        AllReduceFusionPattern.kARResidualRMSNorm,
        AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
    ):
        raise NotImplementedError("Unsupported MNNVL CuTe DSL fusion pattern")
    unsupported = [
        name
        for name, value in (
            ("output", output),
            ("quant_out", quant_out),
            ("scale_out", scale_out),
            ("scale_factor", scale_factor),
            ("layout_code", layout_code),
            ("use_oneshot", use_oneshot),
            ("block_quant_group_size", block_quant_group_size),
            ("moe_reduction_scale_input", moe_reduction_scale_input),
            (
                "moe_reduction_active_experts_token_input",
                moe_reduction_active_experts_token_input,
            ),
            ("moe_reduction_token_input", moe_reduction_token_input),
        )
        if value is not None
    ]
    if fp32_acc:
        unsupported.append("fp32_acc")
    if moe_reduction_device_num_experts is not None:
        unsupported.append("moe_reduction_device_num_experts")
    if unsupported:
        raise ValueError("MNNVL CuTe DSL does not support: " + ", ".join(unsupported))

    if rms_eps != workspace.rms_eps:
        raise ValueError("rms_eps does not match the compiled workspace")
    if weight_bias != workspace.weight_bias:
        raise ValueError("weight_bias does not match the compiled workspace")
    if rms_gamma is None:
        raise ValueError("rms_gamma is required")
    if workspace.add_residual and residual_in is None:
        raise ValueError("residual_in is required by the compiled workspace")
    if not workspace.add_residual and residual_in is not None:
        raise ValueError("residual_in must be None for this compiled workspace")
    if not workspace.write_residual_output and residual_out is not None:
        raise ValueError("residual_out must be None for this compiled workspace")

    device = torch.device("cuda", torch.cuda.current_device())
    hidden = workspace.hidden_dim
    _check_tensor(
        rms_gamma,
        "rms_gamma",
        shape=(hidden,),
        dtype=torch.bfloat16,
        device=device,
        alignment=16,
    )

    if pattern == AllReduceFusionPattern.kARResidualRMSNorm:
        if any(
            value is not None
            for value in (
                expanded_idx_to_permuted_idx,
                expert_scale_factor,
                shared_expert_output,
            )
        ):
            raise ValueError("MoE finalize operands require the finalize pattern")
        _check_tensor(
            input,
            "input",
            shape=(None, hidden),
            dtype=torch.bfloat16,
            device=device,
            alignment=16,
        )
        m = input.shape[0]
        if not 1 <= m <= workspace.max_token_num:
            raise ValueError("input token count exceeds workspace capacity")
        _warn_pdl_mismatch(workspace, pattern, m, launch_with_pdl)
        if residual_in is not None:
            _check_tensor(
                residual_in,
                "residual_in",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        if norm_out is not None:
            _check_tensor(
                norm_out,
                "norm_out",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        if residual_out is not None:
            _check_tensor(
                residual_out,
                "residual_out",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        norm_out, _ = workspace._all_reduce_rms_norm(
            input,
            residual_in,
            rms_gamma,
            input.shape[0],
            norm_output=norm_out,
            residual_output=residual_out,
        )
        return norm_out

    if pattern == AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm:
        if expanded_idx_to_permuted_idx is None:
            raise ValueError("expanded_idx_to_permuted_idx is required")
        if expert_scale_factor is None:
            raise ValueError("expert_scale_factor is required")
        if workspace.include_shared_expert and shared_expert_output is None:
            raise ValueError(
                "shared_expert_output is required by the compiled workspace"
            )
        if not workspace.include_shared_expert and shared_expert_output is not None:
            raise ValueError(
                "shared_expert_output must be None for this compiled workspace"
            )
        m = expanded_idx_to_permuted_idx.shape[0]
        if not 1 <= m <= workspace.max_token_num:
            raise ValueError("input token count exceeds workspace capacity")
        _warn_pdl_mismatch(workspace, pattern, m, launch_with_pdl)
        _check_tensor(
            input,
            "input",
            shape=(None, hidden),
            dtype=torch.bfloat16,
            device=device,
            alignment=16,
        )
        _check_tensor(
            expert_scale_factor,
            "expert_scale_factor",
            shape=(m, workspace.top_k),
            dtype=torch.bfloat16,
            device=device,
            alignment=2,
        )
        _check_tensor(
            expanded_idx_to_permuted_idx,
            "expanded_idx_to_permuted_idx",
            shape=(m, workspace.top_k),
            dtype=torch.int32,
            device=device,
            alignment=4,
        )
        if shared_expert_output is not None:
            _check_tensor(
                shared_expert_output,
                "shared_expert_output",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        if residual_in is not None:
            _check_tensor(
                residual_in,
                "residual_in",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        if norm_out is not None:
            _check_tensor(
                norm_out,
                "norm_out",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        if residual_out is not None:
            _check_tensor(
                residual_out,
                "residual_out",
                shape=(m, hidden),
                dtype=torch.bfloat16,
                device=device,
                alignment=16,
            )
        norm_out, _ = workspace._finalize_all_reduce_rms_norm(
            input,
            expert_scale_factor,
            expanded_idx_to_permuted_idx,
            shared_expert_output,
            residual_in,
            rms_gamma,
            m,
            norm_output=norm_out,
            residual_output=residual_out,
        )
        return norm_out

    raise AssertionError("unreachable")


# Keep the upstream private name intact while exposing SGLang's backend entry point;
# upstream exports it through flashinfer.comm.allreduce_fusion.
mnnvl_cutedsl_allreduce_fusion = _mnnvl_cutedsl_allreduce_fusion

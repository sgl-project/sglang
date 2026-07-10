# Copyright 2024 SGLang Team
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

import logging
import math
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeGuard,
    runtime_checkable,
)

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_parallel

try:
    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
    from triton_kernels.tensor import make_ragged_tensor_metadata
    from triton_kernels.topk import topk as triton_kernels_topk

    def routing(
        logits,
        n_expts_act,
        sm_first=False,
        expt_indx=None,
        simulated_ep=1,
        n_rows=None,
    ):
        if simulated_ep != 1:
            raise NotImplementedError(
                "simulated_ep routing is not supported with triton_kernels 3.6.0"
            )

        if sm_first:
            logits = torch.softmax(logits, dim=-1)

        sparse_logits = triton_kernels_topk(
            logits,
            n_expts_act,
            apply_softmax=not sm_first,
            y_indx=expt_indx,
            n_rows=n_rows,
        )
        dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
        combine_indx = sparse_logits.mask_metadata.col_sorted_indx
        ragged_metadata = make_ragged_tensor_metadata(
            sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0]
        )
        gate_scal = sparse_logits.vals.flatten()[combine_indx]
        routing_data = RoutingData(
            gate_scal,
            ragged_metadata.slice_sizes,
            logits.shape[-1],
            n_expts_act,
            ragged_metadata,
        )
        gather_indx = GatherIndx(combine_indx, dispatch_indx)
        scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
        return routing_data, gather_indx, scatter_indx

except ImportError:
    pass

from sglang.jit_kernel.dsv4 import mask_topk_ids
from sglang.srt.distributed import (
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.utils import (
    has_per_rank_fused_shared_slots,
)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

if TYPE_CHECKING:
    from sglang.srt.layers.quantization import QuantizationConfig


logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_xpu = is_xpu()
_is_npu = is_npu()
_is_xpu = is_xpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_musa = is_musa()

# Experimental: skip the HIP padded-token routing-weight masking entirely.
# Padded (CUDA-graph) rows are discarded downstream and the MoE combine is
# per-token, so zeroing their weights is in principle unnecessary. Gated off by
# default because it is a numerics-affecting change that must be validated with
# an accuracy run before becoming the default.
_skip_hip_pad_mask = get_bool_env_var("SGLANG_MORI_NO_PAD_MASK", "False")


if _is_cuda:
    try:
        from flashinfer.fused_moe import fused_topk_deepseek as _fused_topk_deepseek

        from sglang.srt.utils.custom_op import register_custom_op

        @register_custom_op(
            op_name="fused_topk_deepseek",
            mutates_args=["topk_weights", "topk_ids"],
        )
        def fused_topk_deepseek(
            gating_output: torch.Tensor,
            correction_bias: torch.Tensor,
            num_expert_group: int,
            topk_group: int,
            topk: int,
            scaling_factor: float,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            renormalize: bool,
        ) -> None:
            _fused_topk_deepseek(
                gating_output,
                correction_bias,
                num_expert_group,
                topk_group,
                topk,
                scaling_factor,
                topk_weights,
                topk_ids,
                renormalize,
            )

    except ImportError:
        fused_topk_deepseek = None

if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel import topk_softmax

    try:
        from sgl_kernel import topk_sigmoid
    except ImportError:
        pass
if _use_aiter:
    try:
        from aiter import biased_grouped_topk as aiter_biased_grouped_topk
        from aiter.fused_moe import fused_topk as aiter_fused_topk
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
if _is_musa:
    try:
        from mate import moe_fused_gate
    except ImportError:
        raise ImportError("mate is required for the biased grouped topk.")

    from sglang.srt.hardware_backend.musa.kernels.topk import topk_sigmoid, topk_softmax

# -------------------------------- TopKConfig ---------------------------------------


@dataclass
class TopKConfig:
    top_k: int
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    fused_shared_experts_scaling_factor: Optional[float] = None
    output_format: Optional[TopKOutputFormat] = None
    scoring_func: str = "softmax"
    # Draft-side MoE blocks set this False so they never write the target's
    # process-global routed-experts capture buffer.
    allow_routed_experts_capture: bool = True


# -------------------------------- TopKOutput ---------------------------------------


class TopKOutputChecker:

    @staticmethod
    def format_is_standard(topk_output: TopKOutput) -> TypeGuard[StandardTopKOutput]:
        # ===== TO BE REFACTORED ====
        # The experimental fused topk+pack carrier only exists under the master switch.
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            return isinstance(
                topk_output, (StandardTopKOutput, StandardTopKOutputPacked)
            )
        # ===== END TO BE REFACTORED ====
        return isinstance(topk_output, StandardTopKOutput)

    @staticmethod
    def format_is_triton_kernels(
        topk_output: TopKOutput,
    ) -> TypeGuard[TritonKernelTopKOutput]:
        return isinstance(topk_output, TritonKernelTopKOutput)

    @staticmethod
    def format_is_bypassed(topk_output: TopKOutput) -> TypeGuard[BypassedTopKOutput]:
        return isinstance(topk_output, BypassedTopKOutput)


class TopKOutputFormat(IntEnum):
    STANDARD = auto()
    TRITON_KERNEL = auto()
    BYPASSED = auto()


@runtime_checkable
class TopKOutput(Protocol):
    """Protocol for top-k outputs in different formats."""

    @property
    def format(self) -> TopKOutputFormat:
        """The format of the output."""
        ...


class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


# ===== TO BE REFACTORED ====
# Experimental fused topk+pack (SGLANG_OPT_LORA_FUSED_TOPK_PACK) carrier: the FlashInfer
# routed-MoE packed topk produced fused in the gating kernel. Kept a SEPARATE type rather
# than a 4th StandardTopKOutput field so the OSS `a, b, _ = topk_output` 3-tuple unpack
# stays valid; only the gated experimental MoE dispatch reads .packed_topk_ids (getattr).
class StandardTopKOutputPacked(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor
    packed_topk_ids: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


# ===== END TO BE REFACTORED ====


class TritonKernelTopKOutput(NamedTuple):
    """Triton kernel top-k output format."""

    routing_data: RoutingData
    gather_indx: GatherIndx
    scatter_indx: ScatterIndx

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.TRITON_KERNEL


class BypassedTopKOutput(NamedTuple):
    """Bypassed top-k output format."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: Optional[torch.Tensor] = None
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    def to_standard(self, layer_id: Optional[int] = None) -> StandardTopKOutput:
        """Materialize routing tensors. Used by MoE kernels that need explicit
        topk_ids / topk_weights rather than doing routing internally."""
        return select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            topk_config=self.topk_config,
            layer_id=layer_id,
            num_token_non_padded=self.num_token_non_padded,
            expert_location_dispatch_info=self.expert_location_dispatch_info,
        )


def _make_round_robin_expert_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    layer_id: Optional[int] = None,
) -> torch.Tensor:
    if topk == 0:
        return torch.empty((num_tokens, 0), device=device, dtype=dtype)

    step = max(num_experts // topk, 1)
    layer_offset = 0 if layer_id is None else layer_id
    offsets = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(1)
    steps = torch.arange(topk, device=device, dtype=dtype).unsqueeze(0) * step
    return (offsets + layer_offset + steps) % num_experts


# -------------------------------- TopK ---------------------------------------


class TopK(MultiPlatformOp):
    """
    Parameters:
    --top_k: The all number of top experts selected per token, including the fused shared expert(s).
    --num_fused_shared_experts: num of shared experts, can be activate both in TP or EP mode.
    --routed_scaling_factor: the scaling factor for routed experts in topk_weights.
    --fused_shared_experts_scaling_factor: scaling factor for fused shared experts on AMD-platform.
    """

    def __init__(
        self,
        top_k: int,
        *,
        layer_id: Optional[int] = None,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        quant_config: Optional[QuantizationConfig] = None,
        routed_scaling_factor: Optional[float] = None,
        apply_routed_scaling_factor_on_output: Optional[bool] = False,
        output_format: Optional[TopKOutputFormat] = None,
        fused_shared_experts_scaling_factor: Optional[float] = None,
        is_fp4_experts: bool = False,
        allow_routed_experts_capture: bool = True,
    ):
        # NOTE: scoring_func is not used for now, but we keep it for future use
        # see https://github.com/sgl-project/sglang/pull/4505 for more details
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None

        self.layer_id = layer_id
        from sglang.srt.runtime_context import get_server_args

        self.enable_deepep_waterfill = (
            num_fused_shared_experts > 0 and get_server_args().enable_deepep_waterfill
        )

        self.deepep_waterfill_balancer = None
        if self.enable_deepep_waterfill:
            # TODO(ch-wan): Refactor shared-expert fusion and routed TopK fusion.
            top_k -= num_fused_shared_experts
            num_fused_shared_experts = 0
            output_format = TopKOutputFormat.STANDARD

        # flashinfer_mxfp4 backend only: True -> STANDARD (Mxfp4FlashinferTrtllmMoEMethod
        # consumes), False -> BYPASSED (flashinfer's own mxfp4 kernel). No-op otherwise.
        self.is_fp4_experts = is_fp4_experts
        self.topk_config = TopKConfig(
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            output_format=output_format,
            scoring_func=scoring_func,
            allow_routed_experts_capture=allow_routed_experts_capture,
        )

    def _apply_deepep_waterfill(
        self, topk_output: TopKOutput, num_tokens: int
    ) -> TopKOutput:
        if self.enable_deepep_waterfill and self.deepep_waterfill_balancer is None:
            raise RuntimeError(
                "DeepEP waterfill TopK must be prepared by ModelRunner before forward."
            )
        if self.deepep_waterfill_balancer is None:
            return topk_output
        assert TopKOutputChecker.format_is_standard(topk_output)
        return self.deepep_waterfill_balancer.expand_topk(topk_output, num_tokens)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        self.topk_config.torch_native = True
        topk_output = select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        if self.topk_config.output_format is not None:
            output_format = self.topk_config.output_format
        elif get_moe_runner_backend().is_triton_kernels():
            output_format = TopKOutputFormat.TRITON_KERNEL
        # ===== TO BE REFACTORED ====
        elif get_moe_runner_backend().is_experimental_sgl_trtllm():
            try:
                from sglang.srt.runtime_context import get_server_args

                use_standard_for_lora = bool(get_server_args().enable_lora)
            except ValueError:
                use_standard_for_lora = False
            output_format = (
                TopKOutputFormat.STANDARD
                if use_standard_for_lora
                else TopKOutputFormat.BYPASSED
            )
        # ===== END TO BE REFACTORED ====
        elif get_moe_runner_backend().is_flashinfer_trtllm() or (
            get_moe_runner_backend().is_flashinfer_mxfp4() and not self.is_fp4_experts
        ):
            output_format = TopKOutputFormat.BYPASSED
        else:
            output_format = TopKOutputFormat.STANDARD

        if output_format == TopKOutputFormat.TRITON_KERNEL:
            # renormalize=True is equivalent to sm_first=False
            routing_data, gather_idx, scatter_idx = routing(
                router_logits,
                self.topk_config.top_k,
                sm_first=not self.topk_config.renormalize,
            )
            return TritonKernelTopKOutput(routing_data, gather_idx, scatter_idx)
        elif output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        else:
            self.topk_config.torch_native = False
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                topk_output = select_experts(
                    hidden_states=hidden_states,
                    layer_id=self.layer_id,
                    router_logits=router_logits,
                    topk_config=self.topk_config,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        topk_output = select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])

    def forward_npu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:

        from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu

        return fused_topk_npu(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            layer_id=self.layer_id,
        )

    def empty_topk_output(
        self, device: torch.device, *, layer_id: Optional[int] = None
    ) -> TopKOutput:
        """Return an empty topk output for a rank with zero tokens this forward.

        When ``layer_id`` is provided and the active dispatch algorithm is LP,
        also calls ``LPLBSolver.solve(empty)`` so that this rank participates
        in the EP all-reduce. Without this, an empty rank would skip the
        collective and deadlock under DP-attention.
        """
        if layer_id is not None:
            # Skip the full ExpertLocationDispatchInfo allocation — we only
            # need the per-layer solver to participate in the EP all-reduce.
            from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

            lplb_solver = get_global_lplb_solver(layer_id)
            if lplb_solver is not None:
                lplb_solver.solve(
                    torch.empty(
                        (0, self.topk_config.top_k),
                        dtype=torch.int32,
                        device=device,
                    )
                )
        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
            topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        # FIXME: router_logits should be of size (0, num_experts)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
        if has_per_rank_fused_shared_slots(self.topk_config.num_fused_shared_experts):
            n = self.topk_config.num_fused_shared_experts
            topk_output = topk_output._replace(
                topk_ids=topk_output.topk_ids.new_empty(
                    (0, topk_output.topk_ids.shape[-1] + n)
                ),
                topk_weights=topk_output.topk_weights.new_empty(
                    (0, topk_output.topk_weights.shape[-1] + n)
                ),
            )
        return self._apply_deepep_waterfill(topk_output, 0)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        self.topk_config.torch_native = True
        # [NOTE] XPU device support for topk kernels
        #   - support 'topk_softmax' and 'topk_sigmoid'
        #   - support up to 8 top-k and 256 experts
        self.topk_config.torch_native = not (
            self.topk_config.top_k <= 8 and router_logits.shape[1] <= 256
        )

        return select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )


# ------------------------------- TopK implementation -------------------------------------


def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    def scoring_func_impl(gating_output: torch.Tensor) -> torch.Tensor:
        if scoring_func == "softmax":
            return gating_output.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            return gating_output.sigmoid()
        elif scoring_func == "sqrtsoftplus":
            return F.softplus(gating_output).sqrt()
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = scoring_func_impl(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = scoring_func_impl(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk_softmax_torch_raw_logits(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"

    _, topk_ids = torch.topk(gating_output, k=topk, dim=-1, sorted=False)
    logits = gating_output.float()
    topk_weights = logits.gather(1, topk_ids)
    if renormalize:
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    # TODO: add c++ kernel for cpu
    # The topk_softmax_cpu kernel only handles vanilla softmax scoring with no
    # correction bias. Fall back to the torch-native impl for the rest
    # (e.g. MiniMax sets both correction_bias and scoring_func).
    if correction_bias is not None or scoring_func != "softmax":
        return fused_topk_torch_native(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )

    topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    return topk_weights, topk_ids


def apply_topk_weights_cpu(need_apply, topk_weights, inputs):
    if not need_apply:
        return inputs, topk_weights

    # TODO: fuse below processing in fused_experts_cpu kernel
    inputs = inputs * topk_weights.to(inputs.dtype)
    topk_weights = torch.ones_like(
        topk_weights, dtype=torch.float32
    )  # clear topk_weights as already applied

    return inputs, topk_weights


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
    scoring_func: str = "softmax",
    packed_out: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    if scoring_func == "softmax":
        if _use_aiter:

            # Use fused_topk instead of topk_softmax to auto dispatch to the correct kernel
            topk_weights, topk_ids = aiter_fused_topk(
                hidden_states,
                gating_output,
                topk,
                renormalize,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        # ===== TO BE REFACTORED ====
        elif packed_out is not None:
            # Fused gating + routed pack (SGLANG_OPT_LORA_FUSED_TOPK_PACK): one JIT kernel
            # writes topk_weights/topk_ids AND the FlashInfer packed topk in one launch.
            from sglang.jit_kernel.trtllm_lora_temp.topk_softmax_pack import (
                topk_softmax_pack,
            )

            topk_softmax_pack(
                topk_weights,
                topk_ids,
                packed_out,
                gating_output,
                renormalize,
                num_token_non_padded=num_token_non_padded,
            )
        # ===== END TO BE REFACTORED ====
        elif _is_cuda and envs.SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK.get():
            # Unified Triton router (subsumes the AOT topk_softmax CUDA kernel).
            from sglang.jit_kernel.moe_fused_gate import (
                moe_fused_gate as _jit_moe_fused_gate,
            )

            zero_bias = torch.zeros(
                gating_output.shape[1],
                dtype=torch.float32,
                device=gating_output.device,
            )
            topk_weights, topk_ids = _jit_moe_fused_gate(
                gating_output,
                zero_bias,
                topk,
                scoring_func="softmax",
                renormalize=renormalize,
            )
        else:
            topk_softmax(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
            )
    elif scoring_func == "sigmoid":
        if _use_aiter and correction_bias is not None:
            aiter_biased_grouped_topk(
                gating_output,
                correction_bias.to(dtype=gating_output.dtype),
                topk_weights,
                topk_ids,
                num_expert_group=1,
                topk_group=1,
                need_renorm=renormalize,
            )
        elif _is_cuda and envs.SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK.get():
            # Unified Triton router (subsumes the AOT topk_sigmoid CUDA kernel).
            from sglang.jit_kernel.moe_fused_gate import (
                moe_fused_gate as _jit_moe_fused_gate,
            )

            bias_fp32 = (
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else torch.zeros(
                    gating_output.shape[1],
                    dtype=torch.float32,
                    device=gating_output.device,
                )
            )
            topk_weights, topk_ids = _jit_moe_fused_gate(
                gating_output,
                bias_fp32,
                topk,
                scoring_func="sigmoid",
                renormalize=renormalize,
            )
        else:
            topk_sigmoid(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                correction_bias,
            )
    else:
        raise ValueError(f"Invalid scoring function: {scoring_func}")

    return topk_weights, topk_ids


# This is used by the Deepseek V2/V3/R1 series models
@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    return topk_weights, topk_ids


def grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert not apply_routed_scaling_factor_on_output
    return torch.ops.sgl_kernel.grouped_topk_cpu(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        # num_token_non_padded must be None since it is not supported in kernel
        num_token_non_padded=None,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def kimi_k2_biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """
    Optimized version for num_expert_group=1 case (e.g., Kimi K2 with 384 experts).
    Simplifies the grouped topk logic by removing unnecessary group masking operations.
    Note: This function assumes num_fused_shared_experts=0.
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]

    # When num_expert_group=1, no need for group masking
    # Directly compute scores with correction bias
    tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    # Directly select topk experts (no need to sort since num_fused_shared_experts=0)
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


def biased_topk_jit_kernel_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if _use_aiter and scoring_func == "sqrtsoftplus" and num_fused_shared_experts == 0:
        from aiter import topk_gating

        num_tokens = gating_output.shape[0]
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )

        topk_gating(
            topk_weights,
            topk_ids,
            gating_output,
            correction_bias,
            renormalize,
            routed_scaling_factor,
            score_func="sqrtsoftplus",
        )

        return topk_weights, topk_ids

    else:
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

        topk_weights, topk_ids = moe_fused_gate(
            gating_output,
            correction_bias,
            topk=topk,
            scoring_func=scoring_func,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
        topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(
            torch.int32
        )
        return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    return topk_weights, topk_ids


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


@triton.jit
def _fill_padded_rows_kernel(
    out_ptr,
    num_token_non_padded_ptr,
    n_cols,
    fill_value,
    stride_row,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    n_valid = tl.load(num_token_non_padded_ptr)
    if row >= n_valid:
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < n_cols
        ptrs = out_ptr + row * stride_row + cols
        fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
        tl.store(ptrs, fill, mask=mask)


def _can_fuse_padded_region(x: torch.Tensor) -> bool:
    # The fused kernel uses one program per row and assumes a row-major 2D
    # tensor (columns contiguous); fall back to eager for anything else.
    return x.dim() == 2 and x.stride(1) == 1


def _fill_padded_rows(
    x: torch.Tensor,
    num_token_non_padded: torch.Tensor,
    fill_value,
) -> None:
    """Set ``x[row, :] = fill_value`` for every padded row (row index
    ``>= num_token_non_padded``) using a single Triton launch.

    Replaces the eager ``arange + (>=) + boolean index_put_`` sequence, which
    issues several launch-latency-bound kernels per call. The grid is static
    (one program per row) and the pad count is read from device memory inside
    the kernel, so this is safe to capture inside a CUDA/HIP graph.
    """
    # Metadata-only checks (no device sync): the kernel reads a single scalar
    # routing count from device memory, so it must be a 1-element integer tensor
    # on the same device as ``x``.
    assert isinstance(
        num_token_non_padded, torch.Tensor
    ), "num_token_non_padded must be a torch.Tensor"
    assert num_token_non_padded.numel() == 1, (
        "num_token_non_padded must be a single-element tensor, got shape "
        f"{tuple(num_token_non_padded.shape)}"
    )
    assert (
        not num_token_non_padded.dtype.is_floating_point
    ), f"num_token_non_padded must be an integer tensor, got {num_token_non_padded.dtype}"
    assert (
        num_token_non_padded.device == x.device
    ), "num_token_non_padded and x must be on the same device"
    n_rows, n_cols = x.shape
    _fill_padded_rows_kernel[(n_rows,)](
        x,
        num_token_non_padded,
        n_cols,
        fill_value,
        x.stride(0),
        BLOCK_COLS=triton.next_power_of_2(n_cols),
    )


def _eplb_remap_enabled() -> bool:
    # A real logical->physical mapping only exists when EPLB is enabled, the
    # initial expert placement is non-trivial, or there are redundant physical
    # experts. Otherwise the map is identity and the remap must be skipped (it is
    # both unnecessary and not well-defined over the padded region of topk_ids).
    from sglang.srt.runtime_context import get_server_args

    try:
        server_args = get_server_args()
    except ValueError:
        # Global server args are not initialized outside the server runtime
        # (e.g. in unit tests that call select_experts directly). In that case
        # there is no EPLB mapping, so the remap must be skipped.
        return False
    return (
        server_args.enable_eplb
        or server_args.init_expert_location != "trivial"
        or server_args.ep_num_redundant_experts > 0
    )


@triton.jit
def _fill_padded_rows_kernel(
    out_ptr,
    num_token_non_padded_ptr,
    n_cols,
    fill_value,
    stride_row,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    n_valid = tl.load(num_token_non_padded_ptr)
    if row >= n_valid:
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < n_cols
        ptrs = out_ptr + row * stride_row + cols
        fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
        tl.store(ptrs, fill, mask=mask)


def _can_fuse_padded_region(x: torch.Tensor) -> bool:
    # The fused kernel uses one program per row and assumes a row-major 2D
    # tensor (columns contiguous); fall back to eager for anything else.
    return x.dim() == 2 and x.stride(1) == 1


def _fill_padded_rows(
    x: torch.Tensor,
    num_token_non_padded: torch.Tensor,
    fill_value,
) -> None:
    """Set ``x[row, :] = fill_value`` for every padded row (row index
    ``>= num_token_non_padded``) using a single Triton launch.

    Replaces the eager ``arange + (>=) + boolean index_put_`` sequence, which
    issues several launch-latency-bound kernels per call. The grid is static
    (one program per row) and the pad count is read from device memory inside
    the kernel, so this is safe to capture inside a CUDA/HIP graph.
    """
    # Metadata-only checks (no device sync): the kernel reads a single scalar
    # routing count from device memory, so it must be a 1-element integer tensor
    # on the same device as ``x``. Use explicit raises (not asserts) so the
    # checks survive ``python -O`` and invalid inputs fail loudly instead of
    # turning into opaque Triton/memory errors.
    if not isinstance(num_token_non_padded, torch.Tensor):
        raise TypeError("num_token_non_padded must be a torch.Tensor")
    if num_token_non_padded.numel() != 1:
        raise ValueError(
            "num_token_non_padded must be a single-element tensor, got shape "
            f"{tuple(num_token_non_padded.shape)}"
        )
    if num_token_non_padded.dtype.is_floating_point:
        raise TypeError(
            "num_token_non_padded must be an integer tensor, got "
            f"{num_token_non_padded.dtype}"
        )
    if num_token_non_padded.device != x.device:
        raise ValueError("num_token_non_padded and x must be on the same device")
    n_rows, n_cols = x.shape
    _fill_padded_rows_kernel[(n_rows,)](
        x,
        num_token_non_padded,
        n_cols,
        fill_value,
        x.stride(0),
        BLOCK_COLS=triton.next_power_of_2(n_cols),
    )


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
    fill_value: int = -1,
) -> None:
    if num_token_non_padded is None:
        return
    # TODO: let the kernel support other dtypes
    if _is_cuda and topk_ids.dtype == torch.int32 and fill_value == -1:
        mask_topk_ids(topk_ids, num_token_non_padded)
    elif _is_npu:
        return
    elif _can_fuse_padded_region(topk_ids):
        _fill_padded_rows(topk_ids, num_token_non_padded, fill_value)
    else:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = fill_value


def _zero_topk_weights_padded_region(
    topk_weights: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return
    if _can_fuse_padded_region(topk_weights):
        _fill_padded_rows(topk_weights, num_token_non_padded, 0.0)
        return
    indices = torch.arange(0, topk_weights.shape[0], device=topk_weights.device)
    topk_weights[indices >= num_token_non_padded, :] = 0.0


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(
    topk_ids, expert_location_dispatch_info, num_token_non_padded
):
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids


def biased_grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    experts_per_group = (
        num_experts // num_expert_group if num_expert_group else num_experts
    )

    # topk for routed experts only (shared experts are appended separately below)
    topk_routed = topk - num_fused_shared_experts
    if (
        _is_cuda
        and num_expert_group
        and num_expert_group > 1
        and envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.get()
    ):
        # Opt-in: unified Triton router for DeepSeek-V3 grouped routing. Bit-exact
        # with the flashinfer/AOT paths on DeepSeek-V3.2 e2e (validated); handles any
        # experts-per-group (no <=32 cap). Off by default — see the env-var comment.
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as jit_grouped_gate

        return jit_grouped_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias.to(dtype=torch.float32),
            topk,
            scoring_func="sigmoid",
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            apply_routed_scaling_factor_on_output=bool(
                apply_routed_scaling_factor_on_output
            ),
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
    if (
        _is_cuda
        and fused_topk_deepseek is not None
        and is_power_of_two(num_experts)
        # flashinfer constraints (applied to routed experts only)
        and topk_routed <= 8
        and topk_group <= num_expert_group
        and topk_group * num_expert_group >= topk_routed
        and (
            (experts_per_group <= 32 and experts_per_group * topk_group <= 128)
            if num_expert_group > 1
            else num_experts <= 384
        )
    ):
        # Pre-allocate output tensors (flashinfer mutates them in-place)
        topk_weights = torch.empty(
            (num_tokens, topk_routed), dtype=torch.float32, device=gating_output.device
        )
        topk_ids = torch.empty(
            (num_tokens, topk_routed), dtype=torch.int32, device=gating_output.device
        )

        # flashinfer always applies the scaling_factor internally
        scaling_factor = 1.0
        if routed_scaling_factor is not None and apply_routed_scaling_factor_on_output:
            scaling_factor = routed_scaling_factor

        # flashinfer's fused_topk_deepseek
        fused_topk_deepseek(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk_routed,
            scaling_factor,
            topk_weights,
            topk_ids,
            True,
        )

        if num_fused_shared_experts > 0:
            # Append shared expert columns: ID = num_experts (first shared slot),
            # weight = sum(routed) / scaling_factor (matching biased_grouped_topk_impl).
            # For DeepEP/MegaMOE per-rank shared-slot layout, post-process remaps
            # this placeholder ID and overwrites the shared weight for the active scaling path.
            topk_ids = F.pad(topk_ids, (0, num_fused_shared_experts), value=num_experts)
            topk_weights = F.pad(topk_weights, (0, num_fused_shared_experts))
            if routed_scaling_factor is not None:
                topk_weights[:, topk_routed:] = (
                    topk_weights[:, :topk_routed].sum(dim=-1, keepdim=True)
                    / routed_scaling_factor
                )

        return topk_weights, topk_ids

    elif _is_cuda and num_expert_group > 1:
        # CUDA grouped fallback (flashinfer unavailable / constraints unmet): the
        # unified Triton router replaces the retired AOT moe_fused_gate kernel. It
        # handles any experts-per-group (no MAX_VPT=32 cap) and any num_experts.
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as jit_grouped_gate

        return jit_grouped_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias.to(dtype=torch.float32),
            topk,
            scoring_func="sigmoid",
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            apply_routed_scaling_factor_on_output=bool(
                apply_routed_scaling_factor_on_output
            ),
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )

    elif _use_aiter:
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        token = gating_output.shape[0]
        device = gating_output.device
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch: hidden_states.shape[0] = {hidden_states.shape[0]}, gating_output.shape[0] = {gating_output.shape[0]}"
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        aiter_biased_grouped_topk(
            gating_output,
            correction_bias.to(dtype=gating_output.dtype),
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        )
        return topk_weights, topk_ids
    elif _is_musa and (
        gating_output.shape[1] // num_expert_group <= 32
        or (num_expert_group == 1 and gating_output.shape[1] in {160, 256, 384})
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            True,
            apply_routed_scaling_factor_on_output,
        )
        return topk_weights, topk_ids
    else:
        num_experts = gating_output.shape[1]
        if _is_cuda and num_experts == 384 and num_expert_group == 1:
            # ===== TO BE REFACTORED ====
            _use_jit_bf16_gate = False
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

                _use_jit_bf16_gate = (
                    lora_envs.SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE.get()
                    and lora_envs.SGLANG_OPT_KIMI_GATE_BF16_INPUT.get()
                )
            if _use_jit_bf16_gate:
                from sglang.jit_kernel.trtllm_lora_temp.kimi_k2_moe_fused_gate import (
                    kimi_k2_moe_fused_gate as _kimi_k2_moe_fused_gate,
                )

                # bf16 pass-through: skip the two host-side fp32 upcast kernels.
                return _kimi_k2_moe_fused_gate(
                    gating_output,
                    correction_bias,
                    topk=topk,
                    renormalize=renormalize,
                    routed_scaling_factor=routed_scaling_factor,
                    apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
                )
            # ===== END TO BE REFACTORED ====
            from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as jit_gate

            return jit_gate(
                gating_output.to(dtype=torch.float32),
                correction_bias,
                topk=topk,
                scoring_func="sigmoid",
                num_fused_shared_experts=num_fused_shared_experts,
                renormalize=renormalize,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        elif (
            _is_cuda
            and num_expert_group == 1
            and topk_group == 1
            and num_fused_shared_experts == 0
            and num_experts <= 512
            and topk <= 8
        ):
            # Ungrouped sigmoid (num_expert_group == 1): use the unified Triton
            # router, which subsumes the jit grouped_topk.cuh kernel here.
            from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as jit_gate

            return jit_gate(
                gating_output,
                correction_bias.to(torch.float32),
                topk,
                scoring_func="sigmoid",
                renormalize=renormalize,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        elif (
            _is_xpu
            and num_expert_group == 1
            and topk_group == 1
            and num_fused_shared_experts == 0
            and num_experts <= 256
            and topk <= 8
        ):
            if not apply_routed_scaling_factor_on_output:
                scaling = 1.0

            num_tokens = gating_output.shape[0]

            topk_values = torch.empty(
                (num_tokens, topk), dtype=torch.float32, device=gating_output.device
            )
            topk_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=gating_output.device
            )

            if num_tokens == 0:
                return topk_values, topk_indices

            topk_sigmoid(
                topk_values,
                topk_indices,
                gating_output,
                renormalize,
                correction_bias,
            )
            return topk_values * scaling, topk_indices

        else:
            return biased_grouped_topk_impl(
                hidden_states,
                gating_output,
                correction_bias,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )


def biased_grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    compiled: bool = True,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    return torch.ops.sgl_kernel.biased_grouped_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor if apply_routed_scaling_factor_on_output else None,
        # num_token_non_padded must be None since it is not supported in kernel
        num_token_non_padded=None,
    )


def biased_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    topk_weights, topk_ids = torch.ops.sgl_kernel.biased_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        scoring_func,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output or False,
    )
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


if _is_cpu and _is_cpu_amx_available:
    biased_grouped_topk = biased_grouped_topk_cpu
    grouped_topk = grouped_topk_cpu
    fused_topk_native = fused_topk_cpu
    fused_topk = fused_topk_cpu
else:
    biased_grouped_topk = biased_grouped_topk_gpu
    grouped_topk = grouped_topk_gpu
    fused_topk_native = fused_topk_torch_native


def remap_topk_for_per_rank_shared_slots(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_fused_shared_experts: int,
    num_physical_routed_experts: int,
    topk_config: TopKConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap TopK IDs to a per-rank shared-slot layout.

    DeepEP and MegaMoE dispatch need each rank's shared expert at a unique ID
    so tokens route to the correct rank. The layout is ordered by rank:
    [rank0 routed..., rank0 shared, rank1 routed..., rank1 shared, ...].

    Routed IDs:  e -> e + e // num_local_routed
    Shared IDs:  ep_rank * num_local_experts + num_local_routed
    Shared weight: 1.0 on the aiter path, else 1/routed_scaling_factor (see below).
    """
    if topk_ids.shape[0] == 0:
        return topk_ids, topk_weights

    ep_size = get_parallel().moe_ep_size
    ep_rank = get_parallel().moe_ep_rank
    # Static EPLB may add redundant physical experts. At this point routed
    # topk_ids have already been remapped from logical to physical ids, so the
    # per-rank shared-slot layout must use the physical routed count.
    num_local_routed = num_physical_routed_experts // ep_size
    num_local_experts = num_local_routed + num_fused_shared_experts

    # Remap routed IDs: insert gaps for shared expert slots (single fused op)
    routed = topk_ids[:, :-num_fused_shared_experts]
    topk_ids[:, :-num_fused_shared_experts] = routed + routed // num_local_routed

    # Set shared expert IDs to route to home rank (vectorized)
    topk_ids[:, -num_fused_shared_experts:] = (
        ep_rank * num_local_experts
        + num_local_routed
        + torch.arange(num_fused_shared_experts, device=topk_ids.device)
    )

    # Override the fused shared expert's weight so its net contribution is 1.0x.
    #
    # The correct value depends on whether routed_scaling_factor is applied to
    # the MoE output AFTER the experts run, or already folded into the routed
    # topk weights BEFORE dispatch:
    #
    #   * Post-MoE scaling path (default): DeepseekV2MoE.forward_deepep later
    #     multiplies the whole MoE output by routed_scaling_factor, so the shared
    #     weight must be 1/routed_scaling_factor for (1/rsf) * rsf = 1.0.
    #   * aiter (HIP) path: aiter_biased_grouped_topk folds routed_scaling_factor
    #     into each routed topk weight, and forward_deepep SKIPS the post-MoE
    #     multiply for _use_aiter (see its `not (... or _use_aiter)` guard). The
    #     shared weight must therefore be 1.0 -- applying 1/rsf here would
    #     under-weight the always-on shared expert by routed_scaling_factor and
    #     corrupt every MoE layer.
    #
    # NOTE: forward_deepep also skips the post-MoE multiply for the non-aiter
    # families where routed_scaling_factor is pre-folded in topk
    # (should_fuse_routed_scaling_factor_in_topk / apply_routed_scaling_factor_on_output:
    # ModelOpt NVFP4, cutlass/trtllm-routed fp8), so those would likewise need a
    # 1.0 shared weight. This fix is deliberately scoped to the aiter path (the
    # one validated on AMD MI355X); those other backends are left at their
    # existing behavior and can be addressed by their maintainers.
    routed_scaling_factor = topk_config.routed_scaling_factor
    if _use_aiter:
        topk_weights[:, -num_fused_shared_experts:] = 1.0
    elif routed_scaling_factor is not None and routed_scaling_factor != 0:
        topk_weights[:, -num_fused_shared_experts:] = 1.0 / routed_scaling_factor

    return topk_ids, topk_weights


def capture_routed_experts_if_allowed(
    topk_config: TopKConfig,
    layer_id: Optional[int],
    topk_ids: torch.Tensor,
) -> None:
    """Single capture site for every backend, gated by the per-config opt-out.

    Routing all backends through here keeps the draft-side opt-out from being
    bypassed by an inlined capturer call.
    """
    if not topk_config.allow_routed_experts_capture:
        return
    if (cap := get_global_experts_capturer()) is not None:
        cap.capture(
            layer_id=layer_id,
            topk_indices=topk_ids,
        )


def _post_process_topk_ids(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_config: TopKConfig,
    router_logits: torch.Tensor,
    layer_id: int,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    use_per_rank_shared_slots = has_per_rank_fused_shared_slots(
        num_fused_shared_experts
    )
    fused_shared_experts_scaling_factor = (
        topk_config.fused_shared_experts_scaling_factor
    )
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)
    recorder_topk_ids = None
    if _is_cuda:
        # LP path: solve LP outside torch.compile (the solver contains an
        # EP all-reduce that can't run inside compiled regions).
        log2phy_prob = None
        if (
            expert_location_dispatch_info is not None
            and getattr(expert_location_dispatch_info, "ep_dispatch_algorithm", None)
            == "lp"
        ):
            from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

            lplb_solver = get_global_lplb_solver(layer_id)
            if lplb_solver is not None:
                log2phy_prob = lplb_solver.solve(topk_ids)

        if log2phy_prob is not None:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info, log2phy_prob
            )
            _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
        elif use_per_rank_shared_slots:
            # Shared experts appended as extra columns in topk_ids: their value
            # would be out-of-bounds for the logical-to-physical dispatch table,
            # so split, dispatch the routed cols, recombine.
            shared_cols = topk_ids[:, -num_fused_shared_experts:]
            routed_cols = topk_ids[:, :-num_fused_shared_experts]
            routed_cols = _biased_grouped_topk_postprocess(
                routed_cols, expert_location_dispatch_info, num_token_non_padded
            )
            topk_ids = torch.cat([routed_cols, shared_cols], dim=-1)
            # ExpertDistributionRecorder tracks EPLB physical routed experts.
            # Per-rank shared-slot remap later adds shared slots to the topk ID
            # space, so keep the routed physical ids separately for statistics.
            recorder_topk_ids = routed_cols
        else:
            topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )
    elif _is_hip:
        # On AMD HIP the aiter MoE kernels do not handle topk_ids=-1 safely
        # (negative indices cause illegal memory access). Always fill the padded
        # region with 0 so every kernel sees a valid in-range expert id.
        # Routing weights for padded tokens are zeroed below so their
        # contribution to the hidden state is still zero regardless of the id.
        # Regression: skipping this mask when EPLB is disabled caused garbage
        # MoE routing for models like DeepSeek-R1-MXFP4 (accuracy ~0.09 vs 0.94+).
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded, fill_value=0)
        # The logical->physical remap is only meaningful when a real
        # expert-location mapping exists. With a trivial placement and EPLB off
        # the map is identity so the remap can be skipped safely.
        if _eplb_remap_enabled():
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info
            )
        # NOTE (HIP): padded-token routing-weight zeroing is deferred to the
        # single pass at the end of this function (gated by SGLANG_MORI_NO_PAD_MASK).
        # That final pass re-zeros after any shared-expert append/remap, so a
        # second zeroing here would be redundant (zeroing is idempotent).

    if recorder_topk_ids is None:
        recorder_topk_ids = topk_ids

    _aiter_append = num_fused_shared_experts > 0 and _use_aiter

    if _aiter_append and use_per_rank_shared_slots:
        # Fused path: append shared experts AND apply the per-rank shared-slot
        # remap in a single Triton kernel. This replaces the original
        # fused_append_shared_experts() + eager per-rank shared-slot remap pair,
        # collapsing ~6 launch-bound elementwise kernels/layer (div_floor / add /
        # arange / fill / copy) into the one append kernel that already runs.
        #
        # Shared weight is 1.0 here because this branch is aiter-only:
        # aiter_biased_grouped_topk folds routed_scaling_factor into the routed
        # weights and forward_deepep skips the post-MoE multiply for _use_aiter,
        # so the always-on shared expert must contribute 1.0x. (The eager
        # per-rank shared-slot remap instead sets shared weight to
        # 1/routed_scaling_factor to compensate a post-MoE scale that the aiter
        # path does not apply; see PR #28237.)
        num_physical_routed_experts = (
            expert_location_dispatch_info.num_physical_experts
            if expert_location_dispatch_info is not None
            else router_logits.shape[1]
        )
        ep_size = get_parallel().moe_ep_size
        ep_rank = get_parallel().moe_ep_rank
        num_local_routed = num_physical_routed_experts // ep_size
        num_local_experts = num_local_routed + num_fused_shared_experts
        shared_id_base = ep_rank * num_local_experts + num_local_routed

        # Lazy import to avoid circular-import issues
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_remap_shared_experts_deepep,
        )

        topk_ids, topk_weights = fused_append_remap_shared_experts_deepep(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            1.0,  # shared-expert weight on the aiter path
            shared_id_base,
            num_local_routed,
        )
    elif _aiter_append:
        M, N = router_logits.shape
        scale_factor = (
            1.0
            if fused_shared_experts_scaling_factor is None
            else fused_shared_experts_scaling_factor
        )

        # Lazy import to avoid circular-import issues
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_shared_experts,
        )

        topk_ids, topk_weights = fused_append_shared_experts(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            scale_factor,
            N,  # base id for shared experts
        )

    elif use_per_rank_shared_slots:
        # DeepEP/MegaMOE: remap to per-rank shared-slot layout where each
        # rank's shared expert has a unique ID for dispatch routing.
        num_physical_routed_experts = (
            expert_location_dispatch_info.num_physical_experts
            if expert_location_dispatch_info is not None
            else router_logits.shape[1]
        )
        topk_ids, topk_weights = remap_topk_for_per_rank_shared_slots(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            num_physical_routed_experts,
            topk_config,
        )

    if _is_hip and not _skip_hip_pad_mask:
        # Shared-expert append/remap can introduce non-zero weights after the
        # initial HIP padding mask above. Ensure padded tokens leave this helper
        # with all expert weights zeroed.
        _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)

    return topk_ids, topk_weights, recorder_topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    layer_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:
    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )

    scoring_func = topk_config.scoring_func

    # Set by the fused-gating+pack branch below; None everywhere else.
    packed_topk = None

    (
        router_logits,
        correction_bias,
    ) = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    # remove num_fused_shared_experts from grouped_topk/biased_grouped_topk
    num_routed_topk = top_k - num_fused_shared_experts
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )
    elif custom_routing_function is None:
        if scoring_func != "sqrtsoftplus":
            assert not apply_routed_scaling_factor_on_output, "Not implemented"

        if scoring_func == "sqrtsoftplus":
            if _is_cpu and _is_cpu_amx_available:
                topk_weights, topk_ids = biased_topk_cpu(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    correction_bias=correction_bias,
                    topk=top_k,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                    num_fused_shared_experts=num_fused_shared_experts,
                    routed_scaling_factor=routed_scaling_factor,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                    apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
                )
            else:
                _biased_topk = (
                    biased_topk_jit_kernel_impl
                    if envs.SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK.get()
                    else biased_topk_impl
                )

                topk_weights, topk_ids = _biased_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    correction_bias=correction_bias,
                    topk=num_routed_topk if _use_aiter else top_k,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                    num_fused_shared_experts=num_fused_shared_experts,
                    routed_scaling_factor=routed_scaling_factor,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                    apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
                )
        elif (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            and scoring_func == "softmax"
            and correction_bias is None
        ):
            # flashinfer_trtllm_routed uses raw-logits topk
            topk_weights, topk_ids = fused_topk_softmax_torch_raw_logits(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
            )
        else:
            # Fused gating + routed pack (SGLANG_OPT_LORA_FUSED_TOPK_PACK): only on the plain
            # CUDA softmax path with no EPLB remap / shared experts / bias / routing overrides.
            _fused_topk_pack = False
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

                _fused_topk_pack = lora_envs.SGLANG_OPT_LORA_FUSED_TOPK_PACK.get()
            if (
                _fused_topk_pack
                and _is_cuda
                and not _use_aiter
                and scoring_func == "softmax"
                and correction_bias is None
                and expert_location_dispatch_info is None
                and num_fused_shared_experts == 0
                and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
                and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
            ):
                num_experts = router_logits.shape[-1]
                if num_experts & (num_experts - 1) == 0 and num_experts <= 512:
                    packed_topk = torch.empty(
                        (hidden_states.shape[0], top_k),
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )

            # Qwen3MOE uses fused_topk
            _fused_topk_kwargs = {}
            # ===== TO BE REFACTORED ====
            # Only the experimental fused topk+pack passes packed_out/num_token_non_padded;
            # the default call keeps the upstream signature (fused_topk_cpu lacks these).
            if packed_topk is not None:
                _fused_topk_kwargs = dict(
                    packed_out=packed_topk,
                    num_token_non_padded=num_token_non_padded,
                )
            # ===== END TO BE REFACTORED ====
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                correction_bias=correction_bias,
                scoring_func=scoring_func,
                **_fused_topk_kwargs,
            )
    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
        )

    simulate_uniform_experts = envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
    simulate_round_robin_experts = envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    if simulate_uniform_experts and simulate_round_robin_experts:
        raise ValueError(
            "SGLANG_SIMULATE_UNIFORM_EXPERTS and "
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS are mutually exclusive"
        )

    if simulate_uniform_experts:
        # Benchmark-only: override gating with random-offset uniform expert assignment
        # to avoid expert imbalance from dummy/random weights. Do NOT use in production.
        num_tokens, k = topk_ids.shape
        num_experts = router_logits.shape[1]
        if k > 0:
            offsets = torch.randint(
                0, num_experts, (num_tokens, 1), device=topk_ids.device
            )
            steps = torch.arange(k, device=topk_ids.device).unsqueeze(0)
            step = max(num_experts // k, 1)
            topk_ids = ((offsets + steps * step) % num_experts).to(topk_ids.dtype)
            topk_weights = torch.ones_like(topk_weights) / k
    elif simulate_round_robin_experts:
        # Benchmark-only: override gating with deterministic expert assignment
        # to avoid routing noise from dummy/random weights. Do NOT use in production.
        num_tokens, k = topk_ids.shape
        num_experts = router_logits.shape[1]
        topk_ids = _make_round_robin_expert_ids(
            num_tokens,
            k,
            num_experts,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
            layer_id=layer_id,
        )
        if k > 0:
            topk_weights = torch.full_like(topk_weights, 1.0 / k)

    topk_ids, topk_weights, recorder_topk_ids = _post_process_topk_ids(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

    get_global_expert_distribution_recorder().on_select_experts(
        topk_ids=recorder_topk_ids
    )

    # ===== TO BE REFACTORED ====
    if packed_topk is not None:
        return StandardTopKOutputPacked(
            topk_weights, topk_ids, router_logits, packed_topk
        )
    # ===== END TO BE REFACTORED ====
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


# NOTE: the AOT sgl_kernel::moe_fused_gate and sgl_kernel::kimi_k2_moe_fused_gate
# ops (and their torch.compile fake impls) were retired here — both CUDA gate
# paths now route through the unified Triton router (jit_kernel/moe_fused_gate.py),
# whose Python impl is traceable directly, so no register_fake shim is needed.

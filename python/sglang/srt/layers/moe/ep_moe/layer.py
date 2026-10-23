from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    get_is_extend_in_batch,
    set_is_extend_in_batch,
)
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    moe_forward_piecewise_cuda_graph_impl,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu, is_xpu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


logger = logging.getLogger(__name__)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        backend = self.runner_backend
        is_humming = backend is not None and backend.is_humming()
        if get_moe_a2a_backend().is_deepep_v2():
            self.deprecate_flag = True
        elif is_humming:
            self.deprecate_flag = True
        elif _use_aiter:
            self.deprecate_flag = True
        elif _is_npu:
            self.deprecate_flag = True
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            self.deprecate_flag = True
        elif (
            backend is not None
            and backend.is_flashinfer_cutedsl()
            and quant_config is not None
            and quant_config.get_name() in ("modelopt_fp4", "modelopt_mixed")
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and backend is not None
            and backend.is_deep_gemm()
            and quant_config is not None
            and quant_config.get_name() == "mxfp4"
        ):
            # MXFP4 experts (e.g. Kimi K3) on the DeepGEMM fp8_fp4 W4A8 path:
            # route through the modern FusedMoE runner (Mxfp4MoEMethod.apply).
            self.deprecate_flag = True
        elif (
            quant_config is None
            and self.w13_weight.dtype == torch.bfloat16
            and backend is not None
            and backend.is_deep_gemm()
            and (get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_pplx())
            and not _is_npu
            and not _is_hip
        ):
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, (
                "Unquantized DeepEP MoE requires DeepGEMM BF16"
            )
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False

        self.deepep_mode = get_deepep_mode()
        if _is_xpu and self.deepep_mode.enable_low_latency():
            # Both XPU masked runners are BF16 GEMMs; no quantized masked GEMM.
            assert quant_config is None, (
                "Low-latency expert parallelism on Intel XPU supports "
                "unquantized MoE weights only"
            )
            self.xpu_grouped_activation_type = _xpu_grouped_activation_type(
                activation=self.moe_runner_config.activation,
                is_gated=self.moe_runner_config.is_gated,
                num_local_experts=self.num_local_experts,
            )
            self.xpu_masked_activation = _xpu_masked_activation(
                self.moe_runner_config.activation, self.moe_runner_config.is_gated
            )
        elif (
            self.deepep_mode.enable_low_latency()
            and not _is_npu
            and not _is_hip
            and quant_config is not None
        ):
            # AMD HIP and NPU support low_latency DeepEP without DeepGEMM.
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, (
                f"DeepEP {self.deepep_mode} mode requires deep_gemm"
            )

    def _a2a_forward_with_output_impl(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # eager run under breakable cuda graph
        saved_is_extend_in_batch = get_is_extend_in_batch()
        set_is_extend_in_batch(True)
        try:
            output.copy_(
                self.forward_impl(
                    hidden_states,
                    StandardTopKOutput(topk_weights, topk_ids, router_logits),
                )
            )
        finally:
            set_is_extend_in_batch(saved_is_extend_in_batch)

    def _a2a_forward_capture_stub(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Capture pass only: record the buffer address, skip the
        # rank-coupled a2a. Warmup and replay run the real body.
        output.zero_()

    a2a_forward_with_output = eager_on_graph(
        True, capture_stub=_a2a_forward_capture_stub
    )(_a2a_forward_with_output_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        # DeepEP NORMAL mode is not capturable; run it as an eager node.
        if is_in_breakable_cuda_graph():
            assert TopKOutputChecker.format_is_standard(topk_output), (
                "Only standard topk output is supported for breakable cuda graph"
            )
            output = torch.empty_like(hidden_states)
            self.a2a_forward_with_output(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                output,
            )
            return output
        if is_in_tc_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(topk_output), (
                "Only standard topk output is supported for piecewise cuda graph"
            )
            return moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        return self.dispatcher.combine(combine_input=combine_input)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):

        if self.deprecate_flag:
            return super().run_moe_core(dispatch_output)

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.quant_config is None:
                raise NotImplementedError(
                    "Unquantized DeepEP MoE currently supports low_latency mode only"
                )
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            elif _is_xpu:
                output = self._forward_masked_xpu(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )

        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def _forward_masked_xpu(
        self, dispatch_output: DeepEPLLDispatchOutput
    ) -> torch.Tensor:
        hidden_states, _, topk_ids, _, masked_m, _ = dispatch_output
        if self.xpu_grouped_activation_type is not None:
            return _xpu_masked_grouped_experts(
                hidden_states=hidden_states,
                masked_m=masked_m,
                num_topk=topk_ids.shape[1],
                w13=self.w13_weight,
                w2=self.w2_weight,
                activation_type=self.xpu_grouped_activation_type,
            )
        # Fallback: bmm over each expert's first m rows, m covering the fullest
        # expert; one host sync per layer instead of a capacity-sized GEMM.
        capacity = hidden_states.shape[1]
        # Combine reads only each expert's first masked_m rows, so rows past
        # them may hold anything.
        out = torch.empty_like(hidden_states)
        m = int(masked_m.max().item())
        if m == 0:
            return out
        # Rounded so steps with similar occupancy reuse one GEMM shape.
        m = min(capacity, -(-m // 16) * 16)
        gate_up = torch.bmm(hidden_states[:, :m], self.w13_weight.transpose(1, 2))
        out[:, :m] = torch.bmm(
            self.xpu_masked_activation(gate_up), self.w2_weight.transpose(1, 2)
        )
        return out

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )


def _xpu_grouped_activation_type(
    *, activation: str, is_gated: bool, num_local_experts: int
) -> Optional[int]:
    """sgl-kernel grouped-GEMM activation id, or None where that GEMM does not
    apply: it fuses gated silu/gelu only, over a multiple of 8 experts."""
    if not is_gated or num_local_experts % 8 != 0:
        return None
    return {"silu": 0, "gelu": 1}.get(activation)


def _xpu_masked_grouped_experts(
    *,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    num_topk: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    activation_type: int,
) -> torch.Tensor:
    """Expert MLPs over the masked [num_local_experts, capacity, hidden] layout.

    sgl-kernel's grouped GEMM reads the per-expert row counts on the device, so
    nothing waits on the host and experts without rows read no weights. It needs
    each expert's rows contiguous, hence the compaction and expansion copies.
    """
    from sgl_kernel.utils import is_xe3_arch

    from sglang.kernels.ops.moe.ep_moe_kernels import (
        expand_to_masked_slab,
        masked_slab_to_expand,
    )

    num_local_experts, capacity, hidden_size = hidden_states.shape
    # Each expert holds at most `capacity` rows, and each of the capacity source
    # tokens brings at most num_topk of them.
    max_rows = min(num_topk, num_local_experts) * capacity
    psum = torch.cumsum(masked_m, dim=0, dtype=torch.int32)
    rows = masked_slab_to_expand(
        input_tensor=hidden_states,
        psum_num_recv_tokens_per_expert=psum,
        total_expanded_tokens=max_rows,
        expert_alignment=1,
    )
    grouped_mm = (
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35
        if is_xe3_arch()
        else torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20
    )
    act = rows.new_empty((max_rows, w2.shape[2]))
    grouped_mm(act, rows, w13, None, masked_m, num_local_experts, activation_type, True)
    # GEMM1 was the last reader of `rows`; the down projection overwrites it.
    grouped_mm(rows, act, w2, None, masked_m, num_local_experts, activation_type, False)
    # Combine reads only each expert's first masked_m rows.
    out, _, _ = expand_to_masked_slab(
        recv_x=rows,
        recv_x_scale=None,
        psum_num_recv_tokens_per_expert=psum,
        num_local_experts=num_local_experts,
        max_m=capacity,
        expert_alignment=1,
        check_overflow=False,
    )
    return out


def _xpu_masked_activation(
    activation: str, is_gated: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Activation over a [num_experts, m, width] gate-up output."""
    if not is_gated and activation in ("silu", "gelu"):
        return getattr(torch.nn.functional, activation)
    if is_gated and activation in ("silu", "gelu"):
        from sglang.srt.layers.activation import GeluAndMul, SiluAndMul

        op = SiluAndMul() if activation == "silu" else GeluAndMul()

        def act_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
            num_experts, m, width = gate_up.shape
            return op(gate_up.reshape(-1, width)).reshape(num_experts, m, width // 2)

        return act_and_mul
    raise NotImplementedError(
        f"activation={activation} (gated={is_gated}) is not supported by the "
        "Intel XPU low-latency MoE path"
    )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    if get_moe_a2a_backend().is_mori_epv2():
        return FusedMoE
    # [TODO] kk, temporary solution
    if (
        get_moe_a2a_backend().is_mori()
        or get_moe_a2a_backend().is_deepep()
        or get_moe_a2a_backend().is_deepep_v2()
        or get_moe_a2a_backend().is_mooncake()
        or get_moe_a2a_backend().is_nixl()
        or get_moe_a2a_backend().is_pplx()
    ):
        return DeepEPMoE
    return FusedMoE

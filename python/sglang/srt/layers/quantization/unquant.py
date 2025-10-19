from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    set_weight_attrs,
    use_intel_amx_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x, layer.weight, bias, True  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, use_triton_kernels: bool = False):
        super().__init__()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False

        self.triton_kernel_moe_forward = None
        self.triton_kernel_moe_with_bias_forward = None
        if torch.cuda.is_available() and use_triton_kernels:
            from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_forward as _tk_forward,
            )
            from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_with_bias_forward as _tk_with_bias_forward,
            )

            self.triton_kernel_moe_forward = _tk_forward
            self.triton_kernel_moe_with_bias_forward = _tk_with_bias_forward

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_weight_n, w13_weight_k = 2 * intermediate_size_per_partition, hidden_size
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if self.use_triton_kernels:
            if self.with_bias:
                assert self.triton_kernel_moe_with_bias_forward is not None
                output = self.triton_kernel_moe_with_bias_forward(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    b1=layer.w13_weight_bias,
                    b2=layer.w2_weight_bias,
                    topk_output=topk_output,
                    moe_runner_config=moe_runner_config,
                    w1_pcg=None,
                    w2_pcg=None,
                )
            else:
                assert self.triton_kernel_moe_forward is not None
                output = self.triton_kernel_moe_forward(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_output=topk_output,
                    moe_runner_config=moe_runner_config,
                )
            return StandardCombineInput(hidden_states=output)
        else:
            if _use_aiter:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                output = fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if moe_runner_config.activation == "silu"
                        else ActivationType.Gelu
                    ),
                )
                return StandardCombineInput(hidden_states=output)
            else:

                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    b13=getattr(layer, "w13_weight_bias", None),
                    b2=getattr(layer, "w2_weight_bias", None),
                )
                return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if (
            use_intel_amx_backend(layer)
            and not moe_runner_config.apply_router_weight_on_input
        ):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                False,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                None,  # w1_scale
                None,  # w2_scale
                None,  # block_size
                None,  # a1_scale
                None,  # a2_scale
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        import torch_npu

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
            .view(top_k, -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch_npu.npu_moe_init_routing(
                x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
            )
        )

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )

        expert_tokens = expert_tokens.to(torch.int64)
        if layer.w13_weight.shape[-1] == layer.hidden_size:
            w13 = layer.w13_weight.transpose(1, 2)
            w2 = layer.w2_weight.transpose(1, 2)

        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w13],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "silu":
            hidden_states = torch_npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu

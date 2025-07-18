import importlib
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
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

has_triton_kernels = importlib.util.find_spec("triton_kernels") is not None


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.fused_moe_bf16_asm import ck_moe_2stages
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
            return torch.ops.sgl_kernel.weight_packed_linear(
                x, layer.weight, bias, True  # is_vnni
            )

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, use_triton_kernels: bool = False):
        super().__init__()
        self.use_triton_kernels = use_triton_kernels

        from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

        if torch.cuda.is_available():
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts

            if has_triton_kernels:
                from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                    triton_kernel_moe_forward,
                )
            else:
                triton_kernel_moe_forward = None
        else:
            fused_experts = None  # type: ignore
            triton_kernel_moe_forward = None

        self.moe_forward_native = moe_forward_native
        self.fused_experts = fused_experts
        self.triton_kernel_moe_forward = triton_kernel_moe_forward

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight_n, w13_weight_k = 2 * intermediate_size, hidden_size
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:

        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:

        if self.use_triton_kernels:
            return self.triton_kernel_moe_forward(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
        else:
            from sglang.srt.layers.moe.topk import select_experts

            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                num_fused_shared_experts=num_fused_shared_experts,
                custom_routing_function=custom_routing_function,
                correction_bias=correction_bias,
                routed_scaling_factor=routed_scaling_factor,
            )

            if _use_aiter:
                assert not no_combine, "unsupported"
                if apply_router_weight_on_input:
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

                return fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if activation == "silu"
                        else ActivationType.Gelu
                    ),
                )
            else:
                return self.fused_experts(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=inplace and not no_combine,
                    activation=activation,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    no_combine=no_combine,
                    routed_scaling_factor=routed_scaling_factor,
                )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        assert activation == "silu", f"activation = {activation} is not supported."

        if use_intel_amx_backend(layer):

            from sglang.srt.layers.moe.topk import (
                apply_topk_weights_cpu,
                select_experts,
            )

            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                num_fused_shared_experts=num_fused_shared_experts,
                custom_routing_function=custom_routing_function,
                correction_bias=correction_bias,
                routed_scaling_factor=routed_scaling_factor,
            )
            x, topk_weights = apply_topk_weights_cpu(
                apply_router_weight_on_input, topk_weights, x
            )

            return torch.ops.sgl_kernel.fused_experts_cpu(
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
        else:
            return self.moe_forward_native(
                layer,
                x,
                use_grouped_topk,
                top_k,
                router_logits,
                renormalize,
                topk_group,
                num_expert_group,
                num_fused_shared_experts,
                custom_routing_function,
                correction_bias,
                activation,
                apply_router_weight_on_input,
                inplace,
                no_combine,
                routed_scaling_factor,
            )

    def forward_npu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        return self.moe_forward_native(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            num_fused_shared_experts,
            custom_routing_function,
            correction_bias,
            activation,
            apply_router_weight_on_input,
            inplace,
            no_combine,
            routed_scaling_factor,
        )

    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu


class UnquantizedEPMoEMethod(FusedMoEMethodBase, CustomOp):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        layer.register_parameter("w13_input_scale", None)
        layer.register_parameter("w13_weight_scale", None)

        ones_tensor = torch.ones(num_experts_per_partition, dtype=torch.float32)

        w2_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

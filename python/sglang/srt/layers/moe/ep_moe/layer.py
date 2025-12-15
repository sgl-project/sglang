from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4DynamicMoEMethod,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
elif _is_npu:
    import torch_npu


logger = logging.getLogger(__name__)


if _is_npu:
    import torch_npu


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

        if _use_aiter or _is_npu:
            self.deprecate_flag = False
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
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
            self.use_w4afp8 = False

        self.deepep_mode = get_deepep_mode()

        if (
            self.deepep_mode.enable_low_latency()
            and not _is_npu
            and not (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            )
        ):
            # NPU supports low_latency deepep without deepgemm
            # FP4 quantization with flashinfer_cutedsl also supports low_latency deepep without deepgemm
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward(
                hidden_states,
                topk_output,
            )

        # TODO: can we call super().forward here?
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )

        return hidden_states

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
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
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

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert (
            envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
                if not input_quant and not isinstance(
                    self.quant_method, NPUW4A16Int4DynamicMoEMethod
                ):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")

        return hidden_states


class NpuFuseEPMoE(DeepEPMoE):
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

        self.quant_method.process_weights_after_loading = (
            self._process_weights_after_loading
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
            gmm1_permuted_weight=self.w13_weight,
            gmm1_permuted_weight_scale=self.w13_weight_scale,
            gmm2_weight=self.w2_weight,
            gmm2_weight_scale=self.w2_weight_scale,
        ).hidden_state

    def release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def permute_w13_weight_scale(self, w: torch.Tensor, tile_n: int):
        if tile_n % 2 != 0:
            raise ValueError(f"tile_n must be even, got {tile_n}")

        *dims, n = w.shape
        if n % tile_n != 0:
            raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

        w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)

        # Permute the last two dimensions.
        perm_order = list(range(len(dims))) + [-2, -3, -1]
        w_permuted = w_reshaped.permute(perm_order)

        return w_permuted.reshape(*dims, n)

    def reshape_w13_weight(self, weight: torch.Tensor, dim: int, chunk_size: int = 64):
        # Achieving greater computing power through reshape on Ascend.
        original_shape = weight.shape
        if dim < 0:
            dim += len(original_shape)

        if original_shape[dim] % (2 * chunk_size) != 0:
            raise ValueError(
                f"Dimension {dim} size {original_shape[dim]} must be divisible by {2 * chunk_size}"
            )

        new_shape = (
            *original_shape[:dim],
            2,
            original_shape[dim] // (2 * chunk_size),
            chunk_size,
            *original_shape[dim + 1 :],
        )

        weight = weight.view(new_shape)
        weight = weight.transpose(dim, dim + 1).contiguous()

        return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13 = self.release_weight_cache(layer.w13_weight)
        torch_npu.npu_format_cast_(w13, 2)
        cpu_w13 = w13.cpu()
        w13 = self.reshape_w13_weight(cpu_w13, -1).npu()
        torch_npu.npu_format_cast_(w13, 29)
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)

        w2 = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)

        w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
        w13_scale = self.permute_w13_weight_scale(w13_scale, 128)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale.to(torch.float32), requires_grad=False
        )

        w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale.to(torch.float32), requires_grad=False
        )

        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        return DeepEPMoE
    if get_moe_a2a_backend().is_ascend_fuseep():
        return NpuFuseEPMoE

    if get_moe_runner_backend().is_flashinfer_trtllm():
        # NEW: Direct FP4 detection (bypasses EP requirements)
        # Check for FP4 quantization with TRTLLM flag, regardless of EP
        # FlashInferFP4MoE must be paired with ModelOptNvFp4FusedMoEMethod.
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFP4MoE

            return FlashInferFP4MoE
        elif (
            quant_config is None
            or quant_config.get_name() == "fp8"
            or quant_config.get_name() == "modelopt_fp8"
        ):
            # FlashInferFusedMoE support bf16 and fp8
            return FlashInferFusedMoE

    if get_moe_runner_backend().is_flashinfer_cutlass():
        return FusedMoE
    return FusedMoE

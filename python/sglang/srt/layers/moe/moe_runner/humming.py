from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from weakref import WeakValueDictionary

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.ep_moe.kernels import moe_permute, moe_unpermute
from sglang.srt.layers.moe.fused_moe_triton.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)


try:
    from humming import dtypes
    from humming.config import GemmType as HummingGemmType
    from humming.layer import HummingMethod

    _humming_available = True
except ModuleNotFoundError:
    _humming_available = False


def get_standard_humming_moe_gemm_type() -> HummingGemmType:
    env_gemm_type_str = envs.SGLANG_HUMMING_MOE_GEMM_TYPE.get().lower()
    if env_gemm_type_str == "grouped":
        gemm_type = HummingGemmType.GROUPED_CONTIGUOUS
    elif env_gemm_type_str == "indexed":
        gemm_type = HummingGemmType.INDEXED
    else:
        gemm_type = HummingGemmType.INDEXED

    logger.info_once(f"Using {gemm_type.value} gemm for humming moe")

    return gemm_type


@dataclass
class HummingRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    gemm_type: HummingGemmType
    expert_num_tokens: torch.Tensor | None = None
    expected_m: int | None = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HUMMING


@dataclass
class HummingRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HUMMING


@dataclass
class HummingMoeQuantInfo(MoeQuantInfo):
    pass


@register_custom_op()
def humming_moe_runner_core_run(
    moe_runner_id: int,
    gemm_type: str,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_num_tokens: torch.Tensor | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    runner = HummingRunnerCore.runner_cores[moe_runner_id]
    if gemm_type == "indexed":
        return runner._run_indexed_gemm(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    elif gemm_type == "grouped_contiguous":
        return runner._run_grouped_contiguous_gemm(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    elif gemm_type == "grouped_masked":
        assert expected_m is not None and expert_num_tokens is not None
        return runner._run_grouped_masked_gemm(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expected_m=expected_m,
            expert_num_tokens=expert_num_tokens,
        )
    else:
        raise ValueError(f"Unknown gemm type: {gemm_type}")


class HummingRunnerCore(MoeRunnerCore):
    runner_cores: WeakValueDictionary = WeakValueDictionary()

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert config.layer is not None
        self.layer = config.layer
        assert config.num_local_experts is not None
        assert config.num_experts is not None
        self.num_experts = config.num_local_experts
        self.global_num_experts = config.num_experts
        self.activation = config.activation
        self.swiglu_limit = config.swiglu_limit
        self.humming_gemm_configs = {}
        HummingRunnerCore.runner_cores[id(self)] = self

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HUMMING

    def get_humming_gemm_configs(self, humming_gemm_type: HummingGemmType):
        if humming_gemm_type.value in self.humming_gemm_configs:
            return self.humming_gemm_configs[humming_gemm_type.value]

        compute_config = {
            "use_f16_accum": envs.SGLANG_HUMMING_USE_F16_ACCUM.get(),
            "gemm_type": humming_gemm_type.value,
        }
        w13_tuning_config = HummingMethod.get_default_tuning_configs(
            layer=self.layer,
            use_f16_accum=envs.SGLANG_HUMMING_USE_F16_ACCUM.get(),
            gemm_type=humming_gemm_type,
            sublayer_name="w13",
        )
        w2_tuning_config = HummingMethod.get_default_tuning_configs(
            layer=self.layer,
            use_f16_accum=envs.SGLANG_HUMMING_USE_F16_ACCUM.get(),
            gemm_type=humming_gemm_type,
            sublayer_name="w2",
        )
        self.humming_gemm_configs[humming_gemm_type.value] = {
            "compute_config": compute_config,
            "w13_tuning_config": w13_tuning_config,
            "w2_tuning_config": w2_tuning_config,
            "compute_config_str": json.dumps(compute_config),
            "w13_tuning_config_str": json.dumps(w13_tuning_config),
            "w2_tuning_config_str": json.dumps(w2_tuning_config),
        }

        return self.humming_gemm_configs[humming_gemm_type.value]

    def estimate_local_valid_shape_m(
        self,
        topk_ids: torch.Tensor,
        expected_m: int | None = None,
    ):
        # estimate shape_m for kernel tuning
        if expected_m is not None:
            return expected_m * self.num_experts

        # TODO: update for EP and DP
        return topk_ids.nelement()

    def get_buffer_metas(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ):
        num_experts = self.num_experts
        N = self.layer.intermediate_size_per_partition
        K = self.layer.hidden_size
        assert isinstance(num_experts, int)
        assert isinstance(N, int)
        assert isinstance(K, int)

        # hidden_states
        # (-> quanted_gate_up_input) (if not BF16/FP16 activation)
        # -> gate_up_output
        # -> activation_output
        # (-> quanted_down_input) (if not BF16/FP16 activation)
        # -> down_output
        # (-> output) (if not is_grouped_masked)
        # Neighboring nodes are required to utilize distinct workspaces.
        # The output must be derived from workspace1.

        is_grouped_masked = gemm_type == HummingGemmType.GROUPED_MASKED
        output_shape: tuple[int, ...]
        if gemm_type == HummingGemmType.GROUPED_MASKED:
            if hidden_states.ndim == 3:
                max_num_tokens = hidden_states.size(1)
            else:
                max_num_tokens = hidden_states.size(0) // num_experts
            input_shape_m = num_experts * max_num_tokens
            real_shape_m = num_experts * max_num_tokens
            output_shape = (num_experts, max_num_tokens, K)
        else:
            input_shape_m = hidden_states.size(0)
            real_shape_m = hidden_states.size(0) * topk_ids.size(1)
            if gemm_type == HummingGemmType.GROUPED_CONTIGUOUS:
                input_shape_m = real_shape_m
            output_shape = (hidden_states.size(0), K)

        down_input_size = N
        a_dtype = self.layer.humming_metas["w13"].a_dtype
        c_dtype = self.layer.humming_metas["w13"].c_dtype
        num_bits = a_dtype.num_bits
        torch_dtype_map = {
            dtypes.float16: torch.float16,
            dtypes.bfloat16: torch.bfloat16,
            dtypes.float8e4m3: torch.float8_e4m3fn,
            dtypes.int8: torch.int8,
            dtypes.int4: torch.uint8,
        }

        buffer_metas = {
            "quanted_gate_up_input": {
                "shape": (input_shape_m, K),
                "dtype": torch_dtype_map[a_dtype],
            },
            "gate_up_output": {
                "shape": (real_shape_m, N * 2),
                "dtype": torch_dtype_map[c_dtype],
            },
            "activation_output": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[c_dtype],
            },
            "quanted_down_input": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[a_dtype],
            },
            "down_output": {
                "shape": output_shape if is_grouped_masked else (real_shape_m, K),
                "dtype": torch_dtype_map[c_dtype],
            },
            "output": {
                "shape": output_shape,
                "dtype": torch_dtype_map[c_dtype],
            },
        }

        for key in buffer_metas:
            meta = buffer_metas[key]
            if "quanted" in key and a_dtype.num_bits == 4:
                meta["shape"] = meta["shape"][:-1] + (meta["shape"][-1] // 2,)

        if num_bits == 16:
            required_buffers = ["gate_up_output", "activation_output", "down_output"]
        else:
            required_buffers = [
                "quanted_gate_up_input",
                "gate_up_output",
                "activation_output",
                "quanted_down_input",
                "down_output",
            ]

        # grouped masked moe use down_output as output
        if gemm_type != HummingGemmType.GROUPED_MASKED:
            required_buffers.append("output")

        return buffer_metas, required_buffers

    def _workspace_shapes(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ):
        buffer_metas, required_buffers = self.get_buffer_metas(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=gemm_type,
        )

        workspace1_nbytes = 0
        workspace2_nbytes = 0

        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            nelement = math.prod(buffer_meta["shape"])
            nbytes = nelement * buffer_meta["dtype"].itemsize
            if index % 2 == 0:
                workspace1_nbytes = max(workspace1_nbytes, nbytes)
            else:
                workspace2_nbytes = max(workspace2_nbytes, nbytes)

        output_key = (
            "down_output" if gemm_type == HummingGemmType.GROUPED_MASKED else "output"
        )
        output_shape = buffer_metas[output_key]["shape"]

        return (workspace1_nbytes // 2,), (workspace2_nbytes // 2,), output_shape

    def make_workspaces(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ):
        shapes = self._workspace_shapes(hidden_states, topk_ids, gemm_type)
        workspace1_shape, workspace2_shape, output_shape = shapes
        torch_dtype = self.layer.params_dtype
        device = hidden_states.device
        workspace1 = torch.empty(workspace1_shape, dtype=torch_dtype, device=device)
        workspace2 = torch.empty(workspace2_shape, dtype=torch_dtype, device=device)
        output = workspace1[: math.prod(output_shape)].view(*output_shape)
        return workspace1, workspace2, output

    def prepare_buffers(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ) -> dict[str, torch.Tensor]:
        workspace1, workspace2, output = self.make_workspaces(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=gemm_type,
        )
        buffer_metas, required_buffers = self.get_buffer_metas(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=gemm_type,
        )
        buffers = {"output": output}
        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            workspace = workspace1 if index % 2 == 0 else workspace2
            workspace = workspace.view(buffer_meta["dtype"])
            shape = buffer_meta["shape"]
            tensor = workspace[: math.prod(shape)].view(*shape)
            buffers[name] = tensor

        return buffers

    def apply_activation(self, inputs: torch.Tensor, outputs: torch.Tensor):
        if self.activation == "silu" and self.swiglu_limit is not None:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
                act_and_mul_triton,
            )

            in_2d = inputs.view(-1, inputs.shape[-1])
            out_2d = outputs.view(-1, outputs.shape[-1])
            act_and_mul_triton(
                gateup_output=in_2d,
                down_input=out_2d,
                config={},
                activation="silu",
                swiglu_limit=float(self.swiglu_limit),
            )
            return
        if self.activation == "silu":
            from sgl_kernel import silu_and_mul

            silu_and_mul(inputs, outputs)
        elif self.activation == "gelu":
            from sgl_kernel import gelu_and_mul

            gelu_and_mul(inputs, outputs)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def run(
        self,
        runner_input: HummingRunnerInput,
        quant_info: HummingMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> HummingRunnerOutput:
        if runner_input.hidden_states.size(0) == 0:
            return HummingRunnerOutput(
                hidden_states=torch.empty_like(runner_input.hidden_states)
            )

        # To make it compatible with dynamic shapes in torch.compile,
        # we wrap the main logic inside a torch op.
        # (the moe_block_size selection in indexed gemm would break dynamic shapes).
        output = humming_moe_runner_core_run(
            moe_runner_id=id(self),
            gemm_type=runner_input.gemm_type.value,
            hidden_states=runner_input.hidden_states,
            topk_weights=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            expected_m=runner_input.expected_m,
            expert_num_tokens=runner_input.expert_num_tokens,
        )

        return HummingRunnerOutput(hidden_states=output)

    def _prepare_indexed_gemm_kwargs(
        self, topk_ids: torch.Tensor
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

        configs = self.get_humming_gemm_configs(HummingGemmType.INDEXED)
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        for min_shape_m, max_shape_m, config in configs["w13_tuning_config"]:
            if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
                moe_block_size = config["block_shape"][0]
                break
        else:
            raise ValueError(f"cannot found moe_block_size for shape {valid_shape_m}")

        sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=moe_block_size,
            num_experts=self.num_experts,
            ignore_invalid_expert=True,
        )

        moe_common_kwargs = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "compute_config": configs["compute_config_str"],
            "valid_shape_m": valid_shape_m,
        }

        top_k = topk_ids.size(1)
        moe_kwargs1 = {
            "top_k": top_k,
            "tuning_config": configs["w13_tuning_config_str"],
        }
        moe_kwargs2 = {"top_k": 1, "tuning_config": configs["w2_tuning_config_str"]}
        moe_kwargs1.update(moe_common_kwargs)
        moe_kwargs2.update(moe_common_kwargs)

        return moe_kwargs1, moe_kwargs2

    def _run_indexed_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        buffers = self.prepare_buffers(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=HummingGemmType.INDEXED,
        )

        moe_kwargs1, moe_kwargs2 = self._prepare_indexed_gemm_kwargs(topk_ids)

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            sublayer_name="w13",
            **moe_kwargs1,
        )

        self.apply_activation(
            inputs=buffers["gate_up_output"],
            outputs=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            sublayer_name="w2",
            **moe_kwargs2,
        )

        moe_fused_mul_sum(
            inputs=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            is_ep=self.num_experts != self.global_num_experts,
            routed_scaling_factor=self.config.routed_scaling_factor,
            outputs=buffers["output"],
        )

        return buffers["output"]

    def _run_grouped_contiguous_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        configs = self.get_humming_gemm_configs(HummingGemmType.GROUPED_CONTIGUOUS)
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        buffers = self.prepare_buffers(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=HummingGemmType.GROUPED_CONTIGUOUS,
        )

        hidden_states, src2dst, expert_first_token_offset = moe_permute(
            inputs=hidden_states,
            topk_ids=topk_ids,
            num_experts=self.num_experts,
            is_ep=self.num_experts != self.global_num_experts,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w13_tuning_config_str"],
            sublayer_name="w13",
        )

        self.apply_activation(
            inputs=buffers["gate_up_output"],
            outputs=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w2_tuning_config_str"],
            sublayer_name="w2",
        )

        moe_unpermute(
            outputs=buffers["output"],
            inputs=buffers["down_output"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            src2dst=src2dst,
            routed_scaling_factor=self.config.routed_scaling_factor,
        )

        return buffers["output"]

    def _run_grouped_masked_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        expected_m: int,
    ):
        configs = self.get_humming_gemm_configs(HummingGemmType.GROUPED_MASKED)
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids, expected_m)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        buffers = self.prepare_buffers(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=HummingGemmType.GROUPED_MASKED,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w13_tuning_config_str"],
            sublayer_name="w13",
        )

        self.apply_activation(
            inputs=buffers["gate_up_output"],
            outputs=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w2_tuning_config_str"],
            sublayer_name="w2",
        )

        return buffers["down_output"]


@register_fused_func("none", "humming")
def fused_experts_none_to_humming(
    dispatch_output: StandardDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    runner_input = HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        gemm_type=get_standard_humming_moe_gemm_type(),
    )

    runner_core = HummingRunnerCore(runner_config)
    runner_output = runner_core.run(runner_input, quant_info, {})

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


@register_pre_permute("deepep_ll", "humming")
def pre_permute_deepep_ll_to_humming(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HummingRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.int(),
        expert_num_tokens=dispatch_output.masked_m,
        expected_m=dispatch_output.expected_m,
        gemm_type=HummingGemmType.GROUPED_MASKED,
    )


@register_post_permute("humming", "deepep_ll")
def post_permute_humming_to_deepep_ll(
    runner_output: HummingRunnerOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    topk_weights = running_state["topk_weights"]
    routed_scaling_factor = runner_config.routed_scaling_factor
    # deepep_ll combine is an external weighted sum and cannot fuse the
    # routed_scaling_factor inside its reduce kernel. Pre-scaling topk_weights
    # is mathematically equivalent to scaling the combined output afterwards,
    # since combine is linear in topk_weights: s * sum_k(w_k * x_k) ==
    # sum_k((s * w_k) * x_k).
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=topk_weights,
    )


@register_pre_permute("deepep_normal", "humming")
def pre_permute_deepep_normal_to_humming(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HummingRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.int(),
        gemm_type=get_standard_humming_moe_gemm_type(),
    )


@register_post_permute("humming", "deepep_normal")
def post_permute_humming_to_deepep_normal(
    runner_output: HummingRunnerOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("standard", "humming")
def pre_permute_standard_to_humming(
    dispatch_output: StandardDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HummingRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    return HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.int(),
        gemm_type=get_standard_humming_moe_gemm_type(),
    )


@register_post_permute("humming", "standard")
def post_permute_humming_to_standard(
    runner_output: HummingRunnerOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)

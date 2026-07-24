from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import einops
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import silu_and_mul_masked_post_quant
from sglang.kernels.ops.quantization import per_token_group_quant
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import (
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
)
from sglang.srt.utils.offloader import get_offloader

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

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_musa = is_musa()

# Imported only for the SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False fallback path.
if not (_is_npu or _is_hip) and _is_cuda:
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul as _legacy_silu_and_mul,
    )
elif _is_musa:
    _silu_and_mul_musa = torch.nn.SwishGLU()
else:
    _legacy_silu_and_mul = None


_DEEPGEMM_ON_H20 = get_bool_env_var("SGLANG_DEEPGEMM_ON_H20")


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile(disable=_is_hip or _is_npu)
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu


@dataclass
class DeepGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    use_fp8: bool
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    # DSV4 mxfp4 layout flag; selects recipe_a=(1,128)/recipe_b=(1,32) downstream.
    is_fp4_experts: bool = False
    use_mxfp8: bool = False

    def __post_init__(self):
        if self.use_mxfp8:
            assert self.block_shape == [
                1,
                32,
            ], f"MXFP8 requires block_shape [1, 32], got {self.block_shape}"
            assert (
                deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            ), "MXFP8 requires DEEPGEMM_SCALE_UE8M0=True"


class DeepGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        # SiTU (Kimi K3) is applied outside the GEMMs in python, so it only
        # needs the masked-gemm activation site to branch (see _run_masked_gemm).
        assert self.config.activation in ("silu", "situ")
        assert self.config.is_gated
        self.swiglu_limit = self.config.swiglu_limit
        self.use_swizzle = False
        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
            assert envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get()
            assert envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            self.use_swizzle = True

    def run(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> DeepGemmRunnerOutput:
        weight_dtype = quant_info.w13_weight.dtype
        if not runner_input.use_masked_gemm:
            if weight_dtype == torch.bfloat16:
                hidden_states = self._run_bf16_contiguous_gemm(
                    runner_input, quant_info, running_state
                )
            else:
                hidden_states = self._run_contiguous_gemm(
                    runner_input, quant_info, running_state
                )
        else:
            if weight_dtype == torch.bfloat16:
                hidden_states = self._run_masked_bf16_gemm(
                    runner_input, quant_info, running_state
                )
            else:
                hidden_states = self._run_masked_gemm(
                    runner_input, quant_info, running_state
                )
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant
        from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]
        scale_block_size = 128

        recipe_a, recipe_b = (
            ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
        )

        w13_weight_fp8 = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_fp8 = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if self.config.activation == "situ":
            situ_beta = self.config.gemm1_alpha
            situ_linear_beta = self.config.gemm1_clamp_limit
            assert situ_beta is not None and situ_linear_beta is not None
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
                # Fused SiTU + per-group fp8 quant over the compacted rows,
                # then the proven round-up e8m0 cast (mn-major packed layout).
                rows = gateup_output.shape[0]
                half_n = N // 2
                kg = half_n // scale_block_size
                down_input_fp8 = torch.empty(
                    (rows, half_n),
                    device=gateup_output.device,
                    dtype=torch.float8_e4m3fn,
                )
                s = torch.empty(
                    (rows, kg), device=gateup_output.device, dtype=torch.float32
                )
                _situ_mul_quant_contig_kernel[(rows,)](
                    gateup_output,
                    down_input_fp8,
                    s,
                    half_n,
                    kg,
                    situ_beta,
                    situ_linear_beta,
                    GROUP=scale_block_size,
                    KG_POW2=triton.next_power_of_2(kg),
                    num_warps=8,
                )
                del gateup_output
                down_input_scale = _cast_to_e8m0_with_rounding_up(
                    s.unsqueeze(0)
                ).squeeze(0)
            else:
                from sglang.kernels.ops.quantization.fp8_kernel import (
                    sglang_per_token_group_quant_fp8,
                )

                gate = gateup_output[:, : N // 2].float()
                up = gateup_output[:, N // 2 :].float()
                gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
                up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
                down_input = (gate * up).to(torch.bfloat16)
                del gateup_output

                down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                    down_input,
                    scale_block_size,
                    column_major_scales=False,
                    scale_tma_aligned=False,
                    scale_ue8m0=False,
                )
                del down_input
        elif envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
            swiglu_limit_arg: Optional[float] = self.swiglu_limit

            down_input_fp8 = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(all_tokens, N // 2),
                device=gateup_output.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gateup_output,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=swiglu_limit_arg,
                swizzle=self.use_swizzle,
            )
            del gateup_output
        else:
            # Hacky byte-equal fallback that reproduces the optimize-branch
            # code path exactly: bf16 silu_and_mul then a separate per-token
            # group fp8 quant. Kept behind the mega-moe-memory flag.
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if self.swiglu_limit is not None:
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )

            if not _is_musa:
                down_input = torch.empty(
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            else:
                down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
            del gateup_output

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input

        # Allocate the MoE output in the NCCL symmetric memory pool when symmetric
        # allocation is required, so the downstream all-reduce takes the low-latency
        # symmetric path. Only this final output enters the pool; intermediate
        # buffers stay on the default allocator to bound pool occupancy.
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (all_tokens, K),
                device=hidden_states_device,
                dtype=torch.bfloat16,
            )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = tma_align_input_scale(down_input_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

        return down_output

    def _run_bf16_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:

        hidden_states = runner_input.hidden_states
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight

        # GroupGemm-1: (M, K) (E, N, K) -> (M, N)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        deep_gemm_wrapper.grouped_gemm_nt_bf16_contig(
            hidden_states,
            w13_weight,
            gateup_output,
            m_indices,
        )

        dispose_tensor(hidden_states)

        # Act: (M, N) -> (M, N/2)
        if not _is_musa:
            down_input = torch.empty(
                (
                    all_tokens,
                    N // 2,
                ),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
        else:
            down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
        del gateup_output

        # GroupGemm-2: (M, N/2) (E, K, N/2) -> (M, K)
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (all_tokens, K),
                device=hidden_states_device,
                dtype=torch.bfloat16,
            )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_contig(
            down_input,
            w2_weight,
            down_output,
            m_indices,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers import deep_gemm_wrapper

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        use_mxfp8 = quant_info.use_mxfp8
        scale_block_size = quant_info.block_shape[1] if quant_info.block_shape else 128

        if use_mxfp8:
            recipe_b = tuple(quant_info.block_shape)
            # gran_k is set by the dispatch path (standard=block_shape[1], DeepEP-LL=128),
            # not inferable from K; inferring it silently mis-reads the activation scale.
            gran_k_act = running_state.get(
                "mxfp8_act_gran_k", quant_info.block_shape[1]
            )
            _, _, k_for_recipe = hidden_states.shape
            act_sf_last = hidden_states_scale.shape[-1]
            assert ceil_div(k_for_recipe, gran_k_act * 4) == act_sf_last, (
                f"MXFP8 gateup scale mismatch: gran_k={gran_k_act}, K={k_for_recipe}, "
                f"act_sf_last={act_sf_last}, expected "
                f"{ceil_div(k_for_recipe, gran_k_act * 4)}"
            )
            recipe_a = (quant_info.block_shape[0], gran_k_act)
        elif quant_info.is_fp4_experts:
            recipe_a, recipe_b = (1, 128), (1, 32)
        else:
            recipe_a, recipe_b = None, None

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert (
                    s_mn % 4 == 0 and s_k % 4 == 0
                ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        elif deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            hidden_states_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        swiglu_limit_arg: Optional[float] = None
        if self.swiglu_limit is not None:
            # DeepSeek V4: clamped swiglu requires the DSV4 JIT EP activation.
            assert (
                envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            ), "DeepSeek V4 requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"

            if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
                swiglu_limit_arg = self.swiglu_limit
            else:
                gateup_output = einops.rearrange(
                    gateup_output, "grp tok hidden -> (grp tok) hidden"
                )
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
                gateup_output = einops.rearrange(
                    gateup_output,
                    "(grp tok) hidden -> grp tok hidden",
                    grp=num_groups,
                )

        # Act.
        if self.config.activation == "situ":
            down_input, down_input_scale = _varlen_deep_gemm_situ_mul_quant(
                gateup_output,
                masked_m,
                group_size=128,
                topk=self.config.top_k,
                beta=self.config.gemm1_alpha,
                linear_beta=self.config.gemm1_clamp_limit,
            )
        else:
            topk_ids_rs = running_state.get("topk_ids")
            num_real_tokens = (
                topk_ids_rs.shape[0]
                if (
                    use_mxfp8
                    and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                    and topk_ids_rs is not None
                    and "src2dst" in running_state
                )
                else None
            )
            down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(
                gateup_output,
                masked_m,
                group_size=scale_block_size,
                topk=self.config.top_k,
                swiglu_limit=swiglu_limit_arg,
                swizzle=self.use_swizzle,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit,
                num_real_tokens=num_real_tokens,
            )
        del gateup_output

        # Down activation is quantised locally at scale_block_size (never DeepEP-LL),
        # so its gran_k differs from gateup recipe_a.
        recipe_a_down = recipe_a
        if use_mxfp8:
            recipe_a_down = (quant_info.block_shape[0], scale_block_size)

        # GroupGemm-1
        n = w2_weight.shape[1]

        if (
            use_mxfp8
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and down_input_scale.dtype != torch.int32
        ):
            import deep_gemm.utils.layout

            down_input_scale = (
                deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                    down_input_scale
                )
            )
        elif deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                down_input_scale
            )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
            )

        down_gemm_overlap_args = running_state.get("down_gemm_overlap_args", None)
        if down_gemm_overlap_args is None:
            gemm_overlap_args_dict = {}
        else:
            down_gemm_overlap_args.start_event.record()
            max_block_n = (
                160 if (_DEEPGEMM_ON_H20 and runner_input.expected_m <= 64) else 256
            )
            gemm_overlap_args_dict = {
                "overlap_args": down_gemm_overlap_args,
                "max_block_n": max_block_n,
            }

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a_down,
            recipe_b=recipe_b,
            **gemm_overlap_args_dict,
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        # Returns (block_m, threshold) only with down-gemm overlap, else None;
        # meta_overlap_args may be set without overlap, so guard the unpack.
        if meta_overlap_args is not None and deep_gemm_return_value is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    def _run_masked_bf16_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_fwd
        from sglang.srt.layers import deep_gemm_wrapper

        hidden_states = runner_input.hidden_states
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_masked(
            hidden_states,
            w13_weight,
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)

        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        # Act
        silu_and_mul_masked_fwd(gateup_output, down_input, masked_m)
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
            )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_masked(
            down_input,
            w2_weight,
            down_output,
            masked_m,
            expected_m,
        )
        # Note: BF16 masked gemm doesn't support overlap_args, so no return value unpack

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@register_pre_permute("standard", "deep_gemm")
def pre_permute_standard_to_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import moe_ep_deepgemm_preprocess

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    # The masked grouped GEMM is a capacity-style interface designed for
    # CUDA-graph decode (fixed shapes, unknown m). For prefill-sized eager
    # forwards the contiguous grouped GEMM is the intended interface: tokens
    # are compacted per expert, so no [E, m_max, ...] capacity buffers and
    # far fewer kernels. Small batches (decode eager / graph warmup+capture)
    # stay on the masked path.
    if hidden_states.shape[0] >= 512 and not torch.cuda.is_current_stream_capturing():
        return _pre_permute_standard_contiguous(
            hidden_states, topk_ids, topk_weights, runner_config, running_state
        )

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    topk_weights, topk_ids = topk_weights, topk_ids

    # PreReorder
    output_dtype = (
        torch.bfloat16
        if quant_info.w13_weight.dtype == torch.bfloat16
        else torch.float8_e4m3fn
    )
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (
        moe_ep_deepgemm_preprocess(
            topk_ids,
            runner_config.num_local_experts,
            hidden_states,
            runner_config.top_k,
            quant_info.block_shape,
            output_dtype=output_dtype,
            use_mxfp8=quant_info.use_mxfp8,
        )
    )

    dispose_tensor(hidden_states_ref)

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst
    running_state["mxfp8_act_gran_k"] = (
        quant_info.block_shape[1] if quant_info.block_shape else 128
    )

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


def _pre_permute_standard_contiguous(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    """Standard-dispatch -> contiguous grouped GEMM layout.

    Mirrors the deepep_normal pre-permute (ep_scatter + m_indices), except the
    per-expert counts come from a GPU histogram instead of dispatch metadata.
    To avoid a GPU->CPU sync per layer, buffers are sized to the worst-case
    bound (topk_numel + E * BLOCK_E); rows beyond the real aligned total keep
    m_indices == -1 and are skipped by the GEMM.
    """
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    BLOCK_E = 128  # ep_scatter's per-expert alignment
    num_local = runner_config.num_local_experts
    device = hidden_states.device

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = device
    running_state["contiguous"] = True

    K = hidden_states.shape[1]
    flat = topk_ids.view(-1)
    cnt = torch.zeros(num_local, device=device, dtype=torch.int32)
    cnt.scatter_add_(0, flat.clamp_min(0).to(torch.int64), (flat >= 0).to(torch.int32))
    aligned_cnt = ((cnt + BLOCK_E - 1) // BLOCK_E) * BLOCK_E

    bound = ceil_div(topk_ids.numel(), BLOCK_E) * BLOCK_E + num_local * BLOCK_E
    running_state["all_tokens"] = bound

    # fp8-quantize the tokens once, then scatter quantized rows + scales.
    hs_fp8, hs_scale = per_token_group_quant_fp8(hidden_states, 128)

    input_tensor = torch.empty((bound, K), device=device, dtype=hs_fp8.dtype)
    # Scatter fp32 scales row-major, then do one e8m0 cast over the whole
    # buffer (the GEMM1-proven pipeline). Zero-init: padding rows inside each
    # aligned expert block keep scale == 0 (cast -> 2^-127), so their garbage
    # payload contributes ~exact zeros.
    input_tensor_scale = torch.zeros(
        (bound, K // 128), device=device, dtype=torch.float32
    )
    m_indices = torch.full((bound,), -1, device=device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)
    expert_start_loc = torch.empty_like(cnt)

    ep_scatter(
        hs_fp8,
        hs_scale,
        topk_ids,
        aligned_cnt,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=False,
    )
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        input_tensor_scale = _cast_to_e8m0_with_rounding_up(
            input_tensor_scale.unsqueeze(0)
        ).squeeze(0)
    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


@register_post_permute("deep_gemm", "standard")
def post_permute_deep_gemm_to_standard(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather, post_reorder_deepgemm
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]

    if running_state.get("contiguous", False):
        gather_out = torch.zeros(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        ep_gather(
            runner_output.hidden_states,
            topk_ids,
            topk_weights,
            running_state["output_index"],
            gather_out,
        )
        dispose_tensor(runner_output.hidden_states)
        if runner_config.routed_scaling_factor is not None:
            gather_out *= runner_config.routed_scaling_factor
        return StandardCombineInput(hidden_states=gather_out)

    src2dst = running_state["src2dst"]

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
    post_reorder_deepgemm(
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        runner_config.top_k,
        hidden_states_shape[0],
        hidden_states_shape[1],
        (
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
    )
    dispose_tensor(runner_output.hidden_states)

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("deepep_ll", "deep_gemm")
def pre_permute_deepep_ll_to_deep_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device
    # DeepEP-LL FP8 dispatch quantises activations at a fixed 128 block, not the checkpoint block_shape.
    running_state["mxfp8_act_gran_k"] = 128

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "deepep_ll")
def post_permute_deep_gemm_to_deepep_ll(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "deep_gemm")
def pre_permute_deepep_normal_to_deep_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation in ("silu", "situ")

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # TODO check whether need `zeros`
        input_tensor_scale = torch.zeros(
            (ceil_div(K // 128, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // 128),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )
    dispose_tensor(hidden_states)
    if hidden_states_scale is not None:
        dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


@register_post_permute("deep_gemm", "deepep_normal")
def post_permute_deep_gemm_to_deepep_normal(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]

    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)

    return DeepEPNormalCombineInput(
        hidden_states=gather_out,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


def _varlen_deep_gemm_situ_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int,
    topk: int,
    beta: float,
    linear_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU activation + per-group fp8 quant via CUDA JIT kernel."""
    from sglang.kernels.ops.kimi_k3 import situ_and_mul_masked_post_quant

    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    G = D // group_size
    packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0

    down_input = torch.empty(
        (E, N, D), device=gateup_output.device, dtype=torch.float8_e4m3fn
    )
    if packed_ue8m0:
        down_input_scale = torch.empty(
            (E, G // 4, N), device=gateup_output.device, dtype=torch.int32
        )
    else:
        down_input_scale = torch.empty(
            (E, N, G), device=gateup_output.device, dtype=torch.float32
        )

    situ_and_mul_masked_post_quant(
        gateup_output,
        down_input,
        down_input_scale,
        group_size,
        masked_m,
        beta=beta,
        linear_beta=linear_beta,
        scale_ue8m0=packed_ue8m0,
        topk=topk,
        transposed=packed_ue8m0,
    )

    if packed_ue8m0:
        down_input_scale = down_input_scale.transpose(-1, -2)

    return down_input, down_input_scale


def _varlen_deep_gemm_silu_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    group_size: int,
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
    num_real_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert masked_m is not None
    hidden_states_device = gateup_output.device
    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    del D_2
    G = D // group_size

    # oai-swiglu (gemm1_alpha) stays on the Triton kernel until
    # per_token_group_quant grows an activation-kind axis. The output_scale dtype picks the schedule: packed
    # int32 UE8M0 (no follow-up transform; needs G % 4 == 0 and the
    # num_real_tokens grid bound) when eligible, row-major fp32 otherwise.
    if gemm1_alpha is not None:
        assert (
            swiglu_limit is None
        ), "swiglu_limit and gemm1_alpha are mutually exclusive"
        assert not swizzle, "swizzle is not supported with gemm1_alpha"
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_quant_fwd,
        )

        use_packed = (
            deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and num_real_tokens is not None
            and G % 4 == 0
            and D % (group_size * 4) == 0
        )
        down_input = torch.empty(
            (E, N, D), device=hidden_states_device, dtype=torch.float8_e4m3fn
        )
        down_input_scale = torch.empty(
            (E, G // 4, N) if use_packed else (E, N, G),
            device=hidden_states_device,
            dtype=torch.int32 if use_packed else torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit or 0.0,
            num_real_tokens=num_real_tokens,
            topk=topk,
        )
        if use_packed:
            down_input_scale = down_input_scale.transpose(-1, -2)
        return down_input, down_input_scale

    # DSV4-specific activations (clamped swiglu, swizzled gate|up layout) stay
    # on the DSV4 JIT kernel; it is the only implementation carrying them.
    if swiglu_limit is not None or swizzle:
        assert (
            envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
        ), "swiglu_limit / swizzle require SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"
        assert N % 4 == 0 and G % 4 == 0 and D // 8 >= E, (
            "DSV4 JIT activation requires N % 4 == 0, G % 4 == 0 and "
            f"D // 8 >= num_experts, got N={N} G={G} D={D} E={E}"
        )
        packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        down_input = torch.empty(
            (E, N, D), device=hidden_states_device, dtype=torch.float8_e4m3fn
        )
        down_input_scale = torch.empty(
            (E, G // 4, N) if packed_ue8m0 else (E, N, G),
            device=hidden_states_device,
            dtype=torch.int32 if packed_ue8m0 else torch.float32,
        )
        silu_and_mul_masked_post_quant(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=packed_ue8m0,
            topk=topk,
            transposed=packed_ue8m0,
            swiglu_limit=swiglu_limit,
            swizzle=swizzle,
        )
        if packed_ue8m0:
            down_input_scale = down_input_scale.transpose(-1, -2)
        return down_input, down_input_scale

    # Default plain-silu path: the unified JIT masked fused quant. It allocates
    # the outputs itself, with scales directly in the layout deep_gemm consumes
    # (packed-int32 col-major for UE8M0, TMA-aligned col-major fp32 otherwise),
    # so the caller's get_mn_major transform short-circuits.
    expected_m = ceil_div(num_real_tokens * topk, E) if num_real_tokens else None
    return per_token_group_quant(
        gateup_output,
        group_size=group_size,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        fuse_silu_and_mul=True,
        masked_m=masked_m,
        expected_m=expected_m,
        column_major_scales=True,
    )


@triton.jit
def _situ_mul_quant_contig_kernel(
    g_ptr,  # [rows, 2N] bf16, non-interleaved [gate; up] halves
    q_ptr,  # [rows, N] fp8 out
    s_ptr,  # [rows, KG] fp32 scales out
    N,
    KG,
    situ_beta,
    situ_linear_beta,
    GROUP: tl.constexpr,
    KG_POW2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rows2d = tl.arange(0, KG_POW2)[:, None]
    cols = tl.arange(0, GROUP)[None, :]
    offs = rows2d * GROUP + cols
    mask = rows2d < KG
    gate = tl.load(g_ptr + row * 2 * N + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(g_ptr + row * 2 * N + N + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh(x) == 2*sigmoid(2x) - 1 (avoids a libdevice dependency)
    gate_t = 2.0 * tl.sigmoid(2.0 * gate / situ_beta) - 1.0
    gate = situ_beta * gate_t * tl.sigmoid(gate)
    up_t = 2.0 * tl.sigmoid(2.0 * up / situ_linear_beta) - 1.0
    y = gate * situ_linear_beta * up_t
    amax = tl.clamp(tl.max(tl.abs(y), axis=1), min=1e-10, max=float("inf"))
    q = (y * (448.0 / amax)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * N + offs, q, mask=mask)
    srow = tl.arange(0, KG_POW2)
    tl.store(s_ptr + row * KG + srow, amax / 448.0, mask=srow < KG)


@triton.jit
def _masked_situ_mul_quant_kernel(
    g_ptr,  # [E, M, 2N] bf16, non-interleaved [gate; up] halves
    q_ptr,  # [E, M, N] fp8 out
    s_ptr,  # [E, M, KG] fp32 scales out
    masked_m_ptr,  # [E] int32
    M,
    N,
    KG,
    situ_beta,
    situ_linear_beta,
    GROUP: tl.constexpr,
    KG_POW2: tl.constexpr,
):
    e = tl.program_id(0)
    row = tl.program_id(1)
    if row >= tl.load(masked_m_ptr + e):
        return
    base = (e * M + row).to(tl.int64)
    rows = tl.arange(0, KG_POW2)[:, None]
    cols = tl.arange(0, GROUP)[None, :]
    offs = rows * GROUP + cols
    mask = rows < KG
    gate = tl.load(g_ptr + base * 2 * N + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(g_ptr + base * 2 * N + N + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh(x) == 2*sigmoid(2x) - 1 (avoids a libdevice dependency)
    gate_t = 2.0 * tl.sigmoid(2.0 * gate / situ_beta) - 1.0
    gate = situ_beta * gate_t * tl.sigmoid(gate)
    up_t = 2.0 * tl.sigmoid(2.0 * up / situ_linear_beta) - 1.0
    y = gate * situ_linear_beta * up_t
    amax = tl.clamp(tl.max(tl.abs(y), axis=1), min=1e-10, max=float("inf"))
    q = (y * (448.0 / amax)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + base * N + offs, q, mask=mask)
    srow = tl.arange(0, KG_POW2)
    tl.store(s_ptr + base * KG + srow, amax / 448.0, mask=srow < KG)


def _masked_situ_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SiTU (Kimi K3) gated activation + per-token-group fp8 quant for the
    masked-gemm path: gate' = beta*tanh(gate/beta)*sigmoid(gate),
    up' = linear_beta*tanh(up/linear_beta), out = gate'*up'.

    gateup_output layout is non-interleaved [gate; up] halves (K3 loads w1/w3
    into separate halves, matching silu_and_mul / the marlin SiTU branch).

    For k_groups > 4 (K > 512, i.e. EP>1 full-width experts) this runs a fused
    Triton kernel over the masked (valid) rows only, emitting plain row-major
    fp32 scales that are then packed via _cast_to_e8m0_with_rounding_up — the
    same scale pipeline the first grouped GEMM uses. The packed-UE8M0 layout
    emitted by sglang_per_token_group_quant_8bit (column_major + tma_aligned +
    ue8m0) mismatches what grouped_gemm_nt_f8f8bf16_masked expects once
    k_groups > 4 and produced garbage; EP=1 only ever sees K=384 (k_groups=3),
    which that layout handles correctly, so the unfused torch + packed-quant
    path is kept there. The unfused path also wastes ~E*m_max/valid_tokens of
    elementwise work on garbage rows (88% of prefill GPU time at ep8).
    """
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_8bit,
    )

    assert situ_beta is not None and situ_linear_beta is not None
    E, M, N2 = gateup_output.shape
    N = N2 // 2
    kg = N // group_size
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and kg > 4:
        assert kg % 4 == 0, (
            f"masked SiTU quant: k_groups={kg} > 4 requires kg % 4 == 0 "
            "for the e8m0 scale packing (K3 hits this only at ep_size=2)"
        )
        assert gateup_output.is_contiguous()
        dq = torch.empty(
            (E, M, N), dtype=torch.float8_e4m3fn, device=gateup_output.device
        )
        s = torch.empty((E, M, kg), dtype=torch.float32, device=gateup_output.device)
        _masked_situ_mul_quant_kernel[(E, M)](
            gateup_output,
            dq,
            s,
            masked_m,
            M,
            N,
            kg,
            situ_beta,
            situ_linear_beta,
            GROUP=group_size,
            KG_POW2=triton.next_power_of_2(kg),
            num_warps=8,
        )
        return dq, _cast_to_e8m0_with_rounding_up(s)

    gate = gateup_output[..., :N].float()
    up = gateup_output[..., N:].float()
    gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
    up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
    down_input_bf16 = (gate * up).to(torch.bfloat16)
    return sglang_per_token_group_quant_8bit(
        x=down_input_bf16,
        dst_dtype=torch.float8_e4m3fn,
        group_size=group_size,
        masked_m=masked_m,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )


def _apply_swiglu_limit(
    gateup_output: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    assert swiglu_limit == 10

    num_tokens, hidden_size_x2 = gateup_output.shape
    assert gateup_output.dtype == torch.bfloat16

    gate, up = torch.chunk(gateup_output, chunks=2, dim=-1)
    assert gate.shape == (num_tokens, hidden_size_x2 // 2)
    assert up.shape == (num_tokens, hidden_size_x2 // 2)

    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)

    out = torch.cat([gate, up], dim=-1)
    assert out.shape == (num_tokens, hidden_size_x2)
    return out

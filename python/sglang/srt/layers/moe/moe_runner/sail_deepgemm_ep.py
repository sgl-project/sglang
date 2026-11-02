"""SAIL expert-parallel DeepGEMM runner.

The upstream ``DeepGemmRunnerCore`` dispatches FP8/blockwise, MXFP4 and INT4
grouped GEMMs. SAIL's EP path additionally needs per-channel FP8, INT8 and a
separate MXFP4/INT4 weight layout, none of which upstream supports, so they are
kept here instead of growing the shared runner. ``sail_deepgemm_moe.py`` is the
tensor-parallel counterpart of this module.

SAIL's DeepGEMM also takes the per-expert row counts explicitly (``nopad``
kernels) instead of deriving the pad offsets itself, so the two contiguous
dispatch methods are forked here as well; the shared runner keeps the upstream
``contig`` calls for every other platform.

``runner.py`` instantiates this core on SAIL; every other platform keeps using
``DeepGemmRunnerCore`` unchanged.
"""

import logging
from typing import Optional

import torch
import triton

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    _DEEPGEMM_ON_H20,
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
    DeepGemmRunnerOutput,
    _apply_oai_swiglu,
    _apply_swiglu_limit,
    _cast_to_e8m0_with_rounding_up,
    _is_musa,
    _is_sail,
    _legacy_silu_and_mul,
    _situ_mul_quant_contig_kernel,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import dispose_tensor

logger = logging.getLogger(__name__)


class SailDeepGemmEpRunnerCore(DeepGemmRunnerCore):
    """DeepGEMM runner with the SAIL-only EP kernels plugged in."""

    # (masked, quant family) -> method name. Anything absent falls back to the
    # upstream dispatch, which keeps BF16 and blockwise FP8 on shared code.
    _SAIL_PATHS = {
        (False, "fp8_channel"): "_run_fp8_channel_contiguous_gemm",
        (False, "int8"): "_run_int8_contiguous_gemm",
        (False, "mxfp4"): "_run_fp4_contiguous_gemm",
        (False, "int4"): "_run_int4_contiguous_gemm",
        (True, "fp8_channel"): "_run_masked_fp8_channel_gemm",
        (True, "int8"): "_run_masked_int8_gemm",
        (True, "mxfp4"): "_run_masked_fp4_gemm",
        (True, "int4"): "_run_masked_int4_gemm",
    }

    @staticmethod
    def _sail_quant_family(
        runner_input: DeepGemmRunnerInput, quant_info: DeepGemmMoeQuantInfo
    ) -> Optional[str]:
        """Name the SAIL-only kernel family this layer needs, if any."""
        if quant_info.w13_weight.dtype == torch.bfloat16:
            return None
        if quant_info.use_fp8:
            # Blockwise FP8 is upstream's; only the channel-wise variant diverges.
            return "fp8_channel" if quant_info.per_channel_quant else None
        if quant_info.use_int8:
            return "int8"
        if quant_info.use_mxfp4:
            return "mxfp4"
        if quant_info.use_int4_w4a16 or quant_info.use_mxfp4_w4a16:
            return "int4"
        return None

    def run(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
        hooks: Optional[object] = None,
    ) -> DeepGemmRunnerOutput:
        family = self._sail_quant_family(runner_input, quant_info)
        if family is None:
            return super().run(runner_input, quant_info, running_state, hooks)

        if (
            self.config.silu_mul_keep_fp32
            and quant_info.w13_weight.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("silu_mul_keep_fp32 requires FP8 expert weights")

        masked = runner_input.use_masked_gemm
        gemm = getattr(self, self._SAIL_PATHS[(masked, family)])
        alignment = (
            running_state.get("contiguous_layout_alignment") if not masked else None
        )
        with deep_gemm_wrapper.contiguous_layout_alignment_scope(alignment):
            hidden_states = gemm(runner_input, quant_info, running_state)
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        """Fork of ``DeepGemmRunnerCore._run_contiguous_gemm`` for SAIL.

        SAIL consumes the per-expert row counts (``m_rows``) explicitly and pads
        the compacted activations inside the kernel, so both grouped GEMMs use
        the ``nopad`` variants. The rest is the shared implementation, copied
        verbatim to keep the shared runner free of SAIL branches.
        """
        from sglang.kernels.ops.activation.silu_and_mul_clamp import (
            silu_and_mul_clamp,
            silu_and_mul_contig_post_quant,
        )
        from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
            sglang_per_token_group_quant_fp8,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices
        trace_deepep_v2_contig = (
            envs.SGLANG_DEEPEP_V2_TRACE_CONTIG.get()
            and running_state.get("deepep_v2_expanded", False)
        )
        m_rows = running_state.get("m_rows")

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]
        scale_block_size = quant_info.block_shape[1] if quant_info.use_mxfp8 else 128

        if all_tokens == 0:
            if trace_deepep_v2_contig:
                logger.warning("DeepEP v2 expanded contig runner empty return")
            dispose_tensor(hidden_states)
            dispose_tensor(hidden_states_scale)
            return torch.empty(
                (0, K), device=hidden_states_device, dtype=torch.bfloat16
            )

        recipe_a, recipe_b = quant_info.scale_recipes(
            activation_block_size=runner_input.activation_scale_block_size,
            hidden_size=K,
            activation_scale_width=hidden_states_scale.shape[-1],
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

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_nopad(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            m_indices,
            m_rows,
        )
        if trace_deepep_v2_contig:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 expanded contig gateup GEMM returned")

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
        elif self.config.gemm1_alpha is not None:
            down_input = _apply_oai_swiglu(
                gateup_output,
                self.config.gemm1_alpha,
                self.config.gemm1_clamp_limit,
                self.config.gate_up_interleaved,
            )
            del gateup_output
            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input
        elif self.use_swizzle or self.config.silu_mul_keep_fp32:
            swiglu_limit_arg: Optional[float] = self.swiglu_limit
            use_contig_swizzle = self.use_swizzle and not running_state.get(
                "deepep_v2_disable_contig_swizzle", False
            )

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
                swizzle=use_contig_swizzle,
            )
            del gateup_output
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if not _is_musa:
                down_input = torch.empty(
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                if self.swiglu_limit is not None:
                    # Fuse the SwiGLU limit with the activation. The quantizing
                    # sibling only supports a group size of 128.
                    silu_and_mul_clamp(
                        gateup_output.view(-1, N), down_input, self.swiglu_limit
                    )
                else:
                    _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            else:
                # SAIL never runs on MUSA and the SwishGLU module only exists on
                # MUSA builds, hence the deferred import of the fallback.
                from sglang.srt.layers.moe.moe_runner.deep_gemm import (
                    _silu_and_mul_musa,
                )

                if self.swiglu_limit is not None:
                    gateup_output = _apply_swiglu_limit(
                        gateup_output, swiglu_limit=self.swiglu_limit
                    )
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
        if trace_deepep_v2_contig:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 expanded contig activation returned")

        deepep_v2_expanded = running_state.get("deepep_v2_expanded", False)
        # Folding the row weight into down_input_scale needs a non-power-of-two
        # scale; ue8m0 is a power of two, so it weights down_output before combine.
        # no_combine asks for raw rows, so the fold must not pre-apply the weight.
        fuse_weight_into_scale = (
            deepep_v2_expanded
            and not self.config.no_combine
            and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and running_state.get("topk_weights") is not None
        )
        if fuse_weight_into_scale:
            from sglang.kernels.ops.moe.ep_moe_kernels import scale_expanded_rows_

            scale_expanded_rows_(down_input_scale, running_state["topk_weights"])
            running_state["deepep_v2_weight_prefused"] = True

        # Allocate the MoE output in the NCCL symmetric memory pool when symmetric
        # allocation is required, so the downstream all-reduce takes the low-latency
        # symmetric path. Only this final output enters the pool; intermediate
        # buffers stay on the default allocator to bound pool occupancy.
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (all_tokens, K),
                device=hidden_states_device,
                dtype=torch.bfloat16,
            )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = tma_align_input_scale(down_input_scale)

        # The down activation is quantized here, independently of dispatch.
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_nopad(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
            m_rows,
        )
        if trace_deepep_v2_contig:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 expanded contig down GEMM returned")

        return down_output

    def _run_bf16_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        """Fork of ``DeepGemmRunnerCore._run_bf16_contiguous_gemm`` for SAIL.

        Same ``nopad`` swap as the FP8 path; the oai-swiglu / swiglu_limit
        activation is taken from the shared implementation unchanged.
        """
        hidden_states = runner_input.hidden_states
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices
        m_rows = running_state.get("m_rows")

        if all_tokens <= 0:
            return hidden_states.bfloat16()

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

        deep_gemm_wrapper.grouped_gemm_nt_bf16_nopad(
            hidden_states,
            w13_weight,
            gateup_output,
            m_indices,
            m_rows,
        )

        dispose_tensor(hidden_states)

        # Act: (M, N) -> (M, N/2)
        if self.config.gemm1_alpha is not None and self.config.activation != "situ":
            down_input = _apply_oai_swiglu(
                gateup_output,
                self.config.gemm1_alpha,
                self.config.gemm1_clamp_limit,
                self.config.gate_up_interleaved,
            )
        else:
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
                # SAIL never runs on MUSA and the SwishGLU module only exists on
                # MUSA builds, hence the deferred import of the fallback.
                from sglang.srt.layers.moe.moe_runner.deep_gemm import (
                    _silu_and_mul_musa,
                )

                down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
        del gateup_output

        # GroupGemm-2: (M, N/2) (E, K, N/2) -> (M, K)
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (all_tokens, K),
                device=hidden_states_device,
                dtype=torch.bfloat16,
            )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_nopad(
            down_input,
            w2_weight,
            down_output,
            m_indices,
            m_rows,
        )

        return down_output

    def _run_fp8_channel_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices
        m_rows = running_state.get("m_rows")

        if all_tokens <= 0:
            return hidden_states.bfloat16()

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        w13_weight_quant = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_quant = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        # The "i8i8" entry point also takes fp8-e4m3 activations and weights with
        # channel-wise (per-token) scales, which is exactly what this fp8 path feeds
        # it, so the name is deliberate rather than a leftover of the int8 path.
        deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_nopad(
            (hidden_states, hidden_states_scale),
            w13_weight_quant,
            gateup_output,
            m_indices,
            m_rows,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if not self.config.gate_up_interleaved and self.config.activation != "situ":
            # JIT kernel fusing the activation with per-token fp8 quantization. It
            # covers plain silu, the DeepSeek-V4 clamp and oai-swiglu (MiniMax-M3)
            # with the same semantics as the eager branch below, and avoids the
            # (all_tokens, N/2) bf16 intermediate. SiTU (which reuses gemm1_alpha
            # as situ_beta) and the gpt-oss interleaved layout stay on eager.
            from sglang.kernels.ops.activation.silu_mul_quant import (
                silu_and_mul_post_per_token_quant_fp8,
            )

            down_input_int8, down_input_scale = silu_and_mul_post_per_token_quant_fp8(
                gateup_output,
                swiglu_limit=self.swiglu_limit,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit,
            )
            del gateup_output
        else:
            if self.config.gemm1_alpha is not None and self.config.activation != "situ":
                down_input = _apply_oai_swiglu(
                    gateup_output,
                    self.config.gemm1_alpha,
                    self.config.gemm1_clamp_limit,
                    self.config.gate_up_interleaved,
                )
            else:
                if self.swiglu_limit is not None:
                    gateup_output = _apply_swiglu_limit(
                        gateup_output, swiglu_limit=self.swiglu_limit
                    )
                down_input = torch.empty(
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            del gateup_output

            down_input_int8, down_input_scale = sglang_per_token_quant_fp8(down_input)
            del down_input

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        # Same entry point as the gate-up GEMM above: it serves fp8 channel-wise
        # operands as well.
        deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_nopad(
            (down_input_int8, down_input_scale),
            w2_weight_quant,
            down_output,
            m_indices,
            m_rows,
        )

        return down_output

    def _run_int8_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices
        m_rows = running_state.get("m_rows")

        if all_tokens <= 0:
            return hidden_states.bfloat16()

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        w13_weight_quant = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_quant = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_nopad(
            (hidden_states, hidden_states_scale),
            w13_weight_quant,
            gateup_output,
            m_indices,
            m_rows,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if self.config.gemm1_alpha is not None and self.config.activation != "situ":
            down_input = _apply_oai_swiglu(
                gateup_output,
                self.config.gemm1_alpha,
                self.config.gemm1_clamp_limit,
                self.config.gate_up_interleaved,
            )
        else:
            if self.swiglu_limit is not None:
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
            down_input = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output

        down_input_int8, down_input_scale = per_token_quant_int8(down_input)
        del down_input

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_nopad(
            (down_input_int8, down_input_scale),
            w2_weight_quant,
            down_output,
            m_indices,
            m_rows,
        )

        return down_output

    def _run_fp4_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from deep_gemm import preprocess_mxfp4_scales

        from sglang.kernels.ops.activation.silu_mul_quant import (
            silu_and_mul_post_quant_mxfp4,
        )
        from sglang.kernels.ops.quantization.sail_mxfp4 import downcast_to_mxfp4

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices
        m_rows = running_state.get("m_rows")

        if all_tokens <= 0:
            return torch.zeros(
                (0, 2 * hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        w13_weight_quant = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_quant = (quant_info.w2_weight, quant_info.w2_scale)
        b13 = quant_info.b13
        b2 = quant_info.b2

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        # [WA] before deepep normal mode support uint16_t column major scale
        hidden_states_scale = preprocess_mxfp4_scales(hidden_states_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
            (hidden_states, hidden_states_scale),
            w13_weight_quant,
            b13,
            gateup_output,
            m_indices,
            m_rows,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if self.config.activation == "situ":
            down_input_fp4 = torch.empty(
                (all_tokens, N // 4), device=gateup_output.device, dtype=torch.uint8
            )
            down_input_scale = torch.empty(
                (N // 128, all_tokens), device=gateup_output.device, dtype=torch.uint16
            )
            self._apply_situ_and_mul(
                gateup_output, down_input_fp4, down_input_scale=down_input_scale
            )
            down_input_scale = down_input_scale.t()
        elif self.config.gemm1_alpha is not None:
            # The fused MXFP4 activation has no alpha variant, so activate first
            # and then produce the same MN-major scales.
            down_input = _apply_oai_swiglu(
                gateup_output,
                self.config.gemm1_alpha,
                self.config.gemm1_clamp_limit,
                self.config.gate_up_interleaved,
            )
            down_input_fp4, down_input_scale = downcast_to_mxfp4(down_input, axis=1)
            del down_input
        else:
            down_input_fp4, down_input_scale = silu_and_mul_post_quant_mxfp4(
                gateup_output, swiglu_limit=self.swiglu_limit
            )
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K * 2),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
            (down_input_fp4, down_input_scale),
            w2_weight_quant,
            b2,
            down_output,
            m_indices,
            m_rows,
        )

        return down_output

    def _apply_situ_and_mul(
        self,
        gateup_output: torch.Tensor,
        down_input: torch.Tensor,
        masked_m: Optional[torch.Tensor] = None,
        down_input_scale: Optional[torch.Tensor] = None,
        expected_m: Optional[int] = None,
    ) -> None:
        from sglang.kernels.ops.activation import (
            situ_and_mul,
            situ_and_mul_masked,
            situ_and_mul_masked_post_quant_mxfp4,
            situ_and_mul_post_quant_mxfp4,
        )

        beta = self.config.gemm1_alpha if self.config.gemm1_alpha is not None else 4.0
        linear_beta = self.config.gemm1_clamp_limit
        if down_input_scale is not None:
            if masked_m is None:
                situ_and_mul_post_quant_mxfp4(
                    gateup_output, down_input, down_input_scale, beta, linear_beta
                )
            else:
                situ_and_mul_masked_post_quant_mxfp4(
                    gateup_output,
                    down_input,
                    down_input_scale,
                    masked_m,
                    beta,
                    linear_beta,
                    self.config.top_k,
                    expected_m,
                )
        elif masked_m is None:
            situ_and_mul(gateup_output, down_input, beta, linear_beta)
        else:
            situ_and_mul_masked(
                gateup_output,
                down_input,
                masked_m,
                beta,
                linear_beta,
                self.config.top_k,
                expected_m,
            )

    def _run_int4_contiguous_gemm(
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
        m_rows = running_state.get("m_rows")

        if all_tokens <= 0:
            return hidden_states.bfloat16()

        use_sail_w4a16_mma = (
            _is_sail
            and quant_info.use_mxfp4_w4a16
            and quant_info.w13_weight.dtype == torch.uint8
        )
        N = (
            quant_info.w13_weight.size(1)
            if use_sail_w4a16_mma
            else quant_info.w2_weight.size(1) * 32
        )
        K = hidden_states_shape[1]

        w13_weight_quant = (quant_info.w13_weight, quant_info.w13_scale)
        w2_weight_quant = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_nopad(
            hidden_states,
            w13_weight_quant,
            gateup_output,
            m_indices,
            m_rows,
        )

        dispose_tensor(hidden_states)

        if self.config.gemm1_alpha is not None and self.config.activation != "situ":
            down_input = _apply_oai_swiglu(
                gateup_output,
                self.config.gemm1_alpha,
                self.config.gemm1_clamp_limit,
                self.config.gate_up_interleaved,
            )
        else:
            if self.swiglu_limit is not None:
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
            down_input = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            if self.config.activation == "situ":
                self._apply_situ_and_mul(gateup_output.view(-1, N), down_input)
            else:
                _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_nopad(
            down_input,
            w2_weight_quant,
            down_output,
            m_indices,
            m_rows,
        )

        return down_output

    def _run_masked_fp8_channel_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_quant_fwd_sail,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert s_mn % 4 == 0 and s_k % 4 == 0, (
                    f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                )
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        else:
            pass

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
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        swiglu_limit_arg: Optional[float] = None
        if self.swiglu_limit is not None:
            swiglu_limit_arg = self.swiglu_limit

        # Act
        scale_block_size = gateup_output.shape[2] // 2
        if (
            _is_sail
            and hidden_states.dtype == torch.float8_e4m3fn
            and scale_block_size > 0
            and scale_block_size % 8 == 0
            and envs.SGLANG_SAIL_SILU_MUL_MASKED_QUANT_FP8_CHANNEL_CUDA.get()
        ):
            # SAIL: fused SiLU+Mul + per-token fp8 quant via the JIT kernel.
            from sglang.kernels.ops.activation.silu_mul_quant import (
                silu_and_mul_masked_post_per_token_quant_fp8,
            )

            down_input, down_input_scale = silu_and_mul_masked_post_per_token_quant_fp8(
                gateup_output,
                masked_m,
                swiglu_limit=swiglu_limit_arg,
                expected_m=expected_m,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit,
            )
        else:
            down_input = torch.empty(
                (
                    gateup_output.shape[0],
                    gateup_output.shape[1],
                    gateup_output.shape[2] // 2,
                ),
                device=gateup_output.device,
                dtype=hidden_states.dtype,
            )
            down_input_scale = torch.empty(
                (
                    gateup_output.shape[0],
                    gateup_output.shape[1],
                    1,
                ),
                device=gateup_output.device,
                dtype=torch.float32,
            )
            silu_and_mul_masked_post_quant_fwd_sail(
                gateup_output,
                down_input,
                down_input_scale,
                scale_block_size,
                masked_m,
                use_fp8=hidden_states.dtype == torch.float8_e4m3fn
                or hidden_states.dtype == torch.float8_e5m2,
                use_int8=hidden_states.dtype == torch.int8,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=swiglu_limit_arg,
                gemm1_alpha=self.config.gemm1_alpha or 0.0,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit or 0.0,
            )
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
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
            **gemm_overlap_args_dict,
        )

        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    def _run_masked_int8_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_quant_fwd_sail,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert s_mn % 4 == 0 and s_k % 4 == 0, (
                    f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                )
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        else:
            pass

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        swiglu_limit_arg: Optional[float] = None
        if self.swiglu_limit is not None:
            swiglu_limit_arg = self.swiglu_limit

        # Act
        scale_block_size = gateup_output.shape[2] // 2
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                1,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd_sail(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
            use_fp8=hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e5m2,
            use_int8=hidden_states.dtype == torch.int8,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            swiglu_limit=swiglu_limit_arg,
            gemm1_alpha=self.config.gemm1_alpha or 0.0,
            gemm1_clamp_limit=self.config.gemm1_clamp_limit or 0.0,
        )
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
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

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            **gemm_overlap_args_dict,
        )

        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    def _run_masked_fp4_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        assert self.config.gemm1_alpha is None or self.config.activation == "situ", (
            "oai-swiglu is not supported on the masked mxfp4 path; "
            "use --deepep-mode normal or another MoE runner. "
            "SiTU uses its dedicated beta activation path."
        )
        from sglang.kernels.ops.activation.silu_mul_quant import (
            silu_and_mul_masked_post_quant_mxfp4,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            b13,
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        swiglu_limit_arg: Optional[float] = None
        if self.swiglu_limit is not None:
            swiglu_limit_arg = self.swiglu_limit

        # Act
        if self.config.activation == "situ":
            down_input = torch.empty(
                (num_groups, m, n // 4), device=hidden_states_device, dtype=torch.uint8
            )
            down_input_scale = torch.empty(
                (num_groups, n // 128, m),
                device=hidden_states_device,
                dtype=torch.uint16,
            )
            self._apply_situ_and_mul(
                gateup_output,
                down_input,
                masked_m=masked_m,
                down_input_scale=down_input_scale,
                expected_m=expected_m,
            )
        else:
            down_input, down_input_scale = silu_and_mul_masked_post_quant_mxfp4(
                gateup_output,
                masked_m,
                swiglu_limit=swiglu_limit_arg,
                expected_m=expected_m,
            )
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

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

        # DeepGemm kernel expects [E, T, S//2] for mxfp4 scale
        down_input_scale = down_input_scale.permute(0, 2, 1)
        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            b2,
            down_output,
            masked_m,
            expected_m,
            **gemm_overlap_args_dict,
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    def _run_masked_int4_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_fwd

        assert self.swiglu_limit is None

        hidden_states = runner_input.hidden_states
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        num_groups, m, k = hidden_states.shape
        use_sail_w4a16_mma = (
            _is_sail and quant_info.use_mxfp4_w4a16 and w13_weight.dtype == torch.uint8
        )
        n = w13_weight.size(1) if use_sail_w4a16_mma else w2_weight.size(1) * 32
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_masked(
            hidden_states,
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if self.config.activation == "situ":
            self._apply_situ_and_mul(
                gateup_output, down_input, masked_m, expected_m=expected_m
            )
        else:
            silu_and_mul_masked_fwd(
                gateup_output,
                down_input,
                masked_m,
                gemm1_alpha=self.config.gemm1_alpha or 0.0,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit or 0.0,
            )
        del gateup_output

        # GroupGemm-1
        down_output = torch.empty(
            (num_groups, m, k), device=down_input.device, dtype=torch.bfloat16
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

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_masked(
            down_input,
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            **gemm_overlap_args_dict,
        )

        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

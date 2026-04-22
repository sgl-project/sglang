from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers import asym_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import (
    dispose_tensor,
    is_cuda,
    is_hip,
    is_npu,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()

if not (_is_npu or _is_hip) and _is_cuda:
    from sgl_kernel import silu_and_mul


@dataclass
class AsymGemmBf16RunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None
    offsets: Optional[int] = None
    experts: Optional[int] = None
    list_size: int = 0

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@dataclass
class AsymGemmBf16RunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@dataclass
class AsymGemmBf16MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


class AsymGemmBf16RunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated

    def run(
        self,
        runner_input: AsymGemmBf16RunnerInput,
        quant_info: AsymGemmBf16MoeQuantInfo,
        running_state: dict,
    ) -> AsymGemmBf16RunnerOutput:
        if not runner_input.use_masked_gemm:
            hidden_states = self._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )
        else:
            hidden_states = self._run_masked_gemm(
                runner_input, quant_info, running_state
            )
        return AsymGemmBf16RunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: AsymGemmBf16RunnerInput,
        quant_info: AsymGemmBf16MoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states.to(torch.bfloat16)
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        # GroupGemm-0: BF16 x BF16 -> BF16 (no scales)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        asym_gemm_wrapper.grouped_gemm_nt_bf16bf16bf16_contig(
            hidden_states,
            quant_info.w13_weight,
            gateup_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        dispose_tensor(hidden_states)

        # Activation: SiLU-and-Mul (BF16 -> BF16, no requantization)
        down_input = torch.empty(
            (all_tokens, N // 2),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output

        # GroupGemm-1: BF16 x BF16 -> BF16 (no scales)
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        asym_gemm_wrapper.grouped_gemm_nt_bf16bf16bf16_contig(
            down_input,
            quant_info.w2_weight,
            down_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: AsymGemmBf16RunnerInput,
        quant_info: AsymGemmBf16MoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states.to(torch.bfloat16)
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0: BF16 x BF16 -> BF16
        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        asym_gemm_wrapper.grouped_gemm_nt_bf16bf16bf16_masked(
            hidden_states,
            w13_weight,
            gateup_output,
            masked_m,
            expected_m,
            num_groups,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )
        dispose_tensor(hidden_states)

        # Activation: SiLU-and-Mul (BF16 -> BF16)
        # Apply over entire flattened buffer to avoid masked_m[i].item() which
        # triggers GPU-to-CPU sync and breaks CUDA graph capture.
        # Padding rows are ignored by the downstream masked GEMM.
        down_input = torch.empty(
            (
                gateup_output.shape[0] * gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(
            gateup_output.view(-1, gateup_output.shape[2]),
            down_input,
        )
        down_input = down_input.view(
            gateup_output.shape[0], gateup_output.shape[1], gateup_output.shape[2] // 2
        )
        del gateup_output

        # GroupGemm-1: BF16 x BF16 -> BF16
        n2 = w2_weight.shape[1]
        down_output = torch.empty(
            (num_groups, m, n2), device=hidden_states_device, dtype=torch.bfloat16
        )
        asym_gemm_wrapper.grouped_gemm_nt_bf16bf16bf16_masked(
            down_input,
            w2_weight,
            down_output,
            masked_m,
            expected_m,
            num_groups,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


def fill_gateup_input_bf16(
    hidden_states: torch.Tensor,
    gateup_input: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    """Scatter hidden states (BF16) into the grouped gateup input buffer."""
    import triton
    import triton.language as tl

    @triton.jit
    def _fill_gateup_input_bf16_kernel(
        input_ptr,
        gateup_input_ptr,
        src2dst_ptr,
        topk_ids_ptr,
        topk,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        src_idx_int32 = tl.program_id(0)
        src_idx = src_idx_int32.to(tl.int64)
        src2dst_ptr = src2dst_ptr + src_idx * topk
        topk_ids_ptr = topk_ids_ptr + src_idx * topk
        src_ptr = input_ptr + src_idx * hidden_size

        vec = tl.arange(0, BLOCK_SIZE)
        for idx in range(topk):
            expert_id = tl.load(topk_ids_ptr + idx)
            if expert_id >= 0:
                dst_idx_int32 = tl.load(src2dst_ptr + idx)
                dst_idx = dst_idx_int32.to(tl.int64)
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                    offset = start_offset + vec
                    mask = offset < hidden_size
                    in_data = tl.load(src_ptr + offset, mask=mask)
                    tl.store(dst_ptr + offset, in_data, mask=mask)

    _fill_gateup_input_bf16_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        top_k,
        hidden_states.size(1),
        BLOCK_SIZE=1024,
    )

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.runtime_context import get_context, get_exec, get_lora
from sglang.srt.utils import get_device_sm, is_cuda

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


def resolve_moe_dual_runner_threshold(
    raw_threshold: int,
    runner_config: MoeRunnerConfig,
    *,
    is_fp8: bool,
    scale_ue8m0: Optional[bool] = None,
) -> int:
    """Resolve `SGLANG_MOE_DUAL_RUNNER_THRESHOLD` into a concrete token cutoff.

    `0` disables Dual-Runner, `> 0` preserves an explicit cutoff, and `< 0`
    selects an M_eff-based crossover where DeepGEMM compact's block_e=128
    padding inflation (`1 + 128 / M_eff`) becomes lower than Triton's WGMMA gap.

    On SM90 (`!DEEPGEMM_SCALE_UE8M0`), `target_m_eff=1024` resolves to `16384`
    for `num_local_experts=128, top_k=8`, so FP8 stays on Triton under H200's
    default `chunked_prefill_size=8192` unless raised or overridden positively.
    """
    if raw_threshold >= 0:
        return raw_threshold

    num_local_experts = max(1, int(runner_config.num_local_experts or 1))
    top_k = max(1, int(runner_config.top_k or 1))
    if is_fp8:
        if scale_ue8m0 is None:
            from sglang.srt.layers import deep_gemm_wrapper

            scale_ue8m0 = bool(deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0)
        # On SM90 (!DEEPGEMM_SCALE_UE8M0), DEEPGEMM_NEED_TMA_ALIGNED_SCALES adds
        # two unfused quant + tma_align_input_scale HBM transpose passes per layer,
        # pushing the DeepGEMM crossover to M_eff=1024 (16384 tokens on E=128, k=8).
        target_m_eff = 256 if scale_ue8m0 else 1024
        min_tokens = 1024 if scale_ue8m0 else 2048
    else:
        target_m_eff = 512
        min_tokens = 1024
    return max(min_tokens, (target_m_eff * num_local_experts + top_k - 1) // top_k)


def get_moe_dual_runner_unsupported_reason(
    runner_config: MoeRunnerConfig,
    *,
    with_bias: bool,
) -> Optional[str]:
    """Return a short reason if shared Dual-Runner prerequisites are unmet."""
    from sglang.srt.layers import deep_gemm_wrapper

    moe_runner_backend = get_moe_runner_backend()
    if not moe_runner_backend.is_auto():
        return f"moe_runner_backend is {moe_runner_backend.name}, not auto"
    moe_a2a_backend = get_moe_a2a_backend()
    if not moe_a2a_backend.is_none():
        return f"moe_a2a_backend is {moe_a2a_backend.name}, not none"
    if not is_cuda() or get_device_sm() not in (90, 100):
        return (
            f"requires SM90/SM100 CUDA "
            f"(got sm={get_device_sm() if is_cuda() else 'non-cuda'})"
        )
    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return "ENABLE_JIT_DEEPGEMM is False"
    if not runner_config.is_gated or runner_config.activation != "silu":
        return (
            f"unsupported activation "
            f"({runner_config.activation}, is_gated={runner_config.is_gated})"
        )
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
        or runner_config.swiglu_limit is not None
    ):
        return "custom SwiGLU alpha/clamp/limit is unsupported"
    if (
        runner_config.apply_router_weight_on_input
        or runner_config.no_combine
        or with_bias
    ):
        return "apply_router_weight_on_input, no_combine, or expert bias is unsupported"

    ctx = get_context()
    if ctx.is_config_namespace_published("lora"):
        lora_cfg = get_lora()
        if lora_cfg.enable_lora or lora_cfg.lora_paths:
            return "LoRA is enabled"
    if ctx.is_config_namespace_published("exec"):
        moe_exec = get_exec().moe
        if moe_exec.enable_eplb or moe_exec.enable_paged_experts:
            return "EPLB or paged experts enabled"

    hidden_size = int(runner_config.hidden_size or 0)
    if hidden_size <= 0 or hidden_size % 128 != 0:
        return f"hidden_size={hidden_size} is not divisible by 128"
    return None


class DualMoeRunner:
    """Routing-aware MoE runner dispatching small batches to Triton and large ones to DeepGEMM."""

    def __init__(
        self,
        config: MoeRunnerConfig,
        *,
        threshold: int,
        get_triton_quant_info: Callable[[torch.nn.Module], MoeQuantInfo],
        get_deep_gemm_quant_info: Callable[[torch.nn.Module], MoeQuantInfo],
    ):
        self.config = config
        self.threshold = threshold
        self._get_triton_quant_info = get_triton_quant_info
        self._get_deep_gemm_quant_info = get_deep_gemm_quant_info
        self.runner_triton = MoeRunner(MoeRunnerBackend.TRITON, config)
        self.runner_deep_gemm = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON

    def run(
        self, dispatch_output: DispatchOutput, layer: torch.nn.Module
    ) -> CombineInput:
        # shape[0] is a static Python int during CUDA graph capture (zero GPU-CPU sync).
        if dispatch_output.hidden_states.shape[0] < self.threshold:
            return self.runner_triton.run(
                dispatch_output, self._get_triton_quant_info(layer)
            )
        return self.runner_deep_gemm.run(
            dispatch_output, self._get_deep_gemm_quant_info(layer)
        )

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.runner_triton.set_overlap_args(down_gemm_overlap_args, meta_overlap_args)
        self.runner_deep_gemm.set_overlap_args(
            down_gemm_overlap_args, meta_overlap_args
        )

    def clear_overlap_args(self) -> None:
        self.runner_triton.clear_overlap_args()
        self.runner_deep_gemm.clear_overlap_args()

    @classmethod
    def try_create(
        cls,
        layer: Optional[torch.nn.Module],
        moe_runner_config: MoeRunnerConfig,
        *,
        is_fp8: bool,
        with_bias: bool,
        quant_unsupported_reason_fn: Callable[[], Optional[str]],
        get_triton_quant_info: Callable[[torch.nn.Module], MoeQuantInfo],
        get_deep_gemm_quant_info: Callable[[torch.nn.Module], MoeQuantInfo],
        owner_name: str,
        label: str,
    ) -> Optional[DualMoeRunner]:
        """Create a `DualMoeRunner` when `SGLANG_MOE_DUAL_RUNNER_THRESHOLD` is non-zero and eligible."""
        raw_threshold = envs.SGLANG_MOE_DUAL_RUNNER_THRESHOLD.get()
        if raw_threshold == 0:
            return None

        reason = (
            "layer is None"
            if layer is None
            else get_moe_dual_runner_unsupported_reason(
                moe_runner_config, with_bias=with_bias
            )
        )
        if reason is None:
            reason = quant_unsupported_reason_fn()
        if reason is not None:
            logger.info_once(
                "%s dual-runner disabled on this layer (%s); "
                "falling back to single MoE runner.",
                owner_name,
                reason,
            )
            return None

        resolved_threshold = resolve_moe_dual_runner_threshold(
            raw_threshold, moe_runner_config, is_fp8=is_fp8
        )
        # During `sglang.compile_deep_gemm`, force small warmup requests through
        # the DeepGEMM compact path so its contiguous kernels compile ahead of serving.
        effective_threshold = (
            1
            if envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
            else resolved_threshold
        )
        moe_runner_config.dual_runner_threshold = effective_threshold
        runner = cls(
            moe_runner_config,
            threshold=effective_threshold,
            get_triton_quant_info=get_triton_quant_info,
            get_deep_gemm_quant_info=get_deep_gemm_quant_info,
        )
        logger.info_once(
            "Enabled Routing-Aware Dual-Runner MoE (%s): "
            "threshold=%d (raw=%d, num_local_experts=%s, top_k=%s).",
            label,
            effective_threshold,
            raw_threshold,
            moe_runner_config.num_local_experts,
            moe_runner_config.top_k,
        )
        return runner


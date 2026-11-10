"""V100 FP16 MoE runner backed by TurboMind s884h kernels."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.sm70_paths import sm70_prebuilt
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

logger = logging.getLogger(__name__)

_DEFAULT_OPS_PATH = sm70_prebuilt("_sm70_turbomind_v100.so")
_OPS_LOAD_ATTEMPTED = False
_OPS_AVAILABLE = False
_LOGGED_CONFIGS: set[tuple[int, int, int, int]] = set()


def _load_sm70_ops() -> bool:
    global _OPS_AVAILABLE, _OPS_LOAD_ATTEMPTED
    if _OPS_LOAD_ATTEMPTED:
        return _OPS_AVAILABLE
    _OPS_LOAD_ATTEMPTED = True
    ops_path = Path(envs.SGLANG_SM70_TURBOMIND_OPS_PATH.get() or _DEFAULT_OPS_PATH)
    try:
        torch.ops.load_library(str(ops_path))
        ops = torch.ops.sglang_sm70_turbomind
        required = (
            hasattr(ops, "f16_prepare")
            and hasattr(ops, "f16_moe_build_ptrs")
            and hasattr(ops, "f16_moe_gemm")
            and hasattr(ops, "moe_permute_with_scratch")
            and hasattr(ops, "moe_unpermute")
            and hasattr(ops, "moe_permute_sort_workspace_size")
        )
        if not required:
            raise RuntimeError("one or more required SM70 MoE operators are absent")
    except Exception as exc:
        logger.warning("SM70 FP16 MoE operators unavailable: %s", exc)
        return False
    _OPS_AVAILABLE = True
    return _OPS_AVAILABLE


def can_use_sm70_fp16_moe(params_dtype: torch.dtype) -> bool:
    # Default on after model-level decode, long-prefill, and DFlash validation.
    if not envs.SGLANG_SM70_FP16_MOE.get():
        return False
    if params_dtype != torch.float16 or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 7:
        return False
    return _load_sm70_ops()


class SM70FP16MoEMethod(UnquantizedFusedMoEMethod):
    """Batched FP16 expert GEMMs specialized for V100."""

    def __init__(self):
        if not _load_sm70_ops():
            raise RuntimeError("SM70 FP16 MoE operators are unavailable")
        super().__init__(use_triton_kernels=False)

    def create_moe_runner(self, layer, moe_runner_config):
        # The Triton runner serves configs the TurboMind kernels do not implement.
        super().create_moe_runner(layer, moe_runner_config)
        self.use_turbomind = False

    @staticmethod
    def _turbomind_unsupported_reason(cfg) -> str | None:
        # The gated epilogue is a plain silu(gate) * up; the unpermute only
        # applies the router weights.
        if cfg.activation != "silu" or not cfg.is_gated:
            return f"activation={cfg.activation} is_gated={cfg.is_gated}"
        if cfg.swiglu_limit is not None or cfg.gemm1_clamp_limit is not None:
            return "clamped SwiGLU"
        if cfg.gemm1_alpha is not None or cfg.gemm1_beta is not None:
            return "SwiGLU alpha/beta"
        if cfg.apply_router_weight_on_input or cfg.no_combine:
            return "router weight on input or no combine"
        if cfg.wide_output_scale != 1.0 or cfg.gate_up_input_scale != 1.0:
            return "FP16 range scaling"
        if cfg.num_experts != cfg.num_local_experts:
            return "expert parallelism"
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        reason = self._turbomind_unsupported_reason(self.moe_runner_config)
        if reason is not None:
            logger.info("SM70 FP16 MoE: %s needs the Triton runner", reason)
            super().process_weights_after_loading(layer)
            return
        self.use_turbomind = True
        scale = self.moe_runner_config.routed_scaling_factor
        self.output_scale = (
            None
            if scale in (None, 1.0) or layer.should_fuse_routed_scaling_factor_in_topk
            else float(scale)
        )
        num_experts, w13_n, hidden_size = layer.w13_weight.shape
        intermediate_size = w13_n // 2
        # TurboMind's gated epilogue expects gate/up rows interleaved.
        gate, up = (
            layer.w13_weight[:, :intermediate_size],
            layer.w13_weight[:, intermediate_size:],
        )
        w13 = torch.stack((gate, up), dim=2).reshape(-1, hidden_size)
        ops = torch.ops.sglang_sm70_turbomind
        r13 = ops.f16_prepare(w13)
        w13_k_ld = int(r13[1][0].item())
        w13_tm = r13[0].reshape(num_experts, 2 * intermediate_size, hidden_size)

        w2 = layer.w2_weight.reshape(-1, intermediate_size)
        r2 = ops.f16_prepare(w2)
        w2_k_ld = int(r2[1][0].item())
        w2_tm = r2[0].reshape(num_experts, hidden_size, intermediate_size)

        layer.w13_tm_weight = Parameter(w13_tm, requires_grad=False)
        layer.w2_tm_weight = Parameter(w2_tm, requires_grad=False)
        layer.w13_weight = None
        layer.w2_weight = None

        layer.w13_weight_ptrs = Parameter(
            ops.f16_moe_build_ptrs(
                layer.w13_tm_weight, w13_k_ld, num_experts
            )[0],
            requires_grad=False,
        )
        layer.w2_weight_ptrs = Parameter(
            ops.f16_moe_build_ptrs(
                layer.w2_tm_weight, w2_k_ld, num_experts
            )[0],
            requires_grad=False,
        )

        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_size = hidden_size
        layer.sm70_intermediate_size = intermediate_size
        layer.sm70_top_k = self.moe_runner_config.top_k
        self._allocate_buffers(layer, 32)
        config = (num_experts, hidden_size, intermediate_size, layer.sm70_top_k)
        if config not in _LOGGED_CONFIGS:
            _LOGGED_CONFIGS.add(config)
            logger.info(
                "SM70 FP16 MoE: using TurboMind batched GEMM "
                "(experts=%d hidden=%d intermediate=%d topk=%d)",
                *config,
            )

    @staticmethod
    def _allocate_buffers(layer: torch.nn.Module, max_tokens: int) -> None:
        e, h, i, k = (
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            layer.sm70_intermediate_size,
            layer.sm70_top_k,
        )
        slots = max_tokens * k
        device = layer.w13_tm_weight.device
        layer.sm70_max_tokens = max_tokens
        layer.sm70_output = torch.empty(
            max_tokens, h, dtype=torch.float16, device=device
        )
        layer.sm70_permuted = torch.empty(
            slots, h, dtype=torch.float16, device=device
        )
        layer.sm70_gate_up = torch.empty(
            slots, i, dtype=torch.float16, device=device
        )
        layer.sm70_sorted_output = torch.empty(
            slots, h, dtype=torch.float16, device=device
        )
        layer.sm70_offsets64 = torch.empty(e + 1, dtype=torch.int64, device=device)
        layer.sm70_offsets32 = torch.empty(e + 1, dtype=torch.int32, device=device)
        layer.sm70_inverse = torch.empty(
            max_tokens, k, dtype=torch.int32, device=device
        )
        layer.sm70_ids32 = torch.empty(max_tokens, k, dtype=torch.int32, device=device)
        layer.sm70_token_expert = torch.arange(
            slots, dtype=torch.int32, device=device
        ).view(max_tokens, k)
        layer.sm70_permuted_idx = torch.empty(slots, dtype=torch.int32, device=device)
        workspace_size = (
            torch.ops.sglang_sm70_turbomind.moe_permute_sort_workspace_size(
                slots, e
            )
        )
        layer.sm70_sort_workspace = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
        layer.sm70_permuted_experts = torch.empty(
            max_tokens, k, dtype=torch.int32, device=device
        )
        layer.sm70_sorted_rows = torch.empty(
            max_tokens, k, dtype=torch.int32, device=device
        )
        layer.sm70_ids_for_sort = torch.empty(
            max_tokens, k, dtype=torch.int32, device=device
        )

    def apply(self, layer, dispatch_output):
        if not self.use_turbomind:
            return super().apply(layer, dispatch_output)
        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        tokens = x.shape[0]
        k = layer.sm70_top_k
        slots = tokens * k
        if tokens <= layer.sm70_max_tokens:
            output = layer.sm70_output[:tokens]
            permuted = layer.sm70_permuted[:slots]
            gate_up = layer.sm70_gate_up[:slots]
            sorted_output = layer.sm70_sorted_output[:slots]
            offsets64 = layer.sm70_offsets64
            offsets32 = layer.sm70_offsets32
            inverse = layer.sm70_inverse[:tokens]
            ids32 = layer.sm70_ids32[:tokens]
            token_expert = layer.sm70_token_expert[:tokens]
            permuted_idx = layer.sm70_permuted_idx[:slots]
            sort_workspace = layer.sm70_sort_workspace
            permuted_experts = layer.sm70_permuted_experts[:tokens]
            sorted_rows = layer.sm70_sorted_rows[:tokens]
            ids_for_sort = layer.sm70_ids_for_sort[:tokens]
        else:
            # Prefill workspaces must be transient. Retaining one large buffer
            # set per MoE layer exhausts VRAM before the first request.
            device = x.device
            h, i, e = (
                layer.sm70_hidden_size,
                layer.sm70_intermediate_size,
                layer.sm70_num_experts,
            )
            output = torch.empty(tokens, h, dtype=torch.float16, device=device)
            permuted = torch.empty(slots, h, dtype=torch.float16, device=device)
            gate_up = torch.empty(slots, i, dtype=torch.float16, device=device)
            sorted_output = torch.empty(slots, h, dtype=torch.float16, device=device)
            offsets64 = torch.empty(e + 1, dtype=torch.int64, device=device)
            offsets32 = torch.empty(e + 1, dtype=torch.int32, device=device)
            inverse = torch.empty(tokens, k, dtype=torch.int32, device=device)
            ids32 = torch.empty(tokens, k, dtype=torch.int32, device=device)
            token_expert = torch.arange(
                slots, dtype=torch.int32, device=device
            ).view(tokens, k)
            permuted_idx = torch.empty(slots, dtype=torch.int32, device=device)
            workspace_size = (
                torch.ops.sglang_sm70_turbomind.moe_permute_sort_workspace_size(
                    slots, e
                )
            )
            sort_workspace = torch.empty(
                workspace_size, dtype=torch.int8, device=device
            )
            permuted_experts = torch.empty(
                tokens, k, dtype=torch.int32, device=device
            )
            sorted_rows = torch.empty(tokens, k, dtype=torch.int32, device=device)
            ids_for_sort = torch.empty(tokens, k, dtype=torch.int32, device=device)
        output.zero_()
        if slots == 0:
            return StandardCombineInput(hidden_states=output)

        ids32.copy_(topk.topk_ids)
        ops = torch.ops.sglang_sm70_turbomind
        ops.moe_permute_with_scratch(
            x,
            ids32,
            token_expert,
            None,
            layer.sm70_num_experts,
            layer.sm70_num_experts,
            k,
            permuted,
            offsets64,
            inverse,
            permuted_idx,
            sort_workspace,
            permuted_experts,
            sorted_rows,
            ids_for_sort,
        )
        offsets32.copy_(offsets64)

        ops.f16_moe_gemm(
            gate_up,
            permuted,
            offsets32,
            layer.w13_weight_ptrs,
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            2 * layer.sm70_intermediate_size,
            True,
        )
        ops.f16_moe_gemm(
            sorted_output,
            gate_up,
            offsets32,
            layer.w2_weight_ptrs,
            layer.sm70_num_experts,
            layer.sm70_intermediate_size,
            layer.sm70_hidden_size,
            False,
        )
        ops.moe_unpermute(
            sorted_output,
            topk.topk_weights,
            inverse,
            offsets64,
            k,
            output,
        )
        if self.output_scale is not None:
            output.mul_(self.output_scale)
        return StandardCombineInput(hidden_states=output)

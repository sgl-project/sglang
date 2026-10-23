# SPDX-License-Identifier: Apache-2.0
"""The gfx950 MoE layer of minimax_m3: folds the shared expert into the aiter MoE output and
hands aiter a presorted sort from the fused decode routers. minimax_m3_hip.moe_cls() reaches
it lazily, since it subclasses minimax_m3's MiniMaxM3MoE."""

from contextlib import nullcontext
from typing import Optional, Union

import torch

from sglang.kernels.ops.gemm.router_gemv import (
    router_gemv_partials_supported,
    router_gemv_partials_swiglu,
)
from sglang.kernels.ops.moe.aiter_moe_addend import (
    apply_aiter_moe_addend_patch,
    moe_output_addend,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_sorting_small import presorted_moe_sort
from sglang.kernels.ops.moe.smallm_router_gfx950.m31_router import (
    m31_router,
    m31_router_supported,
    m31_shared_router,
    m31_shared_router_supported,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.quantization.fp8_utils import fp8_input_quant_format
from sglang.srt.models.minimax_common.amd.minimax_m3_hip import Gfx950SwigluFp8
from sglang.srt.models.minimax_m3 import MiniMaxM3MoE, _input_rows
from sglang.srt.runtime_context import get_forward, get_parallel


class MiniMaxM3MoEGfx950(MiniMaxM3MoE):
    """The MoE layer on gfx950; a gfx950 FFN norm hands it ``(fp8, scale, bf16 norm)``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # aiter's reducing stage2 must fold the addend that forward_normal offers
        apply_aiter_moe_addend_patch()
        self._fused_router_ok: Optional[bool] = None

    def forward_normal(
        self, hidden_states: torch.Tensor
    ) -> Union[torch.Tensor, MoeDeferredFinalize]:
        shared_input = hidden_states
        if isinstance(hidden_states, tuple):
            # (fp8, scale) for the shared expert, the bf16 norm for router and experts
            shared_input, hidden_states = hidden_states[:2], hidden_states[2]
        defer_finalize = get_forward().defer_moe_finalize
        if hidden_states.shape[0] > 0:
            # the alt-stream overlap of the parent is CUDA-only (alt_stream is None on ROCm)
            fused = self._fused_shared_router(shared_input, hidden_states)
            if fused is None:
                fused = self._shared_swiglu_router(shared_input, hidden_states)
            if fused is not None:
                shared_output, routed = fused
            else:
                routed = None
                shared_output = self._forward_shared_experts(shared_input)
            # dynamo cannot trace the context-variable handoff
            fold_shared = (
                shared_output is not None
                and not defer_finalize
                and not torch.compiler.is_compiling()
            )
            with (
                moe_output_addend(shared_output) if fold_shared else nullcontext()
            ) as addend:
                final_hidden_states = self._forward_router_experts(
                    hidden_states, defer_finalize=defer_finalize, routed=routed
                )
            if addend is not None and addend.consumed:
                shared_output = None
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)

        if defer_finalize:
            from sglang.srt.layers.moe.utils import sum_post_experts_output

            return MoeDeferredFinalize.from_flashinfer(
                final_hidden_states,
                gated_shared_output=shared_output,
                m=hidden_states.shape[0],
                reduce=sum_post_experts_output,
            )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states

    def _forward_router_experts(
        self,
        hidden_states: torch.Tensor,
        *,
        defer_finalize: bool = False,
        routed=None,
    ):
        """routed: _fused_decode_router()'s result when the caller already routed."""
        fused = (
            routed if routed is not None else self._fused_decode_router(hidden_states)
        )
        if fused is not None:
            topk_output, presorted = fused
        else:
            router_logits = self._compute_router_logits(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            presorted = nullcontext()
        with presorted:
            if defer_finalize:
                return self.experts.forward_deferred_finalize(
                    hidden_states, topk_output
                )
            return self.experts(hidden_states, topk_output)

    def _init_fused_router(self) -> None:
        from sglang.srt.layers.moe.topk import is_expert_remapping_enabled

        cfg = self.topk.topk_config
        backend = get_moe_runner_backend()
        self._fused_router_ok = (
            self.bf16_router_gemm
            and self.num_fused_shared_experts == 0
            and get_parallel().moe_ep_size == 1
            and get_moe_a2a_backend().is_none()
            and (backend.is_auto() or backend.is_aiter())
            and cfg.top_k == 4
            and cfg.scoring_func == "sigmoid"
            and cfg.correction_bias is not None
            and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
            and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
            and not is_expert_remapping_enabled()
        )

    def _fused_decode_router(self, hidden_states: torch.Tensor):
        """A gfx950 router bitwise identical to router_gemv -> moe_fused_gate -> the aiter
        MoE sort, in one launch for <= 16 tokens: (topk_output, context handing the
        presorted sort to the aiter runner), or None."""
        if not self._fused_router_applies(hidden_states) or not m31_router_supported(
            hidden_states, self.gate.weight, self.e_score_correction_bias
        ):
            return None
        out = m31_router(
            hidden_states,
            self.gate.weight,
            self.e_score_correction_bias,
            self.routed_scaling_factor,
        )
        return None if out is None else self._m31_routed(out)

    def _gate_partials(self, partials: torch.Tensor):
        topk_weights, topk_ids = moe_fused_gate(
            partials,
            self.e_score_correction_bias,
            topk=self.topk.topk_config.top_k,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
        )
        # the aiter runner does not read the logits
        return self._routed(topk_weights, topk_ids, None), nullcontext()

    def _fused_router_applies(self, hidden_states: torch.Tensor) -> bool:
        if self._fused_router_ok is None:
            # backend globals are only final at the first forward
            self._init_fused_router()
        return bool(self._fused_router_ok)

    def _m31_routed(self, out):
        topk_weights, topk_ids, router_logits, sorted_bufs = out
        topk_output = self._routed(topk_weights, topk_ids, router_logits)
        presorted = presorted_moe_sort(
            topk_ids,
            sorted_bufs,
            num_experts=self.gate.weight.shape[0],
            block_size=16,
        )
        return topk_output, presorted

    def _fused_shared_router(self, shared_input, hidden_states: torch.Tensor):
        """(shared expert output, _fused_decode_router()'s result) with the shared expert's
        gate_up GEMM in the router's launch, when that GEMM is the rowwise fp8 GEMM of the
        norm's per-token fp8 and m31_router routes these tokens; None otherwise."""
        if not isinstance(shared_input, tuple) or not self._fused_router_applies(
            hidden_states
        ):
            return None
        weight = self.shared_experts.gate_up_proj.weight
        if fp8_input_quant_format(weight, hidden_states.shape[0]) != "fp8_per_token":
            return None
        xq, x_scale = shared_input
        wq = weight._ptpc_weight
        if not m31_shared_router_supported(
            hidden_states,
            self.gate.weight,
            self.e_score_correction_bias,
            xq,
            x_scale,
            wq,
        ):
            return None
        out = m31_shared_router(
            hidden_states,
            self.gate.weight,
            self.e_score_correction_bias,
            self.routed_scaling_factor,
            xq,
            x_scale,
            wq,
            weight._ptpc_scale,
        )
        if out is None:
            return None
        gate_up, routed = out
        mlp = self.shared_experts
        shared_output, _ = mlp.down_proj(mlp.act_fn(gate_up))
        return shared_output, self._m31_routed(routed)

    def _shared_swiglu_router(self, shared_input, hidden_states: torch.Tensor):
        """(shared expert output, _fused_decode_router()'s result) with the shared expert's
        fp8 SwiGLU in the launch of the router GEMV's split-K partials, where
        _fused_decode_router() gates those partials; None otherwise."""
        mlp = self.shared_experts
        if (
            mlp is None
            or not isinstance(mlp.act_fn, Gfx950SwigluFp8)
            or not self._fused_router_applies(hidden_states)
            or not router_gemv_partials_supported(hidden_states, self.gate.weight)
        ):
            return None
        swiglu = mlp.act_fn.args(hidden_states.shape[0])
        if swiglu is None:
            return None
        gate_up, _ = mlp.gate_up_proj(shared_input)
        partials, x = router_gemv_partials_swiglu(
            hidden_states, self.gate.weight, gate_up, *swiglu
        )
        shared_output, _ = mlp.down_proj(x)
        return shared_output, self._gate_partials(partials)

    def _routed(self, topk_weights, topk_ids, router_logits):
        from sglang.srt.layers.moe.topk import (
            StandardTopKOutput,
            capture_routed_experts_if_allowed,
        )

        capture_routed_experts_if_allowed(
            self.topk.topk_config, self.layer_id, topk_ids
        )
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    def _forward_shared_experts(self, hidden_states):
        if _input_rows(hidden_states) > 0 and self.num_fused_shared_experts == 0:
            return self.shared_experts(hidden_states)
        return None

# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    ep_scatter,
    fused_moe_dispatch_index,
    post_reorder_deepgemm,
)
from sglang.kernels.ops.moe.triton_pad_expert_counts import pad_expert_counts
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.moe_runner.deep_gemm import _get_compact_all_tokens
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.quantization.ernie55_low_latency_gemm import (
    Ernie5_5LowLatencyGemm,
)
from sglang.srt.layers.quantization.halops_mxfp8 import (
    HALOPS_MODE,
    swizzle_scale_to_sfatom,
)
from sglang.srt.layers.quantization.halops_mxfp8_linear import (
    halops_w8a8_mxfp8_linear_with_fallback,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once


class Ernie5_5LowLatencyMoE:
    NARROW_TOKENS = 8
    _workspaces: tuple = ()

    def __init__(self, moe: torch.nn.Module):
        self.moe = moe
        self.prepared = False
        self._unpair: dict = {}
        if not moe._shared_sharded:
            return
        from halops import get_workspace_for_moe

        if not self._workspaces:
            group = get_tp_group()
            Ernie5_5LowLatencyMoE._workspaces = tuple(
                get_workspace_for_moe(
                    num_tokens,
                    group.cpu_group.group_name,
                    group.rank_in_group,
                    group.world_size,
                )
                for num_tokens in (
                    self.NARROW_TOKENS,
                    Ernie5_5LowLatencyGemm.MAX_TOKENS,
                )
            )
        shared = moe.shared_experts
        owners = {
            "experts": moe.experts,
            "shared_up": shared.gate_up_proj,
            "shared_dn": shared.down_proj,
            "fc1": moe.fc1_latent_proj,
            "fc2": moe.fc2_latent_proj,
        }
        self._q = dict.fromkeys(owners)
        for key, owner in owners.items():
            qm = owner.quant_method
            orig = qm.process_weights_after_loading

            def wrapped(layer, *a, _orig=orig, _key=key, **k):
                if _key != "experts":
                    _orig(layer, *a, **k)
                self._convert_weights(_key, layer)

            qm.process_weights_after_loading = wrapped
        moe.experts.quant_method.apply = self._apply_experts
        shared.gate_up_proj.quant_method.apply = self._apply_shared_up

    @staticmethod
    def enabled(quant_config) -> bool:
        if not envs.SGLANG_ERNIE55_LOW_LATENCY_DECODE.get():
            return False
        par = get_parallel()
        enabled = (
            getattr(quant_config, "is_fp4_experts", False)
            and par.moe_tp_size == par.tp_size
        )
        if not enabled:
            print_info_once(
                "ERNIE5.5 mega_ll_moe: stock path (needs MXFP4 experts, MoE TP == TP)"
            )
        return enabled

    def _convert_weights(self, key: str, layer: torch.nn.Module) -> None:
        if key == "experts":
            for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, name)
                bits = scale.data.to(torch.float32).contiguous().view(torch.int32)
                plain = ((bits >> 23) & 0xFF).to(torch.uint8)
                scale.data = swizzle_scale_to_sfatom(plain.flatten(0, -2)).view(
                    plain.shape
                )
            self._q[key] = (
                layer.w13_weight.data.view(torch.uint8),
                layer.w13_weight_scale_inv.data,
                layer.w2_weight.data.view(torch.uint8),
                layer.w2_weight_scale_inv.data,
            )
        else:
            plan = getattr(layer, Ernie5_5LowLatencyGemm.PLAN_ATTR)
            if key == "shared_up":
                delattr(layer, Ernie5_5LowLatencyGemm.PLAN_ATTR)
            self._q[key] = (layer.weight.data, plan.sfb)
        if all(q is not None for q in self._q.values()):
            self._interleave_up_weights()
            self._operands = tuple(t for q in self._q.values() for t in q)
            self.prepared = True

    def _interleave_up_weights(self) -> None:
        from halops import transform_weights_for_moe

        (w13, s13, w2, s2), (up, s_up) = self._q["experts"], self._q["shared_up"]
        packed_w13, s13, packed_up, s_up = transform_weights_for_moe(
            w13, s13, up, s_up, routed_use_swizzle=True, shared_use_swizzle=False
        )
        w13.copy_(packed_w13)
        up.copy_(packed_up)
        self.moe.experts.w13_weight_scale_inv.data = s13
        self._q["experts"] = (w13, s13, w2, s2)
        self._q["shared_up"] = (up, s_up)
        orders = [
            torch.arange(w.shape[-2], device=w.device).view(-1, 1) for w in (w13, up)
        ]
        no_scale = torch.empty(0, device=w13.device)
        probe = transform_weights_for_moe(
            orders[0][None], no_scale, orders[1], no_scale
        )
        self._unpair["experts"] = probe[0].flatten().argsort()
        self._unpair["shared_up"] = probe[2].flatten().argsort()

    def should_run(self, num_tokens: int) -> bool:
        return (
            self.prepared
            and 1 <= num_tokens <= Ernie5_5LowLatencyGemm.MAX_TOKENS
            and not get_is_extend_in_batch()
        )

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from halops import mega_ll_moe

        moe, cfg = self.moe, self.moe.config
        m, hidden = hidden_states.shape
        out = torch.empty(m, hidden, dtype=torch.bfloat16, device=hidden_states.device)
        mega_ll_moe(
            hidden_states.contiguous(),
            moe.gate_weight,
            *self._operands,
            moe.latent_norm.weight,
            moe.e_score_correction_bias,
            moe.per_expert_scale,
            out,
            self._workspaces[m > self.NARROW_TOKENS],
            moe.topk.topk_config.top_k,
            scoring_func="sigmoid",
            renormalize=bool(moe.topk.topk_config.renormalize),
            activation="situ",
            situ_beta=float(cfg.activation_situ_beta),
            situ_linear_beta=float(cfg.activation_situ_linear_beta),
            eps=float(cfg.rms_norm_eps),
            routed_use_swizzle=True,
            shared_use_swizzle=False,
            latent_use_swizzle=False,
        )
        return out

    def _apply_experts(self, layer, dispatch_output) -> StandardCombineInput:
        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        w13, s13, w2, s2 = self._q["experts"]
        num_experts = w13.shape[0]
        num_tokens = topk_ids.numel()
        block_e = 64
        rows = num_experts * block_e
        if _get_compact_all_tokens(num_tokens, num_experts, block_e) > rows:
            block_e = 128
            rows = _get_compact_all_tokens(num_tokens, num_experts, block_e)
            if -(-rows // (num_experts * block_e)) % 2 == 0:
                block_e = 256
                rows = _get_compact_all_tokens(num_tokens, num_experts, block_e)
            align = num_experts * block_e
            rows = -(-rows // align) * align
        valid, _ = fused_moe_dispatch_index(topk_ids, num_experts, 1)
        packed = x.new_empty(rows, x.shape[1])
        expert_ids = torch.empty(rows, dtype=torch.int32, device=x.device)
        src2dst = torch.empty_like(topk_ids, dtype=torch.int32)
        ep_scatter(
            x,
            None,
            topk_ids,
            pad_expert_counts(valid, block_e, rows),
            valid,
            torch.empty_like(valid),
            packed,
            torch.empty(rows, 1, dtype=torch.float32, device=x.device),
            expert_ids,
            src2dst,
            expert_alignment=block_e,
        )
        gate_up = self._grouped_gemm(packed, w13, s13, expert_ids.clamp_min_(0))
        act = self.moe.shared_experts.act_fn(
            gate_up.index_select(1, self._unpair["experts"])
        )
        out = torch.empty_like(x)
        post_reorder_deepgemm(
            self._grouped_gemm(act, w2, s2, expert_ids),
            out,
            src2dst,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            x.shape[0],
            x.shape[1],
            layer.moe_runner_config.routed_scaling_factor or 1.0,
        )
        return StandardCombineInput(hidden_states=out)

    def _apply_shared_up(self, layer, x, bias=None) -> torch.Tensor:
        out = halops_w8a8_mxfp8_linear_with_fallback(
            x, layer.weight, self._q["shared_up"][1]
        )
        return out.index_select(-1, self._unpair["shared_up"])

    @staticmethod
    def _grouped_gemm(x, weight, scale, expert_ids) -> torch.Tensor:
        from halops import group_contiguous_gemm, group_wise_quant

        q, sfa = group_wise_quant(x, mode=HALOPS_MODE, use_swizzle=True)
        out = x.new_empty(x.shape[0], weight.shape[1])
        group_contiguous_gemm(q, weight, sfa, scale, expert_ids, out, mode="w4a8-mxfp4")
        return out

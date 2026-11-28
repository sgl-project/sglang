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

from sglang.srt.layers.quantization.ernie55_low_latency_gemm import (
    Ernie5_5LowLatencyGemm,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models.ernie55 import Ernie5_5MlaAttention, Ernie5_5Moe
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once


class Ernie5_5LowLatencyMhcChain:
    _workspaces: dict = {}

    @staticmethod
    def supported() -> bool:
        par = get_parallel()
        if par.attn_tp_size == par.tp_size > 1:
            return True
        print_info_once(
            "ERNIE5.5 low-latency mHC chain: stock path (needs attn TP == TP > 1)"
        )
        return False

    def __init__(self, layers, layer_ids, *, draft: bool = False):
        from halops import get_workspace_for_mhc

        self.layers = [layers[i] for i in layer_ids]
        self.layer_ids = list(layer_ids)
        hidden = self.layers[0].config.hidden_size
        if hidden not in self._workspaces:
            group = get_parallel().attn_tp_group
            self._workspaces[hidden] = get_workspace_for_mhc(
                Ernie5_5LowLatencyGemm.MAX_TOKENS,
                hidden,
                group.cpu_group.group_name,
                group.rank_in_group,
                group.world_size,
            )
        self.workspace = self._workspaces[hidden]
        self.draft = draft
        self._sinkhorn_stream = torch.cuda.Stream()

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        captures=(),
        aux=None,
    ) -> torch.Tensor:
        x, residual, post, comb = None, hidden_states, None, None
        for i, layer_id in enumerate(self.layer_ids):
            x, residual, post, comb = self._layer(
                i, positions, x, residual, post, comb, forward_batch, zero_allocator
            )
            if layer_id in captures:
                from halops import mhc_post_hc_mean

                aux.append(mhc_post_hc_mean(x, residual, post.flatten(1), comb))
        return self.layers[-1]._hc_post(x, residual, post, comb)

    def _layer(
        self, i, positions, x, residual, post, comb, forward_batch, zero_allocator
    ):
        layer = self.layers[i]
        attn_hc, ffn_hc = self._hc_params(layer)
        attn_fused = (get_is_capture_mode() or not self.draft) and (
            1 <= residual.shape[0] <= Ernie5_5LowLatencyGemm.MAX_TOKENS
        )
        moe_fused = x is not None and attn_fused
        defer_attn = moe_fused and not self.draft
        defer_ffn = attn_fused and not self.draft
        attn_kwargs = {"defer_all_reduce": attn_fused}
        if moe_fused:
            residual, x, post, comb, quant = self._fused_mhc(
                layer,
                attn_hc,
                x,
                residual,
                post,
                comb,
                return_logits=defer_attn,
            )
            if get_is_capture_mode():
                attn_kwargs["x_quant"] = quant
        else:
            if x is not None:
                residual = layer._hc_post(x, residual, post, comb)
            x, post, comb = layer._hc_pre(residual, *attn_hc)
        if isinstance(layer.self_attn, Ernie5_5MlaAttention):
            out = layer.self_attn(
                positions, x, forward_batch, zero_allocator, **attn_kwargs
            )
        else:
            out = layer.self_attn(x, positions, forward_batch, **attn_kwargs)
        if attn_fused:
            residual, x, post, comb, _ = self._fused_mhc(
                layer,
                ffn_hc,
                out.partial,
                residual,
                post,
                comb,
                workspace=self.workspace,
                comb_is_logits=defer_attn,
                return_logits=defer_ffn,
            )
        else:
            residual = layer._hc_post(out, residual, post, comb)
            x, post, comb = layer._hc_pre(residual, *ffn_hc)
        done = None
        if defer_ffn:
            from halops import mhc_sinkhorn

            logits = comb
            prefetch = ()
            if isinstance(layer.mlp, Ernie5_5Moe) and i + 1 < len(self.layers):
                fn, scale, base, norm = self._hc_params(self.layers[i + 1])[0]
                prefetch = (fn, scale, base, norm.weight, residual, post)
            main = torch.cuda.current_stream()
            side = self._sinkhorn_stream
            comb = torch.empty_like(logits)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                mhc_sinkhorn(
                    logits,
                    layer.config.hc_eps,
                    layer.config.hc_sinkhorn_iters,
                    out=comb,
                    prefetch=prefetch,
                )
                done = torch.cuda.Event()
                done.record(side)
        x = layer._forward_ffn(x, forward_batch)
        if done is not None:
            torch.cuda.current_stream().wait_event(done)
        return x, residual, post, comb

    @staticmethod
    def _hc_params(layer) -> tuple:
        return (
            (
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.input_layernorm,
            ),
            (
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                layer.post_attention_layernorm,
            ),
        )

    @staticmethod
    def _fused_mhc(layer, hc, x, residual, post, comb, **kwargs):
        from halops import fused_all_reduce_mhc

        fn, scale, base, norm = hc
        config = layer.config
        residual, x, post, comb, q, sf = fused_all_reduce_mhc(
            x.reshape(-1, config.hidden_size),
            residual,
            post.reshape(-1, layer.hc_mult),
            comb,
            fn,
            scale,
            base,
            norm.weight,
            config.rms_norm_eps,
            config.hc_eps,
            config.hc_eps,
            config.hc_post_mult_value,
            config.hc_sinkhorn_iters,
            norm.variance_epsilon,
            **kwargs,
        )
        return residual, x, post, comb, (q, sf)

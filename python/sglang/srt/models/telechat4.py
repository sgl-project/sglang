# Copyright 2023-2024 SGLang Team
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
"""TeleChat4 model for SGLang.

Adapts the TeleChat4 architecture (MLA attention + MoE + mHC residual streams)
on top of sglang's DeepSeek-V2 building blocks.  The mHC module uses sglang's
fused TileLang kernels (``mhc_pre`` / ``mhc_post``) for performance.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn

# Register mhc_pre / mhc_post as torch custom ops with fake (meta) implementations
# so that torch.compile can trace through the mHC module. Without registration,
# dynamo tries to execute the real kernel with FakeTensors and fails because the
# TVM/TileLang C-extension calls attempt to access the data pointer of FakeTensors.
from sglang.kernels.ops.layernorm.mhc import mhc_post as _mhc_post_orig
from sglang.kernels.ops.layernorm.mhc import mhc_pre as _mhc_pre_orig
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.configs.telechat4 import TeleChat4Config
from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_npu, make_layers
from sglang.srt.utils.custom_op import register_custom_op


def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    n_splits_pre: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake (meta) implementation of mhc_pre for torch.compile tracing."""
    num_tokens, hc_mult, hidden_size = residual.shape
    post_mix = torch.empty(
        num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    return (post_mix, comb_mix, layer_input)


@register_custom_op(
    op_name="telechat4_mhc_pre",
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    n_splits_pre: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_npu():
        hc_mult = residual.shape[1]
        if hc_sinkhorn_eps != hc_pre_eps:
            raise ValueError(
                "The AscendC hc_pre kernel uses one hc_eps for both pre and "
                "Sinkhorn; hc_pre_eps and hc_sinkhorn_eps must match."
            )
        if hc_post_mult_value != 2.0:
            raise ValueError("The AscendC hc_pre kernel requires post multiplier 2.0.")
        layer_input, post_mix, comb_mix = torch.ops.npu.hc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=sinkhorn_repeat,
            norm_eps=rms_eps,
            hc_eps=hc_pre_eps,
        )
        return post_mix.unsqueeze(-1), comb_mix, layer_input

    return _mhc_pre_orig(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        n_splits=n_splits,
        n_splits_pre=n_splits_pre,
    )


@register_custom_op(
    op_name="telechat4_mhc_post",
    mutates_args=[],
    out_shape="residual",
)
def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if is_npu():
        return torch.ops.npu.hc_post(
            x, residual, post_layer_mix.squeeze(-1), comb_res_mix
        )

    return _mhc_post_orig(x, residual, post_layer_mix, comb_res_mix)


logger = logging.getLogger(__name__)


class mHCModule(nn.Module):
    """mHC (Manifold-constrained Hyper-Connection) module.

    Backed by sglang's fused TileLang kernels (``mhc_pre`` / ``mhc_post``).
    The fp32 op operands (``fn`` / ``hc_scale`` / ``hc_base``) are materialised
    once in :meth:`finalize` after weight loading.
    """

    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations

        # mHC kernel hyper-parameters
        self.norm_eps = 1e-6
        self.pre_eps = 1e-6
        self.post_mult_value = 2.0
        self.sinkhorn_eps = 1e-6
        # splitk GEMM parallelism. hc_hidden_size = n * hidden_size = 4 * 3584
        # = 14336. 14336 / n_splits_pre must be divisible by hidden_block(256).
        self.n_splits_pre = 8

        out_features = self.n * self.n + 2 * self.n
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, out_features, bias=False
        )

        init_alpha = config.mhc_init_gating_factor
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))

        self.bias = nn.Parameter(torch.zeros(out_features))

        # fp32 op operands, filled by finalize() after weight loading.
        self.register_buffer(
            "fn",
            torch.zeros(out_features, self.n * self.hidden_size),
            persistent=False,
        )
        self.register_buffer("hc_scale", torch.zeros(3), persistent=False)
        self.register_buffer("hc_base", torch.zeros(out_features), persistent=False)
        self._finalized = False

    @torch.no_grad()
    def finalize(self) -> None:
        """Build the fp32 op operands from the loaded parameters."""
        self.fn = self.mapping_proj.weight.detach().to(torch.float32).contiguous()
        self.hc_scale = (
            torch.cat([self.alpha_pre, self.alpha_post, self.alpha_res])
            .detach()
            .to(torch.float32)
            .contiguous()
        )
        self.hc_base = self.bias.detach().to(torch.float32).contiguous()
        self._finalized = True

    def forward(self, hidden_states: torch.Tensor):
        """Compute mHC pre-mixing: aggregate n-stream -> 1-stream.

        Args:
            hidden_states: [S, B, n*C] n-stream hidden states.
        Returns:
            aggregated: [S, B, C] single-stream input for the sub-layer.
            comb_mix: [S*B, n*n] residual mixing matrix.
            post_mix: [S*B, n] stream expansion weights.
        """
        S, B, _ = hidden_states.shape
        n = self.n
        C = self.hidden_size

        if not self._finalized:
            self.finalize()

        # Flatten to [S*B, n, C] and ensure bf16 (kernel requirement).
        residual = hidden_states.reshape(S * B, n, C)
        if residual.dtype != torch.bfloat16:
            residual = residual.to(torch.bfloat16)

        num_tokens = residual.shape[0]
        if num_tokens == 0:
            layer_input = torch.empty(
                0, C, dtype=torch.bfloat16, device=residual.device
            )
            post_mix = torch.empty(0, n, dtype=torch.float32, device=residual.device)
            comb_mix = torch.empty(
                0, n * n, dtype=torch.float32, device=residual.device
            )
        else:
            post_mix, comb_mix, layer_input = mhc_pre(
                residual=residual,
                fn=self.fn,
                hc_scale=self.hc_scale,
                hc_base=self.hc_base,
                rms_eps=self.norm_eps,
                hc_pre_eps=self.pre_eps,
                hc_sinkhorn_eps=self.sinkhorn_eps,
                hc_post_mult_value=self.post_mult_value,
                sinkhorn_repeat=self.sinkhorn_iterations,
                n_splits=1,
                n_splits_pre=self.n_splits_pre,
            )

        aggregated = layer_input.view(S, B, C)
        comb_mix = comb_mix.transpose(-1, -2).contiguous()
        return aggregated, comb_mix, post_mix

    def fused_h_res_h_post_bda_inference(
        self,
        h_res: torch.Tensor,
        original_residual: torch.Tensor,
        h_post: torch.Tensor,
        layer_output_with_bias,
    ) -> torch.Tensor:
        """Fused residual mixing + post expansion + bias-dropout-add.

        Args:
            h_res: comb_mix [S*B, n*n] from forward().
            original_residual: [S, B, n*C] - n-stream input before aggregation.
            h_post: post_mix [S*B, n] from forward().
            layer_output_with_bias: (x [S, B, C], bias None).
        Returns:
            output: [S, B, n*C] - updated n-stream residual.
        """
        x, _ = layer_output_with_bias

        S, B, _ = x.shape
        n = self.n
        C = self.hidden_size

        x_flat = x.reshape(S * B, C)
        if x_flat.dtype != torch.bfloat16:
            x_flat = x_flat.to(torch.bfloat16)
        residual_flat = original_residual.reshape(S * B, n, C)
        if residual_flat.dtype != torch.bfloat16:
            residual_flat = residual_flat.to(torch.bfloat16)

        # comb_mix is stored flattened [S*B, n*n]; view as 3D for mhc_post.
        comb_res_mix = h_res.view(S * B, n, n)

        if x_flat.shape[0] == 0:
            out = torch.empty_like(residual_flat)
        else:
            out = mhc_post(x_flat, residual_flat, h_post, comb_res_mix)

        return out.view(S, B, n * C)


def input_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    s, b, C = x.shape
    expanded = x.unsqueeze(2).expand(s, b, n, C).contiguous()
    return expanded.view(s, b, n * C)


def output_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    s, b, nC = x.shape
    C = nC // n
    x_streams = x.view(s, b, n, C)
    contracted = x_streams.mean(dim=2)
    return contracted


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]


class TeleChat4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: TeleChat4Config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        moe_quant_config_override: QuantizationConfig | None = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        if getattr(config, "rope_parameters", None) is not None:
            rope_theta = config.rope_parameters["rope_theta"]
            rope_type = config.rope_parameters.get("rope_type")
            rope_scaling = config.rope_parameters if rope_type != "default" else None
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings

        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        self.layer_id = layer_id
        self.is_nextn = is_nextn

        mtp_start_layer_idx = config.num_hidden_layers
        self.is_mtp_layer = layer_id >= mtp_start_layer_idx

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        hasattr(config, "index_topk")
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )
        self.use_mha = use_mha

        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_nextn=is_nextn,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )

        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        if (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % moe_layer_freq == 0
        ):
            self.mlp = deepseek_v2.DeepseekV2MoE(
                config=config,
                layer_id=self.layer_id,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
                is_nextn=is_nextn,
                is_deepseek_v4=False,
                dsa_enable_prefill_cp=dsa_enable_prefill_cp,
                mla_enable_prefill_cp=mla_enable_prefill_cp,
            )
        else:
            from sglang.srt.layers.communicator import enable_moe_dense_fully_dp

            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = deepseek_v2.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.n = getattr(config, "num_residual_streams", 1)
        self.enable_mhc = self.n > 1
        if self.enable_mhc and not self.is_mtp_layer:
            self.attn_hc = mHCModule(config, config.num_hidden_layers)
            self.ffn_hc = mHCModule(config, config.num_hidden_layers)

        from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=is_nextn or (layer_id == config.num_hidden_layers - 1),
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor | None,
        zero_allocator,
        gemm_output_zero_allocator=None,
        llama_4_scaling: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        captured_last_layer_outputs: list[torch.Tensor] | None = None,
        next_full_attention_layer_id: int | None = None,
    ) -> torch.Tensor:
        if self.enable_mhc and not self.is_mtp_layer:
            S = hidden_states.shape[0]
            B = hidden_states.shape[1]
            C = self.hidden_size

            origin_hidden_states = hidden_states
            aggregated_hidden_states, attention_res_weights, attention_post_weights = (
                self.attn_hc(origin_hidden_states)
            )

            hidden_states = aggregated_hidden_states.reshape(-1, C)

            # Use prepare_attn to set up attn_inputs_ (needed by MLA attention)
            hidden_states, residual = (
                self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                    hidden_states,
                    None,
                    forward_batch,
                )
            )

            attn_kwargs = {
                "positions": positions,
                "hidden_states": hidden_states,
                "forward_batch": forward_batch,
                "zero_allocator": zero_allocator,
                "layer_scatter_modes": self.layer_scatter_modes,
                "llama_4_scaling": llama_4_scaling,
            }
            hidden_states = self.self_attn(**attn_kwargs)
            if isinstance(hidden_states, tuple):
                hidden_states, topk_indices = hidden_states
            else:
                topk_indices = None

            from sglang.srt.layers.communicator import get_attn_tp_context

            get_attn_tp_context().clear_attn_inputs()

            hidden_states = hidden_states.reshape(S, B, C)

            if (
                isinstance(self.self_attn, deepseek_v2.DeepseekV2AttentionMLA)
                and hidden_states.dtype == torch.float16
            ):
                hidden_states *= 1.0 / self.routed_scaling_factor

            hidden_states = self.attn_hc.fused_h_res_h_post_bda_inference(
                h_res=attention_res_weights,
                original_residual=origin_hidden_states,
                h_post=attention_post_weights,
                layer_output_with_bias=(hidden_states, None),
            )

            origin_hidden_states = hidden_states
            aggregated_hidden_states, mlp_res_weights, mlp_post_weights = self.ffn_hc(
                origin_hidden_states
            )

            hidden_states = aggregated_hidden_states.reshape(-1, C)
            hidden_states = self.post_attention_layernorm(hidden_states)

            with get_forward().scoped(
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=False,
            ):
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    gemm_output_zero_allocator,
                )

            hidden_states = hidden_states.reshape(S, B, C)

            if (
                isinstance(self.mlp, deepseek_v2.DeepseekV2MLP)
                and hidden_states.dtype == torch.float16
            ):
                hidden_states *= 1.0 / self.routed_scaling_factor

            hidden_states = self.ffn_hc.fused_h_res_h_post_bda_inference(
                h_res=mlp_res_weights,
                original_residual=origin_hidden_states,
                h_post=mlp_post_weights,
                layer_output_with_bias=(hidden_states, None),
            )

            return hidden_states, None, topk_indices

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )

        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
            "forward_batch": forward_batch,
            "zero_allocator": zero_allocator,
            "layer_scatter_modes": self.layer_scatter_modes,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        attn_output = self.self_attn(**attn_kwargs)
        if isinstance(attn_output, tuple):
            attn_output, topk_indices = attn_output
        else:
            topk_indices = None

        hidden_states = attn_output

        from sglang.srt.layers.communicator import get_attn_tp_context

        get_attn_tp_context().clear_attn_inputs()

        if (
            isinstance(self.self_attn, deepseek_v2.DeepseekV2AttentionMLA)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_id == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                gemm_output_zero_allocator,
            )

        if (
            not (self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp)
            and fuse_mlp_allreduce
        ):
            hidden_states._sglang_needs_allreduce_fusion = True

        if not fuse_mlp_allreduce:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        if (
            isinstance(self.mlp, deepseek_v2.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual, topk_indices

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor | None,
        zero_allocator,
        tbo_subbatch_index: int | None = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            {
                "forward_batch": forward_batch,
                "positions": positions,
                "zero_allocator": zero_allocator,
                "tbo_subbatch_index": tbo_subbatch_index,
            }
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = {
            "positions": state.positions,
            "hidden_states": hidden_states,
            "residual": residual,
            "forward_batch": state.forward_batch,
            "zero_allocator": state.zero_allocator,
            "tbo_subbatch_index": state.tbo_subbatch_index,
        }

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class TeleChat4Model(nn.Module):
    def __init__(
        self,
        config: TeleChat4Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        dsa_enable_prefill_cp = False
        mla_enable_prefill_cp = False

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: TeleChat4DecoderLayer(
                config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                dsa_enable_prefill_cp=dsa_enable_prefill_cp,
                mla_enable_prefill_cp=mla_enable_prefill_cp,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.num_residual_streams = getattr(config, "num_residual_streams", 1)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if input_embeds is not None:
                hidden_states = input_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to TeleChat4Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors.inputs_embeds

        n_streams = self.num_residual_streams
        if n_streams > 1:
            S = hidden_states.shape[0]
            C = hidden_states.shape[1]
            hidden_states = hidden_states.reshape(S, 1, C)
            hidden_states = input_expand(hidden_states, n_streams)

        residual = None

        from sglang.srt.utils import BumpAllocator

        device = hidden_states.device
        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            hidden_states, residual, _topk_indices = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            return hidden_states

        if n_streams > 1:
            hidden_states = output_contract(hidden_states, n_streams)
            hidden_states = hidden_states.reshape(S, C)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class TeleChat4ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: TeleChat4Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when
        # q_lora_rank is not None (matches DeepseekV2ForCausalLM behaviour).
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)

        self.num_fused_shared_experts = 0
        self.use_dsa = is_deepseek_dsa(config)

        self.model = TeleChat4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        for layer in self.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "num_fused_shared_experts"):
                self.num_fused_shared_experts = layer.mlp.num_fused_shared_experts
                break

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()

        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.logits_processor = LogitsProcessor(config)

        self.capture_aux_hidden_states = False

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.communicator import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, None
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], is_nextn=False):
        processed_weights = []
        hc_weights = []

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "indexer.k_norm_bias" in name:
                name = name.replace("indexer.k_norm_bias", "indexer.k_norm.bias")

            if "attn_hc.mapping_weight" in name:
                name = name.replace(
                    "attn_hc.mapping_weight", "attn_hc.mapping_proj.weight"
                )
            if "ffn_hc.mapping_weight" in name:
                name = name.replace(
                    "ffn_hc.mapping_weight", "ffn_hc.mapping_proj.weight"
                )

            # Skip split bias parameters; the checkpoint stores a merged bias.
            skip_patterns = [
                "attn_hc.bias_pre",
                "attn_hc.bias_post",
                "attn_hc.bias_res",
                "ffn_hc.bias_pre",
                "ffn_hc.bias_post",
                "ffn_hc.bias_res",
            ]
            if any(p in name for p in skip_patterns):
                continue

            if "attn_hc" in name or "ffn_hc" in name:
                hc_weights.append((name, loaded_weight))
            else:
                processed_weights.append((name, loaded_weight))

        params_dict = dict(self.named_parameters())
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        for name, loaded_weight in hc_weights:
            param = None
            target_name = name

            if name in params_dict:
                param = params_dict[name]
            elif f"model.{name}" in params_dict:
                param = params_dict[f"model.{name}"]
                target_name = f"model.{name}"
            elif name.startswith("model.") and name[6:] in params_dict:
                param = params_dict[name[6:]]
                target_name = name[6:]

            if param is None:
                logger.warning("Skip %s, not found in params_dict", name)
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight)
            except Exception as e:
                logger.warning("Failed to load %s -> %s: %s", name, target_name, e)

        self.do_load_weights(processed_weights, is_nextn)

        # Build fp32 op operands (fn / hc_scale / hc_base) for every mHC module
        # so that the fused mhc_pre / mhc_post kernels can run without per-step
        # parameter materialisation.
        for m in self.modules():
            if isinstance(m, mHCModule):
                m.finalize()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        if is_npu():
            torch.npu.empty_cache()
            torch.npu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


EntryClass = [TeleChat4ForCausalLM]

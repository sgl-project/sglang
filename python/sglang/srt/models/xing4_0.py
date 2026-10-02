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
"""Xing4_0 model for SGLang.

Adapts the Xing4_0 architecture (MLA attention + MoE + mHC residual streams)
on top of sglang's DeepSeek-V2 building blocks.  The mHC module uses sglang's
fused TileLang kernels (``mhc_pre`` / ``mhc_post``) for performance.

The Xing4_0 checkpoint stores the
mHC operands (``hc_fn`` / ``hc_scale`` / ``hc_base``) directly, so no
``mapping_proj`` / ``alpha_*`` / ``bias`` materialisation is needed: these
tensors are the exact fp32 op operands consumed by the fused kernels.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch import nn

# Register mhc_pre / mhc_post as torch custom ops with fake (meta) implementations
# so that torch.compile can trace through the mHC module. Without registration,
# dynamo tries to execute the real kernel with FakeTensors and fails because the
# TVM/TileLang C-extension calls attempt to access the data pointer of FakeTensors.
from sglang.kernels.ops.layernorm.mhc import mhc_post as _mhc_post_orig
from sglang.kernels.ops.layernorm.mhc import mhc_pre as _mhc_pre_orig
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.configs.xing4_0 import Xing4_0Config
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layer_boundary import (
    AttentionInputs,
    ProducerReduction,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    make_stages,
)
from sglang.srt.layers.layer_boundary.ops import attn_tp_all_reduce
from sglang.srt.layers.layer_boundary.residual import access as residual_access
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
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
from sglang.srt.utils import add_prefix, is_npu, make_layers
from sglang.srt.utils.custom_op import register_custom_op

_is_npu = is_npu()


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
    op_name="xing4_0_mhc_pre",
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
    if _is_npu:
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
    op_name="xing4_0_mhc_post",
    mutates_args=[],
    out_shape="residual",
)
def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if _is_npu:
        return torch.ops.npu.hc_post(
            x, residual, post_layer_mix.squeeze(-1), comb_res_mix
        )

    return _mhc_post_orig(x, residual, post_layer_mix, comb_res_mix)


logger = logging.getLogger(__name__)


class mHCModule(nn.Module):
    """mHC (Manifold-constrained Hyper-Connection) module.

    Backed by sglang's fused TileLang kernels (``mhc_pre`` / ``mhc_post``).

    The Xing4_0 checkpoint stores the fp32 op operands directly as
    ``hc_fn`` / ``hc_scale`` / ``hc_base`` parameters (they are exactly the
    tensors consumed by the fused kernels). ``finalize`` only upcasts them to
    float32 into non-persistent buffers after weight loading.
    """

    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.n = config.hc_mult
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.hc_sinkhorn_iters

        # mHC kernel hyper-parameters
        self.norm_eps = 1e-6
        self.pre_eps = getattr(config, "hc_eps", 1e-6)
        self.post_mult_value = 2.0
        self.sinkhorn_eps = getattr(config, "hc_eps", 1e-6)
        # splitk GEMM parallelism. hc_hidden_size = n * hidden_size = 4 * 3584
        # = 14336. 14336 / n_splits_pre must be divisible by hidden_block(256).
        self.n_splits_pre = 8

        out_features = self.n * self.n + 2 * self.n

        # Persistent parameters matching the checkpoint weight names
        # (``attn_hc.hc_fn`` / ``attn_hc.hc_scale`` / ``attn_hc.hc_base``).
        # These are exactly the fp32 op operands consumed by the fused kernels.
        self.hc_fn = nn.Parameter(torch.empty(out_features, self.n * self.hidden_size))
        self.hc_scale = nn.Parameter(torch.empty(3))
        self.hc_base = nn.Parameter(torch.empty(out_features))

        # fp32 op operands, filled by finalize() after weight loading.
        self.register_buffer(
            "fn_fp32",
            torch.zeros(out_features, self.n * self.hidden_size),
            persistent=False,
        )
        self.register_buffer("scale_fp32", torch.zeros(3), persistent=False)
        self.register_buffer("base_fp32", torch.zeros(out_features), persistent=False)
        self._finalized = False

    @torch.no_grad()
    def finalize(self) -> None:
        """Build the fp32 op operands from the loaded parameters."""
        self.fn_fp32 = self.hc_fn.detach().to(torch.float32).contiguous()
        self.scale_fp32 = self.hc_scale.detach().to(torch.float32).contiguous()
        self.base_fp32 = self.hc_base.detach().to(torch.float32).contiguous()
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
                fn=self.fn_fp32,
                hc_scale=self.scale_fp32,
                hc_base=self.base_fp32,
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


class Xing4_0DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Xing4_0Config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        moe_quant_config_override: QuantizationConfig | None = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
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
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_nextn=is_nextn,
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
            )
        else:
            from sglang.srt.layers.layer_boundary import is_dense_ffn_fully_dp

            if is_dense_ffn_fully_dp():
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

        self.n = getattr(config, "hc_mult", 1)
        self.enable_mhc = self.n > 1
        if self.enable_mhc and not self.is_mtp_layer:
            self.attn_hc = mHCModule(config, config.num_hidden_layers)
            self.ffn_hc = mHCModule(config, config.num_hidden_layers)

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self._stage_enters_stack = (layer_id) == 0
        self._stage_terminal = (layer_id) == (
            1 if is_nextn else config.num_hidden_layers
        ) - 1
        self._stage_previous_sparse = is_previous_layer_sparse
        self._stage_next_sparse = is_next_layer_sparse

        # KTEPWrapperMethod returns a TP-local partial: GPU expert partitions
        # are present on their respective ranks, while the complete CPU-expert
        # contribution exists only on TP rank 0.  Declare such an FFN output as
        # TAIL_AFTER_SUM so the boundary never defers its post-experts
        # all-reduce to a later layer; the CPU contribution is then counted
        # exactly once inside the MoE's own collective.
        self.is_kt_moe = isinstance(self.mlp, deepseek_v2.DeepseekV2MoE) and isinstance(
            self.mlp.experts.quant_method, KTEPWrapperMethod
        )
        ffn_reduction = (
            ProducerReduction.TAIL_AFTER_SUM
            if self.is_kt_moe
            else ProducerReduction.EXIT_SCOPED
        )

        self.attn_boundary, self.ffn_boundary = make_stages(
            (
                declare_attn(),
                self.input_layernorm,
                {"qkv_latent_func": self.self_attn.prepare_qkv_latent},
            ),
            (
                declare_ffn(
                    sparse=self.is_layer_sparse,
                    next_layer_sparse=self._stage_next_sparse,
                    reduction=ffn_reduction,
                ),
                self.post_attention_layernorm,
            ),
            previous=declare_ffn(
                sparse=self._stage_previous_sparse,
                next_layer_sparse=self.is_layer_sparse,
            )
            if not self._stage_enters_stack
            else None,
            terminal=self._stage_terminal,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
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

            # The legacy ``prepare_attn`` entry (invoked here with
            # residual=None) applied the input RMSNorm on the residual-is-None
            # path before publishing the attention inputs, so the mHC branch
            # always ran MLA on normed rows.  Keep that norm now that this path
            # bypasses the stage boundary.
            hidden_states = self.input_layernorm(hidden_states)

            # Hand the aggregated input to the MLA attention's qkv-latent hook
            # (the new-architecture replacement for the old ``prepare_attn``
            # bookkeeping).  The aggregation above always produces full rows on
            # this rank, so the input is marked pre-gathered even when the
            # forward runs with attn-TP input scattering enabled.
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(
                    hidden_states,
                    forward_batch,
                    self.self_attn.prepare_qkv_latent,
                    is_pre_gathered=True,
                )
            )

            with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                    llama_4_scaling=llama_4_scaling,
                    input_on_attn_tp_slices=False,
                    prev_topk_indices=prev_topk_indices,
                )
            if isinstance(hidden_states, tuple):
                hidden_states, topk_indices = hidden_states
            else:
                topk_indices = None

            get_attn_tp_context().clear_attn_inputs()

            # Complete the attention output's partial sum over the attention-TP
            # group (o_proj is built with reduce_results=False): the fused
            # mhc_post mixing below consumes the full contribution, so this
            # layer never defers the reduction to a stage boundary.
            hidden_states = attn_tp_all_reduce(
                hidden_states, forward_batch, may_quantize=False
            )

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

            return hidden_states, topk_indices

        hidden_states_orig = residual_access.buffer(hidden_states)
        hidden_states = self.attn_boundary.prepare(
            hidden_states,
            forward_batch,
            capture_gathered=captured_last_layer_outputs,
        )

        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            attn_kwargs = {
                "positions": positions,
                "hidden_states": hidden_states,
                "forward_batch": forward_batch,
                "zero_allocator": zero_allocator,
                "input_on_attn_tp_slices": self.attn_boundary.input_on_attn_tp_slices,
                "prev_topk_indices": prev_topk_indices,
            }
            if not self.use_mha:
                attn_kwargs["llama_4_scaling"] = llama_4_scaling
            hidden_states = self.self_attn(**attn_kwargs)
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None

        get_attn_tp_context().clear_attn_inputs()

        if (
            isinstance(self.self_attn, deepseek_v2.DeepseekV2AttentionMLA)
            and hidden_states.dtype == torch.float16
        ):
            # Scaling the attention's partial sum scales its complete sum; at
            # layer 0 the raw embedding sits in the residual stream.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_id == 0:
                stream = forward_batch.residual_stream
                if stream is not None and stream.residual is not None:
                    stream.residual *= 1.0 / self.routed_scaling_factor

        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        if isinstance(self.mlp, deepseek_v2.DeepseekV2MLP):
            gemm_output_zero_allocator = None

        if (
            isinstance(self.mlp, deepseek_v2.DeepseekV2MoE)
            and not self.mlp.experts.moe_runner_config.inplace
            and not torch.compiler.is_compiling()
            # A deferred MoE finalize handoff from the previous layer is no buffer.
            and isinstance(hidden_states_orig, torch.Tensor)
        ):
            from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

            _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
        else:
            _mlp_ctx = nullcontext()

        with self.ffn_boundary.exit(forward_batch) as ffn_exit, _mlp_ctx:
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                gemm_output_zero_allocator,
            )
        hidden_states = ffn_exit.finish(hidden_states)

        mlp_output = residual_access.buffer(hidden_states)
        if (
            isinstance(self.mlp, deepseek_v2.DeepseekV2MLP)
            and mlp_output is not None
            and mlp_output.dtype == torch.float16
        ):
            mlp_output *= 1.0 / self.routed_scaling_factor

        return hidden_states, topk_indices

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
        zero_allocator,
        tbo_subbatch_index: int | None = None,
    ):
        state.hidden_states_after_comm_pre_attn = self.attn_boundary.prepare(
            hidden_states, forward_batch
        )
        if get_moe_a2a_backend().is_mori():
            state.num_tokens = hidden_states.shape[0]
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        hidden_states = self.attn_boundary.finish(
            state.pop("hidden_states_after_attn"), state.forward_batch
        )
        state.hidden_states_mlp_input = self.ffn_boundary.prepare(
            hidden_states, state.forward_batch
        )

    def op_comm_postprocess_layer(self, state):
        hidden_states = self.ffn_boundary.finish_complete_output(
            state.pop("hidden_states_mlp_output"), state.forward_batch
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class Xing4_0Model(nn.Module):
    def __init__(
        self,
        config: Xing4_0Config,
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

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Xing4_0DecoderLayer(
                config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_mult = getattr(config, "hc_mult", 1)

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
                        "to Xing4_0Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            if self.hc_mult > 1:
                # mHC layers carry their own multi-stream residual between
                # layers and never touch the ordinary residual stream.  Start a
                # fresh stream so the terminal complete_output/final_norm (or
                # to_pp) steps pass the tensors through untouched; receiving
                # via attn_boundary.from_pp would instead write the incoming
                # tensor into the stream and trip its output checks.
                hidden_states = pp_proxy_tensors["hidden_states"]
                residual_batch.start(forward_batch)
            else:
                hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                    pp_proxy_tensors, forward_batch
                )

        n_streams = self.hc_mult
        if n_streams > 1:
            S = hidden_states.shape[0]
            C = hidden_states.shape[1]
            hidden_states = hidden_states.reshape(S, 1, C)
            hidden_states = input_expand(hidden_states, n_streams)

        from sglang.srt.utils import BumpAllocator

        device = hidden_states.device
        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        topk_indices = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            hidden_states, topk_indices = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                prev_topk_indices=topk_indices,
            )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)

        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)

        if n_streams > 1:
            hidden_states = output_contract(hidden_states, n_streams)
            hidden_states = hidden_states.reshape(S, C)

        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, self.norm
            )
        return hidden_states


class Xing4_0ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Xing4_0Config,
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

        self.model = Xing4_0Model(
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
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()

        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.logits_processor = LogitsProcessor(config)

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(
            q_lora_rank,
            self.use_dsa,
            is_mhc=getattr(config, "hc_mult", 1) > 1,
        )

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

            # The Xing4_0 checkpoint stores the mHC operands directly
            # (attn_hc.hc_fn / attn_hc.hc_scale / attn_hc.hc_base), so no
            # mapping_weight/alpha/bias remapping or split-bias skipping is
            # needed; the parameter names already match named_parameters().
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

        # Upcast the mHC operands (hc_fn / hc_scale / hc_base) to fp32 buffers
        # so the fused mhc_pre / mhc_post kernels can run without per-step
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
        if _is_npu:
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


EntryClass = [Xing4_0ForCausalLM]

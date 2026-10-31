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
on top of sglang's DeepSeek-V2 building blocks.  The mHC residual streams are
wired through sglang's layer-boundary machinery (``MHCState``, see the usage
in glm5_next) on top of the fused kernels ``hc_pre`` / ``hc_post``.

The Xing4_0 checkpoint stores the
mHC operands (``hc_fn`` / ``hc_scale`` / ``hc_base``) directly, so no
``mapping_proj`` / ``alpha_*`` / ``bias`` materialisation is needed: these
tensors are the exact fp32 op operands consumed by the fused kernels.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch import nn

from sglang.kernels.ops.layernorm.mhc import hc_post as _hc_post_fn
from sglang.kernels.ops.layernorm.mhc import hc_pre as _hc_pre_fn
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.configs.xing4_0 import Xing4_0Config
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layer_boundary import (
    MHCState,
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.residual import access as residual_access
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
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
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, is_npu, make_layers

_is_npu = is_npu()


class mHCModule(nn.Module):
    """mHC (Manifold-constrained Hyper-Connection) module.

    Backed by sglang's fused TileLang kernels through the ``hc_pre`` /
    ``hc_post`` wrappers in ``sglang.kernels`` (the layer-boundary
    ``MHCState`` drives them; see glm5_next for the reference wiring).  On
    Ascend the module calls the ``torch.ops.npu.hc_pre`` / ``hc_post``
    AscendC ops registered by the NPU environment instead, mirroring how
    deepseek_v4 selects its NPU mHC ops at the model layer.

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

    def hc_pre(
        self,
        hidden_states: torch.Tensor,
        out_norm_weight: torch.Tensor | None,
        out_norm_eps: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """``MHCState`` read adapter: mix the n residual streams into the
        sub-layer input (and its norm, fused into the kernel when possible).

        Args:
            hidden_states: [T, n*C] n-stream residual.
            out_norm_weight / out_norm_eps: the sub-layer input RMSNorm.
        Returns:
            (layer_input [T, C], h_res [T, n*n], h_post [T, n], norm_fused).
        """
        if not self._finalized:
            self.finalize()

        s = hidden_states.shape[0]
        n = self.n
        C = self.hidden_size

        if _is_npu:
            if s == 0:
                layer_input = torch.empty(
                    0, C, dtype=hidden_states.dtype, device=hidden_states.device
                )
                h_res = torch.empty(
                    0, n * n, dtype=torch.float32, device=hidden_states.device
                )
                h_post = torch.empty(
                    0, n, dtype=torch.float32, device=hidden_states.device
                )
                return layer_input, h_res, h_post, False
            # The AscendC op takes the streams 3D, uses one hc_eps for both
            # the pre mix and Sinkhorn, and does not fold the out norm;
            # MHCState applies the norm (norm_fused=False).
            residual = hidden_states.view(s, n, C)
            if residual.dtype != torch.bfloat16:
                residual = residual.to(torch.bfloat16)
            layer_input, h_post, comb_mix = torch.ops.npu.hc_pre(
                residual,
                self.fn_fp32,
                self.scale_fp32,
                self.base_fp32,
                hc_mult=n,
                hc_sinkhorn_iters=self.sinkhorn_iterations,
                norm_eps=self.norm_eps,
                hc_eps=self.pre_eps,
            )
            comb_mix = comb_mix.transpose(-1, -2).contiguous()
            return layer_input, comb_mix.reshape(s, n * n), h_post, False

        layer_input, h_res, h_post, norm_fused = _hc_pre_fn(
            x=hidden_states,
            hc_fn=self.fn_fp32,
            hc_scale=self.scale_fp32,
            hc_base=self.base_fp32,
            hc_mult=n,
            rms_eps=self.norm_eps,
            hc_eps=self.pre_eps,
            sinkhorn_iters=self.sinkhorn_iterations,
            post_mult_value=self.post_mult_value,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            n_splits_pre=self.n_splits_pre,
        )
        # The checkpoint's hc_fn was trained with the residual mixing matrix
        # transposed relative to the kernels' convention; without this
        # transpose the model's outputs are gibberish.
        h_res = h_res.view(s, n, n).transpose(1, 2).contiguous().reshape(s, n * n)
        return layer_input, h_res, h_post, norm_fused

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_res: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """``MHCState`` update adapter: write the sub-layer output back into
        the n residual streams.

        Args:
            x: [T, C] sub-layer output.
            residual: [T, n*C] n-stream residual before the sub-layer.
            h_res: [T, n*n] residual mixing matrix (already transposed).
            h_post: [T, n] stream expansion weights.
        Returns:
            The updated n-stream residual [T, n*C].
        """
        n = self.n
        s = x.shape[0]
        if _is_npu:
            if s == 0:
                return residual
            C = x.shape[1]
            out = torch.ops.npu.hc_post(
                x,
                residual.view(s, n, C),
                h_post,
                h_res.view(s, n, n),
            )
            return out.reshape(s, -1)
        return _hc_post_fn(
            x=x,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_mult=n,
        )


class Xing4_0DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Xing4_0Config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        moe_quant_config_override: QuantizationConfig | None = None,
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

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = all(dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))
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
                is_deepseek_v4=False,
                reduce_results=False,
            )
        else:
            self.mlp = deepseek_v2.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                swiglu_limit=getattr(config, "swiglu_limit", None),
                parallel_group="replicated" if is_dense_ffn_fully_dp() else "tp",
                reduce_results=False,
                allow_fused_down=False,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.n = getattr(config, "hc_mult", 1)
        self.enable_mhc = self.n > 1
        mhc = None
        if self.enable_mhc:
            self.attn_hc = mHCModule(config, config.num_hidden_layers)
            self.ffn_hc = mHCModule(config, config.num_hidden_layers)
            mhc = MHCState(
                hc_mult=self.n,
                hc_attn_pre=self.attn_hc.hc_pre,
                hc_ffn_pre=self.ffn_hc.hc_pre,
                hc_post=self.attn_hc.hc_post,
                # The last layer's write-back also contracts the n streams
                # into the hidden states the layer stack hands on.
                is_last_layer=layer_id == config.num_hidden_layers - 1,
            ).residual_ops()

        self.is_layer_sparse = self._is_layer_sparse(layer_id)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1)

        attn, ffn = self.stage_facts(config, layer_id, mhc=mhc)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                attn,
                self.input_layernorm,
                {"qkv_latent_func": self.self_attn.prepare_qkv_latent},
            ),
            (
                ffn,
                self.post_attention_layernorm,
            ),
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

    def _is_layer_sparse(self, layer_id: int) -> bool:
        return self._config_is_layer_sparse(self.config, layer_id)

    @classmethod
    def _config_is_layer_sparse(cls, config, layer_id: int) -> bool:
        return (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )

    @classmethod
    def stage_facts(cls, config, layer_id: int, *, mhc=None):
        """The stages a layer of this class declares, from the config alone:
        the model's shared declaration function, which the layer declares
        with too (see make_layers).  With ``mhc`` (the layer's MHCState
        residual ops) the stages read and write hyper-connection streams
        instead of a plain residual; ``mhc=None`` for a layer that is not
        built falls back to the class-level MHCState facts."""
        residual = mhc
        if residual is None and getattr(config, "hc_mult", 1) > 1:
            residual = MHCState.facts()
        if residual is None:
            return (
                declare_attn(),
                declare_ffn(
                    sparse=cls._config_is_layer_sparse(config, layer_id),
                    next_layer_sparse=cls._config_is_layer_sparse(config, layer_id + 1),
                ),
            )
        return (
            declare_attn(
                read=residual.attn_readout,
                update=residual.attn_update,
            ),
            declare_ffn(
                sparse=cls._config_is_layer_sparse(config, layer_id),
                next_layer_sparse=cls._config_is_layer_sparse(config, layer_id + 1),
                read=residual.ffn_readout,
                update=residual.ffn_update,
            ),
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
        hidden_states = self.ffn_boundary.complete_now(
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
            stage_facts=lambda idx: Xing4_0DecoderLayer.stage_facts(config, idx),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

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
            # With mHC the producer's residual stream already holds the n
            # hyper-connection streams; from_pp reconstructs that stream as
            # declared, so no stream-width special-casing is needed here.
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

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

        # The last layer's mHC write-back contracted the n streams back into
        # plain hidden states (see MHCState(is_last_layer=...)).
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
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
            if name in params_dict:
                param = params_dict[name]
            elif f"model.{name}" in params_dict:
                param = params_dict[f"model.{name}"]
            elif name.startswith("model.") and name[6:] in params_dict:
                param = params_dict[name[6:]]
            else:
                raise ValueError(f"mHC weight {name!r} not found in params_dict")

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self.do_load_weights(processed_weights)

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

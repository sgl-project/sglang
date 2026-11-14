# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Copyright 2026 Upstage AI.

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.srt.configs.solar_open2 import SolarOpen2Config
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.models.solar import SolarMLP as SolarOpen2MLP
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import is_cuda_alike, is_hip


def _validate_supported_config(config) -> None:
    for name in ("use_rope", "use_qk_norm", "kda_use_full_proj"):
        if getattr(config, name, False):
            raise ValueError(f"Solar Open2 does not support {name}=True")
    num_experts = getattr(config, "n_routed_experts", None)
    if (
        num_experts is None
        or num_experts <= 0
        or getattr(config, "first_k_dense_replace", 0) != 0
    ):
        raise ValueError(
            "Solar Open2 requires n_routed_experts > 0 and first_k_dense_replace=0"
        )


def _apply_gqa_output_gate(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Preserve the BF16 sigmoid rounding when fusing the output gate."""
    if (
        not torch.compiler.is_compiling()
        and output.is_cuda
        and not is_hip()
        and gate.device == output.device
        and output.numel() < 2**32
        and output.data_ptr() % 16 == 0
        and gate.data_ptr() % 16 == 0
    ):
        from sglang.kernels.ops.elementwise import mla_output_gate

        if mla_output_gate.covered(output, gate):
            return mla_output_gate.kimi_k3_mla_output_gate(output, gate)
    return output * gate.sigmoid()


class SolarOpen2Attention(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        reduce_results: bool = True,
    ):
        super().__init__()
        parallel = get_parallel()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attn_tp_size, self.attn_tp_rank = (
            parallel.attn_tp_size,
            parallel.attn_tp_rank,
        )
        tp_size = self.attn_tp_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.head_dim = config.head_dim
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.scaling = self.head_dim**-0.5
        attention_bias = getattr(config, "attention_bias", False)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            parallel_group="attn_tp",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            parallel_group="attn_tp",
            reduce_results=reduce_results,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.use_gqa_gate = config.use_gqa_gate
        if self.use_gqa_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=config.use_gqa_gate_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
                parallel_group="attn_tp",
            )

    def forward(self, hidden_states, positions, forward_batch, **kwargs):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        output = self.attn(q, k, v, forward_batch)
        if self.use_gqa_gate:
            gate, _ = self.g_proj(hidden_states)
            output = _apply_gqa_output_gate(output, gate)
        return self.o_proj(output)[0]


class SolarOpen2KDAKernel(TritonKDAKernel):
    # These fused Kimi paths hardcode sigmoid(b), without Solar's factor of two.
    supports_packed_decode = False
    supports_fused_chain_verify = False

    def __init__(self, beta_scale: float):
        self.beta_scale = beta_scale

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=lower_bound,
            beta_scale=self.beta_scale,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        # fused ReplaySSM ring-write (dense verify only; off elsewhere).
        cache_ring: bool = False,
        replayssm_rawv: Optional[torch.Tensor] = None,
        replayssm_rawk: Optional[torch.Tensor] = None,
        replayssm_g: Optional[torch.Tensor] = None,
        replayssm_beta: Optional[torch.Tensor] = None,
        ring_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Keep the committed state intact while saving each draft's state,
        # including tree parents and optional ReplaySSM ring writes.
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=lower_bound,
            cache_ring=cache_ring,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
            beta_scale=self.beta_scale,
            ring_indices=ring_indices,
        )

    def extend(self, q, k, v, g, beta, *, beta_is_raw=False, **kwargs):
        if beta_is_raw:
            beta = beta.float().sigmoid() * self.beta_scale
        return super().extend(q, k, v, g, beta, beta_is_raw=False, **kwargs)


class SolarOpen2KDAAttnBackend(KDAAttnBackend):
    def __init__(self, model_runner):
        config = model_runner.model_config.hf_config
        self._beta_scale = 2.0 if config.kda_allow_neg_eigval else 1.0
        super().__init__(model_runner)

    def _create_kernel_dispatcher(self, backends):
        if self._beta_scale == 1.0:
            return super()._create_kernel_dispatcher(backends)

        # Select the same Triton path used by the scaled-beta implementation.
        # A Triton extend kernel also keeps graph metadata off the direct
        # FlashInfer path, so eager and graph calls share this beta policy.
        triton = LinearAttnKernelBackend.TRITON
        dispatcher = KDAKernelDispatcher(triton, triton, triton)
        kernel = SolarOpen2KDAKernel(self._beta_scale)
        dispatcher.triton_kernel = kernel
        dispatcher.decode_kernel = kernel
        dispatcher.verify_kernel = kernel
        dispatcher.extend_kernel = kernel
        dispatcher.supports_packed_decode = False
        dispatcher.supports_packed_decode_safe_gate = False
        return dispatcher

    def _can_fuse_accept_state(self, verify_backend):
        # Scaled beta uses Triton verify, which commits accepted state normally.
        return self._beta_scale == 1.0 and super()._can_fuse_accept_state(
            verify_backend
        )


class SolarOpen2KimiDeltaAttention(KimiDeltaAttention):
    def __init__(
        self,
        config,
        layer_idx,
        quant_config=None,
        prefix="",
        *,
        rms_norm_eps=1e-5,
        reduce_results: bool = True,
    ):
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            no_kda_lora=False,
            shard_on_attn_tp=True,
            rms_norm_eps=rms_norm_eps,
            reduce_results=reduce_results,
        )
        # HF's dtype describes the checkpoint; the loader's default dtype owns
        # execution. The factored BMM must match its preceding projection.
        if self.do_fuse_qkvbfg:
            self.fused_fg_b_proj.to(dtype=self.fused_qkvbfg_a_proj.weight.dtype)

    def forward(self, hidden_states, positions, forward_batch, **kwargs):
        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, norm_gate = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, norm_gate = self.forward_qkvbfg(hidden_states)
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            forget_gate = forget_gate.unflatten(-1, (-1, self.head_dim)).unsqueeze(0)
        output = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta.unsqueeze(0),
        )
        output = self.o_norm(output, norm_gate.unflatten(-1, (-1, self.head_dim)))
        return self.o_proj(output.squeeze(0).flatten(-2))[0]


def _load_kda_weight(name, weight, attention, params):
    """Pack checkpoint KDA projections and convs into their live parameters.

    Return the destination name, or None for non-packed parameters. Quantization
    scales and repeated weight updates use the same mapping.
    """
    if attention.do_fuse_qkvbfg:
        mapping = [
            (".fused_qkvbfg_a_proj", f".{p}_proj", i)
            for i, p in enumerate(("q", "k", "v", "b", "f_a", "g_a"))
        ] + [(".fused_fg_b_proj", f".{p}_b_proj", i) for i, p in enumerate("fg")]
    else:
        mapping = [(".qkv_proj", f".{p}_proj", p) for p in "qkv"]
    mapping.extend((".qkv_conv1d", f".{p}_conv1d", i) for i, p in enumerate("qkv"))
    for target_module, source, shard_id in mapping:
        target = name.replace(source, target_module)
        if source in name and target in params:
            param = params[target]
            param.weight_loader(param, weight, shard_id)
            return target
    return None


class SolarOpen2RouterGate(ReplicatedLinear):
    def __init__(self, *args, **kwargs):
        kwargs["params_dtype"] = torch.float32
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return linear_bf16_fp32(x, self.weight.float()), None


class SolarOpen2MoE(nn.Module):
    def __init__(
        self, config, layer_idx, quant_config=None, prefix="", alt_stream=None
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        use_deepep = get_moe_a2a_backend().is_deepep()
        self.gate = SolarOpen2RouterGate(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.num_fused_shared_experts = 0
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            num_fused_shared_experts=0,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_idx,
            quant_config=quant_config,
            activation=config.hidden_act,
            routed_scaling_factor=1.0,
            prefix=f"{prefix}.experts",
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_idx,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )
        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = SolarOpen2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                # DeepEP combines routed experts across EP ranks. Shared experts
                # must be complete locally because no TP reduction follows it.
                **(dict(parallel_group="replicated") if use_deepep else {}),
            )

    def _forward_routed(self, hidden_states, forward_batch=None):
        if hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        else:
            router_logits = self.gate(hidden_states)[0]
            routing_kwargs = {}
            if forward_batch is not None:
                routing_kwargs = dict(
                    num_token_non_padded=forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_idx
                    ),
                )
            topk_output = self.topk(hidden_states, router_logits, **routing_kwargs)
        output = self.experts(hidden_states, topk_output)
        if (
            self.routed_scaling_factor is not None
            and self.routed_scaling_factor != 1.0
            and not self.experts.should_fuse_routed_scaling_factor_in_topk
        ):
            output = output * self.routed_scaling_factor
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)

        shared_output = None
        alt_stream = getattr(self, "alt_stream", None)
        if (
            alt_stream is not None
            and self.shared_experts is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            current_stream = torch.cuda.current_stream()
            alt_stream.wait_stream(current_stream)
            shared_output = self.shared_experts(hidden_states.clone())
            with torch.cuda.stream(alt_stream):
                output = self._forward_routed(hidden_states)
            current_stream.wait_stream(alt_stream)
        else:
            if self.shared_experts is not None and hidden_states.shape[0] > 0:
                # FusedMoE may overwrite its input; evaluate the shared branch first.
                shared_output = self.shared_experts(hidden_states)
            output = self._forward_routed(hidden_states)
        if shared_output is not None:
            output = output + shared_output
        if self.tp_size > 1 and not use_reduce_scatter:
            output = tensor_model_parallel_all_reduce(output)
        return output

    def _forward_deepep(self, hidden_states, forward_batch):
        if hidden_states.shape[0] > 0 and forward_batch is None:
            raise ValueError("DeepEP MoE requires a ForwardBatch for token dispatch")
        shared_output = None
        if self.shared_experts is not None and hidden_states.shape[0] > 0:
            shared_output = self.shared_experts(hidden_states)
        output = self._forward_routed(hidden_states, forward_batch)
        if shared_output is not None:
            output = output + shared_output
        # DeepEP combine reduces routed experts; the shared branch is replicated.
        return output


class SolarOpen2DecoderLayer(nn.Module):
    def __init__(
        self, config, layer_idx, quant_config=None, prefix="", alt_stream=None
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        self.self_attn = self._create_attention(
            config, layer_idx, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = self._create_moe(
            config, layer_idx, quant_config, prefix=f"{prefix}.mlp"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn, ffn = self.stage_facts(config, layer_idx)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    def _create_attention(self, config, layer_idx, quant_config, prefix):
        attention_cls = (
            SolarOpen2Attention
            if layer_idx in config.gqa_layers
            else SolarOpen2KimiDeltaAttention
        )
        return attention_cls(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
            # The FFN input boundary completes the attention's output sum.
            reduce_results=False,
        )

    def _create_moe(self, config, layer_idx, quant_config, prefix):
        return SolarOpen2MoE(
            config, layer_idx, quant_config, prefix=prefix, alt_stream=self.alt_stream
        )

    def _forward_attention(self, positions, hidden_states, forward_batch):
        return self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )

    def forward(self, positions, hidden_states, forward_batch):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self._forward_attention(positions, hidden_states, forward_batch)
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        return self.ffn_boundary.finish(hidden_states, forward_batch)

    @staticmethod
    def stage_facts(config, layer_idx):
        return declare_attn(), declare_ffn(
            sparse=True,
            next_layer_sparse=layer_idx + 1 < config.num_hidden_layers,
            output_complete=True,
        )


class SolarOpen2Model(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.alt_stream = get_stream("alt") if is_cuda_alike() else None
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SolarOpen2DecoderLayer(
                config, idx, quant_config, prefix, alt_stream=self.alt_stream
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
            stage_facts=lambda idx: SolarOpen2DecoderLayer.stage_facts(config, idx),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | PPProxyTensors:
        if get_parallel().pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        for i in range(self.start_layer, self.end_layer):
            ctx = get_global_expert_distribution_recorder().with_current_layer(i)
            with ctx:
                hidden_states = self.layers[i](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )

        return hidden_states


class SolarOpen2ForCausalLM(nn.Module):
    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return "Solar Open2 keeps shared experts separate from routed experts."

    @staticmethod
    def num_fusable_shared_experts(hf_config):
        return 0

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        if config.n_routed_experts is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkvbfg_a_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        _validate_supported_config(config)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.model = SolarOpen2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.start_layer, self.end_layer = self.model.start_layer, self.model.end_layer
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            if config.tie_word_embeddings and self.pp_group.is_first_rank:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config=config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        else:
            return hidden_states

    def prepare_context_parallel_metadata_for_dcp(
        self,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens_sum: int,
        kv_buffer_shape: torch.Size,
        kv_cache_dtype,
        kv_cache_device,
        create_chunked_prefix_cache_kv_indices_fn,
    ):
        return prepare_decode_context_parallel_metadata(
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            seq_lens_sum=seq_lens_sum,
            kv_buffer_shape=kv_buffer_shape,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_device=kv_cache_device,
            create_chunked_prefix_cache_kv_indices_fn=create_chunked_prefix_cache_kv_indices_fn,
        )

    def _is_non_local_pp_weight(self, name: str) -> bool:
        if self.pp_group.world_size == 1:
            return False

        layer_id = get_layer_id(name)
        if layer_id is not None:
            return not (self.model.start_layer <= layer_id < self.model.end_layer)
        if name.startswith("model.embed_tokens."):
            return not self.pp_group.is_first_rank
        if name.startswith(("model.norm.", "lm_head.")):
            return not self.pp_group.is_last_rank
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        expert_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
            if getattr(self.config, "n_routed_experts", None)
            else []
        )
        params = dict(self.named_parameters())
        for args in weights:
            name, weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            layer_idx = get_layer_id(name)
            if self._is_non_local_pp_weight(name):
                continue
            if layer_idx is not None and layer_idx >= self.config.num_hidden_layers:
                # Checkpoints may also contain next-token prediction layers.
                continue
            if "rotary_emb." in name:
                continue
            if ".mlp.experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name in name:
                        target = name.replace(weight_name, param_name)
                        param = params[target]
                        param.weight_loader(
                            param,
                            weight,
                            target,
                            expert_id=expert_id,
                            shard_id=shard_id,
                        )
                        break
                else:
                    raise KeyError(f"Unrecognized Solar Open2 expert weight: {name}")
                continue
            # Resolve cache scales before packing projection weights.
            original_name = name
            name = maybe_remap_kv_scale_name(name, params)
            if name is None or (name != original_name and name not in params):
                # Quark exports output scales even when the KV cache is BF16.
                continue
            if layer_idx is not None and ".self_attn." in name:
                attention = self.model.layers[layer_idx].self_attn
                # An MTP head can use KDA where the trunk config selects GQA.
                if isinstance(attention, SolarOpen2KimiDeltaAttention):
                    if _load_kda_weight(name, weight, attention, params) is not None:
                        continue
            for target_module, source, shard_id in stacked_mapping:
                target = name.replace(source, target_module)
                if source not in name or target not in params:
                    continue
                param = params[target]
                param.weight_loader(param, weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params:
                    continue
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight, **kwargs)


EntryClass = SolarOpen2ForCausalLM

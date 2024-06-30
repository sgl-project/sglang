# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
"""Inference-only Grok1 model."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from transformers import PretrainedConfig
from vllm import _custom_ops as ops
from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import print_warning_once

from sglang.srt.layers.fused_moe import fused_moe
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.controller.model_runner import InputMetadata

use_fused = True


class Grok1MLP(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(
            self.hidden_dim, self.ffn_dim, bias=False, quant_config=quant_config
        )
        self.w2 = ReplicatedLinear(
            self.ffn_dim, self.hidden_dim, bias=False, quant_config=quant_config
        )
        self.w3 = ReplicatedLinear(
            self.hidden_dim, self.ffn_dim, bias=False, quant_config=quant_config
        )

        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class Grok1MoEUnfused(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}."
            )
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(
            range(self.num_total_experts), self.tp_size
        )[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList(
            [
                (
                    Grok1MLP(
                        self.num_total_experts,
                        config.hidden_size,
                        config.intermediate_size,
                        quant_config=quant_config,
                    )
                    if idx in self.expert_indicies
                    else None
                )
                for idx in range(self.num_total_experts)
            ]
        )
        self.gate = ReplicatedLinear(
            config.hidden_size, self.num_total_experts, bias=False, quant_config=None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        router_logits = 30 * F.tanh(router_logits / 30)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights.to(hidden_states.dtype)
        hidden_dim = hidden_states.shape[1]

        final_hidden_states = torch.zeros(
            (hidden_states.shape[0], hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_total_experts
        ).permute(2, 1, 0)

        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        return tensor_model_parallel_all_reduce(final_hidden_states)


class Grok1MoE(nn.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config = quant_config

        # FIXME(pcmoritz): Make this more general to support different
        # quantization schemes
        self.use_fp8 = isinstance(quant_config, Fp8Config)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=self.params_dtype,
            quant_config=None,
        )

        if self.use_fp8 and self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=params_dtype,
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=params_dtype,
            )
        )

        set_weight_attrs(
            self.w13_weight,
            {
                "weight_loader": self.weight_loader,
            },
        )
        set_weight_attrs(
            self.w2_weight,
            {
                "weight_loader": self.weight_loader,
            },
        )

        # Used for fp8.
        self.w13_scale = None
        self.w2_scale = None
        self.a13_scale = None
        self.a2_scale = None

        if self.use_fp8:
            # WEIGHT_SCALE (for fp8)
            self.w13_scale = nn.Parameter(
                torch.ones(self.num_total_experts, dtype=torch.float32),
                requires_grad=False,
            )
            self.w2_scale = nn.Parameter(
                torch.ones(self.num_total_experts, dtype=torch.float32),
                requires_grad=False,
            )

            # If loading fp8 checkpoint, pass the weight loaders.
            # If loading an fp16 checkpoint, do not (we will quantize in
            #   process_weights_after_loading()
            if quant_config.is_checkpoint_fp8_serialized:
                set_weight_attrs(
                    self.w13_scale,
                    {
                        "weight_loader": self.weight_loader,
                    },
                )
                set_weight_attrs(
                    self.w2_scale,
                    {
                        "weight_loader": self.weight_loader,
                    },
                )

            # ACT_SCALE (for fp8)
            if quant_config.activation_scheme == "static":
                if not quant_config.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "Found static activation scheme for checkpoint that "
                        "was not serialized fp8."
                    )
                self.a13_scale = nn.Parameter(
                    torch.zeros(self.num_total_experts, dtype=torch.float32),
                    requires_grad=False,
                )
                self.a2_scale = nn.Parameter(
                    torch.zeros(self.num_total_experts, dtype=torch.float32),
                    requires_grad=False,
                )

                set_weight_attrs(
                    self.a13_scale,
                    {
                        "weight_loader": self.weight_loader,
                    },
                )
                set_weight_attrs(
                    self.a2_scale,
                    {
                        "weight_loader": self.weight_loader,
                    },
                )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
        pre_sharded: bool,
    ):
        param_data = param.data
        shard_size = self.intermediate_size
        if pre_sharded:
            # The weight is already sharded. Readl the full shard
            shard = slice(None)
        else:
            tp_rank = get_tensor_model_parallel_rank()
            shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id, shard_size : 2 * shard_size, :] = loaded_weight[
                shard, :
            ]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]
        if "act_scale" in weight_name or "weight_scale" in weight_name:
            param_data[expert_id] = loaded_weight

    def process_weights_after_loading(self):
        # Fp8 is the only case where we need to process after loading.
        if not self.use_fp8:
            return

        # If checkpoint is fp16, quantize here.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w13_weight = torch.empty_like(
                self.w13_weight.data, dtype=torch.float8_e4m3fn
            )
            w2_weight = torch.empty_like(self.w2_weight.data, dtype=torch.float8_e4m3fn)
            for expert in range(self.num_total_experts):
                w13_weight[expert, :, :], self.w13_scale[expert] = ops.scaled_fp8_quant(
                    self.w13_weight.data[expert, :, :]
                )
                w2_weight[expert, :, :], self.w2_scale[expert] = ops.scaled_fp8_quant(
                    self.w2_weight.data[expert, :, :]
                )
            self.w13_weight = nn.Parameter(w13_weight, requires_grad=False)
            self.w2_weight = nn.Parameter(w2_weight, requires_grad=False)

        # If checkpoint is fp8 + static, cleanup act_scales.
        #   Since state_dict has an act_scale per expert but our kernels
        #   are passed one act_scale shared across all experts.
        elif self.quant_config.activation_scheme == "static":
            if self.a13_scale is None or self.a2_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )

            if not all_close_1d(self.a13_scale) or not all_close_1d(self.a2_scale):
                print_warning_once(
                    "Found act_scales that are not equal for fp8 MoE layer. "
                    "Using the maximum across experts for each layer. "
                )

            self.a13_scale = nn.Parameter(self.a13_scale.max(), requires_grad=False)
            self.a2_scale = nn.Parameter(self.a2_scale.max(), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            router_logits,
            self.top_k,
            renormalize=False,
            inplace=True,
            use_fp8=self.use_fp8,
            w1_scale=self.w13_scale,
            w2_scale=self.w2_scale,
            a1_scale=self.a13_scale,
            a2_scale=self.a2_scale,
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


class Grok1Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        logit_cap: float = 30,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = 128
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Grok1DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = Grok1Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
        )
        if use_fused:
            self.block_sparse_moe = Grok1MoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
            )
        else:
            self.block_sparse_moe = Grok1MoEUnfused(
                config=config, quant_config=quant_config
            )
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = (
            self.post_attn_norm(
                self.self_attn(
                    positions=positions,
                    hidden_states=self.pre_attn_norm(hidden_states),
                    input_metadata=input_metadata,
                )
            )
            + hidden_states
        )

        hidden_states = (
            self.post_moe_norm(self.block_sparse_moe(self.pre_moe_norm(hidden_states)))
            + hidden_states
        )

        return hidden_states


class Grok1Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                Grok1DecoderLayer(config, i, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        hidden_states.mul_(self.config.embedding_multiplier_scale)

        for i in range(len(self.layers)):
            hidden_states = self.layers[i](positions, hidden_states, input_metadata)

        hidden_states = self.norm(hidden_states)
        hidden_states.mul_(self.config.output_multiplier_scale)
        return hidden_states


class Grok1ModelForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Grok1Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

        # Monkey patch _prepare_weights to load pre-sharded weights
        setattr(DefaultModelLoader, "_prepare_weights", _prepare_presharded_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, input_metadata, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, input_metadata
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        if use_fused:
            expert_params_mapping = (
                [
                    # These are the weight scales for the experts
                    # (param_name, weight_name, expert_id)
                    (
                        "w13_scale" if weight_name in ["w1", "w3"] else "w2_scale",
                        f"experts.{expert_id}.{weight_name}.weight_scale",
                        expert_id,
                    )
                    for expert_id in range(self.config.num_local_experts)
                    for weight_name in ["w1", "w2", "w3"]
                ]
                + [
                    # These are the weights for the experts
                    # (param_name, weight_name, expert_id)
                    (
                        "w13_weight" if weight_name in ["w1", "w3"] else "w2_weight",
                        f"experts.{expert_id}.{weight_name}.weight",
                        expert_id,
                    )
                    for expert_id in range(self.config.num_local_experts)
                    for weight_name in ["w1", "w2", "w3"]
                ]
                + [
                    # These are the activation scales for the experts
                    # (param_name, weight_name, expert_id)
                    (
                        "a13_scale" if weight_name in ["w1", "w3"] else "a2_scale",
                        f"experts.{expert_id}.{weight_name}.act_scale",
                        expert_id,
                    )
                    for expert_id in range(self.config.num_local_experts)
                    for weight_name in ["w1", "w2", "w3"]
                ]
            )
        else:
            expert_params_mapping = []

        params_dict = dict(self.named_parameters())
        if get_tensor_model_parallel_rank() == 0:
            weights = tqdm.tqdm(weights, total=int(len(params_dict) * 3.4))
        for name, loaded_weight in weights:
            # print(get_tensor_model_parallel_rank(), name)
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name,
                        expert_id=expert_id,
                        pre_sharded=get_tensor_model_parallel_world_size() > 1,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


old_prepare_weights = getattr(DefaultModelLoader, "_prepare_weights")


def _prepare_presharded_weights(
    self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
) -> Tuple[str, List[str], bool]:
    import glob
    import os

    if get_tensor_model_parallel_world_size() == 1:
        return old_prepare_weights(self, model_name_or_path, revision, fall_back_to_pt)

    tp_rank = get_tensor_model_parallel_rank()
    allow_patterns = [f"*-{tp_rank:03d}.bin"]

    hf_folder = model_name_or_path

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    use_safetensors = False

    return hf_folder, hf_weights_files, use_safetensors


EntryClass = Grok1ModelForCausalLM

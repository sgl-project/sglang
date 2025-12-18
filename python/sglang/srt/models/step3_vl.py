import logging
import math
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from sglang.srt.configs.step3_vl import (
    Step3TextConfig,
    Step3VisionEncoderConfig,
    Step3VLConfig,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, log_info_on_rank0, make_layers

logger = logging.getLogger(__name__)


"""
Text Model
"""


class Step3TextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Step3TextMoEMLP(nn.Module):
    # Native
    def __init__(
        self,
        layer_id: int,
        config: Step3TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.topk = TopK(
            top_k=config.moe_top_k,
            renormalize=config.norm_expert_weight,
            use_grouped_topk=False,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            output_size=config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if get_moe_a2a_backend().is_deepep():
            raise NotImplementedError("DeepEP MoE is not supported yet in Step3 model.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class Step3TextAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        share_q_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.all_tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        self.attn_tp_rank = attn_tp_rank
        self.layer_id = layer_id
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = share_q_dim if share_q_dim else head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.q_size, self.kv_size, self.kv_size],
            bias=False,
            quant_config=quant_config,
            tp_rank=0,  # In fact, we need a MergedReplicatedLinear
            tp_size=1,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.inter_norm = RMSNorm(self.q_size, eps=rms_norm_eps)

        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("wq", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.inter_norm(q.contiguous())
        q, _ = self.wq(q)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Step3TextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Step3TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        # TODO: support shared experts fusion
        # self.n_shared_experts = 1
        # self.num_fused_shared_experts = (
        #     0
        #     if global_server_args.disable_shared_experts_fusion
        #     else self.n_shared_experts
        # )
        self.num_fused_shared_experts = 0
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = Step3TextAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            share_q_dim=config.share_q_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
        else:
            # Default to 1dense.
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]

        self.use_moe = False

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_id = layer_id
        self.is_layer_sparse = True if layer_id in moe_layers_idx else False
        self.is_previous_layer_sparse = (
            True if layer_id - 1 in moe_layers_idx else False
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=self.is_previous_layer_sparse,
        )

        if not self.is_layer_sparse:
            self.mlp = Step3TextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.use_moe = True
            if self.num_fused_shared_experts == 0:
                self.moe = Step3TextMoEMLP(
                    layer_id=layer_id,
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp", prefix),
                )
                self.share_expert = Step3TextMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.share_expert_dim,
                    hidden_act="silu",
                    quant_config=quant_config,
                    prefix=add_prefix("share_expert", prefix),
                )
            else:
                self.moe = Step3TextMoEMLP(
                    layer_id=layer_id,
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp", prefix),
                )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def moe_mlp_forward(self, hidden_states):
        if not self.num_fused_shared_experts:
            h = hidden_states.clone()
            hidden_states = self.moe(hidden_states)
            hidden_states += self.share_expert(h)
        else:
            hidden_states = self.moe(hidden_states)
        return hidden_states

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        if self.use_moe:
            hidden_states = self.moe_mlp_forward(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Step3TextModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Step3TextDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


"""
Vision Model
"""


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


class Step3VisionMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        bias: bool = True,
        hidden_act="quick_gelu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Since this is a dense model,
        # the MLP component likewise adopts a DP-MLP approach modeled after DP Attention.
        # This choice may not represent the optimal solution and remains open to further deliberation.
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.fc1 = ColumnParallelLinear(
            dim,
            intermediate_size,
            bias=bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.act = ACT2FN[hidden_act]  # quick_gelu
        self.fc2 = RowParallelLinear(
            intermediate_size,
            dim,
            bias=bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Step3VisionAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        quant_config=None,
        prefix: str = "",
    ) -> None:

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.out_proj = RowParallelLinear(
            dim,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        self.scale = self.head_dim**-0.5

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(hidden_states)
        return attn_output


class Step3VisionEmbeddings(nn.Module):

    def __init__(self, config: Step3VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.pad_tp_size = 4  # hard code for padding
        # To load the pretrained weights, we still use P+1 as the seqlen
        self.position_embedding = torch.nn.Embedding(
            self.num_patches + 1, self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches + 1).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # pad
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), patch_embeds.size(1)
        )
        embeddings = torch.cat(
            [
                embeddings[:, 0, :].unsqueeze(1).repeat(1, self.pad_tp_size - 1, 1),
                embeddings,
            ],
            dim=1,
        )
        return embeddings


class Step3VisionEncoderLayer(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=1e-6)

        self.self_attn = Step3VisionAttention(
            self.embed_dim, num_heads=config.num_attention_heads
        )
        self.mlp = Step3VisionMLP(
            dim=self.embed_dim,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = hidden_states + self.layer_norm1(self.self_attn(hidden_states))
        hidden_states = hidden_states + self.layer_norm2(self.mlp(hidden_states))
        return hidden_states


class Step3VisionTransformer(nn.Module):
    def __init__(self, config: Step3VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.embeddings = Step3VisionEmbeddings(config)
        self.transformer = Step3VisionEncoder(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.transformer(inputs_embeds=hidden_states)
        return hidden_states


class Step3VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Step3VisionEncoderLayer`].

    Args:
        config: StepVisionEncoderConfig
    """

    def __init__(self, config: Step3VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Step3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds,
    ) -> torch.Tensor:

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
            )

        return hidden_states


class Step3VLForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config: Step3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Step3TextModel(
            config.text_config, quant_config, prefix=add_prefix("model", prefix)
        )

        self.vision_model = Step3VisionTransformer(config.vision_config)

        self.vit_downsampler = nn.Conv2d(
            config.vision_config.hidden_size,
            config.vision_config.output_hidden_size,
            kernel_size=2,
            stride=config.understand_projector_stride,
        )
        self.vit_downsampler2 = nn.Conv2d(
            config.vision_config.output_hidden_size,
            config.vision_config.output_hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_large_projector = nn.Linear(
            config.vision_config.output_hidden_size * 2,
            config.hidden_size,
            bias=config.projector_bias,
        )

        # TODO: support shared experts fusion
        # self.n_shared_experts = 1
        # self.num_fused_shared_experts = (
        #     0
        #     if global_server_args.disable_shared_experts_fusion
        #     else self.n_shared_experts
        # )
        self.num_fused_shared_experts = 0
        self.config.tie_word_embeddings = False
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.text_config.vocab_size,
                config.text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config.text_config)

    def _get_vision_model_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)[:, 4:]

    def _flatten_embeddings(self, embeddings) -> torch.Tensor:

        if isinstance(embeddings, torch.Tensor):
            # Flatten all but the last dimension.
            return embeddings.flatten(0, -2)

        return torch.cat(tuple(self._flatten_embeddings(t) for t in embeddings))

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(sqrt(P))
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vit_downsampler(image_features)
        image_features = self.vit_downsampler2(image_features)
        n_dim = image_features.size(1)
        image_features = image_features.view(B, n_dim, -1).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        assert len(items) == 1  # We only have images.

        item = items[0]
        pixel_values = item.feature.type(self.vision_model.dtype)
        num_patches = item.model_specific_data.get("num_patches")
        patch_pixel_values = item.model_specific_data.get("patch_pixel_values", None)
        if patch_pixel_values is not None:
            patch_pixel_values = patch_pixel_values.type(self.vision_model.dtype)

        if patch_pixel_values is not None:
            patch_pixel_values = patch_pixel_values.to("cuda")

        image_features = self._get_vision_model_output(pixel_values)
        patch_image_features = (
            self._get_vision_model_output(patch_pixel_values)
            if patch_pixel_values is not None
            else None
        )

        image_features = self._process_image_features(image_features)
        patch_image_features = (
            self._process_image_features(patch_image_features)
            if patch_image_features is not None
            else None
        )

        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx : cur_patch_idx + num_patch
                ]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0]
            )
        return self._flatten_embeddings(merged_image_features)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", 0),
            (".qkv_proj", ".k_proj", 1),
            (".qkv_proj", ".v_proj", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.text_config.moe_num_experts
            + self.num_fused_shared_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params = set()

        def match_expert_and_shard_ids(name_path: str, weight_path: str) -> bool:
            name_parts = name_path.split(".")
            weight_parts = weight_path.split(".")
            shard_id_matches = name_parts[4] == weight_parts[2]
            return shard_id_matches

        for name, loaded_weight in weights:
            if "vision_model" in name:
                name = name.replace("self_attn", "self_attn.attn")
                name = name.replace("out_proj", "proj")

            # TODO: support vision model
            if self.num_fused_shared_experts > 0 and "share" in name:
                # assert False
                name = name.replace("share_expert", "moe")
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if (
                        expert_id != self.config.text_config.moe_num_experts
                        or not match_expert_and_shard_ids(name, weight_name)
                    ):
                        continue

                    part_name = weight_name.split(".")[-2]
                    fake_weight_name = name.replace(part_name, weight_name[:-1])
                    actual_param_name = name.replace(part_name + ".", param_name)
                    param = params_dict[actual_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "gate." not in name and "moe" in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if "moe" not in name:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                else:
                    if "gate." in name:
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                        continue

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if expert_id == self.config.text_config.moe_num_experts:
                            continue
                        if not match_expert_and_shard_ids(name, weight_name):
                            continue
                        part_name = weight_name.split(".")[-2]
                        fake_weight_name = name.replace(part_name, weight_name[:-1])
                        actual_param_name = name.replace(part_name + ".", param_name)
                        param = params_dict[actual_param_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(actual_param_name)
                        # Don't break here, because this 'loaded_weight' includes all the weights for this layer

    @classmethod
    def get_model_config_for_expert_location(cls, config: Step3VLConfig):
        return ModelConfigForExpertLocation(
            num_layers=config.text_config.num_hidden_layers,
            num_logical_experts=config.text_config.moe_num_experts,
            num_groups=None,
        )


EntryClass = Step3VLForConditionalGeneration

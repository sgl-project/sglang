# SPDX-License-Identifier: Apache-2.0
# Adapted from inclusionAI/Ming-Image (Apache-2.0 modeling code).
"""Ming's Bailing multimodal encoder and non-causal query connector."""

from collections import defaultdict
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.moe import LingBotVideoGroupedExperts
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl_vision import (
    Qwen2_5VLVisionTransformer,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3MLP
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput


def ming_position_ids(
    token_ids: list[int], grids: list[tuple[int, int, int]], image_token: int
):
    """CPU-side centered video-RoPE positions, including learned image queries."""
    positions = []
    cursor = 0
    offset = 0
    for frames, height, width in grids:
        start = token_ids.index(image_token, cursor)
        text_len = start - cursor
        positions.append(torch.arange(offset, offset + text_len).expand(3, -1))
        h, w = height // 2, width // 2
        t = torch.arange(frames).repeat_interleave(h * w) + offset + text_len
        rows = torch.arange(h).repeat_interleave(w).repeat(frames) - (h - 1) // 2
        cols = torch.arange(w).repeat(frames * h) - (w - 1) // 2
        positions.append(torch.stack((t, rows + t, cols + t)))
        offset += text_len + frames
        cursor = start + frames * h * w
    positions.append(
        torch.arange(offset, offset + len(token_ids) - cursor).expand(3, -1)
    )
    return torch.cat(positions, dim=1).unsqueeze(1)


class MingAttention(nn.Module):
    def __init__(self, config, *, connector=False):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads // get_tp_world_size()
        self.num_kv_heads = max(1, config.num_key_value_heads // get_tp_world_size())
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=connector,
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.q_norm = (
            nn.Identity()
            if connector
            else RMSNorm(
                self.head_dim,
                eps=config.rms_norm_eps,
                cast_x_before_out_mul=True,
                force_native=True,
            )
        )
        self.k_norm = (
            nn.Identity()
            if connector
            else RMSNorm(
                self.head_dim,
                eps=config.rms_norm_eps,
                cast_x_before_out_mul=True,
                force_native=True,
            )
        )
        self.attn = LocalAttention(
            self.num_heads, self.head_dim, self.num_kv_heads, causal=not connector
        )
        self.rotary_dim = (
            self.head_dim
            if connector
            else int(self.head_dim * config.partial_rotary_factor)
        )
        self.rope_theta = config.rope_theta
        self.connector = connector

    def forward(self, x, positions):
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            -1,
        )
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        inv = 1.0 / self.rope_theta ** (
            torch.arange(0, self.rotary_dim, 2, device=x.device, dtype=torch.float32)
            / self.rotary_dim
        )
        # the public BF16 inference path casts RoPE buffers before evaluation
        inv = inv.to(x.dtype).float()
        with torch.autocast(x.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * inv
            angles = torch.cat((angles, angles), -1)
        if not self.connector:
            # the spatial frequencies alternate H/W; temporal frequencies are last
            axes = ([1, 2] * 12 + [0] * 8) * 2
            angles = torch.stack(
                [angles[axis, ..., i] for i, axis in enumerate(axes)], -1
            )
        cos, sin = (
            angles.cos().to(x.dtype).unsqueeze(-2),
            angles.sin().to(x.dtype).unsqueeze(-2),
        )

        def rotate(value):
            rot, tail = value[..., : self.rotary_dim], value[..., self.rotary_dim :]
            half = self.rotary_dim // 2
            rotated = torch.cat((-rot[..., half:], rot[..., :half]), -1)
            return torch.cat((rot * cos + rotated * sin, tail), -1)

        output = self.attn(rotate(q), rotate(k), v).flatten(-2)
        return self.o_proj(output)[0]


class MingRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts), requires_grad=False
        )

    def forward(self, x):
        return F.linear(x.float(), self.weight.float()).sigmoid()


class MingMLP(Qwen3MLP):
    def forward(self, x):
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        # preserve the checkpoint's BF16 rounding between SiLU and multiplication
        return self.down_proj(F.silu(gate) * up)[0]


class MingExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = MingRouter(config)
        self.image_gate = MingRouter(config)
        self.intermediate_size = config.moe_intermediate_size // get_tp_world_size()
        self.experts = LingBotVideoGroupedExperts(
            config.num_experts, config.hidden_size, self.intermediate_size
        )
        self.shared_experts = MingMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.num_shared_experts,
            "silu",
        )
        self.runner_config = MoeRunnerConfig(
            num_experts=config.num_experts,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=config.num_experts_per_tok,
            activation="silu_rounded",
            is_gated=True,
            inplace=False,
            no_combine=True,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
        )

    def forward(self, x, image_mask):
        tokens = x.flatten(0, 1)
        mask = image_mask.reshape(-1, 1)
        scores = torch.where(mask, self.image_gate(tokens), self.gate(tokens))
        choice = scores + torch.where(
            mask, self.image_gate.expert_bias, self.gate.expert_bias
        )
        grouped = choice.unflatten(-1, (self.config.n_group, -1))
        groups = (
            grouped.topk(2, dim=-1)
            .values.sum(-1)
            .topk(self.config.topk_group, dim=-1, sorted=False)
            .indices
        )
        group_mask = torch.zeros_like(grouped[..., 0], dtype=torch.bool).scatter_(
            1, groups, True
        )
        choice = choice.masked_fill(
            ~group_mask.unsqueeze(-1).expand_as(grouped).flatten(1), float("-inf")
        )
        ids = choice.topk(self.config.num_experts_per_tok, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = (
            weights
            / (weights.sum(-1, keepdim=True) + 1e-20)
            * self.config.routed_scaling_factor
        )
        topk = StandardTopKOutput(
            weights.float(), ids.int(), torch.empty(0, device=x.device)
        )
        out = fused_experts(
            tokens.contiguous(),
            self.experts.w13_weight,
            self.experts.w2,
            topk,
            self.runner_config,
        )
        if get_tp_world_size() > 1:
            out = get_tp_group().all_reduce(out)
        out = (out.float() * weights.unsqueeze(-1)).sum(1).to(x.dtype)
        return out.view_as(x) + self.shared_experts(x)


class MingEncoderBlock(nn.Module):
    def __init__(self, config, layer_idx, *, connector=False):
        super().__init__()
        self.attention = MingAttention(config, connector=connector)
        self.is_moe = not connector and layer_idx >= config.first_k_dense_replace
        self.mlp = (
            MingExperts(config)
            if self.is_moe
            else MingMLP(config.hidden_size, config.intermediate_size, "silu")
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )

    def forward(self, x, positions, image_mask=None):
        x = x + self.attention(self.input_layernorm(x), positions)
        normalized = self.post_attention_layernorm(x)
        return x + (
            self.mlp(normalized, image_mask) if self.is_moe else self.mlp(normalized)
        )


class MingImageEncoder(TextEncoder):
    @property
    def dtype(self):
        return self.word_embeddings.weight.dtype

    def __init__(self, config):
        super().__init__(config)
        llm = SimpleNamespace(**config.llm_config)
        connector = SimpleNamespace(**config.connector_config)
        projection = config.projection_config
        self.word_embeddings = VocabParallelEmbedding(llm.vocab_size, llm.hidden_size)
        self.layers = nn.ModuleList(
            MingEncoderBlock(llm, i) for i in range(llm.num_hidden_layers)
        )
        self.norm = RMSNorm(
            llm.hidden_size,
            eps=llm.rms_norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.vision = Qwen2_5VLVisionTransformer(
            SimpleNamespace(**config.vision_config)
        )
        # Ming casts vision RoPE buffers with the model, unlike Qwen2.5-VL
        self.vision.rotary_pos_emb.inv_freq = self.vision.rotary_pos_emb.inv_freq.to(
            self.vision.dtype
        )
        self.vision.merger.ln_q.cast_x_before_out_mul = False
        for block in self.vision.blocks:
            block.norm1.cast_x_before_out_mul = False
            block.norm2.cast_x_before_out_mul = False
        self.linear_proj = nn.Sequential(
            nn.Linear(config.vision_config["out_hidden_size"], llm.hidden_size),
            nn.GELU(),
            nn.Linear(llm.hidden_size, llm.hidden_size),
        )
        self.connector = nn.ModuleList(
            MingEncoderBlock(connector, i, connector=True)
            for i in range(connector.num_hidden_layers)
        )
        self.connector_norm = RMSNorm(
            connector.hidden_size,
            eps=connector.rms_norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.proj_in = nn.Linear(llm.hidden_size, connector.hidden_size)
        self.proj_out = nn.Linear(
            connector.hidden_size, projection["diffusion_c_input_dim"]
        )
        self.selected_layers = tuple(projection["selected_hidden_states_layers"])
        direct_dim = llm.hidden_size * len(self.selected_layers)
        self.proj_directvlm = nn.Sequential(
            RMSNorm(
                direct_dim, eps=1e-5, cast_x_before_out_mul=True, force_native=True
            ),
            nn.Linear(direct_dim, projection["diffusion_inner_dim"]),
        )
        self.query_tokens_dict = nn.ParameterDict(
            {
                f"{scale}x{scale}": nn.Parameter(
                    torch.empty(scale * scale, llm.hidden_size)
                )
                for scale in projection["img_gen_scales"]
            }
        )
        self.image_token = llm.image_patch_token
        self.layer_names = ["layers", "vision.blocks", "connector"]

    def forward(
        self, input_ids, position_ids, pixel_values=None, image_grid_thw=None, **kwargs
    ):
        x = self.word_embeddings(input_ids)
        queries = torch.cat(list(self.query_tokens_dict.values()), 0)
        query_count = queries.shape[0]
        image_mask = input_ids == self.image_token
        image_embeddings = queries
        if pixel_values is not None:
            vision_features = self.vision(pixel_values, image_grid_thw)
            with torch.autocast(x.device.type, enabled=False):
                image_embeddings = F.normalize(
                    self.linear_proj(vision_features), dim=-1
                )
            image_embeddings = torch.cat((image_embeddings, queries), 0)
        x = x.masked_scatter(image_mask.unsqueeze(-1), image_embeddings.to(x.dtype))
        selected = []
        for i, layer in enumerate(self.layers):
            if i in self.selected_layers:
                selected.append(x)
            x = layer(x, position_ids, image_mask)
        x = self.norm(x)
        if len(self.layers) in self.selected_layers:
            selected.append(x)
        prompt_length = input_ids.shape[1] - query_count - 2
        direct = self.proj_directvlm(
            torch.cat([hidden[:, :prompt_length] for hidden in selected], -1)
        )
        queries = self.proj_in(x[:, -query_count - 1 : -1])
        positions = torch.arange(query_count, device=x.device).unsqueeze(0)
        for layer in self.connector:
            queries = layer(queries, positions)
        queries = self.proj_out(self.connector_norm(queries))
        if self.config.projection_config["connector_norm"]:
            queries = F.normalize(queries, dim=-1)
        return BaseEncoderOutput(last_hidden_state=queries, hidden_states=(direct,))

    def should_materialize_checkpoint_weight(self, name):
        return not (
            name.endswith("lm_head.weight")
            or ".audio_gate." in name
            or "rotary_emb.inv_freq" in name
        )

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        pieces = defaultdict(set)
        for name, weight in weights:
            if not self.should_materialize_checkpoint_weight(name):
                continue
            name = name.removeprefix("model.model.")
            name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
            name = name.replace("connector.model.layers.", "connector.").replace(
                "connector.model.norm.", "connector_norm."
            )
            if name == "connector.model.embed_tokens.weight":
                continue
            name = (
                name.replace(".self_attn.", ".attention.")
                .replace(".query_key_value.", ".qkv_proj.")
                .replace(".dense.", ".o_proj.")
            )
            shard = None
            if ".experts." in name:
                prefix, suffix = name.split(".experts.")
                expert, projection, _ = suffix.split(".")
                module = self.get_submodule(prefix)
                width = module.intermediate_size
                rank = get_tp_rank()
                if projection == "down_proj":
                    target = f"{prefix}.experts.w2"
                    params[target].data[int(expert)].copy_(
                        weight[:, rank * width : (rank + 1) * width]
                    )
                else:
                    target = f"{prefix}.experts.w13_weight"
                    offset = 0 if projection == "gate_proj" else width
                    params[target].data[int(expert), offset : offset + width].copy_(
                        weight[rank * width : (rank + 1) * width]
                    )
                pieces[target].add((int(expert), projection))
                expected = module.config.num_experts * (
                    1 if projection == "down_proj" else 2
                )
                if len(pieces[target]) == expected:
                    loaded.add(target)
                continue
            for source, target, shard_id in (
                (".gate_proj.", ".gate_up_proj.", 0),
                (".up_proj.", ".gate_up_proj.", 1),
                (".q_proj.", ".qkv_proj.", "q"),
                (".k_proj.", ".qkv_proj.", "k"),
                (".v_proj.", ".qkv_proj.", "v"),
            ):
                if source in name and not name.startswith("vision."):
                    name, shard = name.replace(source, target), shard_id
                    break
            if name not in params:
                raise ValueError(f"Unexpected Ming encoder checkpoint tensor: {name}")
            param = params[name]
            owner = self.get_submodule(name.rsplit(".", 1)[0])
            if (
                isinstance(owner, (LinearBase, VocabParallelEmbedding))
                or name.startswith("vision.")
                and "weight_loader" in param.__dict__
            ):
                if shard is None:
                    param.weight_loader(param, weight)
                else:
                    param.weight_loader(param, weight, shard)
            else:
                default_weight_loader(param, weight)
            if shard is None:
                loaded.add(name)
            else:
                pieces[name].add(shard)
                expected = {"q", "k", "v"} if isinstance(shard, str) else {0, 1}
                if pieces[name] == expected:
                    loaded.add(name)
        return loaded


EntryClass = MingImageEncoder

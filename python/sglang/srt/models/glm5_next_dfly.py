"""AngelSlim DFly2 draft model for GLM-5.3-Flash."""

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.dsa.dfly_kv import gather_dfly_kv
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel
from torch import nn


# The DFly mHC arithmetic follows the public DFly2 implementation in PR #53538.
class _DraftRMSNorm(nn.Module):
    """Apply RMSNorm with FP32 accumulation and a single final cast."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * h).to(x.dtype)


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = (mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _hc_mixes(
    h: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = h.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    return _hc_split_sinkhorn(
        mixes, scale.float(), base.float(), h.shape[-2], iters, eps
    )


def _hc_pre(h: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...k,...kd->...d", pre_mix.float(), h.float()).to(h.dtype)


def _hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    mixed = torch.einsum("...jk,...jd->...kd", comb, residual.float())
    return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)


class _SlotZeroFusion(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.enorm = _DraftRMSNorm(hidden_size, eps)
        self.hnorm = _DraftRMSNorm(hidden_size, eps)
        self.eh_proj = ReplicatedLinear(
            2 * hidden_size, hidden_size, bias=False, prefix="mtp_fusion.eh_proj"
        )
        self.gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, embeds, target_hidden):
        anchor = embeds[:, 0]
        x = torch.cat(
            (self.enorm(anchor), self.hnorm(target_hidden.to(anchor.dtype))), -1
        )
        fused, _ = self.eh_proj(x)
        result = embeds.clone()
        result[:, 0] = anchor + self.gate.to(fused.dtype) * fused
        return result


@dataclass(frozen=True)
class GLM53DFlyConfig:
    num_layers: int
    block_size: int
    mask_token_id: int
    target_kv_layers: tuple[int, ...]
    target_index_layers: tuple[int, ...]
    target_index_topk: int
    correction_layer: int

    @classmethod
    def from_model(cls, config):
        draft = config.dflash_config
        layers = tuple(draft["target_kv_layers"])
        correction = draft["hidden_correction"]["attentive"]
        if (
            not layers
            or len(layers) != config.num_hidden_layers
            or draft.get("mtp_fusion") != "slot0"
            or not draft.get("causal")
            or config.qk_rope_head_dim
            or correction.get("block_mode") != "gated"
            or correction.get("block_kv_in") != "hidden_embed"
        ):
            raise ValueError("Unsupported GLM-5.3-Flash DFly2 checkpoint layout")
        correction_layer = int(correction["source_layer"])
        if not -len(layers) <= correction_layer < len(layers):
            raise ValueError("DFly2 correction source layer is out of range")
        return cls(
            num_layers=config.num_hidden_layers,
            block_size=int(draft["block_size"]),
            mask_token_id=int(draft["mask_token_id"]),
            target_kv_layers=layers,
            target_index_layers=layers,
            # KPool4 can append the newest incomplete pool after the top-k.
            target_index_topk=int(draft["target_index_topk"]) + config.index_kpool - 1,
            correction_layer=correction_layer,
        )


class GLM53DFlyProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_dim = config.qk_nope_head_dim
        self.value_dim = config.v_head_dim
        self.latent_dim = config.kv_lora_rank
        self.heads = config.num_attention_heads // get_parallel().tp_size
        self.q_a_layernorm = _DraftRMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * self.query_dim,
            bias=False,
        )
        self.kv_a_layernorm = _DraftRMSNorm(self.latent_dim, config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.latent_dim,
            config.num_attention_heads * (self.query_dim + self.value_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.value_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

    def project_query(self, low_rank_query):
        query, _ = self.q_b_proj(self.q_a_layernorm(low_rank_query))
        broadcast_shape = (*query.shape[:-1], self.heads, self.query_dim)
        weights = self.kv_b_proj.weight.view(
            self.heads, self.query_dim + self.value_dim, self.latent_dim
        )
        key, value = weights.split((self.query_dim, self.value_dim), dim=1)
        query = torch.einsum("bthd,hdk->bthk", query.view(broadcast_shape), key)
        return query.float() * self.query_dim**-0.5, value

    def project_output(self, latent, weights, dtype):
        values = torch.einsum("bthk,hvk->bthv", latent.to(dtype), weights)
        output, _ = self.o_proj(values.flatten(-2))
        return output


class GLM53DFlyAttention(GLM53DFlyProjection):
    def __init__(self, config):
        super().__init__(config)
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.latent_dim, bias=False
        )

    def forward(self, hidden, history, valid):
        block_kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden)).float()
        query, weights = self.project_query(self.q_a_proj(hidden))
        block_scores = torch.einsum("bthk,bsk->bths", query, block_kv)
        ctx_scores = torch.einsum("bthk,bsk->bths", query, history)
        ctx_scores = ctx_scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        width = hidden.shape[1]
        causal = torch.ones(
            (width, width), dtype=torch.bool, device=hidden.device
        ).tril()
        block_scores = block_scores.masked_fill(~causal[None, :, None, :], -torch.inf)
        probs = torch.cat((ctx_scores, block_scores), dim=-1).softmax(dim=-1)
        ctx_prob, block_prob = probs.split((history.shape[1], width), dim=-1)
        out = torch.einsum("bths,bsk->bthk", ctx_prob, history)
        out += torch.einsum("bths,bsk->bthk", block_prob, block_kv)
        return self.project_output(out, weights, hidden.dtype)


class GLM53DFlyLayer(nn.Module):
    def __init__(self, config, layer_id: int, prefix: str):
        super().__init__()
        self.config = config
        self.self_attn = GLM53DFlyAttention(config)
        self.input_layernorm = _DraftRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _DraftRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = DeepseekV2MoE(
            config=config,
            layer_id=layer_id,
            quant_config=None,
            prefix=f"{prefix}.mlp",
            is_nextn=True,
        )
        streams = config.hc_mult
        mixes = streams * (2 + streams)
        for branch in ("attn", "ffn"):
            for name, shape in (
                ("fn", (mixes, streams * config.hidden_size)),
                ("scale", (3,)),
                ("base", (mixes,)),
            ):
                self.register_parameter(
                    f"hc_{branch}_{name}",
                    nn.Parameter(torch.empty(shape, dtype=torch.float32)),
                )

    def _mix(self, x, branch):
        return _hc_mixes(
            x,
            getattr(self, f"hc_{branch}_fn"),
            getattr(self, f"hc_{branch}_scale"),
            getattr(self, f"hc_{branch}_base"),
            iters=self.config.hc_sinkhorn_iters,
            eps=self.config.hc_eps,
            norm_eps=self.config.rms_norm_eps,
        )

    def forward(self, hidden, collapsed, history, valid, forward_batch):
        pre, post, combination = self._mix(hidden, "attn")
        attention = self.self_attn(self.input_layernorm(collapsed), history, valid)
        hidden = _hc_post(attention, hidden, post, combination)
        ffn_pre, ffn_post, ffn_combination = self._mix(hidden, "ffn")
        h = self.post_attention_layernorm(_hc_pre(hidden, pre))
        # One grouped MoE per layer for all draft positions.
        h = self.mlp(h.flatten(0, 1), forward_batch).view_as(collapsed)
        hidden = _hc_post(h, hidden, ffn_post, ffn_combination)
        return hidden, _hc_pre(hidden, ffn_pre)


class GLM53DFlyCorrectionAttention(GLM53DFlyProjection):
    def __init__(self, config):
        super().__init__(config)
        dim = 2 * config.hidden_size
        self.q_in = nn.Linear(dim, config.q_lora_rank, bias=False)
        self.kv_in = nn.Linear(dim, self.latent_dim, bias=False)
        self.block_gate = ColumnParallelLinear(
            dim, config.num_attention_heads, bias=False
        )
        attentive = config.dflash_config["hidden_correction"]["attentive"]
        slots = int(attentive.get("max_block_slots", 16))
        if config.dflash_config["block_size"] > slots:
            raise ValueError("The DFly2 block exceeds the correction slot count")
        for name, shape in (
            ("block_gate_bias", (self.heads, slots)),
            ("block_rel_bias", (self.heads, slots)),
            ("block_sink", (self.heads,)),
        ):
            param = nn.Parameter(torch.empty(shape))
            param.weight_loader = self._load_heads
            self.register_parameter(name, param)

    def _load_heads(self, param, weight):
        start = self.heads * get_parallel().tp_rank
        default_weight_loader(param, weight.narrow(0, start, self.heads))

    def forward(self, inputs, history, valid, block, slot):
        block[:, slot] = self.kv_a_layernorm(self.kv_in(inputs)).float()
        if slot == 0:
            return None
        query, value = self.project_query(self.q_in(inputs).unsqueeze(1))
        scores = torch.einsum("bthk,bsk->bths", query, history)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        # Keep empty history rows defined without changing non-empty softmaxes.
        any_valid = valid.any(dim=-1)[:, None, None, None]
        scores = torch.where(any_valid, scores, 0)
        context = (
            torch.einsum("bths,bsk->bthk", scores.softmax(dim=-1), history) * any_valid
        )
        local = block[:, : slot + 1]
        scores = torch.einsum("bthk,bsk->bths", query, local)
        scores = scores + self.block_rel_bias[:, : slot + 1].flip(-1)
        normalizer = torch.logaddexp(
            scores.logsumexp(dim=-1, keepdim=True),
            self.block_sink.view(1, 1, self.heads, 1),
        )
        local_read = torch.einsum("bths,bsk->bthk", (scores - normalizer).exp(), local)
        gate, _ = self.block_gate(inputs)
        gate = (gate.float() + self.block_gate_bias[:, slot]).tanh()
        return self.project_output(
            context + gate[:, None, :, None] * local_read, value, inputs.dtype
        )[:, 0]


class GLM53DFlyCorrection(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        dim = config.dflash_config["hidden_correction"]["intermediate_size"]
        self.hidden_norm = _DraftRMSNorm(hidden, config.rms_norm_eps)
        self.embed_norm = _DraftRMSNorm(hidden, config.rms_norm_eps)
        self.gate_proj = nn.Linear(2 * hidden, dim, bias=False)
        self.up_proj = nn.Linear(2 * hidden, dim, bias=False)
        self.down_proj = nn.Linear(dim, hidden, bias=False)
        self.attention = GLM53DFlyCorrectionAttention(config)

    def forward(self, hidden, prev_embed, history, valid, block, slot):
        inputs = torch.cat((self.hidden_norm(hidden), self.embed_norm(prev_embed)), -1)
        attention = self.attention(inputs, history, valid, block, slot)
        if attention is None:
            return hidden
        return hidden + (
            self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))
            + attention
        )


class DFlyGlm53DraftModel(nn.Module):
    """NoPE MLA drafter that shares the target's paged KPool4 selections."""

    def __init__(self, config, quant_config=None, prefix="", *, draft_model_idx=None):
        super().__init__()
        if draft_model_idx not in (None, 0):
            raise ValueError("GLM-5.3 DFly2 accepts a single draft checkpoint")
        if quant_config is not None:
            raise ValueError(
                "GLM-5.3 DFly2 uses BF16 draft weights. "
                "Set --speculative-draft-model-quantization unquant."
            )
        if not get_moe_a2a_backend().is_none():
            raise ValueError(
                "GLM-5.3 DFly2 requires the draft's MoE A2A backend to be none"
            )
        self.config = config
        self.dfly_config = GLM53DFlyConfig.from_model(config)
        self.mtp_fusion = _SlotZeroFusion(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            GLM53DFlyLayer(config, i, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.final_norm = _DraftRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_correction = GLM53DFlyCorrection(config)

    def bind_target(self, embed_tokens, lm_head):
        self.__dict__["embed_tokens"] = embed_tokens
        self.__dict__["lm_head"] = lm_head

    def bind_buffers(self, *, state, candidates_out):
        self.state = state
        self.candidates_out = candidates_out

    def bind_target_cache(self, target):
        if getattr(target.config, "model_type", None) != "glm5_next_text":
            raise ValueError("GLM-5.3 DFly2 requires a GLM-5.3-Flash target")
        scales = []
        for i in self.dfly_config.target_kv_layers:
            attention = target.layers[i].self_attn
            indexer = getattr(attention, "indexer", None)
            if (
                getattr(indexer, "topk_output_width", None)
                != self.dfly_config.target_index_topk
            ):
                raise ValueError(
                    f"DFly2 target layer {i} needs a matching KPool indexer"
                )
            scales.append(float(attention.attn_mqa.k_scale_float or 1.0))
        self.kv_scales = tuple(scales)

    def _select_token(self, hidden):
        head = self.lm_head
        shard = getattr(head, "shard_indices", None)
        if shard is None:
            return F.linear(hidden, head.weight).argmax(dim=-1)
        weight = head.weight[: shard.num_org_elements]
        if weight.shape[0]:
            scores, ids = F.linear(hidden, weight).float().max(dim=-1)
            ids = ids.to(torch.int32) + shard.org_vocab_start_index
        else:
            scores = hidden.new_full(
                (hidden.shape[0],), -torch.inf, dtype=torch.float32
            )
            ids = torch.zeros_like(scores, dtype=torch.int32)
        group = get_parallel().tp_group
        if group.world_size == 1:
            return ids.to(torch.int64)
        # Keep the score's bits and token ID in a single gather.
        pair = torch.stack((scores.view(torch.int32), ids), dim=-1)
        gathered = pair.new_empty((group.world_size * len(pair), 2))
        group.all_gather_into_tensor(gathered, pair.contiguous())
        gathered = gathered.view(group.world_size, len(pair), 2)
        best = gathered[..., 0].contiguous().view(torch.float32).argmax(dim=0)
        return gathered[best, torch.arange(len(pair), device=best.device), 1].long()

    def draft_into_verify(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        width = self.dfly_config.block_size
        ids = input_ids.view(-1, width)
        request_ids = forward_batch.req_pool_indices[: ids.shape[0]]
        embeds = self.embed_tokens(ids.flatten()).view(*ids.shape, -1)
        collapsed = self.mtp_fusion(embeds, self.state.hidden[request_ids])
        hidden = collapsed.unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1)
        pool = get_token_to_kv_pool()
        for i, layer in enumerate(self.layers):
            kv = pool.get_key_buffer(self.dfly_config.target_kv_layers[i])
            if (
                kv.ndim != 3
                or kv.shape[1:] != (1, self.config.kv_lora_rank)
                or kv.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            ):
                raise ValueError("GLM-5.3 DFly2 requires plain NoPE MLA latents")
            history, valid = gather_dfly_kv(
                kv.squeeze(1),
                request_ids,
                self.state.topk[i],
                dtype=torch.float32,
                scale=self.kv_scales[i],
            )
            hidden, collapsed = layer(hidden, collapsed, history, valid, forward_batch)
            if i == self.dfly_config.correction_layer % len(self.layers):
                correction_history, correction_valid = history, valid
        hidden = self.final_norm(collapsed)
        previous_embed = embeds[:, 0]
        block = history.new_empty((ids.shape[0], width, self.config.kv_lora_rank))
        for slot in range(width):
            corrected = self.hidden_correction(
                hidden[:, slot],
                previous_embed,
                correction_history,
                correction_valid,
                block,
                slot,
            )
            if slot:
                token = self._select_token(corrected)
                ids[:, slot].copy_(token)
                if slot + 1 < width:
                    previous_embed = self.embed_tokens(token)
        self.candidates_out[: input_ids.numel()].copy_(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        fused_shared = self.layers[0].mlp.num_fused_shared_experts
        mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + fused_shared,
        )
        loaded = set()
        for name, value in weights:
            name = name.removeprefix("model.")
            if fused_shared and ".mlp.shared_experts." in name:
                name = name.replace(
                    ".mlp.shared_experts.",
                    f".mlp.experts.{self.config.n_routed_experts}.",
                )
            for target, source, expert, shard in mapping:
                if source in name:
                    name = name.replace(source, target)
                    param = params[name]
                    param.weight_loader(
                        param, value, name, shard_id=shard, expert_id=expert
                    )
                    break
            else:
                if name not in params and ".mlp.shared_experts." in name:
                    for shard, proj in enumerate(("gate_proj", "up_proj")):
                        if f".{proj}." in name:
                            name = name.replace(f".{proj}.", ".gate_up_proj.")
                            param = params[name]
                            param.weight_loader(param, value, shard)
                            break
                    else:
                        raise ValueError(
                            f"Unknown GLM-5.3 DFly2 shared weight {name!r}"
                        )
                else:
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(param, value)
            loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Missing GLM-5.3 DFly2 weights: {sorted(missing)}")


EntryClass = [DFlyGlm53DraftModel]

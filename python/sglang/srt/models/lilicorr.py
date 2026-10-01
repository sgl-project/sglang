from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.lilicorr import lilicorr_sample_path
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4A16LinearMethod
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.lilicorr_utils import (
    LiLiCorrConfig,
    parse_lilicorr_draft_config,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _apply_2d(layer: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
    # Quantized linear methods take [tokens, features] input.
    out, _ = layer(x.reshape(-1, x.shape[-1]))
    return out.view(*x.shape[:-1], -1)


class LiLiCorrMLP(nn.Module):
    def __init__(
        self, in_size: int, mid_size: int, out_size: int, quant_config, prefix: str
    ) -> None:
        super().__init__()
        self.up_proj = ReplicatedLinear(
            in_size,
            mid_size,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = ReplicatedLinear(
            mid_size,
            out_size,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_2d(self.down_proj, F.silu(_apply_2d(self.up_proj, x)))


class LiLiCorrLatticeAttention(nn.Module):
    # nn.MultiheadAttention's parameter layout, run via SDPA since that module is not capturable.

    def __init__(
        self, hidden_size: int, num_heads: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}."
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        self.out_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # attention_bias is [1, heads, L, L]; SDPA broadcasts it over the batch.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        out, _ = self.out_proj(out.transpose(1, 2).reshape(-1, self.hidden_size))
        return out.view(bsz, seq_len, self.hidden_size)


class LiLiCorrLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rms_norm_eps: float,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = LiLiCorrLatticeAttention(
            hidden_size, num_heads, quant_config, add_prefix("attn", prefix)
        )
        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = LiLiCorrMLP(
            hidden_size,
            int(hidden_size * mlp_ratio),
            hidden_size,
            quant_config,
            add_prefix("mlp", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), attention_bias
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class LiLiCorrHead(nn.Module):
    # [log_probs, probs, logprob_gap, rank_frac, is_top1], in the trained Linear's order.
    num_candidate_features = 5

    def __init__(
        self,
        *,
        model_hidden_size: int,
        block_size: int,
        rms_norm_eps: float,
        config: LiLiCorrConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.resolve_hidden_size(model_hidden_size=model_hidden_size)
        self.block_size = int(block_size)
        self.num_candidate_slots = self.block_size - 1
        self.num_active_slots = self.num_candidate_slots
        self.candidate_topk = int(config.candidate_topk)
        self.hidden_size = hidden_size
        self.num_heads = int(config.num_heads)
        self.mlp_ratio = float(config.mlp_ratio)
        self.factor_dim = int(config.factor_dim)
        self.vector_eps = float(config.vector_eps)
        self.logit_scale = float(config.logit_scale)

        self.token_proj = (
            None
            if model_hidden_size == hidden_size
            else ReplicatedLinear(
                model_hidden_size,
                hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("token_proj", prefix),
            )
        )
        self.pass_hidden_proj = ReplicatedLinear(
            model_hidden_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("pass_hidden_proj", prefix),
        )
        self.feature_norm = nn.LayerNorm(self.num_candidate_features)
        # Five input features fit no quantized GEMM kernel.
        self.feature_mlp = LiLiCorrMLP(
            self.num_candidate_features,
            hidden_size,
            hidden_size,
            None,
            add_prefix("feature_mlp", prefix),
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_candidate_slots, 1, hidden_size)
        )
        self.rank_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, self.candidate_topk, hidden_size)
        )
        self.relative_slot_bias = nn.Parameter(
            torch.zeros(self.num_heads, 2 * self.block_size - 1)
        )
        self.same_slot_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.context_proj = ReplicatedLinear(
            model_hidden_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("context_proj", prefix),
        )
        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    rms_norm_eps=rms_norm_eps,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(int(config.num_layers))
            ]
        )
        self.output_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Read as raw weights by materialize_inference_buffers, so never quantized.
        self.factor_input_proj = nn.Linear(hidden_size * 3, hidden_size)
        self.out_head = nn.Linear(hidden_size, self.factor_dim)
        self.in_head = nn.Linear(hidden_size, self.factor_dim)
        self.anchor_out_head = ReplicatedLinear(
            hidden_size,
            self.factor_dim,
            quant_config=quant_config,
            prefix=add_prefix("anchor_out_head", prefix),
        )

        self._attn_bias: Optional[torch.Tensor] = None
        self._fused_edge_weight: Optional[torch.Tensor] = None
        self._fused_edge_bias: Optional[torch.Tensor] = None
        self._factor_input_splits: Optional[Tuple[torch.Tensor, ...]] = None
        self._rank_frac_col: Optional[torch.Tensor] = None
        self._is_top1_col: Optional[torch.Tensor] = None

    @torch.no_grad()
    def materialize_inference_buffers(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        # After weight load and before graph capture: host work, not capturable.
        topk = self.candidate_topk
        self._attn_bias = self._build_attention_bias(device=device, dtype=dtype)

        self._fused_edge_weight = (
            torch.cat([self.out_head.weight, self.in_head.weight], dim=0)
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        self._fused_edge_bias = (
            torch.cat([self.out_head.bias, self.in_head.bias], dim=0)
            .to(device=device, dtype=dtype)
            .contiguous()
        )

        weight = self.factor_input_proj.weight
        hdim = self.hidden_size
        self._factor_input_splits = (
            weight[:, :hdim].contiguous(),
            weight[:, hdim : 2 * hdim].contiguous(),
            weight[:, 2 * hdim :].contiguous(),
        )

        if topk > 1:
            rank_frac = torch.arange(topk, device=device, dtype=torch.float32).view(
                1, 1, 1, topk
            ) / float(topk - 1)
        else:
            rank_frac = torch.zeros(1, 1, 1, topk, device=device, dtype=torch.float32)
        is_top1 = torch.zeros(1, 1, 1, topk, device=device, dtype=torch.float32)
        is_top1[..., 0] = 1.0
        self._rank_frac_col = rank_frac.contiguous()
        self._is_top1_col = is_top1.contiguous()

    def _build_attention_bias(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        topk = self.candidate_topk
        slot_ids = torch.arange(
            self.num_active_slots, device=device, dtype=torch.long
        ).repeat_interleave(topk)
        rel = slot_ids.view(-1, 1) - slot_ids.view(1, -1)
        rel = rel.clamp(min=-(self.block_size - 1), max=self.block_size - 1)
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(dtype=bias.dtype) * (
            self.same_slot_bias.view(-1, 1, 1)
        )
        return bias.to(device=device, dtype=dtype).contiguous()

    def set_active_block_size(self, block_size: int) -> None:
        # A prefix of the trained slots; every relative offset it needs was trained.
        if not 2 <= int(block_size) <= self.block_size:
            raise ValueError(
                f"block_size={int(block_size)} is outside [2, {self.block_size}], "
                "the trained LiLiCorr block."
            )
        self.num_active_slots = int(block_size) - 1
        if self._attn_bias is not None:
            self._attn_bias = self._build_attention_bias(
                device=self._attn_bias.device, dtype=self._attn_bias.dtype
            )

    def _require_materialized(self) -> None:
        if self._attn_bias is None:
            raise RuntimeError(
                "LiLiCorr head scored before materialize_inference_buffers(): its "
                "attention bias and fused edge heads are unbuilt."
            )

    def _project_anchor(
        self, anchor_hidden: torch.Tensor, anchor_valid: torch.Tensor
    ) -> torch.Tensor:
        # Branch-free so there is no host sync inside the captured region.
        anchor = _apply_2d(self.context_proj, anchor_hidden)
        return anchor * anchor_valid.unsqueeze(-1).to(anchor.dtype)

    def score(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
        already_projected: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._require_materialized()
        bsz, n_blocks, n_slots, topk = candidate_log_probs.shape
        if topk != self.candidate_topk:
            raise ValueError(
                f"LiLiCorr was built for candidate_topk={self.candidate_topk} but the "
                f"lattice carries {topk}; both are sized for the trained width."
            )

        proj_dtype = self.slot_embedding.dtype
        if token_embeddings.dtype != proj_dtype:
            token_embeddings = token_embeddings.to(proj_dtype)
        if pass_hidden.dtype != proj_dtype:
            pass_hidden = pass_hidden.to(proj_dtype)

        token_states = token_embeddings
        if not already_projected and self.token_proj is not None:
            token_states = _apply_2d(self.token_proj, token_embeddings)
        pass_states = _apply_2d(self.pass_hidden_proj, pass_hidden).unsqueeze(-2)

        log_probs = candidate_log_probs.float()
        features = torch.stack(
            [
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.max(dim=-1, keepdim=True).values,
                self._rank_frac_col.expand_as(log_probs),
                self._is_top1_col.expand_as(log_probs),
            ],
            dim=-1,
        )
        hidden_states = token_states + pass_states
        hidden_states = hidden_states + self.feature_mlp(
            self.feature_norm(features.to(dtype=token_states.dtype))
        )
        hidden_states = hidden_states + self.slot_embedding[:, :, :n_slots]
        hidden_states = hidden_states + self.rank_embedding
        hidden_states = hidden_states.reshape(
            bsz * n_blocks, n_slots * topk, self.hidden_size
        )

        anchor_state = self._project_anchor(anchor_hidden, anchor_valid)
        attention_bias = self._attn_bias.unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        hidden_states = self.output_norm(hidden_states).reshape(
            bsz, n_blocks, n_slots, topk, self.hidden_size
        )
        anchor_state = self.anchor_norm(anchor_state)

        w_self, w_anchor, w_cross = self._factor_input_splits
        anchor_row = anchor_state[:, :, None, None, :]
        pre = F.linear(hidden_states, w_self, self.factor_input_proj.bias)
        pre = pre + F.linear(anchor_row, w_anchor)
        pre = pre + F.linear(hidden_states * anchor_row, w_cross)
        factor_hidden = F.silu(pre)

        edges = F.linear(factor_hidden, self._fused_edge_weight, self._fused_edge_bias)
        out_vec, in_vec = F.normalize(
            edges.unflatten(-1, (2, self.factor_dim)),
            dim=-1,
            eps=self.vector_eps,
        ).unbind(-2)
        anchor_out = F.normalize(
            _apply_2d(self.anchor_out_head, anchor_state), dim=-1, eps=self.vector_eps
        )

        start_scores = (anchor_out[:, :, None, :] * in_vec[:, :, 0, :, :]).sum(dim=-1)
        pair_scores = torch.matmul(
            out_vec[:, :, :-1], in_vec[:, :, 1:].transpose(-1, -2)
        )
        return start_scores, pair_scores

    def log_factors(
        self, start_scores: torch.Tensor, pair_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.logit_scale * start_scores.float(),
            self.logit_scale * pair_scores.float(),
        )

    def select_with_proposal(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        already_projected: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_scores, pair_scores = self.score(
            token_embeddings=token_embeddings.unsqueeze(1),
            candidate_log_probs=candidate_log_probs.unsqueeze(1),
            pass_hidden=pass_hidden.unsqueeze(1),
            anchor_hidden=anchor_hidden.unsqueeze(1),
            anchor_valid=anchor_valid.unsqueeze(1),
            already_projected=already_projected,
        )
        log_start, log_pair = self.log_factors(start_scores, pair_scores)
        return lilicorr_sample_path(
            log_start[:, 0, :],
            log_pair[:, 0],
            candidate_tokens,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )

    @torch.no_grad()
    def build_token_table(self, embed_tokens: nn.Module) -> Optional[torch.Tensor]:
        if self.token_proj is None:
            return None
        weight = embed_tokens.weight
        if self.token_proj.input_size != int(weight.shape[1]):
            return None
        table, _ = self.token_proj(weight.to(self.slot_embedding.dtype))
        return table.contiguous()


def check_head_weight_coverage(head: LiLiCorrHead, seen: set) -> None:
    # The base loader silently drops unresolved weights, which would only show
    # up as a lower acceptance length.
    expected = {f"lilicorr.{name}" for name, _ in head.named_parameters()}
    # NVFP4 W4A16 registers an input_scale it never reads; checkpoints omit it.
    optional = {
        f"lilicorr.{name}.input_scale"
        for name, module in head.named_modules()
        if isinstance(
            getattr(module, "quant_method", None), ModelOptNvFp4A16LinearMethod
        )
    }
    missing = sorted(expected - optional - seen)
    if missing:
        raise ValueError(
            f"LiLiCorr checkpoint is missing {len(missing)} head parameters "
            f"(e.g. {missing[:5]}); a draft with no LiLiCorr head should declare "
            'architectures=["DFlashDraftModel"].'
        )
    unexpected = sorted(seen - expected)
    if unexpected:
        raise ValueError(
            f"LiLiCorr checkpoint carries {len(unexpected)} head tensors this head "
            f"has no parameter for (e.g. {unexpected[:5]}). Check "
            "lilicorr_hidden_size, lilicorr_num_layers and lilicorr_factor_dim in "
            "dflash_config against the checkpoint."
        )


def check_conv_weight_coverage(model: DFlashDraftModel, seen: set) -> None:
    # conv_kernel_size / conv_group_size default to 0, so a config that lost them
    # builds no conv modules and the loader drops every conv tensor silently.
    expected = {
        name
        for name, _ in model.named_parameters()
        if ".attention_conv." in name or ".mlp_conv." in name
    }

    if seen and not expected:
        raise ValueError(
            f"Draft checkpoint carries {len(seen)} grouped-convolution tensors "
            f"(e.g. {sorted(seen)[:3]}) but this draft built no convolution modules: "
            "dflash_config is missing conv_kernel_size / conv_group_size, which both "
            "default to 0 and cannot be inferred from the tensors."
        )
    if expected and not seen:
        raise ValueError(
            f"This draft built {len(expected)} grouped-convolution parameters from "
            "dflash_config, but the checkpoint carries none, so kernel_projection "
            "would serve at its random initialization."
        )

    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise ValueError(
            "Draft checkpoint's grouped-convolution tensors do not correspond to the "
            f"built ones: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}). Check "
            "conv_kernel_size, conv_group_size and num_hidden_layers in dflash_config."
        )

    if expected:
        conv = model.layers[0].attention_conv
        logger.info(
            "DFLASH grouped convolution live: %d taps, group size %d, %d tensors.",
            int(conv.taps),
            int(conv.group_size),
            len(expected),
        )


class LiLiCorrDraftModel(DFlashDraftModel):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.lilicorr = LiLiCorrHead(
            model_hidden_size=int(config.hidden_size),
            block_size=int(self.block_size),
            rms_norm_eps=self.rms_norm_eps,
            config=parse_lilicorr_draft_config(draft_hf_config=config),
            quant_config=quant_config,
            prefix=add_prefix("lilicorr", prefix),
        )

    def set_block_size(self, block_size: int) -> None:
        super().set_block_size(block_size)
        self.lilicorr.set_active_block_size(block_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        seen: set[str] = set()
        seen_conv: set[str] = set()

        def tracking():
            for name, weight in weights:
                stripped = name[len("model.") :] if name.startswith("model.") else name
                if stripped.startswith("lilicorr."):
                    # Older exports index the head MLPs as nn.Sequential.
                    for old, new in (
                        ("feature_mlp.0.", "feature_norm."),
                        ("feature_mlp.1.", "feature_mlp.up_proj."),
                        ("feature_mlp.3.", "feature_mlp.down_proj."),
                        (".mlp.0.", ".mlp.up_proj."),
                        (".mlp.2.", ".mlp.down_proj."),
                    ):
                        stripped = stripped.replace(old, new)
                    name = stripped
                    seen.add(stripped)
                elif ".attention_conv." in stripped or ".mlp_conv." in stripped:
                    seen_conv.add(stripped)
                yield name, weight

        super().load_weights(tracking())

        check_head_weight_coverage(self.lilicorr, seen)
        check_conv_weight_coverage(self, seen_conv)

        parameter = self.lilicorr.slot_embedding
        self.lilicorr.materialize_inference_buffers(parameter.device, parameter.dtype)


EntryClass = [LiLiCorrDraftModel]

# LiLiCorr: lightweight likelihood correlation of parallel drafts.
# Paper: https://arxiv.org/abs/2608.20530

"""DFLASH backbone plus the LiLiCorr candidate-lattice reranker.

DFLASH is trained on per-position marginals rather than on the joint block
distribution, so its drafted tokens are individually plausible yet jointly
incoherent. LiLiCorr keeps the top-``k`` candidates per block position and
processes the whole ``slots x k`` lattice jointly, emitting an ``in`` and an
``out`` vector per candidate; adjacent candidates match when the earlier one's
``out`` has high cosine similarity with the later one's ``in``. One network pass
produces every vector, the pairwise scores are a batched matmul, and only the
greedy left-to-right walk stays sequential.

The head is built here rather than by the worker, so the checkpoint's
``lilicorr.*`` tensors load through the inherited ``load_weights`` like any other
parameter subtree. It is selected by the checkpoint declaring
``architectures=["LiLiCorrDraftModel"]``, the same way ``DFlash2DraftModel``
selects the candidate selector; the serving algorithm stays DFLASH.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.lilicorr import (
    lilicorr_greedy_path,
    lilicorr_sample_path,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.lilicorr_components.lilicorr_config import (
    LiLiCorrConfig,
    parse_lilicorr_draft_config,
)

logger = logging.getLogger(__name__)


class LiLiCorrRMSNorm(nn.Module):
    # Deliberately not sglang.srt.layers.layernorm.RMSNorm: a custom-op norm is
    # opaque to inductor and splits the compiled decode body at every norm, which
    # is most of the pointwise fusion the compile exists to buy. Same math, same
    # weight key.

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)
        self._normalized_shape = (int(hidden_size),)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states,
            self._normalized_shape,
            self.weight,
            self.variance_epsilon,
        )


class LiLiCorrLatticeAttention(nn.Module):
    # Carries nn.MultiheadAttention's exported parameter layout so the trained
    # attn.* tensors load unchanged, but runs SDPA directly: the module's Python
    # control flow and need_weights bookkeeping are what make it uncapturable.

    def __init__(self, hidden_size: int, num_heads: int) -> None:
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
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        # attention_bias arrives [batch * heads, L, L], the eager module's layout.
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias.reshape(bsz, self.num_heads, seq_len, seq_len),
        )
        return self.out_proj(
            out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        )


class LiLiCorrLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = LiLiCorrLatticeAttention(hidden_size, num_heads)
        self.mlp_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, hidden_size),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), attention_bias
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class LiLiCorrHead(nn.Module):
    """Anchor-conditioned chain factors over the per-slot top-k candidates.

    Parameter names match the trained head one for one, and every geometry and
    scaling argument is required: the checkpoint records each one, and a
    defaulted value here would describe a different function of the same weights.
    """

    # Per-candidate features, in the order the trained first Linear expects:
    # [log_probs, probs, logprob_gap, rank_frac, is_top1]. The DFLASH log-probs
    # enter the score only here.
    num_candidate_features = 5

    def __init__(
        self,
        *,
        model_hidden_size: int,
        block_size: int,
        rms_norm_eps: float,
        config: LiLiCorrConfig,
    ) -> None:
        super().__init__()
        hidden_size = config.resolve_hidden_size(model_hidden_size=model_hidden_size)
        self.block_size = int(block_size)
        self.num_candidate_slots = self.block_size - 1
        self.candidate_topk = int(config.candidate_topk)
        self.hidden_size = hidden_size
        self.num_heads = int(config.num_heads)
        self.mlp_ratio = float(config.mlp_ratio)
        self.factor_dim = int(config.factor_dim)
        self.vector_eps = float(config.vector_eps)
        self.logit_scale = float(config.logit_scale)

        # Identity when the head is as wide as the draft, so no redundant matmul.
        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(self.num_candidate_features),
            nn.Linear(self.num_candidate_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
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
        # The anchor is a row of the target hidden state, so it arrives
        # model_hidden_size wide.
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(int(config.num_layers))
            ]
        )
        self.output_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = LiLiCorrRMSNorm(hidden_size, eps=rms_norm_eps)
        # The factor heads read [self, anchor, self*anchor] (3*h).
        self.factor_input_proj = nn.Linear(hidden_size * 3, hidden_size)
        # Named by edge direction: a transition runs out of the previous token and
        # in to the next (pair = out_vec[s] . in_vec[s+1]).
        self.out_head = nn.Linear(hidden_size, self.factor_dim)
        self.in_head = nn.Linear(hidden_size, self.factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, self.factor_dim)

        # Built by materialize_inference_buffers after weight load.
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
        """Precompute every parameter-derived, geometry-fixed buffer once.

        Must run after weight load and before CUDA-graph capture: it does host
        work and host-to-device copies that are not capturable.
        """
        topk = self.candidate_topk
        self._attn_bias = self._build_attention_bias(device=device, dtype=dtype)

        # Row-concatenating the two edge heads is parity-safe: each output row is
        # the same dot product as the split head, and they share their input.
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

        # W . cat([h, a, h*a]) == W1.h + W2.a + W3.(h*a), and the anchor is one row
        # per request, so splitting the projection materializes neither the
        # concatenation nor the anchor's expansion over slots.
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
        # Per-head lattice bias core [num_heads, S, S], S = slots * topk. A pure
        # function of the trained bias parameters and the fixed geometry.
        topk = self.candidate_topk
        slot_ids = torch.arange(
            self.num_candidate_slots, device=device, dtype=torch.long
        ).repeat_interleave(topk)
        rel = slot_ids.view(-1, 1) - slot_ids.view(1, -1)
        rel = rel.clamp(min=-(self.block_size - 1), max=self.block_size - 1)
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(dtype=bias.dtype) * (
            self.same_slot_bias.view(-1, 1, 1)
        )
        return bias.to(device=device, dtype=dtype).contiguous()

    def _require_materialized(self) -> None:
        if self._attn_bias is None:
            raise RuntimeError(
                "LiLiCorr head scored before materialize_inference_buffers(). Its "
                "cached attention bias and fused edge heads are unbuilt, so the "
                "scores would be meaningless rather than wrong-looking."
            )

    def _project_anchor(
        self, anchor_hidden: torch.Tensor, anchor_valid: torch.Tensor
    ) -> torch.Tensor:
        # Branch-free, so there is no host sync inside the captured region: an
        # invalid anchor multiplies to zero rather than selecting another path.
        anchor = self.context_proj(anchor_hidden)
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
        """Score the lattice.

        Shapes: ``token_embeddings [bsz, n_blocks, slots, topk, *]``,
        ``candidate_log_probs [bsz, n_blocks, slots, topk]``, ``pass_hidden [bsz,
        n_blocks, slots, model_hidden]``, ``anchor_hidden [bsz, n_blocks,
        model_hidden]``, ``anchor_valid [bsz, n_blocks]``. Returns
        ``(start_scores [bsz, n_blocks, topk], pair_scores [bsz, n_blocks,
        slots-1, topk, topk])``.

        ``already_projected`` says the caller gathered rows of a precomputed
        ``embed_tokens.weight @ token_proj.weight.T + bias`` table, so
        ``token_proj`` must not be applied twice. It is an argument rather than
        head state because the two call sites disagree, and a sticky flag would
        silently mis-score the eager one.
        """
        self._require_materialized()
        bsz, n_blocks, n_slots, topk = candidate_log_probs.shape
        if topk != self.candidate_topk:
            raise ValueError(
                f"LiLiCorr was built for candidate_topk={self.candidate_topk} but "
                f"the lattice carries {topk}. The rank embedding and the cached "
                "attention bias are both sized for the trained width."
            )

        # A no-op when everything is bf16; the candidate embeddings come from the
        # target's table, which need not share the head's dtype.
        proj_dtype = self.pass_hidden_proj.weight.dtype
        if token_embeddings.dtype != proj_dtype:
            token_embeddings = token_embeddings.to(proj_dtype)
        if pass_hidden.dtype != proj_dtype:
            pass_hidden = pass_hidden.to(proj_dtype)

        token_states = (
            token_embeddings if already_projected else self.token_proj(token_embeddings)
        )
        pass_states = self.pass_hidden_proj(pass_hidden).unsqueeze(-2)

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
            features.to(dtype=token_states.dtype)
        )
        hidden_states = hidden_states + self.slot_embedding
        hidden_states = hidden_states + self.rank_embedding
        hidden_states = hidden_states.reshape(
            bsz * n_blocks, n_slots * topk, self.hidden_size
        )

        anchor_state = self._project_anchor(anchor_hidden, anchor_valid)
        # Materialized rather than handed to SDPA as a batch-broadcast view: the
        # broadcast form is identical math and saves one ~2us copy per block, but
        # a stride-0 mask measured -1.75pp stacked on the compiled body, and the
        # shipped path is always compiled. Measure before changing it back.
        lattice = self._attn_bias.shape[-1]
        attention_bias = (
            self._attn_bias.unsqueeze(0)
            .expand(bsz * n_blocks, -1, -1, -1)
            .reshape(bsz * n_blocks * self.num_heads, lattice, lattice)
        )
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

        # One GEMM over [out | in], then one normalize over the [.., 2, factor_dim]
        # view, which is per-vector because each factor_dim vector is independent.
        edges = F.linear(factor_hidden, self._fused_edge_weight, self._fused_edge_bias)
        out_vec, in_vec = F.normalize(
            edges.unflatten(-1, (2, self.factor_dim)),
            dim=-1,
            eps=self.vector_eps,
        ).unbind(-2)
        anchor_out = F.normalize(
            self.anchor_out_head(anchor_state), dim=-1, eps=self.vector_eps
        )

        start_scores = (anchor_out[:, :, None, :] * in_vec[:, :, 0, :, :]).sum(dim=-1)
        # One batched matmul instead of an elementwise product summed over the
        # factor dim, so the [.., K, K, factor_dim] intermediate never exists.
        pair_scores = torch.matmul(
            out_vec[:, :, :-1], in_vec[:, :, 1:].transpose(-1, -2)
        )
        return start_scores, pair_scores

    def log_factors(
        self, start_scores: torch.Tensor, pair_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # logit_scale is a fixed, non-learnable constant that sets how sharp the
        # chain may become, since a cosine in [-1, 1] is too flat to commit on.
        # fp32 because the decode's argmax runs on these values.
        return (
            self.logit_scale * start_scores.float(),
            self.logit_scale * pair_scores.float(),
        )

    def select(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
        already_projected: bool = False,
    ) -> torch.Tensor:
        """Single-block best-path selection from a precomputed lattice.

        ``candidate_*`` are ``[bs, slots, topk]``, ``token_embeddings`` is ``[bs,
        slots, topk, *]``, ``pass_hidden`` is ``[bs, slots, model_hidden]``,
        ``anchor_hidden`` is ``[bs, feat]`` and ``anchor_valid`` is ``[bs]``.
        Returns the selected tokens ``[bs, slots]``.

        This is the function the worker compiles: static shapes, no collectives,
        no host syncs and a fixed decode trip count.
        """
        start_scores, pair_scores = self.score(
            token_embeddings=token_embeddings.unsqueeze(1),
            candidate_log_probs=candidate_log_probs.unsqueeze(1),
            pass_hidden=pass_hidden.unsqueeze(1),
            anchor_hidden=anchor_hidden.unsqueeze(1),
            anchor_valid=anchor_valid.unsqueeze(1),
            already_projected=already_projected,
        )
        log_start, log_pair = self.log_factors(start_scores, pair_scores)
        return lilicorr_greedy_path(
            log_start[:, 0, :], log_pair[:, 0], candidate_tokens
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
        """``select``, sampling the commit and returning the proposal it used.

        Same arguments plus the per-row sampling state, and returns ``(tokens [bs,
        slots], q_rows [bs, slots, topk])``. ``uniforms`` is ``[bs, slots]``, one draw
        per slot, and ``temperatures`` / ``greedy_mask`` are ``[bs]``.

        This deliberately duplicates ``select``'s four-line prologue instead of
        sharing it.** ``select`` is the shipped path and this method is an overlay, so
        leaving that function and its tests byte-untouched is worth ten repeated lines:
        with ``LILICORR_SAMPLING`` off, the served code is not merely equivalent to the
        greedy path, it is the same function. Exactly one of the two is ever compiled in
        a process, since the mode is a module constant resolved at import.
        """
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
        """Precompute ``embed_tokens.weight @ token_proj.weight.T + bias``.

        ``token_proj`` is affine and its input is a row of the target embedding
        table, so the product is a pure function of the token id. Callers must
        pass ``already_projected`` for rows gathered from the result. None means
        there is nothing to fold, rather than a guess: a silently wrong table
        would score plausibly and wrong.
        """
        if isinstance(self.token_proj, nn.Identity):
            return None
        weight = embed_tokens.weight
        if int(self.token_proj.weight.shape[1]) != int(weight.shape[1]):
            return None
        return F.linear(
            weight.to(self.token_proj.weight.dtype),
            self.token_proj.weight,
            self.token_proj.bias,
        ).contiguous()


def check_head_weight_coverage(head: LiLiCorrHead, seen: set) -> None:
    """Require the checkpoint's ``lilicorr.*`` tensors and the built head to correspond.

    The base loader silently ignores weights it cannot resolve, which is correct
    for HF rotary caches and wrong here in both directions: a missing name leaves
    that parameter randomly initialized, and a surplus one means the served head
    is a different architecture than the trained one. Either reads as a low but
    believable acceptance length.
    """
    expected = {f"lilicorr.{name}" for name, _ in head.named_parameters()}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(
            f"LiLiCorr checkpoint is missing {len(missing)} head parameters "
            f"(e.g. {missing[:5]}). Refusing to serve a partially initialized "
            "head. A checkpoint without a LiLiCorr head should declare "
            'architectures=["DFlashDraftModel"].'
        )
    unexpected = sorted(seen - expected)
    if unexpected:
        raise ValueError(
            f"LiLiCorr checkpoint carries {len(unexpected)} head tensors this head "
            f"has no parameter for (e.g. {unexpected[:5]}). They would be dropped "
            "in silence, so the served head would not be the trained one. Check "
            "lilicorr_hidden_size, lilicorr_num_layers and lilicorr_factor_dim in "
            "dflash_config against the checkpoint."
        )


def check_conv_weight_coverage(model: DFlashDraftModel, seen: set) -> None:
    """Require the checkpoint's backbone conv tensors and the built backbone to agree.

    Separate from the head check because these tensors are backbone parameters
    named ``layers.*.{attention,mlp}_conv.*``, none of them under ``lilicorr.``.
    ``parse_dflash_draft_config`` defaults ``conv_kernel_size`` and
    ``conv_group_size`` to 0, so a config that lost them builds no conv modules
    and the loader drops every conv tensor without a word.

    ``seen`` is the set of conv names the checkpoint offered, stripped of any
    ``model.`` prefix. Both directions raise.
    """
    expected = {
        name
        for name, _ in model.named_parameters()
        if ".attention_conv." in name or ".mlp_conv." in name
    }

    if seen and not expected:
        raise ValueError(
            f"Draft checkpoint carries {len(seen)} grouped-convolution tensors "
            f"(e.g. {sorted(seen)[:3]}) but this draft built no convolution modules, "
            "so every one of them would be dropped in silence and the draft would "
            "serve as its conv-free parent at a believable but wrong acceptance "
            "length. dflash_config is missing conv_kernel_size / conv_group_size: "
            "both default to 0, and the loader cannot infer them from the tensors. "
            "Re-export with the geometry, or add both keys to config.json."
        )
    if expected and not seen:
        raise ValueError(
            f"This draft built {len(expected)} grouped-convolution parameters from "
            "dflash_config, but the checkpoint carries none, so kernel_projection "
            "would serve at its random initialization. Either the config declares a "
            "convolution the trained draft does not have, or the checkpoint is the "
            "wrong one."
        )

    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise ValueError(
            "Draft checkpoint's grouped-convolution tensors do not correspond to the "
            f"built ones: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}). Check "
            "conv_kernel_size, conv_group_size and num_hidden_layers in dflash_config "
            "against the checkpoint."
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
    """DFlash backbone plus the LiLiCorr reranker. Reuses the DFLASH worker."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.lilicorr = LiLiCorrHead(
            model_hidden_size=int(config.hidden_size),
            block_size=int(self.block_size),
            rms_norm_eps=self.rms_norm_eps,
            config=parse_lilicorr_draft_config(draft_hf_config=config),
        )

    def set_block_size(self, block_size: int) -> None:
        super().set_block_size(block_size)
        if int(block_size) != int(self.lilicorr.block_size):
            raise ValueError(
                "LiLiCorr cannot follow a block size the head was not built for: "
                f"the worker resolved block_size={int(block_size)} but the head's "
                f"relative-slot bias and slot embedding are sized for "
                f"{int(self.lilicorr.block_size)}. Drop "
                "--speculative-num-draft-tokens, or serve a head trained at that "
                "block size."
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        seen: set[str] = set()
        seen_conv: set[str] = set()

        def tracking():
            for name, weight in weights:
                stripped = name[len("model.") :] if name.startswith("model.") else name
                if stripped.startswith("lilicorr."):
                    seen.add(stripped)
                elif ".attention_conv." in stripped or ".mlp_conv." in stripped:
                    seen_conv.add(stripped)
                yield name, weight

        super().load_weights(tracking())

        check_head_weight_coverage(self.lilicorr, seen)
        check_conv_weight_coverage(self, seen_conv)

        parameter = next(self.lilicorr.parameters())
        self.lilicorr.materialize_inference_buffers(parameter.device, parameter.dtype)


EntryClass = [LiLiCorrDraftModel]

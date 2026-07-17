from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import can_dflash_slice_qkv_weight
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module
) -> torch.Tensor:
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def run_markov_block(
    head: nn.Module,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits

    sampled_tokens = []
    corrected_logits = []
    prev_tokens = first_prev_tokens.long()
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_logits = head.apply_step_logits(
            base_logits[:, step_idx, :],
            token_ids=prev_tokens,
            hidden_states=step_hidden,
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        torch.cat(corrected_logits, dim=1),
    )


class VanillaMarkov(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


class GatedMarkovHead(VanillaMarkov):

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(int(hidden_size) + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(
            dtype=prev_embeddings.dtype
        )
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + self.hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_tokens)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx, :])
            step_logits = base_logits[:, step_idx, :] + bias
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank <= 0:
        raise ValueError(
            "DSpark requires markov_rank > 0 (the Markov head is the core of the "
            f"semi-AR draft); got markov_rank={markov_rank}."
        )
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    raise ValueError(f"Unsupported DSpark markov_head_type={markov_head_type!r}.")


class DSparkConfidenceHead(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=bias, dtype=dtype)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )
        self._last_confidence_raw: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError(
                    "DSparkConfidenceHead(with_markov=True) requires markov_embed_stack."
                )
            features = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)],
                dim=-1,
            )
        else:
            features = hidden_states
        features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        self._last_confidence_raw = confidence_raw
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


def build_confidence_head(config) -> Optional[nn.Module]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    hidden_size = int(config.hidden_size)
    markov_rank = int(getattr(config, "markov_rank", 0))
    with_markov = bool(getattr(config, "confidence_head_with_markov", markov_rank > 0))
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        with_markov=with_markov,
    )


_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)


class DSparkDraftMixin:

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(dspark_config.resolve_gamma(default=self.block_size))
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.lm_head: Optional[nn.Module] = None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeds with the shared target embedding INSIDE the draft graph
        # (the runner skips the eager input_embeds staging when the draft
        # model exposes forward_embed).
        return self.embed_tokens(input_ids)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        base_logits = gather_and_crop_vocab(local_logits, self.lm_head)
        return base_logits, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in _DSPARK_SKIPPED_WEIGHT_PREFIXES):
                continue
            if name.startswith("confidence_head."):
                if self.confidence_head is None:
                    continue
                confidence_weights.append((name, loaded_weight))
            elif name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        super().load_weights(backbone_weights)

        for name, loaded_weight in markov_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected markov weight {name!r} not found in model "
                    f"parameters (known markov params require a {type(self.markov_head).__name__} head)."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )

    def _load_confidence_weights(
        self,
        *,
        confidence_weights: list,
        params_dict: dict,
    ) -> None:
        if self.confidence_head is None:
            return
        loaded_names = set()
        for name, loaded_weight in confidence_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected confidence weight {name!r} not found in "
                    "model parameters."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_names
        if missing:
            raise ValueError(
                f"DSpark confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        """Stack every layer's KV projection into one weight so the per-step
        target-hidden KV write runs one GEMM instead of one per layer -- the
        input hidden is identical for every layer, so concatenating the output
        columns is exact. Cached after the first call; None (per-layer loop
        fallback) when any layer's QKV weight cannot be sliced (quantized) or
        the layers disagree on norm epsilon / bias presence.
        """
        cached = getattr(self, "_stacked_ctx_kv_cache", False)
        if cached is not False:
            return cached
        weights, biases, k_norm_weights = [], [], []
        eps = None
        for layer in self.layers:
            attn = layer.self_attn
            can_slice, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not can_slice or eps not in (None, attn.k_norm.variance_epsilon):
                self._stacked_ctx_kv_cache = None
                return None
            eps = attn.k_norm.variance_epsilon
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            weights.append(attn.qkv_proj.weight[kv_slice])
            biases.append(
                attn.qkv_proj.bias[kv_slice] if attn.qkv_proj.bias is not None else None
            )
            k_norm_weights.append(attn.k_norm.weight)
        has_bias = [b is not None for b in biases]
        if any(has_bias) and not all(has_bias):
            self._stacked_ctx_kv_cache = None
            return None
        self._stacked_ctx_kv_cache = {
            "weight": torch.cat(weights, dim=0),
            "bias": torch.cat(biases, dim=0) if all(has_bias) else None,
            "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
            "eps": eps,
        }
        return self._stacked_ctx_kv_cache

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        ctx_hidden = self.project_target_hidden(target_hidden)
        stacked = self._stacked_ctx_kv_params()
        if stacked is not None:
            k_all, v_all = self._project_ctx_kv_stacked(
                ctx_hidden=ctx_hidden, positions=positions, stacked=stacked
            )
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            if stacked is not None:
                k = k_all[i]
                v = v_all[i]
            else:
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn.attn,
                    cache_loc_2d,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn.attn,
                    cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def _project_ctx_kv_stacked(
        self,
        *,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        stacked: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn0 = self.layers[0].self_attn
        num_layers = len(self.layers)
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_kv_heads = attn0.num_kv_heads
        tokens = ctx_hidden.shape[0]

        kv_all = F.linear(ctx_hidden, stacked["weight"], stacked["bias"])
        kv_all = kv_all.view(tokens, num_layers, 2, kv_size)
        # Batched per-head k-norm across layers (mirrors RMSNorm.forward_native:
        # fp32 variance, weight multiply in fp32, cast back).
        k32 = (
            kv_all[:, :, 0, :]
            .reshape(tokens, num_layers, num_kv_heads, head_dim)
            .to(torch.float32)
        )
        variance = k32.pow(2).mean(dim=-1, keepdim=True)
        k32 = k32 * torch.rsqrt(variance + stacked["eps"])
        k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
        k_all = k32.to(ctx_hidden.dtype)
        # One RoPE over all layers' heads: every layer shares the rotary params
        # and positions, so the N*kv_heads heads rotate identically.
        k_flat = k_all.reshape(tokens, num_layers * kv_size)
        dummy_q = k_flat.new_empty(k_flat.shape)
        _, k_flat = attn0.rotary_emb(positions, dummy_q, k_flat)
        # [layers, tokens, heads, dim] so per-layer slices are contiguous views.
        k_all = (
            k_flat.view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        v_all = (
            kv_all[:, :, 1, :]
            .view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        return k_all, v_all


class DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel):

    pass


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [DSparkDraftModel, Qwen3DSparkModel]

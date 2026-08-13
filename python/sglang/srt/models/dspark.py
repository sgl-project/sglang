from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.environ import envs
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
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
from sglang.srt.utils import add_prefix
from torch import nn

logger = logging.getLogger(__name__)

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module
) -> torch.Tensor:
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def build_independent_lm_head(
    *, draft_vocab_size: int, hidden_size: int, quant_config, prefix: str
) -> ParallelLMHead:
    """The reduced, checkpoint-owned draft head of a speculators DSpark model.

    A plain ParallelLMHead so the existing vocab-parallel weight loader shards
    and pads it exactly like a target head; org_vocab_size == draft_vocab_size,
    so compute_base_logits' gather_and_crop_vocab crops the all-gathered logits
    back to draft space.
    """
    return ParallelLMHead(
        draft_vocab_size,
        hidden_size,
        quant_config=quant_config,
        prefix=prefix,
    )


def scatter_draft_logits_to_target(
    draft_logits: torch.Tensor,
    *,
    draft_to_target: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Place each draft column j at target column draft_to_target[j], leaving
    unmapped target columns at -inf (zero softmax mass). ``out`` is a
    caller-owned target-width buffer whose leading dims match ``draft_logits``.
    """
    out.fill_(float("-inf"))
    out.index_copy_(out.dim() - 1, draft_to_target, draft_logits)
    return out


def _is_dspark_d2t_weight(name: str) -> bool:
    return name == "d2t" or name.endswith(".d2t")


def _is_dspark_t2d_weight(name: str) -> bool:
    return name == "t2d" or name.endswith(".t2d")


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
        sampled = sampler(step_logits, step_idx)
        # Reduced-vocab checkpoints sample a draft-space id; map it to a target
        # id before storing it and before it conditions the next step's
        # markov_w1 (whose input vocabulary is the target vocabulary). Identity
        # for full-vocab heads, so corrected_logits stays draft-space here.
        next_tokens = head.map_sampled_to_target(sampled)
        sampled_tokens.append(next_tokens)
        corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        torch.cat(corrected_logits, dim=1),
    )


class VanillaMarkov(nn.Module):
    markov_head_type = "vanilla"

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        draft_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        # markov_w1 consumes the previously sampled TARGET token id, so its
        # input row count is always the target vocab. markov_w2 emits the draft
        # distribution: the reduced draft vocab when the checkpoint ships an
        # independent head + d2t map, else the full (target) vocab.
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.draft_vocab_size = (
            int(draft_vocab_size) if draft_vocab_size is not None else self.vocab_size
        )
        self.reduced_vocab = self.draft_vocab_size != self.vocab_size
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.draft_vocab_size, bias=False)
        # Absolute draft->target id map (target = draft + d2t[draft]); filled
        # from the checkpoint's d2t table in load_weights. None (identity map)
        # for full-vocab heads. Non-persistent: reconstructed from d2t on load,
        # never read back from a saved state dict.
        if self.reduced_vocab:
            self.register_buffer(
                "draft_to_target",
                torch.arange(self.draft_vocab_size, dtype=torch.long),
                persistent=False,
            )
        else:
            self.draft_to_target = None
        self._draft_to_target_loaded = False

    def map_sampled_to_target(self, sampled_tokens: torch.Tensor) -> torch.Tensor:
        """Map draft-space sampled ids to target-space ids; identity when the
        head shares the full target vocab."""
        if self.draft_to_target is None:
            return sampled_tokens
        return self.draft_to_target[sampled_tokens.long()]

    @property
    def draft_to_target_loaded(self) -> bool:
        return self._draft_to_target_loaded

    def load_draft_to_target(self, d2t: torch.Tensor) -> None:
        """Load the checkpoint's d2t delta table (target = draft + d2t[draft]),
        following the vLLM/speculators convention. Validates the geometry so a
        mismatched or out-of-range table fails at load, not at sample time."""
        if self.draft_to_target is None:
            raise ValueError(
                "DSpark markov head received a d2t table but was built for the "
                "full target vocab (no reduced draft head)."
            )
        d2t = d2t.view(-1)
        if d2t.numel() != self.draft_vocab_size:
            raise ValueError(
                f"DSpark d2t length {d2t.numel()} != draft_vocab_size "
                f"{self.draft_vocab_size}."
            )
        draft_ids = torch.arange(self.draft_vocab_size, device=d2t.device)
        target_ids = d2t.to(torch.long) + draft_ids
        if int(target_ids.min()) < 0 or int(target_ids.max()) >= self.vocab_size:
            raise ValueError(
                "DSpark d2t maps outside the target vocab "
                f"[0, {self.vocab_size}); got range "
                f"[{int(target_ids.min())}, {int(target_ids.max())}]."
            )
        if torch.unique(target_ids).numel() != self.draft_vocab_size:
            raise ValueError(
                "DSpark d2t must map each draft token to a unique target token; "
                "duplicate target ids cannot be represented by the corrected-logit "
                "scatter."
            )
        self.draft_to_target.copy_(target_ids.to(self.draft_to_target.device))
        self._draft_to_target_loaded = True

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

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        draft_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            draft_vocab_size=draft_vocab_size,
        )
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

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        draft_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            draft_vocab_size=draft_vocab_size,
        )
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
            sampled = sampler(step_logits, step_idx)
            # Draft-space id -> target-space id (identity for full-vocab); the
            # target id both gets stored and conditions the next RNN step's
            # markov_w1. See run_markov_block for the rationale.
            next_tokens = self.map_sampled_to_target(sampled)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )


def build_markov_head(
    config, *, draft_vocab_size: Optional[int] = None
) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank <= 0:
        raise ValueError(
            "DSpark requires markov_rank > 0 (the Markov head is the core of the "
            f"semi-AR draft); got markov_rank={markov_rank}."
        )
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    # vocab_size is the target vocab (markov_w1 input); draft_vocab_size, when
    # set, is the reduced markov_w2 output. They are equal for full-vocab heads.
    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    if markov_head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            draft_vocab_size=draft_vocab_size,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
            draft_vocab_size=draft_vocab_size,
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
            draft_vocab_size=draft_vocab_size,
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


# embed_tokens is always shared from the target; lm_head is handled separately
# (kept for a reduced draft head, skipped when the target head is shared).
_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "rotary_emb.",
)


class DSparkDraftMixin:
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._fused_kv_write_cache = None
        self.logits_mup_width_multiplier = None
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        # speculators-trained checkpoints (dspark_config.speculators_convention)
        # use a `gamma + 1`-wide draft block with the anchor as a separate
        # bonus token, rather than DeepSpec's `gamma`-wide anchor-first block.
        # That width difference is handled downstream in dspark_draft.py's
        # DraftBlockProposer/DsparkDraftSampler (see `bonus_anchor` there),
        # not in this model class -- run_markov_block itself is unaffected
        # either way, since the caller always hands it exactly `gamma` real
        # draft-hidden slots regardless of which convention produced them.
        self.gamma = int(dspark_config.resolve_gamma(default=self.block_size))
        self.markov_head = build_markov_head(
            config, draft_vocab_size=dspark_config.draft_vocab_size
        )
        self.confidence_head = build_confidence_head(config)
        # A speculators checkpoint with a reduced draft vocab ships its own
        # small lm_head (loaded in load_weights); a full-vocab draft leaves this
        # None and shares the target head via attach_shared_modules.
        if self.markov_head.reduced_vocab:
            self.lm_head: Optional[nn.Module] = build_independent_lm_head(
                draft_vocab_size=int(self.markov_head.draft_vocab_size),
                hidden_size=int(config.hidden_size),
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = None
        # Reused eager buffer for lifting draft-space corrected logits into
        # target-vocab columns (reduced-vocab probabilistic verify only).
        self._corrected_target_scratch: Optional[torch.Tensor] = None

    @property
    def uses_reduced_draft_vocab(self) -> bool:
        return self.markov_head is not None and self.markov_head.reduced_vocab

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        # Full-vocab drafts share the target lm_head; a reduced-vocab draft keeps
        # the independent head built (and loaded) for it in __init__.
        if self.lm_head is None:
            self.lm_head = lm_head

    def scatter_corrected_to_target(
        self, corrected_logits: torch.Tensor, *, target_width: int
    ) -> torch.Tensor:
        """Lift draft-space markov-corrected block logits into target-vocab
        columns (unmapped columns -inf) so target-space rejection sampling and
        the block-accept estimator index them by the same target ids the drafts
        were mapped to. Reuses one eager scratch buffer; only ever called
        outside cuda-graph capture (the folded accept path is greedy-only)."""
        draft_to_target = self.markov_head.draft_to_target
        assert draft_to_target is not None, "scatter requires a reduced draft head"
        bs, gamma_rows, _ = corrected_logits.shape
        need = (bs, gamma_rows, int(target_width))
        buf = self._corrected_target_scratch
        if (
            buf is None
            or tuple(buf.shape) != need
            or buf.dtype != corrected_logits.dtype
            or buf.device != corrected_logits.device
        ):
            buf = torch.empty(
                need, dtype=corrected_logits.dtype, device=corrected_logits.device
            )
            self._corrected_target_scratch = buf
        return scatter_draft_logits_to_target(
            corrected_logits, draft_to_target=draft_to_target, out=buf
        )

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeds with the shared target embedding INSIDE the draft graph
        # (the runner skips the eager input_embeds staging when the draft
        # model exposes forward_embed).
        return self.embed_tokens(input_ids)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project the draft's raw final hidden through the lm_head into base
        logits: target-vocab-wide for a shared head, draft-vocab-wide for a
        reduced independent head (org_vocab_size == draft vocab).

        muP targets (Inkling) train the draft against a FOLDED head (weights
        pre-divided by logits_mup_width_multiplier) while serving attaches the
        target's unfolded head, so the division happens here — exactly once,
        keeping base logits in the scale the markov bias and confidence head
        were trained against. DSparkWorkerV2 wires the multiplier from the
        target config; it stays None for non-muP targets. A reduced independent
        head is the draft's own trained head (not the target's), so the folding
        does not apply to it.
        """
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        if self.logits_mup_width_multiplier and not self.uses_reduced_draft_vocab:
            hidden = hidden / self.logits_mup_width_multiplier
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        base_logits = gather_and_crop_vocab(local_logits, self.lm_head)
        return base_logits, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        lm_head_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        reduced_vocab = self.uses_reduced_draft_vocab
        for name, loaded_weight in weights:
            if _is_dspark_d2t_weight(name):
                # Draft->target id map; only meaningful for a reduced draft head.
                if reduced_vocab:
                    self.markov_head.load_draft_to_target(loaded_weight)
                continue
            if _is_dspark_t2d_weight(name):
                # Inverse (target->draft) map; training-only, never used at inference.
                continue
            if name.startswith("lm_head."):
                # Keep the checkpoint's own head only for a reduced draft vocab;
                # full-vocab drafts share the target head, so drop it here.
                if reduced_vocab:
                    lm_head_weights.append((name, loaded_weight))
                continue
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

        self._load_independent_lm_head(
            lm_head_weights=lm_head_weights, params_dict=params_dict
        )
        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )
        if reduced_vocab and not self.markov_head.draft_to_target_loaded:
            raise ValueError(
                "DSpark reduced-vocab draft (draft_vocab_size is set) requires a "
                "d2t draft->target map in the checkpoint, but none was found. "
                "Provide the speculators checkpoint's d2t table; the target head "
                "cannot be substituted for a reduced draft vocab."
            )

    def _load_independent_lm_head(
        self,
        *,
        lm_head_weights: list,
        params_dict: dict,
    ) -> None:
        if not self.uses_reduced_draft_vocab:
            return
        loaded_names = set()
        for name, loaded_weight in lm_head_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected lm_head weight {name!r} not found in model "
                    "parameters."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        expected = {name for name in params_dict if name.startswith("lm_head.")}
        missing = expected - loaded_names
        if missing:
            raise ValueError(
                "DSpark reduced-vocab draft is missing independent lm_head weights "
                f"{sorted(missing)}. The checkpoint must ship its own reduced "
                "lm_head; the shared target head cannot cover a reduced draft vocab."
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

    def _fused_kv_write_bundle(self, pool):
        cached = self._fused_kv_write_cache
        if cached is not None and cached[0] == id(pool):
            return cached[1]
        bundle = self._build_fused_kv_write_bundle(pool)
        self._fused_kv_write_cache = (id(pool), bundle)
        return bundle

    def _build_fused_kv_write_bundle(self, pool):
        layers = list(self.layers)
        if not layers:
            return None
        if not (hasattr(pool, "get_key_buffer") and hasattr(pool, "get_value_buffer")):
            return None
        attn0 = layers[0].self_attn
        head_dim = attn0.head_dim
        kv_size = attn0.kv_size
        rotary = attn0.rotary_emb
        if type(rotary).__name__ != "RotaryEmbedding":
            return None
        if not getattr(rotary, "is_neox_style", False):
            return None
        if getattr(rotary, "rotary_dim", None) != head_dim:
            return None
        eps = attn0.k_norm.variance_epsilon
        weights, knws, meta_rows = [], [], []
        for layer in layers:
            attn = layer.self_attn
            ok, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not ok:
                return None
            if attn.qkv_proj.bias is not None:
                return None
            if attn.attn.k_scale is not None or attn.attn.v_scale is not None:
                return None
            if attn.head_dim != head_dim or attn.kv_size != kv_size:
                return None
            if attn.rotary_emb is not rotary and not torch.equal(
                attn.rotary_emb.cos_sin_cache, rotary.cos_sin_cache
            ):
                return None
            if attn.k_norm.variance_epsilon != eps:
                return None
            k_buf = pool.get_key_buffer(attn.attn.layer_id)
            v_buf = pool.get_value_buffer(attn.attn.layer_id)
            nh = kv_size // head_dim
            for buf in (k_buf, v_buf):
                if buf.dtype != torch.bfloat16:
                    return None
                if buf.shape[1:] != (nh, head_dim):
                    return None
                if buf.stride(1) != head_dim or buf.stride(2) != 1:
                    return None
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            w = attn.qkv_proj.weight[kv_slice]
            if w.dtype != torch.bfloat16:
                return None
            weights.append(w)
            knws.append(attn.k_norm.weight.data)
            meta_rows.append(
                [k_buf.data_ptr(), v_buf.data_ptr(), k_buf.stride(0), v_buf.stride(0)]
            )
        device = weights[0].device
        w_all = torch.cat(weights, dim=0).contiguous()
        knw = torch.stack(knws).to(device)
        meta = torch.tensor(meta_rows, dtype=torch.int64, device=device)
        cos_sin = rotary.cos_sin_cache.to(device)
        return (w_all, meta, knw, cos_sin, eps, len(layers), kv_size, head_dim)

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        """Stack every layer's KV projection into one weight (exact: the input
        hidden is shared, so concatenating output columns is equivalent).
        Cached; None (per-layer fallback) when a QKV weight cannot be sliced
        (quantized) or layers disagree on norm epsilon / bias presence.
        """
        if not envs.SGLANG_DSPARK_STACKED_CTX_KV.get():
            return None
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

        bundle = self._fused_kv_write_bundle(pool)
        if bundle is not None:
            from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
                fused_kv_norm_rope_write,
            )

            w_all, meta, knw, cos_sin, eps, num_layers, kv_size, head_dim = bundle
            kv_all = F.linear(ctx_hidden, w_all)
            if cache_loc_2d is not None and commit_lens is not None:
                locs = cache_loc_2d.reshape(-1)
                write_commit_lens = commit_lens
                locs_row_width = cache_loc_2d.shape[1]
            else:
                locs = cache_loc
                write_commit_lens = None
                locs_row_width = None
            fused_kv_norm_rope_write(
                kv_all,
                meta,
                knw,
                cos_sin,
                positions,
                locs,
                num_layers,
                kv_size,
                head_dim,
                eps,
                commit_lens=write_commit_lens,
                locs_row_width=locs_row_width,
            )
            return

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
        # Batched per-head k-norm across layers (fp32 variance + weight, cast back).
        k32 = (
            kv_all[:, :, 0, :]
            .reshape(tokens, num_layers, num_kv_heads, head_dim)
            .to(torch.float32)
        )
        variance = k32.pow(2).mean(dim=-1, keepdim=True)
        k32 = k32 * torch.rsqrt(variance + stacked["eps"])
        k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
        k_all = k32.to(ctx_hidden.dtype)
        # One RoPE over all layers' heads (shared rotary params + positions).
        k_flat = k_all.reshape(tokens, num_layers * kv_size)
        dummy_q = k_flat.new_empty(k_flat.shape)
        _, k_flat = attn0.rotary_emb(positions, dummy_q, k_flat)
        # [layers, tokens, heads, dim]: per-layer slices are contiguous views.
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
    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.o_proj = None
        torch.cuda.empty_cache()


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [Qwen3DSparkModel, DSparkDraftModel]

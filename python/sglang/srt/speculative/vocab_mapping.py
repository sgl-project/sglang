# SPDX-License-Identifier: Apache-2.0
"""
Vocabulary mapping for Token-Level Intersection (TLI) speculative decoding.

Based on the ICML 2025 oral paper:
  "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for
   Heterogeneous Vocabularies" — Timor et al., https://arxiv.org/abs/2502.05202

This module builds a normalized token intersection between the target and draft
model vocabularies and provides:
- logit masking (constrain draft logits to the intersection)
- target-to-draft token ID mapping
- draft-to-target token ID mapping
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Tokenizer-specific space-prefix characters used by BPE tokenizers.
# BPE vocabularies often encode a leading space as one of these Unicode chars.
# Used as a static fallback; runtime detection is preferred (see _detect_space_sign).
_KNOWN_SPACE_PREFIXES = ("\u0120", "\u2581")


def _detect_space_sign(tokenizer) -> Optional[str]:
    """Detect the space-prefix character used by a tokenizer by encoding a literal space.

    Returns the first character of the first token produced for " ", or None if the
    tokenizer does not encode a leading space as a prefix character.
    """
    try:
        space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
        if space_ids:
            tok_str = tokenizer.convert_ids_to_tokens(space_ids)[0]
            if tok_str and tok_str[0] in _KNOWN_SPACE_PREFIXES:
                return tok_str[0]
    except Exception:
        pass
    return None


def _normalize_token(token: str, space_sign: Optional[str] = None) -> str:
    """Normalize a BPE token by converting its space-prefix character to a plain space.

    If ``space_sign`` is provided it is used directly; otherwise all known prefixes
    are tried (static fallback for tokenizers that weren't probed at startup).
    """
    if space_sign is not None:
        if token.startswith(space_sign):
            return " " + token[len(space_sign) :]
        return token
    for prefix in _KNOWN_SPACE_PREFIXES:
        if token.startswith(prefix):
            return " " + token[len(prefix) :]
    return token


class VocabMapping:
    """Maps token IDs between target and draft model vocabularies via intersection.

    The intersection is computed by normalizing token strings (so that
    different BPE space-prefix representations still match), then finding
    tokens that appear in both vocabularies.

    Args:
        target_tokenizer: HuggingFace tokenizer for the target model.
        draft_tokenizer:  HuggingFace tokenizer for the draft model.
        target_vocab_size: Vocabulary size of the target model.
        draft_vocab_size:  Vocabulary size of the draft model.
        device: Torch device to place mapping tensors on.
    """

    def __init__(
        self,
        target_tokenizer,
        draft_tokenizer,
        target_vocab_size: int,
        draft_vocab_size: int,
        device: torch.device,
    ):
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device

        # Resolve unk_token_id for out-of-intersection fallback.
        # Llama 3, SmolLM2, and other models may not define unk_token_id;
        # we fall back to eos_token_id (always present) so that such models
        # are still supported. Using EOS as the fallback is safe: the
        # out-of-intersection token will be fed back into the draft KV cache
        # but will always be rejected by the target's acceptance criterion,
        # preserving losslessness.
        self.target_unk_token_id: int = (
            target_tokenizer.unk_token_id
            if target_tokenizer.unk_token_id is not None
            else target_tokenizer.eos_token_id
        )
        self.draft_unk_token_id: int = (
            draft_tokenizer.unk_token_id
            if draft_tokenizer.unk_token_id is not None
            else draft_tokenizer.eos_token_id
        )
        if self.target_unk_token_id is None or self.draft_unk_token_id is None:
            raise ValueError(
                "Target or draft tokenizer has neither unk_token_id nor eos_token_id. "
                "Cannot determine a safe fallback token for out-of-intersection IDs."
            )
        if target_tokenizer.unk_token_id is None:
            logger.warning(
                "Target tokenizer has no unk_token_id; using eos_token_id=%d as fallback "
                "for out-of-intersection tokens (e.g. Llama 3, SmolLM2).",
                self.target_unk_token_id,
            )
        if draft_tokenizer.unk_token_id is None:
            logger.warning(
                "Draft tokenizer has no unk_token_id; using eos_token_id=%d as fallback "
                "for out-of-intersection tokens.",
                self.draft_unk_token_id,
            )

        # Detect space-prefix characters dynamically so that unusual tokenizers
        # (e.g. those that use a different prefix not in _KNOWN_SPACE_PREFIXES)
        # are still matched correctly.
        target_space_sign = _detect_space_sign(target_tokenizer)
        draft_space_sign = _detect_space_sign(draft_tokenizer)

        # Build normalized vocabularies.  When two tokens normalize to the
        # same string we keep only the first occurrence (lowest token ID).
        target_vocab = target_tokenizer.get_vocab()
        draft_vocab = draft_tokenizer.get_vocab()

        target_normalized: dict[str, int] = {}
        for token, tid in target_vocab.items():
            norm = _normalize_token(token, target_space_sign)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized: dict[str, int] = {}
        for token, tid in draft_vocab.items():
            norm = _normalize_token(token, draft_space_sign)
            if norm not in draft_normalized:
                draft_normalized[norm] = tid

        # Intersection: tokens that appear in both vocabularies.
        common_tokens = set(target_normalized.keys()) & set(draft_normalized.keys())

        # Build mapping tensors (initialized to -1 = "not in intersection").
        draft_to_target = torch.full((draft_vocab_size,), -1, dtype=torch.long)
        target_to_draft = torch.full((target_vocab_size,), -1, dtype=torch.long)
        intersection_mask_draft = torch.zeros(draft_vocab_size, dtype=torch.bool)

        for norm_token in common_tokens:
            t_id = target_normalized[norm_token]
            d_id = draft_normalized[norm_token]
            if t_id < target_vocab_size and d_id < draft_vocab_size:
                draft_to_target[d_id] = t_id
                target_to_draft[t_id] = d_id
                intersection_mask_draft[d_id] = True

        self.draft_to_target_ids = draft_to_target.to(device)
        self.target_to_draft_ids = target_to_draft.to(device)
        # Boolean mask over the draft vocabulary: True iff the token is in the intersection.
        self.intersection_mask_draft = intersection_mask_draft.to(device)
        self.intersection_size = int(intersection_mask_draft.sum().item())
        # Pre-allocated fallback scalars for CUDA-graph-safe token mapping
        # (avoids .any()/.item() CPU sync inside map_*_ids methods).
        self._target_unk_tensor = torch.tensor(
            self.target_unk_token_id, dtype=torch.long, device=device
        )
        self._draft_unk_tensor = torch.tensor(
            self.draft_unk_token_id, dtype=torch.long, device=device
        )
        # Sorted draft token IDs in the intersection: shape [intersection_size].
        # Used for LM head pruning (see TLIWorker._try_prune_draft_lm_head).
        self.intersection_draft_ids = intersection_mask_draft.nonzero(as_tuple=True)[
            0
        ].to(device)

        logger.info(
            "VocabMapping initialized: target_vocab=%d, draft_vocab=%d, "
            "intersection=%d (%.1f%% of draft, %.1f%% of target)",
            target_vocab_size,
            draft_vocab_size,
            self.intersection_size,
            100.0 * self.intersection_size / max(draft_vocab_size, 1),
            100.0 * self.intersection_size / max(target_vocab_size, 1),
        )

        if self.intersection_size < 100:
            logger.warning(
                "Very small vocabulary intersection (%d tokens). "
                "TLI acceptance rate will be very low.",
                self.intersection_size,
            )

    def map_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Map target model token IDs to draft model token IDs.

        Tokens not in the intersection are mapped to ``draft_unk_token_id``.

        Args:
            target_ids: Integer tensor of target token IDs.

        Returns:
            Integer tensor of draft token IDs with the same shape and dtype.
        """
        draft_ids = self.target_to_draft_ids[target_ids]
        # torch.where is CUDA-graph compatible; avoids .any()/.item() CPU sync.
        draft_ids = torch.where(draft_ids >= 0, draft_ids, self._draft_unk_tensor)
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft model token IDs to target model token IDs.

        Tokens not in the intersection are mapped to ``target_unk_token_id``.

        Args:
            draft_ids: Integer tensor of draft token IDs.

        Returns:
            Integer tensor of target token IDs with the same shape and dtype.
        """
        target_ids = self.draft_to_target_ids[draft_ids]
        # torch.where is CUDA-graph compatible; avoids .any()/.item() CPU sync.
        target_ids = torch.where(target_ids >= 0, target_ids, self._target_unk_tensor)
        return target_ids.to(draft_ids.dtype)

    def constrain_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Zero out (−inf) logits for tokens outside the intersection.

        Args:
            logits: Float tensor of shape ``(..., draft_vocab_size)``.

        Returns:
            A cloned tensor with non-intersection logits set to ``-inf``.
        """
        # masked_fill is CUDA-graph compatible; avoids boolean indexed
        # assignment which may invoke non-capturable CUDA ops.
        return logits.masked_fill(~self.intersection_mask_draft, float("-inf"))

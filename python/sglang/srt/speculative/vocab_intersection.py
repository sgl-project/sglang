"""
Vocabulary Intersection Mapper for Heterogeneous Vocabulary Speculative Decoding.

Implements TLI algorithm from "Lossless Speculative Decoding for Heterogeneous
Vocabularies" (ICML 2025 Oral). https://arxiv.org/abs/2502.05202
"""

import logging
from typing import Dict, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


class VocabIntersectionMapper:
    """Manages vocabulary intersection for heterogeneous vocab speculative decoding."""

    def __init__(
        self,
        draft_tokenizer,
        target_tokenizer,
        device: str = "cuda",
    ):
        self.device = device
        self.draft_vocab_size = len(draft_tokenizer)
        self.target_vocab_size = len(target_tokenizer)

        (
            self.shared_tokens,
            self.draft_to_target_map,
            self.target_to_draft_map,
        ) = self._compute_vocab_intersection(draft_tokenizer, target_tokenizer)

        self.intersection_size = len(self.shared_tokens)
        self.draft_vocab_mask = self._create_draft_vocab_mask()
        self.draft_to_target_tensor = self._create_draft_to_target_tensor()

        logger.info(
            f"VocabIntersectionMapper: draft={self.draft_vocab_size}, "
            f"target={self.target_vocab_size}, "
            f"intersection={self.intersection_size} "
            f"({100 * self.intersection_size / self.draft_vocab_size:.1f}%)"
        )

    def _compute_vocab_intersection(
        self, draft_tokenizer, target_tokenizer
    ) -> Tuple[Set[str], Dict[int, int], Dict[int, int]]:
        draft_vocab = draft_tokenizer.get_vocab()
        target_vocab = target_tokenizer.get_vocab()
        shared_tokens = set(draft_vocab.keys()) & set(target_vocab.keys())

        draft_to_target_map = {}
        target_to_draft_map = {}
        for token in shared_tokens:
            draft_id = draft_vocab[token]
            target_id = target_vocab[token]
            draft_to_target_map[draft_id] = target_id
            target_to_draft_map[target_id] = draft_id

        return shared_tokens, draft_to_target_map, target_to_draft_map

    def _create_draft_vocab_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.draft_vocab_size, dtype=torch.bool, device=self.device)
        for draft_id in self.draft_to_target_map.keys():
            mask[draft_id] = True
        return mask

    def _create_draft_to_target_tensor(self) -> torch.Tensor:
        mapping = torch.full(
            (self.draft_vocab_size,), -1, dtype=torch.long, device=self.device
        )
        for draft_id, target_id in self.draft_to_target_map.items():
            mapping[draft_id] = target_id
        return mapping

    def mask_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.clone()
        masked_logits[..., ~self.draft_vocab_mask] = float("-inf")
        return masked_logits

    def mask_draft_logits_inplace(self, logits: torch.Tensor) -> torch.Tensor:
        logits[..., ~self.draft_vocab_mask] = float("-inf")
        return logits

    def map_draft_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.draft_to_target_tensor[draft_token_ids]

    def map_target_to_draft(
        self, target_token_ids: torch.Tensor, fallback_id: int = 0
    ) -> torch.Tensor:
        """Map target vocab token IDs to draft vocab token IDs.

        For tokens not in the intersection, returns fallback_id (default: 0).
        This can happen when target model predicts a token outside the intersection.
        """
        if not hasattr(self, "_target_to_draft_tensor"):
            # Use fallback_id for tokens not in intersection instead of -1
            self._target_to_draft_tensor = torch.full(
                (self.target_vocab_size,), fallback_id, dtype=torch.long, device=self.device
            )
            for target_id, draft_id in self.target_to_draft_map.items():
                self._target_to_draft_tensor[target_id] = draft_id
        return self._target_to_draft_tensor[target_token_ids]

    def get_intersection_ratio(self) -> float:
        return self.intersection_size / self.draft_vocab_size


def create_vocab_mapper(
    draft_tokenizer,
    target_tokenizer,
    device: str = "cuda",
    min_intersection_ratio: float = 0.1,
) -> Optional[VocabIntersectionMapper]:
    mapper = VocabIntersectionMapper(draft_tokenizer, target_tokenizer, device)

    if mapper.get_intersection_ratio() < min_intersection_ratio:
        logger.warning(
            f"Vocabulary intersection too small: {mapper.intersection_size} tokens "
            f"({100 * mapper.get_intersection_ratio():.1f}%). "
            f"May have very low acceptance rate."
        )

    return mapper

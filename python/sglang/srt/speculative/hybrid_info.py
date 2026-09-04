from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class HybridVerifyInput(SpecInput):
    """Next-iteration state shared by Hybrid's EAGLE and NGRAM routes.

    Each route keeps using its native input type. This wrapper only composes
    their scheduler and overlap-relay lifecycles so a route switch does not
    overload either native type with state owned by the other algorithm.
    """

    def __init__(
        self,
        eagle_draft_input: EagleDraftInput,
        ngram_verify_input: NgramVerifyInput,
    ) -> None:
        super().__init__(SpecInputType.HYBRID_VERIFY)
        self.eagle_draft_input = eagle_draft_input
        self.ngram_verify_input = ngram_verify_input

    @property
    def future_indices(self) -> Optional[torch.Tensor]:
        return self.eagle_draft_input.future_indices

    @future_indices.setter
    def future_indices(self, value: Optional[torch.Tensor]) -> None:
        self.eagle_draft_input.future_indices = value
        self.ngram_verify_input.future_indices = value

    @property
    def dsa_topk_indices(self) -> Optional[torch.Tensor]:
        return self.eagle_draft_input.dsa_topk_indices

    @dsa_topk_indices.setter
    def dsa_topk_indices(self, value: Optional[torch.Tensor]) -> None:
        self.eagle_draft_input.dsa_topk_indices = value

    @property
    def future_dsa_topk_indices_available(self) -> bool:
        return self.eagle_draft_input.future_dsa_topk_indices_available

    @future_dsa_topk_indices_available.setter
    def future_dsa_topk_indices_available(self, value: bool) -> None:
        self.eagle_draft_input.future_dsa_topk_indices_available = value

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ) -> None:
        self.eagle_draft_input.filter_batch(new_indices, new_indices_cpu)
        self.ngram_verify_input.filter_batch(new_indices, new_indices_cpu)

    def merge_batch(self, spec_info: HybridVerifyInput) -> None:
        if not isinstance(spec_info, HybridVerifyInput):
            raise TypeError(
                "HybridVerifyInput can only merge another HybridVerifyInput."
            )
        self.eagle_draft_input.merge_batch(spec_info.eagle_draft_input)
        self.ngram_verify_input.merge_batch(spec_info.ngram_verify_input)

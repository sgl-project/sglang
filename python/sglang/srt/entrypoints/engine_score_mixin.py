# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Engine mixin that exposes score() and async_score() on the Engine class.

These methods delegate to TokenizerManager.score_request() which is provided
by TokenizerManagerScoreMixin.
"""

from typing import List, Optional, Union

import torch

from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult


class EngineScoreMixin:
    def score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        embed_override_token_id: Optional[int] = None,
        query_embed_overrides: Optional[List[torch.Tensor]] = None,
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]] = None,
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """
        Score items against a query using the loaded model.

        For generation (CausalLM) models, returns the probability of each label_token_id
        being generated after the query+item prompt. Example:
            query = "<|user|>Is the following city the capital of France? "
            items = ["Paris <|assistant|>", "London <|assistant|>"]
            label_token_ids = [2332, 1223]  # "Yes" / "No"
            # -> [[0.9, 0.1], [0.2, 0.8]]

        For SequenceClassification models, returns the pooled class logits directly from
        the classification head. label_token_ids is optional and ignored.

        Args:
            query: The query text or pre-tokenized token IDs.
            items: The item text(s) or pre-tokenized token IDs.
            label_token_ids: Token IDs to score (required for CausalLM; ignored for
                SequenceClassification).
            apply_softmax: Whether to normalize scores using softmax.
            item_first: If True, prepend items before query (single-item mode only).
            embed_override_token_id: Placeholder token ID used to locate override positions.
            query_embed_overrides: Embedding vectors replacing placeholder tokens in query.
            item_embed_overrides: Per-item embedding vectors replacing placeholder tokens in items.
            return_pooled_hidden_states: Whether to include raw pooled transformer
                hidden states (before the task head) in the result. Only supported
                for non-generation models (SequenceClassification, RewardModel).

        Returns:
            ScoreResult with scores (one list per item), prompt token count, and
            optional pooled_hidden_states tensors.
        """
        return self.loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                embed_override_token_id=embed_override_token_id,
                query_embed_overrides=query_embed_overrides,
                item_embed_overrides=item_embed_overrides,
                request=None,
                return_pooled_hidden_states=return_pooled_hidden_states,
            )
        )

    async def async_score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        embed_override_token_id: Optional[int] = None,
        query_embed_overrides: Optional[List[torch.Tensor]] = None,
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]] = None,
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """Asynchronous version of score(). See score() for full documentation."""
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            embed_override_token_id=embed_override_token_id,
            query_embed_overrides=query_embed_overrides,
            item_embed_overrides=item_embed_overrides,
            request=None,
            return_pooled_hidden_states=return_pooled_hidden_states,
        )

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

from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult


class EngineScoreMixin:
    def score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> ScoreResult:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair. For example:
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223] # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        The api would then return the probabilities of the model producing "Yes" and "No" as the next token.
        The output would look like:
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]


        Args:
            query: The query text or pre-tokenized query token IDs. Must be provided.
            items: The item text(s) or pre-tokenized item token IDs. Must be provided.
            label_token_ids: List of token IDs to compute probabilities for. If None, no token probabilities will be computed.
            apply_softmax: Whether to normalize probabilities using softmax.
            item_first: If True, prepend items to query. Otherwise append items to query.

        Returns:
            ScoreResult with:
                scores: List of lists containing probabilities for each item and each label token
                prompt_tokens: The number of prompt tokens processed.

        Raises:
            ValueError: If query is not provided, or if items is not provided,
                      or if token IDs are out of vocabulary, or if logprobs are not available for the specified tokens.
        """
        return self.loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    async def async_score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> ScoreResult:
        """
        Asynchronous version of score method.

        See score() for detailed documentation.
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    scores: List[List[float]]
    prompt_tokens: int = 0


class TokenizerManagerScoreMixin:
    async def score_prompts(
        self,
        prompts: Union[str, List[str], List[List[int]]],
        label_token_ids: List[int],
        apply_softmax: bool = False,
        request: Optional[Any] = None,
    ) -> ScoreResult:
        """
        Score probabilities of specified token IDs after each *full prompt*.

        This is a thin wrapper over `score_request` that treats `prompts` as
        already-composed inputs (i.e., no query/item concatenation needed).

        Args:
            prompts: A single prompt string, a list of prompt strings, or a list of
                pre-tokenized prompt token ID sequences.
            label_token_ids: Token IDs to compute probabilities for.
            apply_softmax: Whether to normalize probabilities using softmax.
            request: Optional FastAPI request object.

        Returns:
            ScoreResult with:
                scores: List of score lists, one for each prompt, each in the order of label_token_ids.
                prompt_tokens: The number of prompt tokens processed.
        """
        # Text prompts
        if isinstance(prompts, str) or (
            isinstance(prompts, list) and (not prompts or isinstance(prompts[0], str))
        ):
            return await self.score_request(
                query="",
                items=prompts,  # type: ignore[arg-type]
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=False,
                request=request,
            )

        # Tokenized prompts
        if isinstance(prompts, list) and (not prompts or isinstance(prompts[0], list)):
            return await self.score_request(
                query=[],
                items=prompts,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=False,
                request=request,
            )

        raise ValueError("Invalid prompts type for score_prompts.")

    def _initialize_multi_item_delimiter_text(self):
        """Initialize multi-item delimiter text from token ID after tokenizer is loaded."""
        if (
            hasattr(self.server_args, "multi_item_scoring_delimiter")
            and self.server_args.multi_item_scoring_delimiter is not None
            and self.tokenizer is not None
        ):
            try:
                self.multi_item_delimiter_text = self.tokenizer.decode(
                    [self.server_args.multi_item_scoring_delimiter],
                    skip_special_tokens=False,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to decode delimiter token {self.server_args.multi_item_scoring_delimiter}: {e}"
                )
                self.multi_item_delimiter_text = None

    def _build_multi_item_token_sequence(
        self, query: List[int], items: List[List[int]], delimiter_token_id: int
    ) -> List[int]:
        """
        Build a single token sequence for multi-item scoring.
        Format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>

        Args:
            query: Query token IDs
            items: List of item token ID sequences
            delimiter_token_id: Token ID to use as delimiter

        Returns:
            Combined token sequence
        """
        combined_sequence = query[:]  # Start with query

        for item in items:
            combined_sequence.append(delimiter_token_id)  # Add delimiter
            combined_sequence.extend(item)  # Add item tokens

        # Add final delimiter after the last item for logprob extraction
        combined_sequence.append(delimiter_token_id)

        return combined_sequence

    def _batch_tokenize_query_and_items(
        self,
        query: Optional[Union[str, List[int]]],
        items: Optional[Union[str, List[str], List[List[int]]]],
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Tokenize query and items into token IDs.

        Args:
            query: The query text (str) or pre-tokenized token IDs (List[int]).
            items: Item texts or pre-tokenized token IDs.

        Returns:
            (query_ids, items_ids): query token IDs and list of per-item token IDs.
        """
        if isinstance(query, str):
            query_ids = self.tokenizer.encode(query)
        else:
            query_ids = list(query)

        items_list = [items] if isinstance(items, str) else items

        items_ids = []
        for item in items_list:
            if isinstance(item, str):
                items_ids.append(self.tokenizer.encode(item))
            else:
                items_ids.append(list(item))

        return query_ids, items_ids

    def _process_multi_item_scoring_results(
        self,
        results: Any,
        items: List,
        label_token_ids: Optional[List[int]],
        apply_softmax: bool,
        batch_request=None,
    ) -> ScoreResult:
        """
        Process results from multi-item scoring request.

        Extracts per-delimiter scores from whichever field the scheduler
        populated (input_token_ids_logprobs for generation models,
        embedding for classification models), then uniformly validates,
        skips the query-boundary delimiter, and normalizes.

        Args:
            results: Results from generate_request
            items: List of items being scored
            label_token_ids: Token IDs to extract scores for
            apply_softmax: Whether to apply softmax normalization
            batch_request: The original batch request containing input sequence

        Returns:
            ScoreResult with:
                scores: List of score lists, one for each prompt, each in the order of label_token_ids.
                prompt_tokens: The number of prompt tokens processed.
        """
        single_result = results[0] if isinstance(results, list) else results
        meta_info = single_result.get("meta_info", {})
        num_items = len(items) if isinstance(items, list) else 1
        expected_count = num_items + 1
        request_id = meta_info.get("id", "<unknown>")
        prompt_tokens = meta_info.get("prompt_tokens", 0)

        # Extract per-delimiter scores from whichever field has them
        input_logprobs = meta_info.get("input_token_ids_logprobs", [])
        embedding = single_result.get("embedding")

        if input_logprobs:
            # Generation model: extract label-token logprobs at each delimiter
            per_delimiter_scores = []
            for logprobs_data in input_logprobs:
                logprobs = self._extract_logprobs_for_tokens(
                    logprobs_data, label_token_ids
                )
                score_list = self._convert_logprobs_to_scores(
                    logprobs, label_token_ids, apply_softmax
                )
                per_delimiter_scores.append(score_list)
        elif embedding is not None:
            # Classification model: scores are directly in 2D embedding.
            if apply_softmax:
                scores_tensor = (
                    torch.tensor(embedding)
                    if isinstance(embedding, list)
                    else embedding
                )
                scores_tensor = torch.nn.functional.softmax(scores_tensor, dim=-1)
                per_delimiter_scores = scores_tensor.tolist()
            else:
                per_delimiter_scores = (
                    embedding if isinstance(embedding, list) else embedding.tolist()
                )
        else:
            raise RuntimeError(
                f"No scoring data found for multi-item scoring request {request_id}. "
                "Expected either input_token_ids_logprobs or embedding."
            )

        # Validate delimiter count
        if len(per_delimiter_scores) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} delimiter entries for multi-item scoring "
                f"with {num_items} items, but got {len(per_delimiter_scores)}. "
                f"Request ID: {request_id}"
            )

        # Skip the first delimiter (query-item boundary)
        scores = per_delimiter_scores[1:]

        return ScoreResult(scores=scores, prompt_tokens=prompt_tokens)

    def _process_single_item_scoring_results(
        self, results: Any, label_token_ids: Optional[List[int]], apply_softmax: bool
    ) -> ScoreResult:
        """
        Process results from single-item scoring request.

        For generation (CausalLM) models: reads output_token_ids_logprobs.
        For non-generation (SequenceClassification) models: reads the embedding field
        which contains pooled class logits from the classification head.

        Args:
            results: Results from generate_request
            label_token_ids: Token IDs to extract scores for (generation models only)
            apply_softmax: Whether to apply softmax normalization

        Returns:
            ScoreResult with:
                scores: List of score lists, one for each prompt, each in the order of label_token_ids.
                prompt_tokens: The number of prompt tokens processed.
        """
        scores = []
        prompt_tokens = 0

        is_generation = getattr(self, "is_generation", True)
        if is_generation:
            for result in results:
                # For single-item scoring, logprobs are in output_token_ids_logprobs
                output_logprobs = result["meta_info"].get(
                    "output_token_ids_logprobs", []
                )
                prompt_tokens += result["meta_info"].get("prompt_tokens", 0)

                if not output_logprobs or len(output_logprobs) == 0:
                    raise RuntimeError(
                        f"output_logprobs is empty for request "
                        f"{result['meta_info'].get('id', '<unknown>')}."
                    )

                # Extract logprobs for the first (and only) position
                logprobs = self._extract_logprobs_for_tokens(
                    output_logprobs[0], label_token_ids
                )
                score_list = self._convert_logprobs_to_scores(
                    logprobs, label_token_ids, apply_softmax
                )
                scores.append(score_list)
        else:
            for result in results:
                embedding = result.get("embedding", None)
                if embedding is None:
                    raise ValueError("Embedding not found in the result.")

                prompt_tokens += result.get("meta_info", {}).get("prompt_tokens", 0)

                if apply_softmax:
                    embedding = torch.softmax(
                        torch.as_tensor(embedding), dim=-1
                    ).tolist()

                # The classification head produces per-token logits, which the pooler reduces
                # into a single vector per input. That vector is returned in the `.embeddings`
                # field — not as semantic embeddings, but as pooled classification logits.
                # The field name is reused for compatibility with the existing
                # EmbeddingPoolerOutput API.
                scores.append(embedding)

        return ScoreResult(scores=scores, prompt_tokens=prompt_tokens)

    async def score_request(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        request: Optional[Any] = None,
    ) -> ScoreResult:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair.

        This method supports two scoring approaches:
        1. Single-Item scoring (default): Process each query+item pair independently
        2. Multi-Item scoring: When multi_item_scoring_delimiter is set, combine query and
           multiple items into a single sequence using delimiter for efficient processing.
           Note: item_first parameter is ignored in multi-item scoring mode since it uses
           a fixed format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>

           Multi-item scoring works with both text and pre-tokenized inputs:
           - Text: query<delimiter_text>item1<delimiter_text>item2<delimiter_text>item3<delimiter_text>
           - Tokens: query<delimiter_token_id>item1<delimiter_token_id>item2<delimiter_token_id>item3<delimiter_token_id>

        Supports two model types:
        - Generation (CausalLM): Requires label_token_ids; returns logprob-based scores.
        - SequenceClassification: label_token_ids is optional; returns pooled class logits.

        Args:
            query: The query text or pre-tokenized query token IDs
            items: The item text(s) or pre-tokenized item token IDs
            label_token_ids: List of token IDs to compute probabilities for
            apply_softmax: Whether to normalize probabilities using softmax
            item_first: If True, prepend items to query. Ignored for multi-item scoring.
            request: Optional FastAPI request object

        Returns:
            ScoreResult with:
                scores: List of score lists, one for each prompt, each in the order of label_token_ids.
                prompt_tokens: The number of prompt tokens processed.
        """
        is_generation = getattr(self, "is_generation", True)

        if is_generation and label_token_ids is None:
            raise ValueError(
                "label_token_ids is required for generation (CausalLM) models."
            )

        if items is None:
            raise ValueError("items must be provided")
        if not items:
            return ScoreResult(scores=[], prompt_tokens=0)

        if self.tokenizer is not None and label_token_ids is not None:
            vocab_size = self.tokenizer.vocab_size
            for token_id in label_token_ids:
                if token_id >= vocab_size:
                    raise ValueError(
                        f"Token ID {token_id} is out of vocabulary (vocab size: {vocab_size})"
                    )

        # Check if multi-item scoring is enabled by presence of delimiter
        use_multi_item_scoring = (
            self.server_args.multi_item_scoring_delimiter is not None
            and self.multi_item_delimiter_text is not None
        )

        input_ids = None
        text_prompts = None

        # Handle string or tokenized query/items
        if isinstance(query, str) and (
            isinstance(items, str)
            or (isinstance(items, list) and (not items or isinstance(items[0], str)))
        ):
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items

            if use_multi_item_scoring:
                # Multi-item scoring: tokenize separately then combine at token level
                # to ensure the delimiter token ID is inserted exactly once per boundary
                # (a text-level roundtrip through the tokenizer can alter boundary tokens)
                delimiter_token_id = self.server_args.multi_item_scoring_delimiter
                query_ids, items_ids = self._batch_tokenize_query_and_items(
                    query, items_list
                )
                combined_input_ids = self._build_multi_item_token_sequence(
                    query_ids, items_ids, delimiter_token_id
                )
                input_ids = [combined_input_ids]
            else:
                # Single-item scoring: create separate prompts for each item
                if item_first:
                    text_prompts = [f"{item}{query}" for item in items_list]
                else:
                    text_prompts = [f"{query}{item}" for item in items_list]

        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):
            # Both query and items are token IDs
            if use_multi_item_scoring:
                # Multi-item scoring: concatenate with delimiter token ID
                # Format: query<delimiter_token_id>item1<delimiter_token_id>item2<delimiter_token_id>item3<delimiter_token_id>
                delimiter_token_id = self.server_args.multi_item_scoring_delimiter
                combined_input_ids = self._build_multi_item_token_sequence(
                    query, items, delimiter_token_id
                )
                input_ids = [combined_input_ids]
            else:
                # Single-item scoring: process each item separately
                if item_first:
                    input_ids = [item + query for item in items]
                else:
                    input_ids = [query + item for item in items]
        else:
            raise ValueError(
                "Invalid combination of query/items types for score_request."
            )

        # Create the appropriate request type
        if is_generation:
            batch_request = GenerateReqInput(
                text=text_prompts,
                input_ids=input_ids,
                token_ids_logprob=label_token_ids,
                return_logprob=True,
                # Set logprob_start_len=0 for multi-item scoring since we want logprobs at all delimiter positions
                logprob_start_len=0 if use_multi_item_scoring else -1,
                stream=False,
                sampling_params={"max_new_tokens": 0},
            )
        else:
            batch_request = EmbeddingReqInput(
                text=text_prompts,
                input_ids=input_ids,
            )

        results = await self.generate_request(batch_request, request).__anext__()

        if use_multi_item_scoring:
            # Multi-item scoring: extract scores from input_token_ids_logprobs or embedding
            return self._process_multi_item_scoring_results(
                results, items, label_token_ids, apply_softmax, batch_request
            )
        else:
            # Single-item scoring: process each result separately
            return self._process_single_item_scoring_results(
                results, label_token_ids, apply_softmax
            )

    def _convert_logprobs_to_scores(
        self,
        logprobs: Dict[int, float],
        label_token_ids: List[int],
        apply_softmax: bool,
    ) -> List[float]:
        """
        Convert logprobs dictionary to ordered score list.

        Args:
            logprobs: Dictionary mapping token_id to logprob
            label_token_ids: Token IDs in desired order
            apply_softmax: Whether to apply softmax normalization

        Returns:
            List of scores in the same order as label_token_ids
        """
        score_list = [
            logprobs.get(token_id, float("-inf")) for token_id in label_token_ids
        ]

        if apply_softmax:
            score_list = torch.softmax(torch.tensor(score_list), dim=0).tolist()
        else:
            # Convert logprobs to probabilities if not using softmax
            score_list = [
                math.exp(x) if x != float("-inf") else 0.0 for x in score_list
            ]

        return score_list

    def _extract_logprobs_for_tokens(
        self, logprobs_data: List, label_token_ids: List[int]
    ) -> Dict[int, float]:
        """
        Extract logprobs for specified token IDs from logprobs data.

        Args:
            logprobs_data: List of (logprob, token_id, text) tuples
            label_token_ids: Token IDs to extract logprobs for

        Returns:
            Dictionary mapping token_id to logprob
        """
        logprobs = {}
        if logprobs_data:
            for logprob, token_id, _ in logprobs_data:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob
        return logprobs

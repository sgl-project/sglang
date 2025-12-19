import logging
import math
from typing import Any, Dict, List, Optional, Union

from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


class TokenizerManagerMultiItemMixin:
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

    def _process_multi_item_scoring_results(
        self,
        results: Any,
        items: List,
        label_token_ids: List[int],
        apply_softmax: bool,
        batch_request=None,
    ) -> List[List[float]]:
        """
        Process results from multi-item scoring request.
        Extracts logprobs at delimiter positions from input_token_ids_logprobs.

        Args:
            results: Results from generate_request
            items: List of items being scored
            label_token_ids: Token IDs to extract scores for
            apply_softmax: Whether to apply softmax normalization
            batch_request: The original batch request containing input sequence

        Returns:
            List of score lists, one for each item
        """
        single_result = results[0] if isinstance(results, list) else results

        # For multi-item scoring, logprobs are in input_token_ids_logprobs
        input_logprobs = single_result["meta_info"].get("input_token_ids_logprobs", [])

        if not input_logprobs:
            raise RuntimeError(
                f"input_token_ids_logprobs is empty for multi-item scoring request {single_result['meta_info'].get('id', '<unknown>')}. "
                "This indicates token_ids_logprobs were not computed properly for Mutil Item Scoring."
            )

        scores = []
        num_items = len(items) if isinstance(items, list) else 1

        # Check if we have the expected number of logprobs
        expected_logprobs_count = num_items + 1
        if len(input_logprobs) != expected_logprobs_count:
            raise RuntimeError(
                f"Expected {expected_logprobs_count} input_token_ids_logprobs for multi-item scoring "
                f"with {num_items} items, but got {len(input_logprobs)}. "
                f"Request ID: {single_result['meta_info'].get('id', '<unknown>')}"
            )

        # Skip the first delimiter (between query and first item) and process remaining delimiter positions
        # We want to exclude the first one since it represents the boundary between query and first item, not an item boundary
        start_idx = 1 if len(input_logprobs) > 1 else 0

        # Process logprobs for each item position (excluding first delimiter)
        for item_idx in range(num_items):
            logprob_idx = start_idx + item_idx
            item_logprobs_data = input_logprobs[logprob_idx]
            logprobs = self._extract_logprobs_for_tokens(
                item_logprobs_data, label_token_ids
            )
            score_list = self._convert_logprobs_to_scores(
                logprobs, label_token_ids, apply_softmax
            )
            scores.append(score_list)

        return scores

    def _process_single_item_scoring_results(
        self, results: Any, label_token_ids: List[int], apply_softmax: bool
    ) -> List[List[float]]:
        """
        Process results from single-item scoring request.
        Single-item scoring results are stored in output_token_ids_logprobs.

        Args:
            results: Results from generate_request
            label_token_ids: Token IDs to extract scores for
            apply_softmax: Whether to apply softmax normalization

        Returns:
            List of score lists, one for each result
        """
        scores = []

        for result in results:
            # For single-item scoring, logprobs are in output_token_ids_logprobs
            output_logprobs = result["meta_info"].get("output_token_ids_logprobs", [])

            if not output_logprobs or len(output_logprobs) == 0:
                raise RuntimeError(
                    f"output_logprobs is empty for request {result['meta_info'].get('id', '<unknown>')}."
                )

            # Extract logprobs for the first (and only) position
            logprobs = self._extract_logprobs_for_tokens(
                output_logprobs[0], label_token_ids
            )
            score_list = self._convert_logprobs_to_scores(
                logprobs, label_token_ids, apply_softmax
            )
            scores.append(score_list)

        return scores

    async def score_request(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        request: Optional[Any] = None,
    ) -> List[List[float]]:
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

        Args:
            query: The query text or pre-tokenized query token IDs
            items: The item text(s) or pre-tokenized item token IDs
            label_token_ids: List of token IDs to compute probabilities for
            apply_softmax: Whether to normalize probabilities using softmax
            item_first: If True, prepend items to query. Ignored for multi-item scoring.
            request: Optional FastAPI request object

        Returns:
            List of lists containing probabilities for each item and each label token
        """
        if label_token_ids is None:
            raise ValueError("label_token_ids must be provided")

        if self.tokenizer is not None:
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

        batch_request = GenerateReqInput(
            token_ids_logprob=label_token_ids,
            return_logprob=True,
            # Set logprob_start_len=0 for multi-item scoring since we want logprobs at all delimiter positions
            logprob_start_len=0 if use_multi_item_scoring else -1,
            stream=False,
            sampling_params={"max_new_tokens": 0},
        )

        # Handle string or tokenized query/items
        if isinstance(query, str) and (
            isinstance(items, str)
            or (isinstance(items, list) and (not items or isinstance(items[0], str)))
        ):
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items

            if use_multi_item_scoring:
                # Multi-item scoring: create single prompt with delimiter text
                # Always use format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>
                # (item_first is ignored for multi-item scoring)
                delimiter = self.multi_item_delimiter_text
                combined_items = delimiter.join(items_list)
                # Add final delimiter after the last item for logprob extraction
                single_prompt = f"{query}{delimiter}{combined_items}{delimiter}"
                batch_request.text = [single_prompt]
            else:
                # Single-item scoring: create separate prompts for each item
                if item_first:
                    prompts = [f"{item}{query}" for item in items_list]
                else:
                    prompts = [f"{query}{item}" for item in items_list]
                batch_request.text = prompts

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
                batch_request.input_ids = [combined_input_ids]
            else:
                # Single-item scoring: process each item separately
                if item_first:
                    input_ids_list = [item + query for item in items]
                else:
                    input_ids_list = [query + item for item in items]
                batch_request.input_ids = input_ids_list
        else:
            raise ValueError(
                "Invalid combination of query/items types for score_request."
            )

        results = await self.generate_request(batch_request, request).__anext__()

        if use_multi_item_scoring:
            # Multi-item scoring: extract scores from input_token_ids_logprobs
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
        import torch

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

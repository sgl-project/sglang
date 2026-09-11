import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import torch

from sglang.srt.configs.model_config import (
    is_cross_encoding_pooler_model,
    is_score_and_pool_model,
)
from sglang.srt.constants import MIS_DELIMITER_TOKEN_ID
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule, get_serving

logger = logging.getLogger(__name__)


ScoreRow: TypeAlias = List[float]
ScoreMatrix: TypeAlias = List[ScoreRow]
# Last-token pooling: one score row per item ([num_items, num_labels]).
# Multi-position pooling (the head is read at each score-extraction token): one
# [N_i, num_labels] matrix per item.
ScoreOutput: TypeAlias = Union[ScoreMatrix, List[ScoreMatrix]]

PooledHiddenStateRows: TypeAlias = List[Optional[torch.Tensor]]
# Last-token / MIS pooling: one optional vector per item. Multi-position pooling:
# one optional matrix (a vector per score-extraction token) per item.
PooledHiddenStateOutput: TypeAlias = Union[
    PooledHiddenStateRows, List[Optional[List[torch.Tensor]]]
]


@dataclass(frozen=True, slots=True)
class ScoreResult:
    # Last-token pooling returns a score matrix with one row per item.
    # Multi-position pooling (the head is read at each score-extraction token
    # instead of the last token) is always nested: one score matrix per item,
    # regardless of item count or --enable-mis. This is what powers setwise
    # scoring, where a set's candidates are concatenated into one item with one
    # score-extraction token per candidate.
    scores: ScoreOutput
    prompt_tokens: int = 0
    # Pre-head CPU tensors, parallel to scores: one optional vector per item for
    # last-token / MIS pooling, or one optional list of per-extraction-token
    # vectors per item for multi-position pooling. Kept as tensors so in-process
    # consumers avoid a .tolist() round-trip; the HTTP path converts tensors
    # recursively before JSON serialization.
    pooled_hidden_states: Optional[PooledHiddenStateOutput] = None


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

    def _build_multi_item_token_sequence(
        self, query: List[int], items: List[List[int]], delimiter_token_id: int
    ) -> Tuple[List[int], List[int]]:
        """
        Build a single token sequence for multi-item scoring.
        Format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>
        """
        combined_sequence = query[:]  # Start with query
        delimiter_indices = []

        for item in items:
            delimiter_indices.append(len(combined_sequence))
            combined_sequence.append(delimiter_token_id)  # Add delimiter
            combined_sequence.extend(item)  # Add item tokens

        # Add final delimiter after the last item for logprob extraction
        delimiter_indices.append(len(combined_sequence))
        combined_sequence.append(delimiter_token_id)

        return combined_sequence, delimiter_indices

    def _batch_tokenize_query_and_items(
        self,
        query: Optional[Union[str, List[int]]],
        items: Optional[Union[str, List[str], List[List[int]]]],
    ) -> Tuple[List[int], List[List[int]]]:
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
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """
        Process results from multi-item scoring request.

        Extracts per-delimiter scores from whichever field the scheduler
        populated (input_token_ids_logprobs for generation models,
        embedding for classification models), then uniformly validates,
        skips the query-boundary delimiter, and normalizes.
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

        phs_list = None
        if return_pooled_hidden_states:
            raw_phs = single_result.get("pooled_hidden_state")
            if raw_phs is not None and len(raw_phs) == expected_count:
                phs_list = list(raw_phs[1:])

        return ScoreResult(
            scores=scores,
            prompt_tokens=prompt_tokens,
            pooled_hidden_states=phs_list,
        )

    # ------------------------------------------------------------------
    # Multi-position pooling result helpers — the head is read at every
    # score-extraction token instead of the last token. Shared by the batched
    # and fused (--enable-mis) result processors. These power setwise scoring: a
    # set's candidates are concatenated into one item with one score-extraction
    # token per candidate.
    # ------------------------------------------------------------------

    def _multi_position_score_rows(
        self, embedding: Any, apply_softmax: bool
    ) -> List[List[float]]:
        """Validate a multi-position result embedding and return its per-token rows.

        The head pooled at every score-extraction token yields a 2-D
        ``[num_positions, num_labels]`` embedding. This validates that shape and
        returns it as a list of rows (softmaxed over labels when requested). The
        original list values are preserved when no softmax is applied (avoiding a
        float round-trip).
        """
        embedding_tensor = torch.as_tensor(embedding)
        if embedding_tensor.ndim != 2:
            # A 1-D result means the model pooled a single vector (e.g. a
            # reward/embedding head that reads the last token) instead of
            # per-position scores. Raise ValueError so the /v1/score handler
            # returns a clean 400, not a 500.
            raise ValueError(
                "Multi-position scoring expected a 2-D "
                "[num_positions, num_labels] result, but got shape "
                f"{tuple(embedding_tensor.shape)}. This model may not support "
                "score_extraction_token_id (only SequenceClassification heads that "
                "pool per position do)."
            )
        if apply_softmax:
            return torch.softmax(embedding_tensor, dim=-1).tolist()
        return embedding if isinstance(embedding, list) else embedding_tensor.tolist()

    def _multi_position_phs_matrix(self, phs: Any, expected_rows: int) -> torch.Tensor:
        """Validate multi-position pooled hidden states are a 2-D tensor with one
        row per score-extraction token and return it as a tensor."""
        phs_tensor = torch.as_tensor(phs)
        if phs_tensor.ndim != 2 or phs_tensor.shape[0] != expected_rows:
            raise ValueError(
                "Multi-position pooled hidden states must be a 2-D tensor with "
                "one row per score position."
            )
        return phs_tensor

    def _process_multi_item_extraction_results(
        self,
        results: Any,
        per_item_anchor_counts: List[int],
        apply_softmax: bool,
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """Process a fused multi-item score-extraction request (``--enable-mis``).

        The items are fused into one multi-item sequence, so the scheduler returns
        a single result whose ``embedding`` is the ``[ΣNᵢ, num_labels]`` matrix of
        every anchor's pooled logits, in item order. This splits that flat matrix
        back into one ``[Nᵢ, num_labels]`` matrix per item using the per-item
        anchor counts, yielding a nested ``scores`` list (one matrix per item).

        Args:
            results: Result(s) from generate_request (single fused request).
            per_item_anchor_counts: Number of extraction tokens per item, in order.
            apply_softmax: Whether to softmax over labels for each row.
            return_pooled_hidden_states: Whether to group pooled hidden states per
                item (parallel to ``scores``).

        Returns:
            ScoreResult with ``scores`` nested per item.
        """
        single_result = results[0] if isinstance(results, list) else results
        meta_info = single_result.get("meta_info", {})
        request_id = meta_info.get("id", "<unknown>")
        prompt_tokens = meta_info.get("prompt_tokens", 0)

        embedding = single_result.get("embedding")
        if embedding is None:
            raise ValueError("Embedding not found in the result.")

        # One flat [ΣNᵢ, num_labels] matrix of every anchor's logits, in item order.
        rows = self._multi_position_score_rows(embedding, apply_softmax)
        total_anchors = sum(per_item_anchor_counts)
        if len(rows) != total_anchors:
            raise RuntimeError(
                f"Expected {total_anchors} anchor rows across "
                f"{len(per_item_anchor_counts)} items, but got {len(rows)}. "
                f"Request ID: {request_id}"
            )

        # Split the flat matrix into one [Nᵢ, num_labels] matrix per item.
        scores = []
        offset = 0
        for count in per_item_anchor_counts:
            scores.append(rows[offset : offset + count])
            offset += count

        phs_grouped = None
        if return_pooled_hidden_states:
            raw_phs = single_result.get("pooled_hidden_state")
            if raw_phs is not None:
                phs_matrix = self._multi_position_phs_matrix(raw_phs, total_anchors)
                phs_grouped = [
                    list(group) for group in phs_matrix.split(per_item_anchor_counts)
                ]

        return ScoreResult(
            scores=scores,
            prompt_tokens=prompt_tokens,
            pooled_hidden_states=phs_grouped,
        )

    def _process_single_item_scoring_results(
        self,
        results: Any,
        label_token_ids: Optional[List[int]],
        apply_softmax: bool,
        return_pooled_hidden_states: bool = False,
        per_item_matrix: bool = False,
    ) -> ScoreResult:
        """
        Process results from single-item scoring request.

        For generation (CausalLM) models: reads output_token_ids_logprobs.
        For non-generation (SequenceClassification) models: reads the embedding field
        which contains pooled class logits from the classification head.

        per_item_matrix (setwise / multi-position mode): each result carries a 2-D
        ``[num_positions, num_labels]`` embedding (the head pooled at every
        extraction token). Each such matrix is kept as one item's entry, so
        ``scores`` is nested one score matrix per item. When False (pointwise),
        each result is one score row and ``scores`` is ``[num_items x num_labels]``.
        """
        scores = []
        phs_list = []
        has_phs = False
        prompt_tokens = 0

        is_generation = self.is_generation
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

                # The classification head produces per-token logits, which the pooler
                # reduces into a single vector per input (pointwise) or one row per
                # extraction token (setwise). The field name is reused for
                # compatibility with the existing EmbeddingPoolerOutput API.
                if per_item_matrix:
                    # Multi-position: one 2-D [num_positions, num_labels] matrix per
                    # item — keep the item's rows grouped instead of flattening
                    # across items.
                    rows = self._multi_position_score_rows(embedding, apply_softmax)
                    scores.append(rows)
                else:
                    # Pointwise: one score vector per item.
                    if apply_softmax:
                        embedding = torch.softmax(
                            torch.as_tensor(embedding), dim=-1
                        ).tolist()
                    scores.append(embedding)

                if return_pooled_hidden_states:
                    phs = result.get("pooled_hidden_state")
                    if per_item_matrix and phs is not None:
                        # One pooled-hidden-states matrix per item, parallel to scores.
                        phs_list.append(
                            list(self._multi_position_phs_matrix(phs, len(rows)))
                        )
                        has_phs = True
                    else:
                        phs_list.append(phs)
                        if phs is not None:
                            has_phs = True

        return ScoreResult(
            scores=scores,
            prompt_tokens=prompt_tokens,
            pooled_hidden_states=phs_list if has_phs else None,
        )

    # ------------------------------------------------------------------
    # Embed override position resolution
    # ------------------------------------------------------------------

    def _resolve_overrides_for_sequence(
        self,
        token_ids: List[int],
        embeds: Optional[List[torch.Tensor]],
        embed_override_token_id: int,
        position_offset: int = 0,
        label: str = "input",
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Scan token_ids for placeholder occurrences and pair with embeddings.
        Returns empty lists when embeds is None."""
        if embeds is None:
            return [], []
        positions = [
            idx + position_offset
            for idx, tok in enumerate(token_ids)
            if tok == embed_override_token_id
        ]
        if len(positions) != len(embeds):
            raise ValueError(
                f"{label} contains {len(positions)} occurrences of "
                f"embed_override_token_id={embed_override_token_id}, "
                f"but {len(embeds)} override embeddings were provided."
            )
        return embeds, positions

    def _resolve_embed_overrides_for_request(
        self,
        query: List[int],
        item: List[int],
        embed_override_token_id: int,
        query_embed_overrides: Optional[List[torch.Tensor]],
        item_embeds: Optional[List[torch.Tensor]],
        item_position_offset: int,
        item_label: str,
    ) -> Optional[PositionalEmbeds]:
        """Resolve embed overrides for a query+item pair; None when no overrides exist."""
        q_embeds, q_positions = self._resolve_overrides_for_sequence(
            query,
            query_embed_overrides,
            embed_override_token_id,
            position_offset=0,
            label="query",
        )
        i_embeds, i_positions = self._resolve_overrides_for_sequence(
            item,
            item_embeds,
            embed_override_token_id,
            position_offset=item_position_offset,
            label=item_label,
        )
        all_embeds = q_embeds + i_embeds
        all_positions = q_positions + i_positions
        if not all_embeds:
            return None
        return PositionalEmbeds(embeds=all_embeds, positions=all_positions)

    # ------------------------------------------------------------------
    # Input preparation (tokenization + input_ids construction)
    # ------------------------------------------------------------------

    def _build_token_id_inputs(
        self,
        query: List[int],
        items: List[List[int]],
        item_first: bool,
        use_multi_item_scoring: bool,
        embed_override_token_id: Optional[int],
        query_embed_overrides: Optional[List[torch.Tensor]],
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]],
    ) -> Tuple[None, List[List[int]], Optional[list], Optional[List[int]]]:
        """Build input_ids and resolve embed overrides for token-ID inputs.

        Multi-item-scoring and single-item modes differ only in how input_ids
        are assembled and what position offset each item gets.
        """
        # Both query and items are token IDs
        has_embeds = (
            query_embed_overrides is not None or item_embed_overrides is not None
        )

        # Query placeholder positions are invariant across items — resolve once.
        # (No-op returning ([], []) if has_embeds is False or query_embed_overrides is None.)
        q_embeds, q_positions = self._resolve_overrides_for_sequence(
            query,
            query_embed_overrides,
            embed_override_token_id,
            position_offset=0,
            label="query",
        )

        if use_multi_item_scoring:
            # Multi-item scoring: concatenate with placeholder delimiter token.
            # Positions are derived from item lengths (delimiter_indices), not
            # by scanning for this token — it exists only for FlashInfer compat.
            delimiter_token_id = MIS_DELIMITER_TOKEN_ID
            combined_input_ids, delimiter_indices = (
                self._build_multi_item_token_sequence(query, items, delimiter_token_id)
            )
            input_ids = [combined_input_ids]

            if not has_embeds:
                return None, input_ids, None, delimiter_indices

            # Resolve embed overrides across the combined multi-item-scoring sequence.
            all_embeds: List[torch.Tensor] = list(q_embeds)
            all_positions: List[int] = list(q_positions)
            current_offset = len(query) + 1  # +1 for first delimiter
            for i, item in enumerate(items):
                item_embs = item_embed_overrides[i] if item_embed_overrides else None
                i_embeds, i_positions = self._resolve_overrides_for_sequence(
                    item,
                    item_embs,
                    embed_override_token_id,
                    position_offset=current_offset,
                    label=f"items[{i}]",
                )
                all_embeds.extend(i_embeds)
                all_positions.extend(i_positions)
                current_offset += len(item) + 1  # +1 for delimiter

            if all_embeds:
                # PositionalEmbeds.__post_init__ does the single torch.cat stack.
                positional_embed_overrides = [
                    PositionalEmbeds(embeds=all_embeds, positions=all_positions)
                ]
            else:
                positional_embed_overrides = None
            return None, input_ids, positional_embed_overrides, delimiter_indices

        else:
            # Single-item scoring: process each item separately
            if item_first:
                input_ids = [item + query for item in items]
            else:
                input_ids = [query + item for item in items]

            if not has_embeds:
                return None, input_ids, None, None

            positional_embed_overrides = []
            any_overrides = False
            for i, item in enumerate(items):
                item_embs = item_embed_overrides[i] if item_embed_overrides else None
                i_embeds, i_positions = self._resolve_overrides_for_sequence(
                    item,
                    item_embs,
                    embed_override_token_id,
                    position_offset=len(query),
                    label=f"items[{i}]",
                )
                combined_embeds = q_embeds + i_embeds
                if combined_embeds:
                    positional_embed_overrides.append(
                        PositionalEmbeds(
                            embeds=combined_embeds,
                            positions=q_positions + i_positions,
                        )
                    )
                    any_overrides = True
                else:
                    positional_embed_overrides.append(None)

            return (
                None,
                input_ids,
                positional_embed_overrides if any_overrides else None,
                None,
            )

    # ------------------------------------------------------------------
    # Multi-position pooling readout (setwise scoring)
    # ------------------------------------------------------------------

    def _resolve_score_extraction_token_id(self, token: str) -> int:
        """Resolve the score-extraction token string to a dedicated token id.

        Runs in the tokenizer-manager process (which owns the tokenizer). Used by
        the HTTP score handler to turn a client-supplied token string into the id
        that ``score_request`` scans for.

        Raises:
            ValueError: if no tokenizer is available or the token does not resolve
                to a dedicated (non-unk) id.
        """
        if self.tokenizer is None:
            raise ValueError(
                "A tokenizer is required to resolve score_extraction_token."
            )
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = self.tokenizer.unk_token_id
        if token_id is None or (unk_id is not None and token_id == unk_id):
            raise ValueError(
                f"score_extraction_token {token!r} did not resolve to a dedicated "
                f"token id."
            )
        return token_id

    def _validate_score_extraction(
        self,
        is_generation: bool,
        item_first: bool,
    ) -> None:
        """Validate that multi-position pooling readout is applicable.

        Multi-position pooling (read the head at each score-extraction token) is
        supported only by models whose forward routes the head through
        ``score_and_pool`` (per-position pooling) -- the SequenceClassification
        heads in ``is_score_and_pool_model``. Generation (CausalLM), cross-encoder
        (``CrossEncodingPooler``), reward, and embedding models are rejected here,
        before inference, because they pool a single vector and would ignore the
        readout positions.

        Multiple items (candidate sets) are supported in both execution modes and
        return one score matrix per item: ``--enable-mis`` fuses them into one
        multi-item sequence (block-diagonal mask, shared query-prefix KV), while
        without MIS each item is scored as an independent ``query + item`` sequence
        in one batch. As with all MIS scoring, the fused path injects delimiter
        tokens into each item's attention context, so its scores differ slightly
        from the independent batch path (the standard MIS trade-off); pick the mode
        the deployment already uses.
        """
        if is_generation:
            raise ValueError(
                "score_extraction_token_id is only supported for "
                "SequenceClassification models, not generation (CausalLM) models."
            )
        architectures = self.model_config.hf_config.architectures or []
        if not is_score_and_pool_model(architectures):
            raise ValueError(
                "score_extraction_token_id is only supported for "
                "SequenceClassification models that pool the head per position "
                "(score_and_pool); model architecture(s) "
                f"{architectures} do not (cross-encoder, reward, and embedding "
                "models pool a single vector and ignore the readout positions)."
            )
        if item_first:
            raise ValueError(
                "item_first is not supported with score_extraction_token_id."
            )
        if not get_memory().disable_radix_cache:
            raise ValueError(
                "score_extraction_token_id requires --disable-radix-cache because "
                "pooling positions are relative to the full prompt."
            )
        if get_schedule().chunked_prefill_size != -1:
            raise ValueError(
                "score_extraction_token_id requires --chunked-prefill-size -1 "
                "because all pooling positions must be processed in one prefill."
            )
        if get_serving().allow_auto_truncate:
            raise ValueError(
                "score_extraction_token_id is not supported with "
                "--allow-auto-truncate: pooling positions are computed from the "
                "full prompt, and truncating the sequence would leave them "
                "pointing past the truncated length."
            )

    def _reject_query_prefix_anchor(
        self,
        query_token_ids: List[int],
        score_extraction_token_id: int,
    ) -> None:
        """Reject a shared query prefix that itself contains the extraction token.

        The non-MIS path scans the whole ``query + item`` sequence for anchors, so
        a query-side hit would emit an extra, misaligned score row instead of one
        row per candidate. Mirrors the MIS guard in ``_anchor_counts_per_item``
        that confines anchors to the item blocks.
        """
        if score_extraction_token_id in query_token_ids:
            raise ValueError(
                "score-extraction token found in the shared query prefix; it must "
                "appear only within items, one per candidate."
            )

    def _build_score_extraction_text_inputs(
        self,
        query: str,
        items: Union[str, List[str]],
        score_extraction_token_id: int,
    ) -> List[List[int]]:
        """Tokenize each complete query-item prompt once for extraction scanning."""
        items_list = [items] if isinstance(items, str) else items
        if not all(isinstance(item, str) for item in items_list):
            raise ValueError(
                "query and items must both be text or both be token IDs when "
                "score_extraction_token_id is set."
            )
        if query:
            self._reject_query_prefix_anchor(
                self.tokenizer.encode(query), score_extraction_token_id
            )
        return [self.tokenizer.encode(f"{query}{item}") for item in items_list]

    def _resolve_score_extraction_indices(
        self,
        input_ids: List[List[int]],
        score_extraction_token_id: int,
    ) -> List[List[int]]:
        """Scan each built sequence for the score-extraction token.

        The classification head is later pooled AT these positions (via
        ``token_indices_to_pool``) instead of the last token. Returns one list of
        positions per sequence (parallel to ``input_ids``).

        Raises:
            ValueError: if any sequence contains no extraction token (the client
                must render one per candidate; a missing token would silently
                drop that sequence's scores and lose request-to-score alignment).
        """
        token_indices_to_pool = [
            [i for i, t in enumerate(seq) if t == score_extraction_token_id]
            for seq in input_ids
        ]
        if any(not seq_indices for seq_indices in token_indices_to_pool):
            raise ValueError(
                f"No score-extraction token (id={score_extraction_token_id}) "
                "found in one or more input(s); the client must render one per "
                "candidate."
            )
        return token_indices_to_pool

    def _anchor_counts_per_item(
        self,
        delimiter_indices: List[int],
        anchor_positions: List[int],
    ) -> List[int]:
        """Count extraction-token anchors falling inside each fused item's block.

        ``--enable-mis`` fuses the items into one sequence as
        ``query <delim> item0 <delim> item1 … <delim>`` (``delimiter_indices`` has
        one entry per delimiter, so item ``i`` spans ``(delimiter_indices[i],
        delimiter_indices[i + 1])``). This buckets the fused sequence's anchor
        positions back to their items so the flat score matrix can be split per
        item.

        Raises:
            ValueError: if any item contains no anchor, or if an anchor falls
                outside every item block (e.g. in the shared query prefix). Both
                are validated here (before inference) so the client gets a clean
                400 instead of a post-forward RuntimeError.
        """
        counts = [
            sum(1 for p in anchor_positions if lo < p < hi)
            for lo, hi in zip(delimiter_indices, delimiter_indices[1:])
        ]
        # Every anchor must fall inside an item block. The extraction scan covers
        # the whole fused sequence, including the shared query prefix before the
        # first delimiter; an anchor there is not attributable to any item and
        # would leave the pooled forward with more rows than sum(counts). Reject it
        # up front rather than crashing in _process_multi_item_extraction_results.
        if sum(counts) != len(anchor_positions):
            raise ValueError(
                "score-extraction token found outside the candidate items (e.g. in "
                "the shared query prefix) under --enable-mis multi-item scoring; it "
                "must appear only within items, one per candidate."
            )
        if any(c == 0 for c in counts):
            raise ValueError(
                "Each item (candidate set) must contain at least one "
                "score-extraction token under --enable-mis multi-item scoring."
            )
        return counts

    def _resolve_multi_position_pooling(
        self,
        input_ids: List[List[int]],
        score_extraction_token_id: int,
        use_multi_item_scoring: bool,
        delimiter_indices: Optional[List[int]],
    ) -> Tuple[List[List[int]], Optional[List[int]]]:
        """Resolve where the head is pooled for a score-extraction request.

        Scans each built sequence for the score-extraction token to get the head
        pooling positions (``token_indices_to_pool``). Under ``--enable-mis`` the
        items are fused into one sequence, so it also buckets the anchor positions
        back to their items (``per_item_anchor_counts``) so the fused score matrix
        can be split per item. Returns ``(token_indices_to_pool,
        per_item_anchor_counts)`` — the latter is None outside MIS.
        """
        token_indices_to_pool = self._resolve_score_extraction_indices(
            input_ids, score_extraction_token_id
        )
        per_item_anchor_counts = None
        if use_multi_item_scoring:
            per_item_anchor_counts = self._anchor_counts_per_item(
                delimiter_indices, token_indices_to_pool[0]
            )
        return token_indices_to_pool, per_item_anchor_counts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def score_request(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        embed_override_token_id: Optional[int] = None,
        query_embed_overrides: Optional[List[torch.Tensor]] = None,
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]] = None,
        request: Optional[Any] = None,
        return_pooled_hidden_states: bool = False,
        score_extraction_token_id: Optional[int] = None,
    ) -> ScoreResult:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair.

        This method supports two scoring approaches:
        1. Single-Item scoring (default): Process each query+item pair independently
        2. Multi-Item scoring: When --enable-mis is set, combine query and
           multiple items into a single sequence using delimiter for efficient processing.
           Note: item_first parameter is ignored in multi-item scoring mode since it uses
           a fixed format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>

           Multi-item scoring works with both text and pre-tokenized inputs:
           - Text: query<delimiter_text>item1<delimiter_text>item2<delimiter_text>item3<delimiter_text>
           - Tokens: query<delimiter_token_id>item1<delimiter_token_id>item2<delimiter_token_id>item3<delimiter_token_id>

        Supports two model types:
        - Generation (CausalLM): Requires label_token_ids; returns logprob-based scores.
        - SequenceClassification: label_token_ids is optional; returns pooled class logits.

        Setwise scoring (SequenceClassification-only) is expressed via
        score_extraction_token_id: when set, the classification head is pooled AT
        every occurrence of this token in each ``query + item`` sequence (instead of
        the default last token). Each item is one candidate set, and ``scores`` is
        returned nested — one ``[Nᵢ x num_labels]`` matrix per item. Multiple items
        are supported in both execution modes: with ``--enable-mis`` the items are
        fused into one multi-item sequence (block-diagonal mask, shared query-prefix
        KV); without MIS each item is scored as an independent ``query + item``
        sequence in one batch. Incompatible with generation models.

        return_pooled_hidden_states is only supported for non-generation models
        (SequenceClassification, RewardModel); raises ValueError for CausalLM.
        """
        is_generation = self.is_generation

        if is_generation and label_token_ids is None:
            raise ValueError(
                "label_token_ids is required for generation (CausalLM) models."
            )
        if items is None:
            raise ValueError("items must be provided")
        if not items:
            return ScoreResult(scores=[], prompt_tokens=0)

        # Normalize an omitted query by item type so the input-path selection below
        # sees a concrete prefix: [] for pre-tokenized (token-ID) items so they take
        # the token-ID path, "" for text items so they take the text path. Without
        # this, token-ID items with the query omitted fall through to the text
        # builder (and None reaches the tokenizer helpers as list(None)).
        if query is None:
            items_are_token_ids = isinstance(items, list) and isinstance(items[0], list)
            query = [] if items_are_token_ids else ""

        has_embeds = (
            query_embed_overrides is not None or item_embed_overrides is not None
        )
        if has_embeds and embed_override_token_id is None:
            raise ValueError(
                "embed_override_token_id is required when query_embed_overrides "
                "or item_embed_overrides are supplied."
            )
        if item_first and has_embeds:
            raise ValueError("item_first is not supported when embeddings are supplied")
        if item_embed_overrides is not None and len(item_embed_overrides) != len(items):
            raise ValueError(
                f"item_embed_overrides length ({len(item_embed_overrides)}) "
                f"must match items length ({len(items)})."
            )
        if self.tokenizer is not None and label_token_ids is not None:
            vocab_size = self.tokenizer.vocab_size
            for token_id in label_token_ids:
                if token_id >= vocab_size:
                    raise ValueError(
                        f"Token ID {token_id} is out of vocabulary (vocab size: {vocab_size})"
                    )

        # Check if multi-item scoring is enabled
        use_multi_item_scoring = get_exec().features.enable_mis

        # Multi-position pooling readout (setwise): pool the head at every
        # occurrence of score_extraction_token_id instead of the last token.
        use_score_extraction = score_extraction_token_id is not None
        if use_score_extraction:
            self._validate_score_extraction(is_generation, item_first)

        input_ids = None
        text_prompts = None
        positional_embed_overrides = None
        delimiter_indices = None

        use_text_prompts = (
            isinstance(query, str) and not has_embeds and not use_score_extraction
        )

        if use_text_prompts:
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items
            if use_multi_item_scoring:
                # Tokenize separately, then combine at token level with placeholder
                # delimiter. Positions come from item lengths (delimiter_indices),
                # not from scanning for this token — it's for FlashInfer compat only.
                delimiter_token_id = MIS_DELIMITER_TOKEN_ID
                query_ids, items_ids = self._batch_tokenize_query_and_items(
                    query, items_list
                )
                combined_input_ids, delimiter_indices = (
                    self._build_multi_item_token_sequence(
                        query_ids, items_ids, delimiter_token_id
                    )
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
            # Both query and items are token IDs — tokenize text inputs if needed for embed overrides
            query_ids, items_ids = query, items
            if use_score_extraction and not use_multi_item_scoring:
                # MIS confines anchors to item blocks in _anchor_counts_per_item;
                # the non-MIS path concatenates query + item, so reject a
                # query-prefix anchor here (the text path does so in
                # _build_score_extraction_text_inputs).
                self._reject_query_prefix_anchor(query_ids, score_extraction_token_id)
            _, input_ids, positional_embed_overrides, delimiter_indices = (
                self._build_token_id_inputs(
                    query_ids,
                    items_ids,
                    item_first,
                    use_multi_item_scoring,
                    embed_override_token_id,
                    query_embed_overrides,
                    item_embed_overrides,
                )
            )
        elif (
            use_score_extraction
            and not use_multi_item_scoring
            and not has_embeds
            and (query is None or isinstance(query, str))
        ):
            # Single-set setwise: text (or omitted) query + text items, one
            # sequence per item (whole candidate block in one item, no shared
            # prefix when query=None). Multi-item scoring (--enable-mis) instead
            # fuses the items below via _build_token_id_inputs.
            input_ids = self._build_score_extraction_text_inputs(
                query or "", items, score_extraction_token_id
            )
        elif has_embeds or use_score_extraction:
            # Tokenize text inputs to token IDs, then build via the token-id path.
            # has_embeds needs positions to resolve embed overrides; score
            # extraction needs token IDs to scan for the extraction token. Under
            # --enable-mis this fuses the items (candidate sets) into one
            # multi-item sequence with delimiter tokens and a block-diagonal mask.
            query_ids, items_ids = self._batch_tokenize_query_and_items(
                query or "" if use_multi_item_scoring else query, items
            )
            if use_score_extraction and not use_multi_item_scoring:
                # Same query-prefix anchor guard as the pre-tokenized branch, for
                # the tokenized non-MIS path (e.g. text query + embed overrides).
                self._reject_query_prefix_anchor(query_ids, score_extraction_token_id)
            _, input_ids, positional_embed_overrides, delimiter_indices = (
                self._build_token_id_inputs(
                    query_ids,
                    items_ids,
                    item_first,
                    use_multi_item_scoring,
                    embed_override_token_id,
                    query_embed_overrides,
                    item_embed_overrides,
                )
            )
        else:
            raise ValueError(
                "Invalid combination of query/items types for score_request."
            )

        # Multi-position pooling readout: scan each built sequence for the
        # extraction token; the pooler reads the head AT those positions. Under
        # --enable-mis this also returns the per-item anchor counts used to split
        # the fused score matrix back into one matrix per item.
        token_indices_to_pool = None
        per_item_anchor_counts = None
        if use_score_extraction:
            token_indices_to_pool, per_item_anchor_counts = (
                self._resolve_multi_position_pooling(
                    input_ids,
                    score_extraction_token_id,
                    use_multi_item_scoring,
                    delimiter_indices,
                )
            )

        if return_pooled_hidden_states:
            if is_generation:
                raise ValueError(
                    "return_pooled_hidden_states is not supported for CausalLM models. "
                    "It requires a model with a task-specific head "
                    "(e.g. SequenceClassification or RewardModel)."
                )
            model_config = self.model_config
            if model_config is not None:
                archs = getattr(model_config.hf_config, "architectures", []) or []
                if is_cross_encoding_pooler_model(archs):
                    raise ValueError(
                        f"return_pooled_hidden_states is not supported for "
                        f"{archs[0]}. This model uses CrossEncodingPooler which "
                        f"does not expose pre-head hidden states."
                    )

        # Create the appropriate request type
        mis_delimiter_indices = [delimiter_indices] if use_multi_item_scoring else None
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
                positional_embed_overrides=positional_embed_overrides,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )
        else:
            batch_request = EmbeddingReqInput(
                text=text_prompts,
                input_ids=input_ids,
                positional_embed_overrides=positional_embed_overrides,
                return_pooled_hidden_states=return_pooled_hidden_states,
                multi_item_delimiter_indices=mis_delimiter_indices,
                token_indices_to_pool=token_indices_to_pool,
            )

        results = await self.generate_request(batch_request, request).__anext__()

        if use_multi_item_scoring and use_score_extraction:
            # Multi-item setwise: the items were fused into one sequence and the
            # head pooled at every anchor. Split the flat [ΣNᵢ, num_labels] result
            # back into one matrix per item (nested scores).
            return self._process_multi_item_extraction_results(
                results,
                per_item_anchor_counts,
                apply_softmax,
                return_pooled_hidden_states,
            )
        elif use_multi_item_scoring:
            # Multi-item scoring: extract scores from input_token_ids_logprobs or embedding
            return self._process_multi_item_scoring_results(
                results,
                items,
                label_token_ids,
                apply_softmax,
                batch_request,
                return_pooled_hidden_states,
            )
        else:
            # Single-item scoring: process each result separately. With
            # score_extraction_token_id, each item is scored as an independent
            # query+item sequence and its 2-D embedding ([positions x num_labels])
            # is kept as one score matrix per item (nested scores).
            return self._process_single_item_scoring_results(
                results,
                label_token_ids,
                apply_softmax,
                return_pooled_hidden_states,
                per_item_matrix=use_score_extraction,
            )

    def _convert_logprobs_to_scores(
        self,
        logprobs: Dict[int, float],
        label_token_ids: List[int],
        apply_softmax: bool,
    ) -> List[float]:
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
        """Extract logprobs for label_token_ids from (logprob, token_id, text) tuples."""
        logprobs = {}
        if logprobs_data:
            for logprob, token_id, _ in logprobs_data:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob
        return logprobs

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestValidatorConfig:
    context_len: int
    num_reserved_tokens: int
    is_generation: bool
    validate_total_tokens: bool
    allow_auto_truncate: bool
    enable_return_hidden_states: bool
    enable_custom_logit_processor: bool
    limit_mm_data_per_request: Optional[Dict[str, int]]
    is_matryoshka: bool
    matryoshka_dimensions: Optional[List[int]]
    hidden_size: int
    model_path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestValidator:
    """Request consistency / length / vocab / quota validation."""

    config: RequestValidatorConfig

    def validate_one(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_ids: List[int],
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""
        # FIXME: unify the length validation logic with the one in the scheduler.
        _max_req_len = self.config.context_len
        input_token_num = len(input_ids) if input_ids is not None else 0
        input_token_num += self.config.num_reserved_tokens

        # Validate input length
        if input_token_num >= self.config.context_len:
            if self.config.allow_auto_truncate:
                logger.warning(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.config.context_len} tokens). "
                    "Truncating the input."
                )
                del input_ids[_max_req_len:]
                input_token_num = len(input_ids)
            else:
                raise ValueError(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.config.context_len} tokens)."
                )

        # Validate total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            self.config.validate_total_tokens
            and max_new_tokens is not None
            and (max_new_tokens + input_token_num) > _max_req_len
        ):
            if self.config.allow_auto_truncate:
                logger.warning(
                    f"Requested token count ({input_token_num} input + {max_new_tokens} new) "
                    f"exceeds the model's context length ({self.config.context_len} tokens). "
                    "Truncating max_new_tokens."
                )
                obj.sampling_params["max_new_tokens"] = max(
                    0, _max_req_len - input_token_num
                )
            else:
                total_tokens = max_new_tokens + input_token_num
                error_msg = (
                    f"Requested token count exceeds the model's maximum context length "
                    f"of {self.config.context_len} tokens. You requested a total of {total_tokens} "
                    f"tokens: {input_token_num} tokens from the input messages and "
                    f"{max_new_tokens} tokens for the completion. Please reduce the number "
                    f"of tokens in the input messages or the completion to fit within the limit."
                )
                raise ValueError(error_msg)

        # Validate embedding requests
        if isinstance(obj, EmbeddingReqInput) and self.config.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        # Validate Matryoshka embeddings
        if isinstance(obj, EmbeddingReqInput):
            self._validate_for_matryoshka_dim(obj)

        # Validate custom logit processor
        if isinstance(obj, GenerateReqInput):
            if obj.return_hidden_states and not self.config.enable_return_hidden_states:
                raise ValueError(
                    "The server is not configured to return the hidden states. "
                    "Please set `--enable-return-hidden-states` to enable this feature."
                )
            if (
                obj.custom_logit_processor
                and not self.config.enable_custom_logit_processor
            ):
                raise ValueError(
                    "The server is not configured to enable custom logit processor. "
                    "Please set `--enable-custom-logit-processor` to enable this feature."
                )

    def _validate_mm_limits(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> None:
        if not self.config.limit_mm_data_per_request:
            return

        for modality, limit in self.config.limit_mm_data_per_request.items():
            data = getattr(obj, f"{modality}_data", None)
            if data:
                count = len(data) if isinstance(data, list) else 1
                if count > limit:
                    raise ValueError(
                        f"{modality.capitalize()} count {count} exceeds limit {limit} per request."
                    )

    def _validate_for_matryoshka_dim(self, obj: EmbeddingReqInput) -> None:
        """Validate the request for Matryoshka dim if it has the field set."""
        if obj.dimensions is None:
            return

        if not self.config.is_matryoshka:
            raise ValueError(
                f"Model '{self.config.model_path}' does not support matryoshka representation, "
                f"changing output dimensions will lead to poor results."
            )

        if obj.dimensions < 1:
            raise ValueError("Requested dimensions must be greater than 0")

        if (
            self.config.matryoshka_dimensions
            and obj.dimensions not in self.config.matryoshka_dimensions
        ):
            raise ValueError(
                f"Model '{self.config.model_path}' only supports {self.config.matryoshka_dimensions} matryoshka dimensions, "
                f"using other output dimensions will lead to poor results."
            )

        if obj.dimensions > self.config.hidden_size:
            raise ValueError(
                f"Provided dimensions are greater than max embedding dimension: {self.config.hidden_size}"
            )

    def validate_input_ids_in_vocab(
        self,
        input_ids: Union[List[int], List[List[int]]],
        vocab_size: int,
    ) -> None:
        if isinstance(input_ids[0], list):
            # Batch of sequences
            for seq in input_ids:
                if any(id >= vocab_size for id in seq):
                    raise ValueError(
                        f"The input_ids {seq} contains values greater than the vocab size ({vocab_size})."
                    )
        else:
            # Single sequence
            if any(id >= vocab_size for id in input_ids):
                raise ValueError(
                    f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
                )

    def validate_batch_tokenization_constraints(
        self,
        batch_size: int,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> None:
        """Validate constraints for batch tokenization processing."""
        for i in range(batch_size):
            if self.config.is_generation and obj[i].contains_mm_input():
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

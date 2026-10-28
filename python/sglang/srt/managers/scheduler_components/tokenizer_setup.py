"""Loading the scheduler's tokenizer and processors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import get_disagg, get_mm, get_model, get_serving
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
    resolve_image_processor_backend,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchedulerTokenizer:
    """The tokenizer and processors the scheduler loads; all None when it skips
    tokenizer init."""

    tokenizer: Any
    processor: Any
    # The multimodal processor for the M-RoPE fallback computation.
    mm_processor: Any


def load_tokenizer(
    *, model_config: ModelConfig, server_args: ServerArgs, skip_tokenizer_init: bool
) -> SchedulerTokenizer:
    """Load the tokenizer and processors; with a reasoning parser, record its
    think-end token ids on the model config."""

    if skip_tokenizer_init:
        tokenizer = processor = None
    else:
        if model_config.is_multimodal:
            processor = get_processor(
                get_serving().tokenizer_path,
                tokenizer_mode=get_serving().tokenizer_mode,
                trust_remote_code=get_model().trust_remote_code,
                revision=get_model().revision,
                image_processor_backend=resolve_image_processor_backend(get_mm()),
                tokenizer_backend=get_serving().tokenizer_backend,
                model_name=get_model().model_path,
            )
            tokenizer = get_tokenizer_from_processor(processor)
        else:
            processor = None
            tokenizer = get_tokenizer(
                get_serving().tokenizer_path,
                tokenizer_mode=get_serving().tokenizer_mode,
                trust_remote_code=get_model().trust_remote_code,
                revision=get_model().revision,
                tokenizer_backend=get_serving().tokenizer_backend,
            )

    # Load multimodal processor for M-RoPE fallback computation.
    mm_processor = None
    if (
        model_config.is_multimodal
        and processor is not None
        and not get_disagg().language_model_only
    ):
        try:
            import_processors("sglang.srt.multimodal.processors")
            mm_processor = get_mm_processor(
                model_config.hf_config,
                server_args,
                processor,
                "default",
                skip_mm_pool=True,
            )
        except Exception:
            logger.warning(
                "Failed to load multimodal processor in scheduler; "
                "M-RoPE fallback will not be available."
            )

    if get_serving().reasoning_parser and tokenizer:
        reasoning_parser = ReasoningParser(
            model_type=get_serving().reasoning_parser,
            stream_reasoning=False,
            tokenizer=tokenizer,
        )
        think_end_ids = reasoning_parser.detector.get_think_end_token_ids(tokenizer)
        if think_end_ids:
            model_config.think_end_ids = think_end_ids
            model_config.self_opened_think_start_ids = (
                reasoning_parser.detector.get_self_opened_think_start_token_ids(
                    tokenizer
                )
            )
        else:
            logger.warning(
                "Reasoning parser think_end_token %r could not be encoded; "
                "grammar-gated reasoning is disabled.",
                reasoning_parser.detector.think_end_token,
            )

        selectable_tokens = getattr(
            reasoning_parser.detector,
            "request_selectable_think_end_tokens",
            (),
        )
        if selectable_tokens:
            selectable_sequences = []
            for end_token in selectable_tokens:
                token_ids = tokenizer.encode(end_token, add_special_tokens=False)
                if not token_ids:
                    raise ValueError(
                        f"Request-selectable reasoning terminator {end_token!r} "
                        "could not be encoded"
                    )
                selectable_sequences.append(token_ids)
            model_config.request_selectable_think_end_id_sequences = [
                list(sequence)
                for sequence in dict.fromkeys(
                    tuple(sequence) for sequence in selectable_sequences
                )
            ]

    return SchedulerTokenizer(
        tokenizer=tokenizer, processor=processor, mm_processor=mm_processor
    )

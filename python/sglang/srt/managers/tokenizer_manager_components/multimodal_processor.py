from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)
from typing import Any, Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class MultimodalProcessorConfig:
    language_only: bool
    encoder_transfer_backend: str
    enable_adaptive_dispatch_to_encoder: bool
    encoder_dispatch_min_items: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MultimodalProcessor:
    mm_processor: Optional[Any]
    mm_receiver: Optional[Any]
    config: MultimodalProcessorConfig

    @classmethod
    def from_server_args(
        cls,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        mm_processor: Optional[Any],
    ) -> "MultimodalProcessor":
        if server_args.language_only:
            mm_receiver = create_mm_receiver(
                server_args,
                dtype=model_config.dtype,
                hf_config=model_config.hf_config,
            )
        else:
            mm_receiver = None
        return cls(
            mm_processor=mm_processor,
            mm_receiver=mm_receiver,
            config=MultimodalProcessorConfig(
                language_only=server_args.language_only,
                encoder_transfer_backend=server_args.encoder_transfer_backend,
                enable_adaptive_dispatch_to_encoder=server_args.enable_adaptive_dispatch_to_encoder,
                encoder_dispatch_min_items=envs.SGLANG_ENCODER_DISPATCH_MIN_ITEMS.get(),
            ),
        )

    def should_dispatch_to_encoder(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> bool:
        """Check if the request should be dispatched to encoder for processing.

        Returns True if the request should be dispatched to encoder (multiple multimodal items),
        False if it should be processed locally (single multimodal item or no multimodal items).

        Args:
            obj: The request input object

        Returns:
            bool: True if should dispatch to encoder, False otherwise
        """
        if obj.batch_size > 1:
            logger.warning(
                "Batch request (batch_size=%d) is not supported in EPD disaggregation mode; skipping encoder dispatch.",
                obj.batch_size,
            )
            return False
        if not isinstance(obj, GenerateReqInput) or not obj.contains_mm_input():
            return False

        # Count image / video / audio items for dispatch threshold
        def _count_mm_items(data):
            return (
                len(data) if isinstance(data, list) else (1 if data is not None else 0)
            )

        total_mm_items = (
            _count_mm_items(getattr(obj, "image_data", None))
            + _count_mm_items(getattr(obj, "video_data", None))
            + _count_mm_items(getattr(obj, "audio_data", None))
        )
        return total_mm_items >= self.config.encoder_dispatch_min_items

    def maybe_dispatch_to_encoder(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Handle EPD-disaggregation mode encoding request."""
        if isinstance(obj, GenerateReqInput) and obj.contains_mm_input():
            # dispatch to encoder by default
            should_dispatch = True
            if self.config.enable_adaptive_dispatch_to_encoder:
                should_dispatch = self.should_dispatch_to_encoder(obj)

            # Set need_wait_for_mm_inputs flag based on whether we dispatch to encoder
            # This flag will be used in _tokenize_one_request to determine processing path
            if should_dispatch:
                obj.need_wait_for_mm_inputs = True
                if self.config.encoder_transfer_backend in [
                    "zmq_to_scheduler",
                    "mooncake",
                ]:
                    self.mm_receiver.send_encode_request(obj)
            else:
                obj.need_wait_for_mm_inputs = False

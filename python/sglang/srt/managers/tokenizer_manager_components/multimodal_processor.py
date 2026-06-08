from __future__ import annotations

from dataclasses import dataclass
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

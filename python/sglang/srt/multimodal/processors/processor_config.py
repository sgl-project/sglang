from typing import Any

import msgspec


class MultimodalProcessorConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Settings a multimodal processor reads once while it initializes."""

    image_processor_backend: str = "auto"
    disable_fast_image_processor: bool = False
    mm_process_config: dict[str, Any] = {}

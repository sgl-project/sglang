from typing import Any

import msgspec


class MultimodalProcessorConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Settings a multimodal processor reads once while it initializes."""

    image_processor_backend: str = "auto"
    disable_fast_image_processor: bool = False
    mm_process_config: dict[str, Any] = {}
    mm_processor_worker_num: int = 0
    mm_io_worker_num: int = 0

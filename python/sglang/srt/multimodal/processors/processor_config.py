from typing import Any

import msgspec


class MultimodalProcessorConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Settings a multimodal processor reads once while it initializes."""

    image_processor_backend: str = "auto"
    disable_fast_image_processor: bool = False
    mm_process_config: dict[str, Any] = {}
    mm_processor_worker_num: int = 0
    mm_io_worker_num: int = 0
    allowed_media_domains: list[str] = []
    media_url_max_file_size_mb: int = 64
    # Keep preprocessed features on the GPU; serving sets it for GPU transports.
    keep_mm_features_on_device: bool = False

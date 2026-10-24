from typing import Any

import msgspec


class MultimodalProcessorConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Settings a multimodal processor reads once while it initializes."""

    image_processor_backend: str = "auto"
    disable_fast_image_processor: bool = False
    mm_process_config: dict[str, Any] = {}
    mm_processor_worker_num: int = 0
    mm_io_worker_num: int = 0
    # 0 reads SGLANG_CPU_WORKERS (default: all cores) whenever a pool is created.
    cpu_worker_num: int = 0
    # Serving always passes this (fork, or spawn for cuda_vmm); the default serves
    # trainer hosts, where CUDA is usually initialized and fork is unsafe.
    cpu_process_start_method: str = "spawn"
    allowed_media_domains: list[str] = []
    media_url_max_file_size_mb: int = 64

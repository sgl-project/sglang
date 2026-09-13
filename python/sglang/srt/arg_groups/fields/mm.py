"""Config fields of the ``mm`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``mm`` bag, which is what ``get_mm()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
import json
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)


@dataclasses.dataclass
class Mm:
    """Namespace ``mm``."""

    _NS_PATH = "mm"
    enable_multimodal: A[
        Optional[bool],
        "Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
    ] = None
    mm_attention_backend: A[
        Optional[str],
        Arg(
            help="Set multimodal attention backend.",
            choices=[
                "sdpa",
                "fa3",
                "fa4",
                "triton_attn",
                "ascend_attn",
                "aiter_attn",
                "flashinfer_cudnn",
                "amx_attn",
                "xpu_attn",
            ],
        ),
    ] = None

    # -------------------------------------------------------------------------
    # Multi-modal optimization configs
    # -------------------------------------------------------------------------
    enable_broadcast_mm_inputs_process: A[
        bool,
        "Enable broadcast mm-inputs process in scheduler.",
    ] = False
    enable_prefix_mm_cache: A[
        bool, "Enable prefix multimodal cache. Currently only supports mm-only."
    ] = False
    mm_enable_dp_encoder: A[
        bool,
        "Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically.",
    ] = False
    mm_process_config: A[
        Optional[Dict[str, Any]],
        Arg(
            help="Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio`",
            type_parser=json.loads,
        ),
    ] = None
    mm_processor_worker_num: A[
        int,
        "Number of threads for multimodal processor calls. 0 selects the "
        "model-specific default. Only processors with isolated-worker support "
        "can use more than one thread.",
    ] = 0
    mm_io_worker_num: A[
        int,
        "Number of threads for multimodal data loading and decoding. 0 selects "
        "the model-specific default. SGLANG_IO_WORKERS remains supported as an "
        "environment override when this argument is 0.",
    ] = 0
    allowed_media_domains: A[
        List[str],
        "Restrict client-supplied HTTP(S) image, video, and audio URLs to these "
        "exact hostnames. Redirect destinations are checked against the same "
        "allowlist. When unset, remote media from any domain is allowed.",
    ] = dataclasses.field(default_factory=list)
    media_url_max_file_size_mb: A[
        int,
        "Maximum size in MiB for one client-supplied remote media download. "
        "The limit is enforced while streaming; set to 0 to disable it.",
    ] = 64
    mm_preprocess_cache_size_mb: A[
        Optional[int],
        "CPU memory budget for content-addressed multimodal preprocessing "
        "artifacts. Unset selects a model-specific default (256 MiB for "
        "Kimi-K3); 0 disables the cache. The budget is divided across "
        "tokenizer workers and does not reserve GPU memory.",
    ] = None
    trust_mm_content_hashes: A[
        bool,
        "Trust caller-provided multimodal SHA-256 content hashes. This can "
        "skip reading media on a hot metadata-cache hit; only enable it when "
        "the caller guarantees that hashes identify immutable media bytes.",
    ] = False
    limit_mm_data_per_request: A[
        Optional[Union[str, Dict[str, int]]],
        Arg(
            help='Limit the number of multimodal inputs per request. e.g. \'{"image": 1, "video": 1, "audio": 1}\'',
            type_parser=json.loads,
        ),
    ] = None
    enable_mm_global_cache: A[
        bool,
        "Enable global multimodal embedding cache to skip redundant ViT inference.",
    ] = False
    image_processor_backend: A[
        Literal["auto", "torchvision", "pil"],
        "Image processor backend. 'auto' lets Transformers select the best "
        "available backend.",
    ] = "auto"
    mm_global_cache_backend: A[
        str,
        Arg(
            help="Storage backend for the multimodal global embedding cache. "
            "Used when --enable-mm-global-cache is set.",
            choices=["mooncake"],
        ),
    ] = "mooncake"
    disable_fast_image_processor: A[
        bool, "Deprecated. Use --image-processor-backend=pil instead."
    ] = False
    mm_feature_transport: A[
        Optional[Literal["cpu", "cuda_ipc", "cuda_vmm"]],
        "Transport multimodal features through CPU memory, a bounded CUDA IPC "
        "pool, or a bounded CUDA VMM pool. "
        "Unset uses cpu except for validated multi-node GB200/GB300 MNNVL models, "
        "which use cuda_vmm when an IMEX channel is available. Select cuda_ipc "
        "explicitly for single-node GPU transport. GPU transports reserve "
        "SGLANG_MM_FEATURE_CACHE_MB (default 1024 MiB) on the base GPU and fall "
        "back to CPU transport when the pool is full.",
    ] = None
    keep_mm_feature_on_device: A[
        bool,
        "Deprecated. Use --mm-feature-transport=cuda_ipc for bounded GPU-resident "
        "multimodal feature transport.",
    ] = False

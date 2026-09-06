"""Config fields of the ``disagg`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``disagg`` bag, which is what ``get_disagg()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import (
    List,
    Literal,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.arg_groups.choices import DISAGG_TRANSFER_BACKEND_CHOICES
from sglang.srt.utils.common import json_list_type


@dataclasses.dataclass
class Disagg:
    """Namespace ``disagg``."""

    _NS_PATH = "disagg"

    # Decoupled speculative decoding: draft and verify run as
    # separate engines, currently connected by a ZMQ IPC mesh.
    decoupled_spec_bind_endpoint: A[
        Optional[str],
        "ZMQ endpoint this engine binds for its inbound channel in decoupled "
        "speculative decoding (verifier: result PULL; drafter: control PULL).",
    ] = None
    decoupled_spec_connect_endpoints: A[
        Optional[List[str]],
        Arg(
            help="Peer inbound (bind) endpoints to connect to, ordered by peer "
            "rank, for decoupled speculative decoding.",
            type_parser=json_list_type,
        ),
    ] = None
    decoupled_spec_rank: A[
        Optional[int],
        "This engine's rank within its own role space (verifier-rank or "
        "drafter-rank) for decoupled speculative decoding.",
    ] = None
    decoupled_spec_role: A[
        Literal["null", "verifier", "drafter"],
        "Role in decoupled speculative decoding: 'null' disables it, 'verifier' "
        "runs the target/verify half, 'drafter' runs the draft half.",
    ] = "null"

    # -------------------------------------------------------------------------
    # PD disaggregation
    # -------------------------------------------------------------------------
    disaggregation_mode: A[
        Literal["null", "prefill", "decode"],
        'Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
    ] = "null"
    disaggregation_transfer_backend: A[
        str,
        Arg(
            help="The backend for disaggregation transfer. Default is mooncake.",
            choices=DISAGG_TRANSFER_BACKEND_CHOICES,
        ),
    ] = "mooncake"
    disaggregation_bootstrap_port: A[
        int, "Bootstrap server port on the prefill server. Default is 8998."
    ] = 8998
    disaggregation_ib_device: A[
        Optional[str],
        'The InfiniBand devices for disaggregation transfer. Supports a single device (e.g., --disaggregation-ib-device mlx5_0), a shared comma-separated list (e.g., --disaggregation-ib-device mlx5_0,mlx5_1), a per-GPU JSON mapping (e.g., --disaggregation-ib-device \'{"0": "mlx5_0,mlx5_1", "1": "mlx5_2"}\'), or a path to a JSON file containing that mapping. Default is None, which triggers automatic device detection when mooncake backend is enabled.',
    ] = None
    disaggregation_decode_enable_radix_cache: A[
        bool,
        "Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake.",
    ] = False
    disaggregation_decode_enable_offload_kvcache: A[
        bool, "Enable async KV cache offloading on decode server (PD mode)."
    ] = False
    disaggregation_decode_retraction_backup: A[
        Optional[str],
        Arg(
            help=(
                "Storage backend for KV preserved across PD decode retraction. "
                "'cpu_tensor' uses per-request CPU tensors. 'host_pool' uses "
                "a reserved HiCache pool and does not fall back on exhaustion. "
                "If omitted, the backend is inferred from the decode KV pool."
            ),
            choices=["cpu_tensor", "host_pool"],
        ),
    ] = None
    num_reserved_decode_tokens: A[
        int,
        "Number of decode tokens that will have memory reserved when adding new request to the running batch.",
    ] = 512
    disaggregation_decode_extra_slots: A[
        Optional[int],
        "Number of extra decode req_to_token slots pre-allocated for in-transfer requests (PD mode). If unset, defaults to 0 (or 2x the per-worker running batch for small batches).",
    ] = None
    disaggregation_decode_polling_interval: A[
        int,
        "The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this.",
    ] = 1
    optimistic_prefill_attempts: A[
        int, "Number of optimistic prefill forward passes that skip the bootstrap wait."
    ] = 0

    # -------------------------------------------------------------------------
    # Encode prefill disaggregation
    # -------------------------------------------------------------------------
    encoder_only: A[
        bool,
        "For MLLM with an encoder, launch an encoder-only server",
    ] = False
    language_only: A[
        bool,
        "For VLM, load weights for the language model only.",
    ] = False
    language_model_only: A[
        bool,
        "Skip the multimodal encoder entirely: its weights are never loaded and the "
        "tower is never built, freeing that GPU memory for KV cache. Multimodal "
        "requests are rejected. Unlike --language-only this is a standalone mode, "
        "not part of encoder/decoder disaggregation.",
    ] = False
    encoder_transfer_backend: A[
        str,
        Arg(
            help="The backend for encoder disaggregation transfer. Auto selects a model- and TP-aware backend.",
            choices=["auto", "zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"],
        ),
    ] = "auto"
    encoder_urls: A[List[str], "List of encoder server urls."] = dataclasses.field(
        default_factory=list
    )
    encoder_bootstrap_port: A[
        int,
        "Port for the EncoderBootstrapServer that runs in the language-only tokenizer manager process. Encoders register here, and language-only receivers fetch the current URL list from here.",
    ] = 8997
    encoder_register_urls: A[
        List[str],
        "One or more EncoderBootstrapServer URLs to register this encoder with on startup, for dynamic encoder discovery. Example: --encoder-register-urls http://prefill0:8997 http://prefill1:8997. Used with --encoder-only servers.",
    ] = dataclasses.field(default_factory=list)
    enable_adaptive_dispatch_to_encoder: A[
        bool,
        "When enabled, adaptively dispatch: multi-image requests go to encoder in language_only epd mode, single-image requests are processed locally.",
    ] = False

    # -------------------------------------------------------------------------
    # PD-Multiplexing
    # -------------------------------------------------------------------------
    enable_pdmux: A[
        bool,
        "Enable PD-Multiplexing, PD running on greenctx stream.",
    ] = False
    pdmux_config_path: A[
        Optional[str],
        "The path of the PD-Multiplexing config file.",
    ] = None
    sm_group_num: A[int, "Number of sm partition groups."] = 8

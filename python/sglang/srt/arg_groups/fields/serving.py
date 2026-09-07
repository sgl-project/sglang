"""Config fields of the ``serving`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``serving`` bag, which is what ``get_serving()`` returns, so a reader
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
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.utils.common import json_list_type


@dataclasses.dataclass
class Serving:
    """Namespace ``serving``."""

    _NS_PATH = "serving"
    tokenizer_path: A[Optional[str], "The path of the tokenizer."] = None
    tokenizer_mode: A[
        str,
        Arg(
            help="Tokenizer mode. 'auto' will use the fast tokenizer if available, "
            "and 'slow' will always use the slow tokenizer.",
            choices=["auto", "slow"],
        ),
    ] = "auto"
    tokenizer_backend: A[
        str,
        Arg(
            help="Tokenizer backend. 'huggingface' uses the default HuggingFace "
            "tokenizers library, and 'fastokens' uses the fastokens library "
            "for faster tokenization. Requires the fastokens package to be installed.",
            choices=["huggingface", "fastokens"],
        ),
    ] = "huggingface"
    tokenizer_worker_num: A[
        int,
        "The worker num of the tokenizer manager.",
    ] = 1
    detokenizer_worker_num: A[
        int,
        "The worker num of the detokenizer manager.",
    ] = 1
    skip_tokenizer_init: A[
        bool, "If set, skip init tokenizer and pass input_ids in generate request."
    ] = False

    # -------------------------------------------------------------------------
    # HTTP server
    # -------------------------------------------------------------------------
    host: A[str, "The host of the HTTP server."] = "127.0.0.1"
    port: A[int, "The port of the HTTP server."] = 30000
    fastapi_root_path: A[
        str,
        "App is behind a path based routing proxy.",
    ] = ""
    smg_grpc_mode: A[
        bool,
        "Use the legacy SMG gRPC server (smg-grpc-servicer) instead of the HTTP "
        "server. Replaces the deprecated --grpc-mode.",
    ] = False
    grpc_mode: A[
        bool, "(Deprecated, use --smg-grpc-mode) Legacy SMG gRPC server selector."
    ] = False
    grpc_port: A[
        Optional[int],
        "Port for the native gRPC server, started alongside HTTP. Setting this "
        "(or SGLANG_GRPC_PORT) enables the native gRPC server; it is off by "
        "default. In legacy --smg-grpc-mode this is the SMG server port and "
        "defaults to --port + 10000.",
    ] = None
    # Env-only (SGLANG_GRPC_WORKER_THREADS); a field so the projection sees it.
    grpc_worker_threads: A[Optional[int], Arg(no_cli=True)] = None
    sidecar: A[
        Optional[str],
        "Start a locally managed sidecar against the native gRPC server. "
        "The selected module must expose main(argv) and read the resolved "
        "native gRPC endpoint from SGLANG_GRPC_ENDPOINT. Requires --grpc-port "
        "or SGLANG_GRPC_PORT.",
    ] = None
    sidecar_args: A[
        Optional[List[str]],
        Arg(
            help="JSON array passed to the selected sidecar module's "
            "main(argv) function. --sidecar-shutdown-timeout SECONDS is "
            "consumed by SGLang.",
            type_parser=json_list_type,
        ),
    ] = None
    skip_server_warmup: A[bool, "If set, skip warmup."] = False
    warmups: A[
        Optional[str],
        "Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
    ] = None
    enable_http2: A[
        bool,
        "Use Granian instead of Uvicorn as the ASGI server, enabling HTTP/1.1 and HTTP/2 auto-negotiation. Clients may use h2c (cleartext HTTP/2) or plain HTTP/1.1. Requires 'pip install sglang[http2]'.",
    ] = False
    http2_max_concurrent_streams: A[
        int,
        "Maximum number of concurrent streams advertised on each HTTP/2 "
        "connection (1 to 2^32 - 1). Only applies with --enable-http2.",
    ] = 200
    http2_initial_connection_window_size: A[
        int,
        "Initial connection-level HTTP/2 receive window in bytes (1024 to "
        "2^31 - 1). Only applies with --enable-http2.",
    ] = 1024 * 1024

    # -------------------------------------------------------------------------
    # SSL/TLS
    # -------------------------------------------------------------------------
    ssl_keyfile: A[
        Optional[str],
        "The file path to the SSL key file.",
    ] = None
    ssl_certfile: A[
        Optional[str],
        "The file path to the SSL certificate file.",
    ] = None
    ssl_ca_certs: A[Optional[str], "The CA certificates file."] = None
    ssl_keyfile_password: A[
        Optional[str],
        "The password to decrypt the SSL keyfile.",
    ] = None
    enable_ssl_refresh: A[
        bool,
        "Enable automatic SSL certificate hot-reloading when cert/key files change on disk. Requires --ssl-certfile and --ssl-keyfile.",
    ] = False

    # -------------------------------------------------------------------------
    # API related
    # -------------------------------------------------------------------------
    api_key: A[
        Optional[str],
        "Set API key of the server. It is also used in the OpenAI API compatible server.",
    ] = None
    admin_api_key: A[
        Optional[str],
        "Set admin API key for sensitive management endpoints (e.g. /clear_hicache_storage_backend). When set, admin endpoints require this key and do NOT accept --api-key.",
    ] = None
    served_model_name: A[
        Optional[str],
        "Override the model name returned by the v1/models endpoint in OpenAI API server.",
    ] = None
    weight_version: A[
        str,
        "Version identifier for the model weights. Defaults to 'default' if not specified.",
    ] = "default"
    chat_template: A[
        Optional[str],
        "The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
    ] = None
    hf_chat_template_name: A[
        Optional[str],
        "When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used.",
    ] = None
    completion_template: A[
        Optional[str],
        "The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
    ] = None
    file_storage_path: A[
        str,
        "The path of the file storage in backend.",
    ] = "sglang_storage"
    enable_cache_report: A[
        bool,
        "Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
    ] = False
    return_input_ids: A[
        bool,
        "Return prompt (input) token ids on the response-level sglext extension for every chat completion request, as if return_input_ids_in_sglext were set on the request.",
    ] = False
    return_output_ids: A[
        bool,
        "Return sampled output token ids on the response-level sglext extension for every chat completion request, as if return_output_ids_in_sglext were set on the request.",
    ] = False
    reasoning_parser: Optional[str] = None
    default_chat_template_kwargs: A[
        Optional[Dict[str, Any]],
        Arg(
            help="Default chat template kwargs applied to every request when not "
            "overridden per-request. Keys must match what the model's chat template "
            "expects (e.g. enable_thinking, thinking, reasoning_effort). Per-request "
            "chat_template_kwargs takes precedence.",
            type_parser=json.loads,
        ),
    ] = None
    strip_thinking_cache: A[
        bool,
        "Skip caching reasoning-model output (thinking + answer) in the radix tree on finish; keep only the prompt prefix. Opt-in: changes cache contents.",
    ] = False
    enable_strict_thinking: A[
        bool,
        "Enable strict token filtering during the thinking phase. Blocks model-specific excluded tokens (e.g., tool call markers) during reasoning. Requires a grammar backend that supports token filtering.",
    ] = False
    tool_call_parser: Optional[str] = None
    tool_server: A[
        Optional[str],
        "Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
    ] = None
    sampling_defaults: A[
        str,
        Arg(
            help="Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'.",
            choices=["openai", "model"],
        ),
    ] = "model"
    asr_max_buffer_seconds: A[
        int,
        "Maximum seconds of PCM audio the streaming ASR WebSocket handler will accumulate before closing the session with a buffer_overflow error. Guards against OOM when a client streams audio faster than inference can consume it. Default 60s.",
    ] = 60
    asr_max_concurrent_sessions: A[
        int,
        "Maximum number of concurrent realtime ASR WebSocket sessions served by /v1/realtime. New connections beyond this cap are accepted, sent an error{code:too_many_sessions} frame, and closed. Default 32.",
    ] = 32
    preferred_sampling_params: A[
        Optional[str],
        Arg(
            help="json-formatted sampling settings that will be returned in /get_model_info",
            type_parser=json.loads,
        ),
    ] = None
    allow_auto_truncate: A[
        bool,
        "Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
    ] = False

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------
    stream_interval: A[
        int,
        "The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
    ] = 1
    batch_notify_size: A[
        int,
        "Number of streaming notifications to batch before yielding to the event loop. Reduces asyncio wakeup overhead under high concurrency.",
    ] = 16
    stream_response_default_include_usage: A[
        bool,
        "Include usage in every streaming response (even when stream_options is not specified).",
    ] = False
    incremental_streaming_output: A[
        bool,
        "Whether to output as a sequence of disjoint segments.",
    ] = False
    enable_streaming_session: A[
        bool, "Enable streaming session mode and StreamingSession wrapper."
    ] = False

    # -------------------------------------------------------------------------
    # Constrained decoding
    # -------------------------------------------------------------------------
    constrained_json_whitespace_pattern: A[
        Optional[str],
        "(outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
    ] = None
    constrained_json_disable_any_whitespace: A[
        bool,
        "(xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output.",
    ] = False

    # -------------------------------------------------------------------------
    # Dynamic batch tokenizer
    # -------------------------------------------------------------------------
    enable_dynamic_batch_tokenizer: A[
        bool,
        "Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.",
    ] = False
    dynamic_batch_tokenizer_batch_size: A[
        int,
        "[Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.",
    ] = 32
    dynamic_batch_tokenizer_batch_timeout: A[
        float,
        "[Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.",
    ] = 0.002
    enable_tokenizer_batch_encode: A[
        bool,
        "Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
    ] = False
    disable_tokenizer_batch_decode: A[
        bool, "Disable batch decoding when decoding multiple completions."
    ] = False

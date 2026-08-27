"""Launch the standalone Rust OpenAI frontend."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path


def render(args, extra_argv):
    del args
    from sglang.cli.serve import normalize_positional_model_path
    from sglang.srt.managers.utils import compute_num_reserved_tokens
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    if any(arg in ("-h", "--help") for arg in extra_argv):
        print(
            "Usage: sglang render MODEL --engine-url URL [server arguments]\n\n"
            "Launch the standalone Rust OpenAI frontend. The renderer owns "
            "Chat, Completion, tokenization, and OpenAI response formatting; "
            "URL must point to a token-only SGLang /generate server."
        )
        return

    engine_url, server_argv = extract_engine_url(extra_argv)
    load_plugins()
    normalized_argv, _ = normalize_positional_model_path(server_argv)
    server_args = prepare_server_args(normalized_argv)
    if server_args.skip_tokenizer_init:
        raise ValueError("sglang render requires a tokenizer")
    if server_args.preferred_sampling_params:
        raise ValueError(
            "sglang render does not yet apply --preferred-sampling-params; "
            "send those values in each request"
        )
    if server_args.api_key:
        raise ValueError("sglang render does not yet implement --api-key")
    if server_args.ssl_keyfile or server_args.ssl_certfile:
        raise ValueError("sglang render does not yet implement TLS")

    model_config = server_args.get_model_config()
    defaults = model_config.get_default_sampling_params()
    config = {
        "http_addr": f"{server_args.host}:{server_args.port}",
        "http_workers": 2,
        "tokenizer_workers": server_args.tokenizer_worker_num,
        "queue_capacity": 128,
        "engine_url": engine_url,
        "renderer": {
            "served_model_name": server_args.served_model_name,
            "tokenizer_path": server_args.tokenizer_path,
            "revision": server_args.revision,
            "model_path": server_args.model_path,
            "chat_template": server_args.chat_template,
            "tool_call_parser": server_args.tool_call_parser,
            "reasoning_parser": server_args.reasoning_parser,
            "stream_response_default_include_usage": (
                server_args.stream_response_default_include_usage
            ),
            "skip_tokenizer_init": False,
            "vocab_size": model_config.vocab_size,
            "default_sampling_params": {
                "temperature": defaults.get("temperature"),
                "top_p": defaults.get("top_p"),
            },
            "limits": {
                "skip_tokenizer_init": False,
                "vocab_size": model_config.vocab_size,
                "context_len": model_config.context_len,
                "num_reserved_tokens": compute_num_reserved_tokens(),
                "allow_auto_truncate": server_args.allow_auto_truncate,
                "enable_return_hidden_states": (
                    server_args.enable_return_hidden_states
                ),
            },
        },
    }

    binary = os.environ.get("SGLANG_RENDERER_BIN") or shutil.which(
        "sglang-renderer"
    )
    if binary is None:
        raise RuntimeError(
            "sglang-renderer executable was not found; install a wheel built "
            "with the Rust renderer or set SGLANG_RENDERER_BIN"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_path = write_renderer_config(config)
    process = subprocess.Popen([binary, "--config", str(config_path)])
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        return_code = process.wait()
    finally:
        config_path.unlink(missing_ok=True)
    if return_code:
        raise SystemExit(return_code)


def extract_engine_url(argv):
    remaining = []
    engine_url = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--engine-url":
            if engine_url is not None or index + 1 >= len(argv):
                raise ValueError("--engine-url must be provided exactly once")
            engine_url = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--engine-url="):
            if engine_url is not None:
                raise ValueError("--engine-url must be provided exactly once")
            engine_url = arg.partition("=")[2]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    if not engine_url:
        raise ValueError("sglang render requires --engine-url URL")
    return engine_url, remaining


def write_renderer_config(config):
    fd, path = tempfile.mkstemp(prefix="sglang-renderer-", suffix=".json")
    config_path = Path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(config, file)
    except BaseException:
        config_path.unlink(missing_ok=True)
        raise
    return config_path

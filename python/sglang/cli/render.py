"""Standalone text request preprocessing server."""

import os
import threading


def render(args, extra_argv):
    del args
    from sglang.cli.serve import _normalize_positional_model_path
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    if any(arg in ("-h", "--help") for arg in extra_argv):
        print(
            "Usage: sglang render <model-name-or-path> [server arguments]\n\n"
            "Launch the text-only, GPU-less Rust preprocessing service. Only "
            "/v1/chat/completions/render and /v1/completions/render are served."
        )
        try:
            prepare_server_args(["--help"])
        except SystemExit:
            pass
        return

    load_plugins()
    normalized_argv, _ = _normalize_positional_model_path(extra_argv)
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

    from sglang.srt.managers.rust_server import build_rust_server_args
    from sglang.srt.rust_extensions import load_rust_extension

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_config = server_args.get_model_config()

    Renderer = load_rust_extension("sglang.srt.rust_extensions._server").Renderer
    renderer = Renderer(
        server_args=build_rust_server_args(server_args, model_config),
        http_addr=f"{server_args.host}:{server_args.port}",
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        renderer.shutdown()

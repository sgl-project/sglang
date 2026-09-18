# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from sglang.cli.serve_backends import (
    SERVE_BACKEND_API_VERSION,
    ServeBackend,
    ServeBackendDetection,
    ServeBackendRegistry,
    ServeRequest,
)
from sglang.cli.utils import get_is_diffusion_model, get_model_path, try_get_model_path
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

logger = logging.getLogger(__name__)


def _extract_model_type_override(extra_argv):
    """Extract and remove the backend selector from argv.

    Validation is deferred to :class:`ServeBackendRegistry` because installed
    out-of-tree entry points dynamically extend the accepted values.
    """

    backend_name = "auto"
    filtered_argv = []
    i = 0
    while i < len(extra_argv):
        arg = extra_argv[i]
        if arg == "--model-type":
            if i + 1 >= len(extra_argv):
                raise ValueError("Error: --model-type requires a value.")
            backend_name = extra_argv[i + 1]
            i += 2
            continue

        if arg.startswith("--model-type="):
            backend_name = arg.split("=", 1)[1]
            i += 1
            continue

        filtered_argv.append(arg)
        i += 1

    if not backend_name:
        raise ValueError("Error: --model-type requires a non-empty value.")
    return backend_name, filtered_argv


def _normalize_positional_model_path(extra_argv):
    """Allow `sglang serve <model>` while preserving existing flag parsing."""
    if extra_argv and not extra_argv[0].startswith("-"):
        return ["--model-path", extra_argv[0], *extra_argv[1:]], True
    return extra_argv, False


def _print_llm_help(_request: ServeRequest) -> None:
    from sglang.srt.server_args import prepare_server_args

    try:
        prepare_server_args(["--help"])
    except SystemExit:
        pass  # argparse --help calls sys.exit


def _print_diffusion_help(_request: ServeRequest) -> None:
    try:
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
        )

        parser = argparse.ArgumentParser(
            prog="sglang serve",
            description="SGLang Diffusion Model Serving",
        )
        add_multimodal_gen_serve_args(parser)
        parser.print_help()
    except ImportError:
        print(
            "Diffusion model support is not available. "
            'Install with: pip install "sglang[diffusion]"'
        )


def _run_llm(request: ServeRequest) -> None:
    if any(arg in request.argv for arg in ("-h", "--help")):
        _print_llm_help(request)
        return

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(list(request.argv))
    run_server(server_args)


def _detect_diffusion(request: ServeRequest) -> ServeBackendDetection:
    if request.model_path is None:
        return ServeBackendDetection.UNKNOWN
    if get_is_diffusion_model(request.model_path):
        return ServeBackendDetection.MATCH
    return ServeBackendDetection.NO_MATCH


def _run_diffusion(request: ServeRequest) -> None:
    if any(arg in request.argv for arg in ("-h", "--help")):
        _print_diffusion_help(request)
        return

    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
        add_multimodal_gen_serve_args,
        execute_serve_cmd,
    )

    parser = argparse.ArgumentParser(description="SGLang Diffusion Model Serving")
    add_multimodal_gen_serve_args(parser)
    parsed_args, remaining_argv = parser.parse_known_args(list(request.argv))
    if request.model_path_is_positional:
        parsed_args._sglang_explicit_arg_names = {"model_path"}

    execute_serve_cmd(parsed_args, remaining_argv)


def _create_backend_registry() -> ServeBackendRegistry:
    return ServeBackendRegistry(
        {
            "llm": ServeBackend(
                api_version=SERVE_BACKEND_API_VERSION,
                run=_run_llm,
            ),
            "diffusion": ServeBackend(
                api_version=SERVE_BACKEND_API_VERSION,
                run=_run_diffusion,
                detect=_detect_diffusion,
            ),
        }
    )


def _print_general_help(registry: ServeBackendRegistry) -> None:
    available = ",".join(("auto", *registry.available_names))
    print(
        "Usage: sglang serve <model-name-or-path> [additional-arguments]\n"
        "   or: sglang serve --model-path <model-name-or-path> "
        "[additional-arguments]\n\n"
        "The serving backend is detected from the model by default. Installed "
        "out-of-tree projects can add more backends.\n"
        f"Optional override: --model-type {{{available}}} "
        "(default: auto, fallback to LLM when no backend matches).\n"
        "Use `sglang serve --model-type BACKEND --help` for backend-specific "
        "help."
    )

    print("\n--- Help for Standard Language Model Server ---")
    _print_llm_help(ServeRequest(argv=("--help",), model_path=None))

    print("\n--- Help for Diffusion Model Server ---")
    _print_diffusion_help(ServeRequest(argv=("--help",), model_path=None))


def serve(args, extra_argv):
    del args  # The top-level parser currently has no serve-owned fields.

    backend_name, dispatch_argv = _extract_model_type_override(extra_argv)
    dispatch_argv, positional_model_path = _normalize_positional_model_path(
        dispatch_argv
    )
    request = ServeRequest(
        argv=tuple(dispatch_argv),
        model_path=try_get_model_path(dispatch_argv),
        model_path_is_positional=positional_model_path,
    )
    registry = _create_backend_registry()

    if any(h in request.argv for h in ("-h", "--help")):
        if backend_name == "auto":
            _print_general_help(registry)
        else:
            registry.get(backend_name).backend.run(request)
        return

    from sglang.srt.plugins import load_plugins

    load_plugins()

    try:
        if backend_name == "auto":
            registered = registry.auto_detect(request)
            logger.info("Selected serve backend %r", registered.name)
        else:
            registered = registry.get(backend_name)
            logger.info(
                "Dispatch override enabled: --model-type=%s (skip auto detection)",
                backend_name,
            )

        if registered.backend.requires_model_path and request.model_path is None:
            get_model_path(request.argv)  # Raise the existing actionable CLI error.

        registered.backend.run(request)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

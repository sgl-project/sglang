"""Early compatibility checks for the SGLang surfaces used by the simulator."""

import inspect
from importlib import metadata

SIMULATOR_SERVER_ARG_OVERRIDES = {
    # The simulator models the target deployment topology separately through
    # sim_config.scheduler. Keep the SGLang runtime single-process so host-side
    # simulator work is not multiplied by the modeled parallel world size.
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "attn_cp_size": 1,
    "dcp_size": 1,
    "disable_overlap_schedule": True,
    "disable_cuda_graph": True,
    "attention_backend": "torch_native",
    "prefill_attention_backend": "torch_native",
    "decode_attention_backend": "torch_native",
}


class SGLangCompatibilityError(RuntimeError):
    pass


def _sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return "source-checkout"


def _require_parameters(function, required: set[str], surface: str) -> None:
    parameters = set(inspect.signature(function).parameters)
    missing = required - parameters
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing {surface} parameters: "
            f"{', '.join(sorted(missing))}. The simulator must be adapted to "
            "this SGLang revision before it can run."
        )


def apply_simulator_server_args(target) -> None:
    """Apply simulator-owned values before constructing the final ServerArgs."""
    if isinstance(target, dict):
        target.update(SIMULATOR_SERVER_ARG_OVERRIDES)
        return

    for name, value in SIMULATOR_SERVER_ARG_OVERRIDES.items():
        setattr(target, name, value)


def validate_simulator_server_args(server_args) -> None:
    """Fail early if a process bypassed a simulator-owned launch entry point."""
    mismatches = [
        f"{name}={getattr(server_args, name, None)!r} (expected {expected!r})"
        for name, expected in SIMULATOR_SERVER_ARG_OVERRIDES.items()
        if getattr(server_args, name, None) != expected
    ]
    if mismatches:
        raise SGLangCompatibilityError(
            "SGLang Simulator server arguments were not prepared by a supported "
            f"entry point: {', '.join(mismatches)}"
        )


def validate_launch_runtime() -> None:
    from sglang.srt.entrypoints.http_server import launch_server

    _require_parameters(
        launch_server,
        {"server_args", "run_scheduler_process_func", "run_detokenizer_process_func"},
        "launch_server",
    )


def validate_benchmark_runtime() -> None:
    from sglang.benchmark import serving

    missing = [
        name
        for name in (
            "BenchmarkMetrics",
            "calculate_metrics",
            "cli_main",
            "get_request",
            "run_benchmark",
        )
        if not hasattr(serving, name)
    ]
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing benchmark surfaces: "
            f"{', '.join(missing)}. The simulator must be adapted to this "
            "SGLang revision before it can run."
        )


def request_prefix_length(req) -> int:
    """Read the cached-prefix representation used by the installed Req API.

    Newer Req exposes a scalar; earlier Req stores the cached token indices.
    Neither field is a default for the full input length or a cost fallback.
    """
    value = getattr(req, "prefix_len", None)
    if value is not None:
        if type(value) is not int or value < 0:
            raise SGLangCompatibilityError(
                "Req.prefix_len must be a nonnegative integer"
            )
        return value
    indices = getattr(req, "prefix_indices", None)
    if indices is not None:
        return len(indices)
    raise SGLangCompatibilityError(
        "Req has neither prefix_len nor prefix_indices; adapt the Simulator to this Req API"
    )


def request_extend_length(req) -> int:
    """Read the actual prefill token count from a supported Req representation.

    The Range API exposes ``extend_range.length``; its start/end pair is not
    itself a two-token sequence. Missing metadata is never replaced by zero.
    """
    for name in ("extend_input_len", "extend_len"):
        value = getattr(req, name, None)
        if value is not None:
            if type(value) is not int or value < 0:
                raise SGLangCompatibilityError(
                    f"Req.{name} must be a nonnegative integer"
                )
            return value
    span = getattr(req, "extend_range", None)
    value = getattr(span, "length", None)
    if type(value) is int and value >= 0:
        return value
    raise SGLangCompatibilityError(
        "Req has no valid extend_input_len, extend_len or extend_range.length; "
        "adapt the Simulator to this Req API"
    )

"""Early compatibility checks for the SGLang surfaces used by the simulator."""

import inspect
import ipaddress
import json
import os
from importlib import metadata
from urllib.parse import urlsplit

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


def _native_predictor_config(target=None):
    """Read the explicit predictor selection before native argument resolution.

    A CLI config path is established here because ServerArgs resolution invokes
    the CPU DSPark handler before launch_server's later config logging.
    """
    path = os.getenv("SGLANG_SIMULATOR_CONFIG_PATH")
    if not path or not os.path.isfile(path):
        candidate = (
            target.get("sim_config_path")
            if isinstance(target, dict)
            else getattr(target, "sim_config_path", None)
        )
        if candidate:
            path = os.path.abspath(candidate)
            os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = path
    if not path:
        return {}
    with open(path, encoding="utf-8") as stream:
        return json.load(stream).get("predictor", {})


def _joint_reference_requested(predictor) -> bool:
    # This selects an entry path, not a source attestation. The existing
    # predictor constructor validates every original asset before any forward.
    return (
        predictor.get("name") == "infercast_dspark"
        and type(predictor.get("native_joint_source_binding")) is dict
        and predictor.get("cost_mode") == "HYBRID"
        and predictor.get("progress_kind") == "source_golden351"
        and predictor.get("native_analytic_model") is not None
    )


def native_joint_reference_requested(target=None) -> bool:
    """Select only the explicit two-singleton joint source entry domain."""
    return _joint_reference_requested(_native_predictor_config(target))


def native_reference_requested(target=None) -> bool:
    predictor = _native_predictor_config(target)
    return (
        predictor.get("name") == "infercast_dspark"
        and predictor.get("native_segment_profile_id") is not None
    ) or _joint_reference_requested(predictor)


def _simulator_overrides(target=None):
    overrides = SIMULATOR_SERVER_ARG_OVERRIDES
    if native_joint_reference_requested(target):
        return {
            **overrides,
            "tp_size": 2,
            "attn_dp_size": 2,
            "nnodes": 1,
            "disable_overlap_schedule": False,
        }
    if native_reference_requested(target):
        overrides = {**overrides, "disable_overlap_schedule": False}
    return overrides


def validate_coordinator_endpoint(endpoint: str) -> str:
    """Limit the initial single-driver protocol to explicit loopback TCP."""
    try:
        parsed = urlsplit(endpoint)
        address = ipaddress.ip_address(parsed.hostname or "")
        valid = (
            parsed.scheme == "tcp"
            and address.is_loopback
            and parsed.port is not None
            and 0 < parsed.port < 65536
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("coordinator endpoint must be tcp://<loopback IP>:<port>")
    return endpoint


def _validate_coordinator_server_args(args, endpoint: str) -> None:
    validate_coordinator_endpoint(endpoint)
    if os.getenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE").upper() != "OFFLINE":
        raise ValueError("coordinator requires OFFLINE")
    for name in (
        "enable_hierarchical_cache",
        "enable_lora",
    ):
        if getattr(args, name, None):
            raise ValueError(f"coordinator does not support {name}")
    if getattr(
        args, "enable_dp_attention", None
    ) and not native_joint_reference_requested(args):
        raise ValueError("coordinator does not support enable_dp_attention")
    algorithm = getattr(args, "speculative_algorithm", None)
    if algorithm and not (algorithm == "DSPARK" and native_reference_requested(args)):
        raise ValueError("coordinator requires source-bound native DSPark speculation")
    interval = getattr(args, "prefill_decode_interval", None)
    if interval is not None and (type(interval) is not int or interval < 0):
        raise ValueError(
            "coordinator requires a nonnegative integer prefill_decode_interval"
        )
    if getattr(args, "disaggregation_mode", "null") != "null":
        raise ValueError("coordinator does not support disaggregation")
    if not getattr(args, "skip_server_warmup", False):
        raise ValueError("coordinator requires skip_server_warmup")
    if not getattr(args, "enable_cache_report", False):
        raise ValueError("coordinator requires enable_cache_report")
    for name in ("SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT"):
        if float(os.getenv(name, "-1")) > 0:
            raise ValueError(
                f"coordinator requires a virtual deadline instead of {name}"
            )


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
    overrides = _simulator_overrides(target)
    if os.getenv("SGLANG_SIMULATOR_COORDINATOR"):
        overrides = {
            **overrides,
            "skip_server_warmup": True,
            "enable_cache_report": True,
        }
    if isinstance(target, dict):
        target.update(overrides)
        return

    for name, value in overrides.items():
        setattr(target, name, value)


def resolve_simulator_server_args(server_args) -> None:
    """Run native deferred resolution before launching or reading declarations.

    Native ServerArgs construction preserves raw input. Its resolve_once method
    is the owner of model defaults, speculative handlers and resolved overlays.
    Never write those decisions into the raw record in the simulator.
    """
    resolve = getattr(server_args, "resolve_once", None)
    if not callable(resolve):
        raise SGLangCompatibilityError(
            "SGLang Simulator requires native ServerArgs.resolve_once"
        )
    resolve()


def validate_simulator_server_args(server_args) -> None:
    """Fail early if a process bypassed a simulator-owned launch entry point."""
    if endpoint := os.getenv("SGLANG_SIMULATOR_COORDINATOR"):
        _validate_coordinator_server_args(server_args, endpoint)
    mismatches = [
        f"{name}={getattr(server_args, name, None)!r} (expected {expected!r})"
        for name, expected in _simulator_overrides(server_args).items()
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

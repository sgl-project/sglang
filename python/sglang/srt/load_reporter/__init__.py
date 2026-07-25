"""Embedded SGLang load reporter."""

from typing import Any, Optional

__all__ = ["LoadReporterRuntime", "describe_optional_dependency_error"]


def describe_optional_dependency_error(exc: BaseException) -> Optional[str]:
    """Describe a missing or incompatible optional reporter dependency.

    Args:
        exc: Exception raised while importing the load-reporter runtime.

    Returns:
        A user-facing dependency error, or ``None`` when the exception is not
        caused by the optional gRPC/protobuf dependency boundary.
    """
    if isinstance(exc, ModuleNotFoundError):
        root_name = (exc.name or "").split(".", 1)[0]
        if root_name in {"google", "grpc"}:
            return (
                "load reporting requires grpcio>=1.78.0 and "
                "protobuf>=6.31.1 in the runtime image"
            )
        return None

    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        is_grpc_version_error = (
            "grpc" in message and "generated" in message and "version" in message
        )
        is_protobuf_version_error = (
            "protobuf" in message
            and "version" in message
            and ("gencode" in message or "runtime" in message)
        )
        if is_grpc_version_error or is_protobuf_version_error:
            return (
                "load reporting requires grpcio>=1.78.0 and "
                "protobuf>=6.31.1 in the runtime image"
            )
    return None


def __getattr__(name: str) -> Any:
    """Lazily expose runtime classes without importing optional gRPC packages.

    Args:
        name: Attribute requested from the package.

    Returns:
        The requested load-reporter public object.

    Raises:
        AttributeError: If the attribute is not part of the public API.
        ModuleNotFoundError: If the runtime is requested without its optional
            gRPC/protobuf dependencies installed.
    """
    if name == "LoadReporterRuntime":
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        return LoadReporterRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

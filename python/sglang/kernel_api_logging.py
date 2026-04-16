"""Kernel API crash debugging helpers for SGLang.

This module was developed with reference to FlashInfer's kernel API logging utility:
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/api_logging.py
"""

from __future__ import annotations

import fnmatch
import functools
import inspect
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch


def _substitute_process_id(path: str) -> str:
    if "%i" in path:
        return path.replace("%i", str(os.getpid()))
    return path


_KERNEL_API_LOG_LEVEL = int(os.environ.get("SGLANG_KERNEL_API_LOGLEVEL", "0"))
_KERNEL_API_LOG_DEST = _substitute_process_id(
    os.environ.get("SGLANG_KERNEL_API_LOGDEST", "stdout")
)
_DUMP_DIR = Path(
    _substitute_process_id(
        os.environ.get("SGLANG_KERNEL_API_DUMP_DIR", "sglang_kernel_api_dumps")
    )
)
_DUMP_INCLUDE_PATTERNS = [
    p.strip()
    for p in os.environ.get("SGLANG_KERNEL_API_DUMP_INCLUDE", "").split(",")
    if p.strip()
]
_DUMP_EXCLUDE_PATTERNS = [
    p.strip()
    for p in os.environ.get("SGLANG_KERNEL_API_DUMP_EXCLUDE", "").split(",")
    if p.strip()
]

_logger = logging.getLogger("sglang.kernel_api")
_dump_call_counter: dict[str, int] = {}


def _setup_logger() -> None:
    for handler in list(_logger.handlers):
        _logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    if _KERNEL_API_LOG_LEVEL == 0:
        _logger.addHandler(logging.NullHandler())
        _logger.setLevel(logging.CRITICAL + 1)
        return

    _logger.setLevel(logging.DEBUG)

    if _KERNEL_API_LOG_DEST == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif _KERNEL_API_LOG_DEST == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(_KERNEL_API_LOG_DEST, mode="a")

    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.propagate = False


_setup_logger()


def _is_compiling() -> bool:
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            return bool(torch.compiler.is_compiling())
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
            return bool(torch._dynamo.is_compiling())
    except Exception:
        return False
    return False


def _timestamp() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _is_cuda_graph_capture_active() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _append_line(lines: list[str], indent: int, text: str) -> None:
    lines.append(" " * indent + text)


def _should_dump_function(func_name: str) -> bool:
    if _DUMP_INCLUDE_PATTERNS and not any(
        fnmatch.fnmatch(func_name, pattern) for pattern in _DUMP_INCLUDE_PATTERNS
    ):
        return False
    if _DUMP_EXCLUDE_PATTERNS and any(
        fnmatch.fnmatch(func_name, pattern) for pattern in _DUMP_EXCLUDE_PATTERNS
    ):
        return False
    return True


def _serialize_tensor(tensor: torch.Tensor) -> list[str]:
    lines = ["Tensor("]
    _append_line(lines, 2, f"shape={tuple(tensor.shape)}")
    _append_line(lines, 2, f"dtype={tensor.dtype}")
    _append_line(lines, 2, f"device={tensor.device}")
    _append_line(lines, 2, f"requires_grad={tensor.requires_grad}")
    _append_line(lines, 2, f"is_contiguous={tensor.is_contiguous()}")

    if _KERNEL_API_LOG_LEVEL >= 5:
        if tensor.numel() == 0:
            _append_line(lines, 2, "statistics=[empty tensor]")
        elif tensor.device.type == "cuda" and _is_cuda_graph_capture_active():
            _append_line(
                lines, 2, "statistics=[skipped: CUDA graph capture in progress]"
            )
        else:
            try:
                detached = tensor.detach()
                if detached.is_complex():
                    stats_source = detached.abs().float()
                    nan_count = int(torch.isnan(detached).sum().item())
                    inf_count = int(torch.isinf(detached).sum().item())
                else:
                    stats_source = detached.float()
                    if detached.is_floating_point():
                        nan_count = int(torch.isnan(detached).sum().item())
                        inf_count = int(torch.isinf(detached).sum().item())
                    else:
                        nan_count = 0
                        inf_count = 0

                _append_line(lines, 2, f"min={stats_source.min().item():.6f}")
                _append_line(lines, 2, f"max={stats_source.max().item():.6f}")
                _append_line(lines, 2, f"mean={stats_source.mean().item():.6f}")
                _append_line(lines, 2, f"nan_count={nan_count}")
                _append_line(lines, 2, f"inf_count={inf_count}")
            except Exception as exc:
                _append_line(
                    lines, 2, f"statistics=[unavailable: {type(exc).__name__}]"
                )

    lines.append(")")
    return lines


def _serialize_value(value: Any, depth: int = 0) -> list[str]:
    if depth >= 2:
        return [f"{type(value).__name__}(...)"]

    if isinstance(value, torch.Tensor):
        return _serialize_tensor(value)

    if isinstance(value, (str, int, float, bool, type(None))):
        return [repr(value)]

    if isinstance(value, (list, tuple)):
        opener = "[" if isinstance(value, list) else "("
        closer = "]" if isinstance(value, list) else ")"
        lines = [opener]
        for idx, item in enumerate(value[:4]):
            item_lines = _serialize_value(item, depth + 1)
            lines.append(f"  [{idx}] {item_lines[0]}")
            for extra in item_lines[1:]:
                lines.append(f"      {extra}")
        if len(value) > 4:
            lines.append(f"  ... ({len(value) - 4} more items)")
        lines.append(closer)
        return lines

    if isinstance(value, dict):
        lines = ["{"]
        items = list(value.items())
        for key, item in items[:8]:
            item_lines = _serialize_value(item, depth + 1)
            lines.append(f"  {key!r}: {item_lines[0]}")
            for extra in item_lines[1:]:
                lines.append(f"      {extra}")
        if len(items) > 8:
            lines.append(f"  ... ({len(items) - 8} more items)")
        lines.append("}")
        return lines

    summary = [f"{type(value).__name__}("]
    for attr in ("shape", "dtype", "device"):
        if hasattr(value, attr):
            try:
                _append_line(summary, 2, f"{attr}={getattr(value, attr)}")
            except Exception:
                pass
    if len(summary) == 1:
        _append_line(summary, 2, f"repr={repr(value)[:200]}")
    summary.append(")")
    return summary


def _serialize_json_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return {"type": "torch.dtype", "value": str(value)}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_json_value(item) for item in value[:16]]
    if isinstance(value, dict):
        return {
            str(key): _serialize_json_value(item)
            for key, item in list(value.items())[:32]
        }
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _collect_dump_entries(
    prefix: str,
    value: Any,
    tensor_entries: dict[str, torch.Tensor],
    metadata_entries: dict[str, Any],
) -> None:
    if isinstance(value, torch.Tensor):
        tensor_entries[prefix] = value.detach().cpu()
        return

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _collect_dump_entries(
                f"{prefix}_{idx}", item, tensor_entries, metadata_entries
            )
        metadata_entries[f"{prefix}__container"] = {
            "type": type(value).__name__,
            "length": len(value),
        }
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _collect_dump_entries(
                f"{prefix}_{str(key)}", item, tensor_entries, metadata_entries
            )
        metadata_entries[f"{prefix}__container"] = {
            "type": "dict",
            "keys": [str(k) for k in value.keys()],
        }
        return

    metadata_entries[prefix] = _serialize_json_value(value)


def _dump_metadata_path(dump_dir: Path) -> Path:
    return dump_dir / "metadata.json"


def _write_dump_metadata(dump_dir: Path, metadata: dict[str, Any]) -> None:
    _dump_metadata_path(dump_dir).write_text(json.dumps(metadata, indent=2))


def _read_dump_metadata(dump_dir: Path) -> dict[str, Any]:
    return json.loads(_dump_metadata_path(dump_dir).read_text())


def _dump_function_inputs(
    func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Path | None:
    if not _should_dump_function(func_name):
        return None

    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    call_index = _dump_call_counter.get(func_name, 0) + 1
    _dump_call_counter[func_name] = call_index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_func_name = func_name.replace("/", "_").replace("<", "_").replace(">", "_")
    dump_dir = (
        _DUMP_DIR
        / f"{timestamp}_pid{os.getpid()}_{safe_func_name}_call{call_index:04d}"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)

    tensor_entries: dict[str, torch.Tensor] = {}
    metadata_entries: dict[str, Any] = {}
    for idx, arg in enumerate(args):
        _collect_dump_entries(f"arg_{idx}", arg, tensor_entries, metadata_entries)
    for key, value in kwargs.items():
        _collect_dump_entries(f"kwarg_{key}", value, tensor_entries, metadata_entries)

    if tensor_entries:
        torch.save(tensor_entries, dump_dir / "inputs.pt")

    metadata = {
        "function_name": func_name,
        "timestamp": timestamp,
        "process_id": os.getpid(),
        "execution_status": "inputs_saved",
        "input_metadata": metadata_entries,
        "input_tensor_keys": list(tensor_entries.keys()),
        "output_metadata": {},
        "output_tensor_keys": [],
    }
    _write_dump_metadata(dump_dir, metadata)
    _logger.debug("Dumped inputs to: %s", dump_dir)
    return dump_dir


def _dump_function_outputs(dump_dir: Path, result: Any) -> None:
    tensor_entries: dict[str, torch.Tensor] = {}
    metadata_entries: dict[str, Any] = {}
    _collect_dump_entries("result", result, tensor_entries, metadata_entries)
    if tensor_entries:
        torch.save(tensor_entries, dump_dir / "outputs.pt")

    metadata = _read_dump_metadata(dump_dir)
    metadata["execution_status"] = "completed"
    metadata["output_metadata"] = metadata_entries
    metadata["output_tensor_keys"] = list(tensor_entries.keys())
    _write_dump_metadata(dump_dir, metadata)
    _logger.debug("Dumped outputs to: %s", dump_dir)


def _mark_dump_exception(dump_dir: Path, exc: Exception) -> None:
    metadata = _read_dump_metadata(dump_dir)
    metadata["execution_status"] = "exception"
    metadata["exception"] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    _write_dump_metadata(dump_dir, metadata)


def _log_section(title: str, data: dict[str, Any]) -> None:
    _logger.debug(title)
    for key, value in data.items():
        lines = _serialize_value(value)
        _logger.debug("  %s=%s", key, lines[0])
        for line in lines[1:]:
            _logger.debug("    %s", line)


def _infer_func_name(func: Callable) -> str:
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
    qualname = qualname.replace(".<locals>.", ".").replace("<locals>.", "")

    module = getattr(func, "__module__", "")
    for prefix in ("sglang.", "sgl_kernel."):
        if module.startswith(prefix):
            module = module[len(prefix) :]
            break

    if module and module not in {"__main__", "builtins"}:
        return f"{module}.{qualname}"

    source_path = inspect.getsourcefile(func)
    if source_path is not None:
        return f"{Path(source_path).stem}.{qualname}"

    return qualname


def debug_kernel_api(
    func: Callable | None = None,
    *,
    op_name: str | None = None,
) -> Callable:
    if _KERNEL_API_LOG_LEVEL == 0:
        if func is None:
            return lambda f: f
        return func

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _is_compiling():
                return f(*args, **kwargs)

            func_name = op_name or _infer_func_name(f)
            dump_dir: Path | None = None
            positional_args = args
            try:
                parameters = tuple(inspect.signature(f).parameters.values())
            except (TypeError, ValueError):
                parameters = ()
            if args and parameters and parameters[0].name in {"self", "cls"}:
                positional_args = args[1:]
            _logger.debug("=" * 80)
            _logger.debug("%s SGLang Kernel API Call: %s", _timestamp(), func_name)

            if _KERNEL_API_LOG_LEVEL >= 3:
                if positional_args:
                    _log_section(
                        "Positional input arguments:",
                        {f"arg[{idx}]": arg for idx, arg in enumerate(positional_args)},
                    )
                if kwargs:
                    _log_section("Keyword input arguments:", kwargs)

            if _KERNEL_API_LOG_LEVEL >= 10:
                if _is_cuda_graph_capture_active():
                    _logger.debug("Tensor dump skipped: CUDA graph capture in progress")
                else:
                    dump_dir = _dump_function_inputs(func_name, positional_args, kwargs)

            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                if dump_dir is not None:
                    _mark_dump_exception(dump_dir, exc)
                _logger.debug(
                    "%s SGLang Kernel API Exception: %s (%s: %s)",
                    _timestamp(),
                    func_name,
                    type(exc).__name__,
                    exc,
                )
                raise

            if dump_dir is not None:
                _dump_function_outputs(dump_dir, result)
            if _KERNEL_API_LOG_LEVEL >= 3:
                _log_section("Output:", {"return": result})
            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def debug_torch_op(op_name: str, *, namespace: str = "sglang") -> Callable:
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(getattr(torch.ops, namespace), op_name)(*args, **kwargs)

    return debug_kernel_api(call, op_name=f"{namespace}.custom_op.{op_name}")


def wrap_method_with_debug_kernel_once(
    obj: Any,
    method_name: str,
    *,
    op_name: str,
    marker_attr: str | None = None,
) -> Any:
    if marker_attr is None:
        marker_attr = f"_debug_kernel_{method_name}_wrapped"

    if getattr(obj, marker_attr, False):
        return obj

    setattr(
        obj,
        method_name,
        debug_kernel_api(getattr(obj, method_name), op_name=op_name),
    )
    setattr(obj, marker_attr, True)
    return obj

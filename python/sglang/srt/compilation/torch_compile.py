"""Lazy per-callsite torch compile/export decorator."""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

from sglang.srt.compilation.export_artifact import make_export_artifact_spec
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs

TorchCompileStrategy = Literal["compile", "noop", "export"]


@dataclass
class TorchCompileConfig:
    enabled: bool | None = None
    dynamic: bool | None = None
    fullgraph: bool | None = None
    mode: str | None = None
    backend: Any | None = None
    key: str | None = None
    strategy_override: TorchCompileStrategy | None = None
    dynamic_shapes: Any | None = None
    shape_policy: str | None = None
    export_format: str | None = None
    export_artifact_mode: str | None = None
    run_exported: bool | None = None
    copy_output_to_arg_index: int | None = None
    forced_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.forced_fields, frozenset):
            self.forced_fields = frozenset(self.forced_fields)


@dataclass(frozen=True)
class TorchCompileTarget:
    key: str
    fn: Callable
    config: TorchCompileConfig


TORCH_COMPILE_TARGET_REGISTRY: dict[str, TorchCompileTarget] = {}

_CONFIG_FIELDS = (
    "enabled",
    "dynamic",
    "fullgraph",
    "mode",
    "backend",
    "key",
    "strategy_override",
    "dynamic_shapes",
    "shape_policy",
    "export_format",
    "export_artifact_mode",
    "run_exported",
    "copy_output_to_arg_index",
)
_LIBRARY_DEFAULTS = TorchCompileConfig(
    enabled=True,
    dynamic=True,
    fullgraph=False,
    shape_policy="infer_dynamic",
    export_format="torch",
    export_artifact_mode="build_if_missing",
    run_exported=False,
)


class _FunctionModule(torch.nn.Module):
    def __init__(
        self,
        fn: Callable,
        copy_output_to_arg_index: int | None = None,
    ):
        super().__init__()
        self.fn = fn
        self.copy_output_to_arg_index = copy_output_to_arg_index

    def forward(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)
        if result is None and self.copy_output_to_arg_index is not None:
            return args[self.copy_output_to_arg_index]
        return result


def sgl_compile(fn=None, **kwargs):
    """Decorate a function with platform-selected compile/noop/export behavior."""
    config = TorchCompileConfig(**kwargs)
    if fn is None:
        return lambda actual_fn: _decorate(actual_fn, config)
    return _decorate(fn, config)


def _decorate(target, config: TorchCompileConfig):
    if isinstance(target, staticmethod):
        return staticmethod(_decorate_callable(target.__func__, config))
    if isinstance(target, classmethod):
        return classmethod(_decorate_callable(target.__func__, config))
    return _decorate_callable(target, config)


def _decorate_callable(fn: Callable, decorator_config: TorchCompileConfig):
    target_key = _target_key(fn)
    TORCH_COMPILE_TARGET_REGISTRY[target_key] = TorchCompileTarget(
        key=target_key,
        fn=fn,
        config=decorator_config,
    )
    state: dict[str, Callable | None] = {"callable": None}
    lock = threading.Lock()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cached = state["callable"]
        if cached is not None:
            return cached(*args, **kwargs)

        if _is_transient_compile_context():
            return fn(*args, **kwargs)

        with lock:
            cached = state["callable"]
            if cached is None:
                if _is_transient_compile_context():
                    return fn(*args, **kwargs)
                cached = _build_callable(fn, decorator_config, args, kwargs)
                state["callable"] = cached
        return cached(*args, **kwargs)

    wrapper._sgl_compile_state = state
    wrapper._sgl_compile_config = decorator_config
    wrapper._sgl_compile_key = target_key
    return wrapper


def _build_callable(
    fn: Callable,
    decorator_config: TorchCompileConfig,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Callable:
    platform = _get_current_platform()
    config = _resolve_config(platform.torch_compile_defaults(), decorator_config)

    strategy = config.strategy_override or platform.torch_compile_strategy()
    if not config.enabled or envs.SGLANG_DISABLE_TORCH_COMPILE.get():
        strategy = "noop"

    if strategy == "noop":
        return fn
    if strategy == "compile":
        backend = config.backend
        if backend is None:
            backend = platform.get_compile_backend(config.mode)
        compile_kwargs = {
            "dynamic": config.dynamic,
            "fullgraph": config.fullgraph,
            "backend": backend,
        }
        if config.mode is not None:
            compile_kwargs["mode"] = config.mode
        return torch.compile(fn, **compile_kwargs)
    if strategy == "export":
        return _export_callable(fn, config, args, kwargs, platform)

    raise ValueError(f"Unknown torch compile strategy: {strategy}")


def _resolve_config(
    platform_config: TorchCompileConfig,
    decorator_config: TorchCompileConfig,
) -> TorchCompileConfig:
    values = {
        field_name: getattr(_LIBRARY_DEFAULTS, field_name)
        for field_name in _CONFIG_FIELDS
    }
    for field_name in _CONFIG_FIELDS:
        value = getattr(platform_config, field_name)
        if value is not None:
            values[field_name] = value
    for field_name in _CONFIG_FIELDS:
        if field_name in platform_config.forced_fields:
            continue
        value = getattr(decorator_config, field_name)
        if value is not None:
            values[field_name] = value
    return TorchCompileConfig(
        **values,
        forced_fields=platform_config.forced_fields,
    )


def _export_callable(
    fn: Callable,
    config: TorchCompileConfig,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    platform,
) -> Callable:
    key = config.key or _target_key(fn)
    config.key = key
    artifact = make_export_artifact_spec(
        key=key,
        export_format=config.export_format,
        mode=config.export_artifact_mode,
        shape_policy=config.shape_policy,
        copy_output_to_arg_index=config.copy_output_to_arg_index,
        args=args,
    )

    torch_program_path = artifact.torch_program_path
    if (
        artifact.mode != "export_only"
        and torch_program_path is not None
        and torch_program_path.exists()
    ):
        artifact.validate_metadata()
        exported_program = torch.export.load(torch_program_path)
    else:
        if artifact.mode == "load_only":
            raise FileNotFoundError(
                f"Torch export artifact {torch_program_path} is required by "
                "load_only mode."
            )
        export_args = args
        call_kwargs = kwargs
        torch_export_kwargs = {}
        if config.dynamic_shapes is not None:
            torch_export_kwargs["dynamic_shapes"] = config.dynamic_shapes
        elif config.dynamic and artifact.shape_policy == "infer_dynamic":
            export_args, call_kwargs = _prepare_dynamic_export_inputs(args, kwargs)
            dynamic_shapes = _infer_dynamic_shapes(export_args, call_kwargs)
            if dynamic_shapes is not None:
                torch_export_kwargs["dynamic_shapes"] = dynamic_shapes
        exported_program = torch.export.export(
            _FunctionModule(fn, config.copy_output_to_arg_index),
            export_args,
            kwargs=call_kwargs or None,
            **torch_export_kwargs,
        )
        if torch_program_path is not None and artifact.mode in (
            "build_if_missing",
            "export_only",
        ):
            artifact.ensure_export_dir()
            torch.export.save(exported_program, torch_program_path)

    if config.run_exported or artifact.mode == "export_only":
        runtime = platform.get_export_runtime(config)
        runtime_callable = runtime.prepare_runtime(exported_program, artifact, config)
        if artifact.mode == "export_only":
            return fn
        return runtime_callable
    return fn


def _prepare_dynamic_export_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if kwargs:
        return args, kwargs

    return tuple(_dynamic_export_arg(arg) for arg in args), kwargs


def _dynamic_export_arg(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    export_value = value.detach()
    if export_value.dim() > 0 and export_value.shape[0] == 1:
        repeats = [1] * export_value.dim()
        repeats[0] = 2
        return export_value.repeat(*repeats)
    return export_value.clone()


def _infer_dynamic_shapes(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any | None:
    if kwargs:
        return None

    dim_cache: dict[tuple[int, int], Any] = {}
    specs = tuple(_dynamic_shape_spec(arg, dim_cache) for arg in args)
    if not any(spec is not None for spec in specs):
        return None

    # _FunctionModule.forward accepts *args, so torch.export sees one varargs
    # input whose structure is the positional args tuple.
    return (specs,)


def _dynamic_shape_spec(value: Any, dim_cache: dict[tuple[int, int], Any]) -> Any | None:
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 0:
        return None

    spec = {}
    for dim_index, dim_size in enumerate(value.shape):
        if dim_size <= 1:
            spec[dim_index] = torch.export.Dim.AUTO
        else:
            cache_key = (dim_index, int(dim_size))
            dim = dim_cache.get(cache_key)
            if dim is None:
                dim = torch.export.Dim(
                    f"arg_dim_{dim_index}_{int(dim_size)}",
                    min=1,
                )
                dim_cache[cache_key] = dim
            spec[dim_index] = dim
    return spec


def _target_key(fn: Callable) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def _get_current_platform():
    import sglang.srt.platforms as platforms

    return platforms.current_platform


def _is_transient_compile_context() -> bool:
    if is_in_piecewise_cuda_graph():
        return True

    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        return True

    dynamo_is_compiling = getattr(torch._dynamo, "is_compiling", None)
    return bool(callable(dynamo_is_compiling) and dynamo_is_compiling())


__all__ = [
    "TORCH_COMPILE_TARGET_REGISTRY",
    "TorchCompileConfig",
    "TorchCompileStrategy",
    "sgl_compile",
]

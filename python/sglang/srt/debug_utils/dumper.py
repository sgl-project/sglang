import enum
import functools
import json
import os
import random
import re
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, List, Literal, Optional, Union, get_args, get_type_hints

import torch
import torch.distributed as dist

# -------------------------------------- config base ------------------------------------------


@dataclass(frozen=True)
class _BaseConfig(ABC):
    def __post_init__(self) -> None:
        self._verify_types()

    def _verify_types(self) -> None:
        hints = get_type_hints(type(self))
        cls_name = type(self).__name__
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            expected = self._unwrap_type(hints[f.name])
            if not isinstance(value, expected):
                raise TypeError(
                    f"{cls_name}.{f.name}: expected {expected.__name__}, "
                    f"got {type(value).__name__}"
                )

    @classmethod
    @abstractmethod
    def _env_prefix(cls) -> str: ...

    @classmethod
    def _env_name(cls, field_name: str) -> str:
        return f"{cls._env_prefix()}{field_name.upper()}"

    @classmethod
    def from_env(cls) -> "_BaseConfig":
        return cls(
            **{
                f.name: cls._parse_env_field(cls._env_name(f.name), f.default)
                for f in fields(cls)
            }
        )

    def with_defaults(self, **kwargs) -> "_BaseConfig":
        cls = type(self)
        actual = {
            key: value
            for key, value in kwargs.items()
            if os.getenv(cls._env_name(key)) is None
        }
        return replace(self, **actual) if actual else self

    @staticmethod
    def _unwrap_type(hint) -> type:
        args = get_args(hint)
        if args:
            return next(a for a in args if a is not type(None))
        return hint

    @classmethod
    def _parse_env_field(cls, env_name: str, default):
        return cls._parse_env_value(os.getenv(env_name), default)

    @staticmethod
    def _parse_env_value(raw, default):
        if raw is None or not raw.strip():
            return default
        if isinstance(default, bool):
            return raw.lower() in ("true", "1")
        if isinstance(default, int):
            return int(raw)
        return raw

    @classmethod
    def from_kv_pairs(cls, pairs: Optional[List[str]]) -> "_BaseConfig":
        return cls(**cls._kv_pairs_to_dict(pairs))

    @classmethod
    def _kv_pairs_to_dict(cls, pairs: Optional[List[str]]) -> dict:
        if not pairs:
            return {}

        missing = object()
        defaults = {f.name: f.default for f in fields(cls)}
        result: dict = {}

        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep:
                raise ValueError(f"Invalid config pair (missing '='): {pair!r}")
            default = defaults.get(key, missing)
            if default is missing:
                raise ValueError(
                    f"Unknown config key {key!r}. Valid keys: {sorted(defaults)}"
                )
            try:
                result[key] = cls._parse_env_value(value, default)
            except (ValueError, TypeError) as exc:
                field_type = type(default).__name__
                raise TypeError(f"{key}: expected {field_type}, got {value!r}") from exc

        return result


_DEFAULT_EXP_NAME_PREFIX = "dump_"


@dataclass(frozen=True)
class DumperConfig(_BaseConfig):
    enable: bool = False
    filter: Optional[str] = None
    dir: str = "/tmp/dumper"
    enable_output_file: bool = True
    enable_output_console: bool = True
    enable_value: bool = True
    enable_grad: bool = False
    enable_model_value: bool = False
    enable_model_grad: bool = False
    exp_name: Optional[str] = None
    cleanup_previous: bool = False
    collective_timeout: int = 60
    server_port: str = "-1"
    non_intrusive_mode: str = "core"
    source_patcher_config: Optional[str] = None

    @classmethod
    def _env_prefix(cls) -> str:
        # NOTE: should not be `SGLANG_DUMPER_`, otherwise it is weird when dumping Megatron in Miles
        return "DUMPER_"

    @property
    def server_port_parsed(self) -> Optional[Union[int, Literal["reuse"]]]:
        raw = self.server_port
        if raw == "reuse":
            return "reuse"
        port = int(raw)
        if port <= 0:
            return None
        return port


# -------------------------------------- dumper core ------------------------------------------


@dataclass
class _DumperState:
    dump_index: int = 0
    step: int = 0
    global_ctx: dict = field(default_factory=dict)
    captured_output_data: Optional[dict] = None
    cleanup_previous_handled: bool = False


class _Dumper:
    """Utility to dump tensors, which can be useful when comparison checking models.

    Example usage:
    dumper.dump("layer_start__hidden_states", hidden_states, layer_id=self.layer_id)
    dumper.step()

    Import from non-SGLang system:
    ```
    import sys
    sys.path.append("/YOUR_PATH/sglang/python/sglang/srt/debug_utils")
    from dumper import dumper
    ```

    Then run the program:
    `DUMPER_ENABLE=1 python ...`

    Auto-cleanup old dumps before first write:
    `DUMPER_CLEANUP_PREVIOUS=1 python ...`

    Alternatively, disable at startup and configure via HTTP:
    1. `python ...`
    2. sglang mode:  `curl -X POST http://localhost:30000/dumper/configure -d '{"enable": true}'`
       standalone:   `curl -X POST http://localhost:40000/dumper/configure -d '{"enable": true}'`
    3. `curl -X POST http://localhost:30000/dumper/configure -d '{"enable": true, "filter": "layer_id=[0-3]"}'`
    4. `curl -X POST http://localhost:30000/dumper/reset`

    Related: `sglang.srt.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(self, *, config: DumperConfig):
        self._config = config
        self._state = _DumperState()
        self._non_intrusives: list["_NonIntrusiveDumper"] = []

    # ------------------------------- public :: core ---------------------------------

    @property
    def may_enable(self) -> bool:
        return self._config.enable or self._config.server_port_parsed is not None

    def step(self):
        """This should be called on all ranks at the end of each iteration."""

        self._http_manager  # noqa: B018

        if not self._config.enable:
            return

        # Users may want to `dump` only on some ranks, thus determine name here
        self._ensure_exp_name()

        self._state.step += 1
        print(f"[Dumper] [{time.time()}] step={self._state.step}")

    def dump(
        self,
        name: str,
        value,
        save: bool = True,
        dims: Optional[str] = None,
        dims_grad: Optional[str] = None,
        **kwargs,
    ) -> None:
        value_meta: dict = {}
        grad_meta: dict = {}
        if dims is not None:
            value_meta["dims"] = dims
            grad_meta["dims"] = dims
        if dims_grad is not None:
            value_meta["dims_grad"] = dims_grad
            grad_meta["dims"] = dims_grad

        self._dump_inner(
            name=name,
            value=value,
            extra_kwargs=kwargs,
            save=save,
            enable_value=self._config.enable_value,
            enable_curr_grad=False,
            enable_future_grad=self._config.enable_grad,
            value_tag="Dumper.Value",
            grad_tag="Dumper.Grad",
            value_meta_only_fields=value_meta,
            grad_meta_only_fields=grad_meta,
        )

    def dump_model(
        self,
        model: "torch.nn.Module",
        name_prefix: str = "param",
        save: bool = True,
        **kwargs,
    ) -> None:
        for param_name, param in model.named_parameters():
            self._dump_inner(
                name=f"{name_prefix}__{param_name}",
                value=param,
                extra_kwargs=kwargs,
                save=save,
                enable_value=self._config.enable_model_value,
                enable_curr_grad=self._config.enable_model_grad,
                enable_future_grad=False,
                value_tag="Dumper.ParamValue",
                grad_tag="Dumper.ParamGrad",
            )

    def dump_dict(self, name_prefix, data, save: bool = True, **kwargs):
        data = _obj_to_dict(data)
        for name, value in data.items():
            self.dump(f"{name_prefix}_{name}", value, save=save, **kwargs)

    def set_ctx(self, **kwargs):
        """
        Example:

        dumper.configure_default(filter='layer_id=[0-3]')
        dumper.set_ctx(layer_id=self.layer_id)
        ...
        dumper.set_ctx(layer_id=None)
        """
        self._state.global_ctx = {
            k: v for k, v in (self._state.global_ctx | kwargs).items() if v is not None
        }

    def ctx(
        self,
        _extractor: Optional[Callable[..., dict]] = None,
        **static_ctx: Any,
    ) -> Callable:
        """Decorator that sets context before calling the wrapped function and clears it after.

        Two forms:
            @dumper.ctx(lambda self: dict(layer_id=self.layer_id))
            def forward(self, x): ...

            @dumper.ctx(phase="decode")
            def decode_step(self, x): ...
        """
        if _extractor is not None and static_ctx:
            raise ValueError("cannot mix lambda extractor with static kwargs")
        if _extractor is None and not static_ctx:
            raise ValueError("must provide either a lambda or static kwargs")

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx_dict: dict = _extractor(args[0]) if _extractor else static_ctx
                self.set_ctx(**ctx_dict)
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.set_ctx(**{k: None for k in ctx_dict})

            return wrapper

        return decorator

    def apply_source_patches(self) -> None:
        """Apply source patches from DUMPER_SOURCE_PATCHER_CONFIG if set.

        Automatically injects ``from sglang.srt.debug_utils.dumper import dumper``
        into every replacement block so users don't need to write it in YAML.
        """
        config_path = self._config.source_patcher_config
        if not config_path:
            return

        from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

        yaml_content: str = Path(config_path).read_text()
        print(f"[source_patcher] loading config from {config_path}")
        apply_patches_from_config(
            yaml_content,
            extra_imports=["from sglang.srt.debug_utils.dumper import dumper"],
        )

    def register_non_intrusive_dumper(
        self,
        model: "torch.nn.Module",
    ) -> Optional["_NonIntrusiveDumper"]:
        self._http_manager  # noqa: B018
        mode = self._config.non_intrusive_mode
        if mode == "off":
            return None
        non_intrusive = _NonIntrusiveDumper(dumper=self, model=model, mode=mode)
        self._non_intrusives.append(non_intrusive)
        return non_intrusive

    # ------------------------------- public :: secondary ---------------------------------

    def configure(self, **kwargs) -> None:
        self._config = replace(self._config, **kwargs)

    def configure_default(self, **kwargs) -> None:
        self._config = self._config.with_defaults(**kwargs)

    def reset(self) -> None:
        for non_intrusive in self._non_intrusives:
            non_intrusive.remove()
        self._non_intrusives.clear()
        self._state = _DumperState()

    @contextmanager
    def capture_output(self):
        assert self._state.captured_output_data is None
        self._state.captured_output_data = {}
        try:
            yield self._state.captured_output_data
        finally:
            self._state.captured_output_data = None

    def get_state(self) -> dict:
        return {
            "config": asdict(self._config),
            "dump_index": self._state.dump_index,
            "step": self._state.step,
        }

    @cached_property
    def _http_manager(self) -> Optional["_DumperHttpManager"]:
        if self._config.server_port_parsed is None:
            return None
        return _DumperHttpManager(self)

    # ------------------------- private :: related to dump -----------------------------

    def _dump_inner(
        self,
        *,
        name: str,
        value,
        extra_kwargs: dict,
        save: bool,
        enable_value: bool,
        enable_curr_grad: bool,
        enable_future_grad: bool,
        value_tag: str,
        grad_tag: str,
        value_meta_only_fields: Optional[dict] = None,
        grad_meta_only_fields: Optional[dict] = None,
    ) -> None:
        self._http_manager  # noqa: B018

        if not self._config.enable:
            return

        recompute_status = _detect_recompute_status()
        tags = dict(
            name=name,
            recompute_status=recompute_status.value,
            **extra_kwargs,
            **self._state.global_ctx,
        )

        if (f := self._config.filter) is not None and re.search(
            f, _format_tags(tags)
        ) is None:
            return

        if not (enable_value or enable_curr_grad or enable_future_grad):
            return

        recompute_meta = recompute_status.to_pseudo_parallel_meta()
        value = _materialize_value(value)

        if enable_value:
            self._dump_single(
                tag=value_tag,
                tags=tags,
                value=value,
                save=save,
                meta_only_fields={**(value_meta_only_fields or {}), **recompute_meta},
            )

        if (
            enable_curr_grad
            and isinstance(value, torch.Tensor)
            and (g := value.grad) is not None
        ):
            self._dump_single(
                tag=grad_tag,
                tags={**tags, "name": f"grad__{name}"},
                value=g,
                save=save,
                meta_only_fields={**(grad_meta_only_fields or {}), **recompute_meta},
            )

        if enable_future_grad:
            self._register_dump_grad_hook(
                name=name,
                tensor=value,
                extra_kwargs=extra_kwargs,
                save=save,
                meta_only_fields=grad_meta_only_fields or {},
            )

    def _register_dump_grad_hook(
        self,
        *,
        name: str,
        tensor,
        extra_kwargs: dict,
        save: bool,
        meta_only_fields: Optional[dict] = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if not tensor.requires_grad:
            return

        captured_step = self._state.step
        captured_tags = dict(
            name=f"grad__{name}",
            **deepcopy(extra_kwargs),
        )
        captured_meta_only = meta_only_fields or {}

        def grad_hook(grad: torch.Tensor) -> None:
            self._dump_single(
                tag="Dumper.Grad",
                tags=captured_tags,
                value=grad,
                save=save,
                step=captured_step,
                meta_only_fields=captured_meta_only,
            )

        tensor.register_hook(grad_hook)

    def _dump_single(
        self,
        *,
        tag: str,
        tags: dict,
        value,
        save: bool,
        step: Optional[int] = None,
        meta_only_fields: Optional[dict] = None,
    ) -> None:
        self._ensure_exp_name()
        self._state.dump_index += 1

        rank = _get_rank()
        full_kwargs = dict(
            step=(step if step is not None else self._state.step),
            rank=rank,
            dump_index=self._state.dump_index,
            **tags,
        )
        full_filename = _format_tags(full_kwargs) + ".pt"
        path = Path(self._config.dir) / self._config.exp_name / full_filename

        if self._config.enable_output_console:
            print(
                f"[{tag}] [{rank}, {time.time()}] {path} "
                f"type={type(value)} "
                f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
                f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
                f"device={value.device if isinstance(value, torch.Tensor) else None} "
                f"id={id(value)} "
                f"sample_value={get_truncated_value(value)}"
            )

        capturing = self._state.captured_output_data is not None
        if save and (self._config.enable_output_file or capturing):
            output_data = {
                "value": value,
                "meta": dict(
                    **full_kwargs,
                    **self._static_meta,
                    **(meta_only_fields or {}),
                ),
            }

            if capturing:
                output_data["value"] = _deepcopy_or_clone(output_data["value"])
                self._state.captured_output_data[tags["name"]] = output_data
            else:
                if (
                    not self._state.cleanup_previous_handled
                    and self._config.cleanup_previous
                ):
                    self._state.cleanup_previous_handled = True
                    _cleanup_old_dumps(
                        Path(self._config.dir), exp_name=self._config.exp_name
                    )

                path.parent.mkdir(parents=True, exist_ok=True)
                _torch_save(output_data, str(path))

    # ------------------------------- private :: misc ---------------------------------

    @cached_property
    def _static_meta(self) -> dict:
        return _compute_static_meta()

    def _ensure_exp_name(self):
        if self._config.exp_name is None:
            name = _get_default_exp_name(
                timeout_seconds=self._config.collective_timeout
            )
            self.configure(exp_name=name)
            print(f"[Dumper] Choose exp_name={name}")


# -------------------------------------- hook dumper ------------------------------------------


class _NonIntrusiveDumper:
    _NAME_PREFIX = "non_intrusive__"
    _LAYER_NAME_RE = re.compile(r"(?:.+\.)?layers\.(\d+)$")

    def __init__(
        self,
        dumper: _Dumper,
        model: "torch.nn.Module",
        mode: str,
    ):
        self._dumper = dumper
        self._mode = mode
        self._handles: list = []
        self._core_fields: frozenset[str] = frozenset().union(
            *(p.core_fields() for p in _plugins)
        )

        for module_name, module in model.named_modules():
            if ctx := self._detect_module_ctx(module_name, module):
                self._register_ctx_hooks(module, ctx=ctx)

            is_root = module_name == ""
            pre_hook = self._make_forward_pre_hook(
                module_name=module_name, is_root=is_root
            )
            hook = self._make_forward_hook(module_name=module_name, is_root=is_root)
            self._handles += _register_forward_hook_or_replace_fn(
                module,
                pre_hook=pre_hook,
                hook=hook,
                mode="replace_fn" if is_root else "hook",
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @classmethod
    def _detect_module_ctx(
        cls, module_name: str, module: "torch.nn.Module"
    ) -> Optional[dict]:
        match = cls._LAYER_NAME_RE.fullmatch(module_name)
        if match:
            for plugin in _plugins:
                layer_id = plugin.detect_layer_id(module)
                if layer_id is not None:
                    return {"layer_id": layer_id}
            return {"layer_id": int(match.group(1))}
        return None

    def _register_ctx_hooks(self, module: "torch.nn.Module", *, ctx: dict) -> None:
        clear_ctx = {k: None for k in ctx}
        self._handles.append(
            module.register_forward_pre_hook(
                lambda _mod, _input, _ctx=ctx: self._dumper.set_ctx(**_ctx)
            )
        )
        self._handles.append(
            module.register_forward_hook(
                lambda _mod, _input, _output, _clear=clear_ctx: self._dumper.set_ctx(
                    **_clear
                )
            )
        )

    def _make_forward_pre_hook(self, *, module_name: str, is_root: bool):
        def _hook(_module, args, kwargs):
            for i, item in enumerate(args):
                self._dump_value(
                    module_name, item, sub_name=f"inputs.{i}", is_root=is_root
                )
            for name, value in kwargs.items():
                self._dump_value(
                    module_name,
                    value,
                    sub_name=f"inputs.{name}",
                    is_root=is_root,
                )

        return _hook

    def _make_forward_hook(self, *, module_name: str, is_root: bool):
        def _hook(_module, input, output):
            if output is not None:
                self._dump_value(module_name, output, sub_name="output", is_root=False)

        return _hook

    def _dump_value(
        self, module_name: str, value: Any, sub_name: str, *, is_root: bool
    ) -> None:
        for key, item in self._convert_value(
            value, skip_forward_batch=(not is_root)
        ).items():
            effective_key = key or sub_name.rsplit(".", 1)[-1]
            if effective_key in self._core_fields:
                self._dumper.dump(effective_key, item)
            elif self._mode == "all":
                parts = [p for p in (module_name, sub_name, key) if p]
                self._dumper.dump(self._NAME_PREFIX + ".".join(parts), item)

    @staticmethod
    def _convert_value(value, *, skip_forward_batch: bool = False) -> dict[str, Any]:
        if isinstance(value, torch.Tensor):
            return {"": value}

        if isinstance(value, (tuple, list)):
            tensors = [t for t in value if isinstance(t, torch.Tensor)]
            if len(tensors) == 1:
                return {"": tensors[0]}
            return {str(i): t for i, t in enumerate(tensors)}

        for plugin in _plugins:
            result = plugin.convert_value(value, skip_forward_batch=skip_forward_batch)
            if result is not None:
                return result

        return {}


def _register_forward_hook_or_replace_fn(
    module: "torch.nn.Module",
    *,
    pre_hook,
    hook,
    mode: str,
) -> list:
    """Attach pre/post forward hooks to *module*.

    mode="hook"       — standard ``register_forward_pre_hook`` / ``register_forward_hook``
                        (fires only via ``__call__``).
    mode="replace_fn" — monkey-patch ``module.forward`` so hooks fire even when
                        callers invoke ``.forward()`` directly (as sglang does for the
                        root model).

    Returns a list of handle objects with a ``.remove()`` method that undoes
    the registration.
    """
    if mode == "hook":
        return [
            module.register_forward_pre_hook(pre_hook, with_kwargs=True),
            module.register_forward_hook(hook),
        ]
    elif mode == "replace_fn":
        original_forward = module.forward

        @functools.wraps(original_forward)
        def _wrapped(*args, **kwargs):
            pre_hook(module, args, kwargs)
            output = original_forward(*args, **kwargs)
            hook(module, args, output)
            return output

        module.forward = _wrapped

        class _Handle:
            def remove(self) -> None:
                assert module.forward is _wrapped
                module.forward = original_forward

        return [_Handle()]
    else:
        raise ValueError(f"Unknown mode {mode!r}")


# -------------------------------------- util fn ------------------------------------------


def _torch_save(value, path: str):
    value = _clone_if_view(value)
    try:
        try:
            return torch.save(value, path)
        except RuntimeError as e:
            if "not pickleable" in str(e):
                stripped = _strip_parameter(value)
                if stripped is not value:
                    print(f"[Dumper] Observe error={e} and try pickling .data")
                    return _torch_save(stripped, path)
            raise
    except Exception as e:
        print(f"[Dumper] Observe error={e} when saving data, skip the tensor")


def _map_tensor(value, fn: Callable[[torch.Tensor], torch.Tensor]):
    if isinstance(value, dict):
        return {k: _map_tensor(v, fn) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return fn(value)
    return value


def _clone_if_view(value):
    def _fn(t: torch.Tensor) -> torch.Tensor:
        if t.untyped_storage().nbytes() > t.nelement() * t.element_size():
            return t.clone()
        return t

    return _map_tensor(value, _fn)


def _strip_parameter(value):
    def _fn(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.nn.Parameter):
            return t.data
        return t

    return _map_tensor(value, _fn)


def _collective_with_timeout(fn, operation_name: str, timeout_seconds: int = 60):
    completed = threading.Event()

    def watchdog():
        if not completed.wait(timeout=timeout_seconds):
            print(
                f"\n[Dumper] WARNING: '{operation_name}' has not completed after "
                f"{timeout_seconds}s. This usually means not all ranks are "
                f"participating in this collective operation.\n",
                flush=True,
            )

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        completed.set()


def _get_default_exp_name(timeout_seconds: int = 60):
    rank = _get_rank()
    now = time.time()
    ms = int((now % 1) * 1000)
    rand_suffix = random.randint(0, 999)
    object_list = [
        (
            (
                f"{_DEFAULT_EXP_NAME_PREFIX}"
                f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime(now))}"
                f"_{ms:03d}{rand_suffix:03d}"
            )
            if rank == 0
            else None
        )
    ]

    if dist.is_initialized():
        _collective_with_timeout(
            lambda: dist.broadcast_object_list(object_list, device="cuda"),
            operation_name="broadcast_object_list in _get_default_exp_name",
            timeout_seconds=timeout_seconds,
        )

    return object_list[0]


def _cleanup_old_dumps(base_dir: Path, exp_name: Optional[str] = None) -> None:
    import shutil

    if _get_rank() == 0:
        targets = {entry for entry in base_dir.glob(f"{_DEFAULT_EXP_NAME_PREFIX}*")}
        if exp_name:
            targets.add(base_dir / exp_name)
        targets = {d for d in targets if d.is_dir()}

        for entry in targets:
            shutil.rmtree(entry)
            print(f"[Dumper] Cleaned up {entry}")

    if dist.is_initialized():
        _collective_with_timeout(
            dist.barrier,
            operation_name="barrier in _cleanup_old_dumps",
        )


def _get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def _get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def _obj_to_dict(obj):
    if isinstance(obj, dict):
        return obj
    ret = {}
    for k in dir(obj):
        if k.startswith("__") and k.endswith("__"):
            continue
        try:
            v = getattr(obj, k)
            if not callable(v):
                ret[k] = v
        except Exception:
            # Skip attributes that raise an exception on access
            continue
    return ret


def _materialize_value(value):
    if callable(value):
        value = value()
    return value


def _format_tags(kwargs: dict) -> str:
    return "___".join(f"{k}={v}" for k, v in kwargs.items())


def _deepcopy_or_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    return deepcopy(x)


# -------------------------------------- static meta ------------------------------------------


def _compute_static_meta():
    result = {
        "world_rank": _get_rank(),
        "world_size": _get_world_size(),
    }

    for plugin in _plugins:
        if info := plugin.collect_parallel_info():
            result[f"{plugin.name}_parallel_info"] = info

    for plugin in _plugins:
        tokenizer_path: Optional[str] = plugin.get_tokenizer_path()
        if tokenizer_path is not None:
            result["tokenizer_path"] = tokenizer_path
            break

    return result


# -------------------------------------- http manager ------------------------------------------


class _DumperHttpManager:
    def __init__(self, dumper: "_Dumper"):
        self._dumper = dumper
        http_port = self._dumper._config.server_port_parsed

        rpc_broadcast = _create_zmq_rpc_broadcast(
            self,
            timeout_seconds=self._dumper._config.collective_timeout,
        )

        if _get_rank() == 0:
            assert rpc_broadcast is not None
            self._rpc_broadcast = rpc_broadcast

            if http_port == "reuse":
                print(
                    "[Dumper] Standalone HTTP server disabled, reusing existing ports"
                )
            else:
                _start_http_server(prefix="/dumper/", target=self, http_port=http_port)
                print(f"[Dumper] HTTP server started on port {http_port}")

    # ------------------------------- public ---------------------------------

    def handle_request(self, *, method: str, body: dict[str, Any]) -> list[dict]:
        return self._rpc_broadcast._handle_request_inner(method=method, body=body)

    # ------------------------------- private ---------------------------------

    def _handle_request_inner(self, *, method: str, body: dict[str, Any]) -> dict:
        if method == "get_state":
            return self._dumper.get_state()
        elif method == "configure":
            self._dumper.configure(**body)
            return {}
        elif method == "reset":
            self._dumper.reset()
            return {}
        else:
            raise ValueError(f"Unknown dumper control method: {method!r}")


# -------------------------------------- http control server ------------------------------------------


def _start_http_server(*, prefix: str, target: object, http_port: int):
    handler_class = _make_http_handler(prefix=prefix, target=target)
    server = HTTPServer(("0.0.0.0", http_port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _make_http_handler(*, prefix: str, target):
    class _HTTPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.startswith(prefix):
                self.send_error(404)
                return
            method = self.path[len(prefix) :]
            try:
                req_body = self._get_request_body()
                print(f"[Dumper#{_get_rank()}] HTTP {self.path} {req_body=}")
                result = target.handle_request(method=method, body=req_body)
                resp_body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
            except Exception as e:
                self.send_error(400, str(e))

        def _get_request_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            return json.loads(self.rfile.read(content_length))

    return _HTTPHandler


# -------------------------------------- zmq rpc ------------------------------------------


def _create_zmq_rpc_broadcast(
    handler, timeout_seconds: int = 60
) -> Optional["_ZmqRpcBroadcast"]:
    """A general-purpose minimal RPC to support broadcasting executions to multi processes"""
    import zmq

    rank = _get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:0")
    bound_port = int(sock.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
    local_addr = f"tcp://{_get_local_ip_by_remote()}:{bound_port}"

    def serve_loop():
        while True:
            try:
                req = sock.recv_pyobj()
                result = getattr(handler, req["method"])(*req["args"], **req["kwargs"])
                resp = {"result": result, "error": None}
            except Exception as e:
                print(f"[Dumper.ZmqRpc] error inside handler: {e}")
                resp = {"result": None, "error": str(e)}
            sock.send_pyobj(resp)

    thread = threading.Thread(target=serve_loop, daemon=True)
    thread.start()
    print(f"[Dumper.ZmqRpc] rank={rank} server started at {local_addr}")

    if dist.is_initialized():
        all_addresses = [None] * world_size
        _collective_with_timeout(
            lambda: dist.all_gather_object(all_addresses, local_addr),
            operation_name="all_gather_object in _create_zmq_rpc_broadcast",
            timeout_seconds=timeout_seconds,
        )
    else:
        all_addresses = [local_addr]
    print(f"[Dumper.ZmqRpc] rank={rank} all_addresses={all_addresses}")

    if rank == 0:
        handles = []
        for i, addr in enumerate(all_addresses):
            req_socket = ctx.socket(zmq.REQ)
            req_socket.connect(addr)
            handles.append(_ZmqRpcHandle(req_socket, debug_name=f"rank-{i}"))
        return _ZmqRpcBroadcast(handles)
    else:
        return None


class _ZmqRpcHandle:
    """Proxy object to call remote handler methods via ZMQ."""

    def __init__(self, socket, debug_name: str):
        self._socket = socket
        self._debug_name = debug_name

    def __getattr__(self, method_name: str):
        def call(*args, **kwargs):
            self._socket.send_pyobj(
                {
                    "method": method_name,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            response = self._socket.recv_pyobj()
            if response["error"]:
                raise RuntimeError(
                    f"RPC error on {self._debug_name}: {response['error']}"
                )
            return response["result"]

        return call


class _RpcBroadcastBase:
    """Base for broadcasting method calls to dumper instance(s)."""

    def __getattr__(self, method_name: str):
        raise NotImplementedError

    def __init__(self, handles: List[_ZmqRpcHandle]):
        self._handles = handles


class _ZmqRpcBroadcast(_RpcBroadcastBase):
    """Broadcasts method calls to all ZMQ RPC handles.

    Returns a list of results, one per rank (ordered by rank).
    """

    def __init__(self, handles: List[_ZmqRpcHandle]):
        self._handles = handles

    def __getattr__(self, method_name: str):
        def call(*args, **kwargs):
            return [
                getattr(handle, method_name)(*args, **kwargs)
                for handle in self._handles
            ]

        return call


# --------------------------------- copied code (avoid dependency) --------------------------------------


def _get_local_ip_by_remote() -> Optional[str]:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        print("Can not get local ip by remote")
    return None


# -------------------------------------- framework plugins ------------------------------------------


class _RecomputeStatus(enum.Enum):
    DISABLED = "disabled"
    ORIGINAL = "original"  # inside checkpoint, original forward
    RECOMPUTE = "recompute"  # inside checkpoint, recompute forward

    def to_pseudo_parallel_meta(self) -> dict[str, Any]:
        if self == _RecomputeStatus.DISABLED:
            return {}
        return {
            "recompute_pseudo_rank": 1 if self == _RecomputeStatus.RECOMPUTE else 0,
            "recompute_pseudo_size": 2,
        }


class _FrameworkPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def collect_parallel_info(self) -> dict: ...

    @abstractmethod
    def convert_value(
        self, value: Any, *, skip_forward_batch: bool
    ) -> Optional[dict[str, Any]]:
        """Return converted dict, or None if this plugin doesn't handle the value."""
        ...

    @abstractmethod
    def detect_layer_id(self, module: "torch.nn.Module") -> Optional[int]:
        """Return 0-indexed layer_id, or None if not detectable."""
        ...

    def core_fields(self) -> frozenset[str]:
        return frozenset()

    def get_tokenizer_path(self) -> Optional[str]:
        return None

    def detect_recompute_status(self) -> _RecomputeStatus:
        return _RecomputeStatus.DISABLED


class _SGLangPlugin(_FrameworkPlugin):
    _available = True
    try:
        from sglang.srt import distributed as _dist
        from sglang.srt.layers import dp_attention as _dp_attn
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            PPProxyTensors,
        )
    except ImportError:
        _available = False

    @property
    def name(self) -> str:
        return "sglang"

    def collect_parallel_info(self) -> dict:
        if not self._available:
            return {}

        info = {}

        try:
            info["tp_rank"] = self._dist.get_tensor_model_parallel_rank()
            info["tp_size"] = self._dist.get_tensor_model_parallel_world_size()
            info["pp_rank"] = self._dist.get_pipeline_model_parallel_rank()
            info["pp_size"] = self._dist.get_pipeline_model_parallel_world_size()
            info["moe_ep_rank"] = self._dist.get_moe_expert_parallel_rank()
            info["moe_ep_size"] = self._dist.get_moe_expert_parallel_world_size()
            info["moe_tp_rank"] = self._dist.get_moe_tensor_parallel_rank()
            info["moe_tp_size"] = self._dist.get_moe_tensor_parallel_world_size()
        except (AttributeError, AssertionError):
            info["distributed_error"] = True

        try:
            info["enable_dp_attention"] = self._dp_attn.is_dp_attention_enabled()
            info["attn_tp_rank"] = self._dp_attn.get_attention_tp_rank()
            info["attn_tp_size"] = self._dp_attn.get_attention_tp_size()
            info["attn_dp_rank"] = self._dp_attn.get_attention_dp_rank()
            info["attn_dp_size"] = self._dp_attn.get_attention_dp_size()
            info["local_attn_dp_rank"] = self._dp_attn.get_local_attention_dp_rank()
            info["local_attn_dp_size"] = self._dp_attn.get_local_attention_dp_size()
        except (AttributeError, AssertionError):
            info["dp_attention_error"] = True

        return info

    def convert_value(
        self, value: Any, *, skip_forward_batch: bool
    ) -> Optional[dict[str, Any]]:
        if not self._available:
            return None

        if isinstance(value, self.LogitsProcessorOutput):
            return {"next_token_logits": value.next_token_logits}
        if isinstance(value, self.ForwardBatch):
            if skip_forward_batch:
                return {}
            result = {
                "input_ids": value.input_ids,
                "seq_lens": value.seq_lens,
                "positions": value.positions,
                "req_pool_indices": value.req_pool_indices,
            }
            if value.rids is not None:
                result["rids"] = value.rids
            return result
        if isinstance(value, self.PPProxyTensors):
            return {k: v for k, v in value.tensors.items()}

        return None

    def detect_layer_id(self, module: "torch.nn.Module") -> Optional[int]:
        if hasattr(module, "layer_id"):
            return module.layer_id
        return None

    def core_fields(self) -> frozenset[str]:
        return frozenset(
            {"input_ids", "positions", "seq_lens", "req_pool_indices", "rids"}
        )

    def get_tokenizer_path(self) -> Optional[str]:
        if not self._available:
            return None

        try:
            from sglang.srt.server_args import get_global_server_args

            args = get_global_server_args()
            if args is None:
                return None

            return args.tokenizer_path
        except Exception:
            return None


class _MegatronPlugin(_FrameworkPlugin):
    _available = True
    try:
        from megatron.core import parallel_state as _mpu
        from megatron.core.packed_seq_params import PackedSeqParams
    except ImportError:
        _available = False

    @property
    def name(self) -> str:
        return "megatron"

    def collect_parallel_info(self) -> dict:
        if not self._available:
            return {}

        info = {}
        try:
            info["tp_rank"] = self._mpu.get_tensor_model_parallel_rank()
            info["tp_size"] = self._mpu.get_tensor_model_parallel_world_size()
            info["pp_rank"] = self._mpu.get_pipeline_model_parallel_rank()
            info["pp_size"] = self._mpu.get_pipeline_model_parallel_world_size()
            info["dp_rank"] = self._mpu.get_data_parallel_rank()
            info["dp_size"] = self._mpu.get_data_parallel_world_size()
            info["cp_rank"] = self._mpu.get_context_parallel_rank()
            info["cp_size"] = self._mpu.get_context_parallel_world_size()
            info["vpp_rank"] = self._mpu.get_virtual_pipeline_model_parallel_rank()
            info["vpp_size"] = (
                self._mpu.get_virtual_pipeline_model_parallel_world_size()
            )
            info["ep_rank"] = self._mpu.get_expert_model_parallel_rank()
            info["ep_size"] = self._mpu.get_expert_model_parallel_world_size()
            info["etp_rank"] = self._mpu.get_expert_tensor_parallel_rank()
            info["etp_size"] = self._mpu.get_expert_tensor_parallel_world_size()
            info["edp_rank"] = self._mpu.get_expert_data_parallel_rank()
            info["edp_size"] = self._mpu.get_expert_data_parallel_world_size()
            info["tcp_rank"] = self._mpu.get_tensor_and_context_parallel_rank()
            info["tcp_size"] = self._mpu.get_tensor_and_context_parallel_world_size()
            info["etmp_rank"] = self._mpu.get_expert_tensor_and_model_parallel_rank()
            info["etmp_size"] = (
                self._mpu.get_expert_tensor_and_model_parallel_world_size()
            )
            info["tp_src_rank"] = self._mpu.get_tensor_model_parallel_src_rank()
            info["mp_src_rank"] = self._mpu.get_model_parallel_src_rank()
            info["dp_src_rank"] = self._mpu.get_data_parallel_src_rank()
        except (AttributeError, AssertionError):
            info["megatron_error"] = True

        # Megatron sequence parallel reuses the TP group (no dedicated parallel state API).
        # When sequence_parallel=True, inject sp_rank/sp_size for the comparator unsharder.
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
            if getattr(args, "sequence_parallel", False) and "tp_rank" in info:
                info["sp_rank"] = info["tp_rank"]
                info["sp_size"] = info["tp_size"]
        except (ImportError, AssertionError, AttributeError):
            pass

        return info

    def convert_value(
        self, value: Any, *, skip_forward_batch: bool
    ) -> Optional[dict[str, Any]]:
        if not self._available:
            return None
        if isinstance(value, self.PackedSeqParams):
            return {
                "cu_seqlens_q": value.cu_seqlens_q,
                "cu_seqlens_kv": value.cu_seqlens_kv,
                "qkv_format": value.qkv_format,
            }
        return None

    def detect_layer_id(self, module: "torch.nn.Module") -> Optional[int]:
        if hasattr(module, "layer_number"):
            return module.layer_number - 1
        return None

    def core_fields(self) -> frozenset[str]:
        return frozenset(
            {"input_ids", "position_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"}
        )

    def detect_recompute_status(self) -> _RecomputeStatus:
        if not self._available:
            return _RecomputeStatus.DISABLED
        try:
            from megatron.core.tensor_parallel.random import is_checkpointing

            if not is_checkpointing():
                return _RecomputeStatus.DISABLED
            if torch.is_grad_enabled():
                return _RecomputeStatus.RECOMPUTE
            return _RecomputeStatus.ORIGINAL
        except (ImportError, AttributeError):
            return _RecomputeStatus.DISABLED


_plugins: list[_FrameworkPlugin] = [_SGLangPlugin(), _MegatronPlugin()]


def _detect_recompute_status() -> _RecomputeStatus:
    for plugin in _plugins:
        info = plugin.detect_recompute_status()
        if info != _RecomputeStatus.DISABLED:
            return info
    return _RecomputeStatus.DISABLED


# -------------------------------------- singleton ------------------------------------------


dumper = _Dumper(config=DumperConfig.from_env())


# -------------------------------------- other utility functions ------------------------------------------


def get_truncated_value(value):
    if value is None:
        return None

    if isinstance(value, tuple):
        return [get_truncated_value(x) for x in value]

    if not isinstance(value, torch.Tensor):
        return value

    if value.numel() < 200:
        return value

    slices = [slice(0, 5) if dim_size > 50 else slice(None) for dim_size in value.shape]
    return value[tuple(slices)]


def get_tensor_info(x):
    """
    from sglang.srt.debug_utils.dumper import get_tensor_info
    """
    if not isinstance(x, torch.Tensor):
        return f"type={type(x)} value={x}"
    min = x.float().min() if x.numel() > 0 else None
    max = x.float().max() if x.numel() > 0 else None
    mean = x.float().mean() if x.numel() > 0 else None
    torch.set_printoptions(precision=10)
    x_sample_head = str(x.flatten()[:5])
    x_sample_tail = str(x.flatten()[-5:])
    torch.set_printoptions(precision=4)
    return (
        f"type={type(x)} "
        f"shape={x.shape} "
        f"dtype={x.dtype} "
        f"device={x.device} "
        f"stride={x.stride()} "
        f"req_grad={x.requires_grad} "
        f"min={min} "
        f"max={max} "
        f"mean={mean} "
        f"x_sample_head={x_sample_head} "
        f"x_sample_tail={x_sample_tail}"
    )

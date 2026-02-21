import json
import os
import re
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from functools import cached_property
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, List, Literal, Optional, Union, get_args, get_type_hints

import torch
import torch.distributed as dist

# -------------------------------------- frozen config base ------------------------------------------


@dataclass(frozen=True)
class _FrozenConfig(ABC):
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
    def from_env(cls) -> "_FrozenConfig":
        return cls(
            **{
                f.name: cls._parse_env_field(cls._env_name(f.name), f.default)
                for f in fields(cls)
            }
        )

    def with_defaults(self, **kwargs) -> "_FrozenConfig":
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

    @staticmethod
    def _parse_env_field(env_name: str, default):
        return _FrozenConfig._parse_env_value(os.getenv(env_name), default)

    @staticmethod
    def _parse_env_value(raw, default):
        if raw is None or not raw.strip():
            return default
        if isinstance(default, bool):
            return raw.lower() in ("true", "1")
        if isinstance(default, int):
            return int(raw)
        return raw


@dataclass(frozen=True)
class _DumperConfig(_FrozenConfig):
    enable: bool = False
    filter: Optional[str] = None
    dir: str = "/tmp"
    enable_output_file: bool = True
    enable_output_console: bool = True
    enable_value: bool = True
    enable_grad: bool = False
    enable_model_value: bool = True
    enable_model_grad: bool = True
    partial_name: Optional[str] = None
    enable_http_server: bool = True
    cleanup_previous: bool = False
    collective_timeout: int = 60
    server_port: str = "-1"

    @classmethod
    def _env_prefix(cls) -> str:
        return "SGLANG_DUMPER_"

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


class _Dumper:
    """Utility to dump tensors, which can be useful when comparison checking models.

    Example usage:
    dumper.on_forward_pass_start()
    dumper.dump("layer_start__hidden_states", hidden_states, layer_id=self.layer_id)

    Import from non-SGLang system:
    ```
    import sys
    sys.path.append("/YOUR_PATH/sglang/python/sglang/srt/debug_utils")
    from dumper import dumper
    ```

    Then run the program:
    `SGLANG_DUMPER_ENABLE=1 python ...`

    Auto-cleanup old dumps before first write:
    `SGLANG_DUMPER_CLEANUP_PREVIOUS=1 python ...`

    Alternatively, disable at startup and configure via HTTP:
    1. `python ...`
    2. sglang mode:  `curl -X POST http://localhost:30000/dumper/configure -d '{"enable": true}'`
       standalone:   `curl -X POST http://localhost:40000/dumper/configure -d '{"enable": true}'`
    3. `curl -X POST http://localhost:30000/dumper/configure -d '{"enable": true, "filter": "layer_id=[0-3]"}'`
    4. `curl -X POST http://localhost:30000/dumper/reset`

    Related: `sglang.srt.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(self, *, config: _DumperConfig):
        self._config = config

        self._http_server_handled = not config.enable_http_server
        self._cleanup_previous_handled = not config.cleanup_previous

        self._dump_index = 0
        self._forward_pass_id = 0
        self._global_ctx: dict = {}
        self._captured_output_data: Optional[dict] = None
        self._rpc_broadcast: "_RpcBroadcastBase" = _LocalOnlyBroadcast(self)

    def on_forward_pass_start(self):
        """This should be called on all ranks."""

        # Even if SGLANG_DUMPER_ENABLE=0, users may want to use HTTP endpoint to enable it
        self._ensure_http_server()

        if not self._config.enable:
            return

        # Users may want to `dump` only on some ranks, thus determine name here
        self._ensure_partial_name()

        self._forward_pass_id += 1
        print(
            f"[Dumper] [{time.time()}] on_forward_pass_start id={self._forward_pass_id}"
        )

    def _ensure_http_server(self):
        if self._http_server_handled:
            return
        self._http_server_handled = True

        http_port = self._config.server_port_parsed
        if http_port is None:
            return

        rpc_broadcast = _create_zmq_rpc_broadcast(
            self,
            base_port=get_int_env_var("SGLANG_DUMPER_ZMQ_BASE_PORT", 16800),
            timeout_seconds=self._config.collective_timeout,
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

    def _ensure_partial_name(self):
        if self._config.partial_name is None:
            name = _get_partial_name(timeout_seconds=self._config.collective_timeout)
            self.configure(partial_name=name)
            print(f"[Dumper] Choose partial_name={name}")

    def set_ctx(self, **kwargs):
        """
        Example:

        dumper.configure_default(filter='layer_id=[0-3]')
        dumper.set_ctx(layer_id=self.layer_id)
        ...
        dumper.set_ctx(layer_id=None)
        """
        self._global_ctx = {
            k: v for k, v in (self._global_ctx | kwargs).items() if v is not None
        }

    def reset(self) -> None:
        self._dump_index = 0
        self._forward_pass_id = 0
        self._global_ctx = {}

    @contextmanager
    def capture_output(self):
        assert self._captured_output_data is None
        self._captured_output_data = {}
        try:
            yield self._captured_output_data
        finally:
            self._captured_output_data = None

    def get_state(self) -> dict:
        return {
            "config": asdict(self._config),
            "dump_index": self._dump_index,
            "forward_pass_id": self._forward_pass_id,
        }

    def _handle_http_control_request(
        self, *, method: str, body: dict[str, Any]
    ) -> list[dict]:
        return self._rpc_broadcast._handle_http_control_request_inner(
            method=method, body=body
        )

    def _handle_http_control_request_inner(
        self, *, method: str, body: dict[str, Any]
    ) -> dict:
        if method == "get_state":
            return self.get_state()
        elif method == "configure":
            self.configure(**body)
            return {}
        elif method == "reset":
            self.reset()
            return {}
        else:
            raise ValueError(f"Unknown dumper control method: {method!r}")

    def configure(self, **kwargs) -> None:
        self._config = replace(self._config, **kwargs)

    def configure_default(self, **kwargs) -> None:
        self._config = self._config.with_defaults(**kwargs)

    def dump_dict(self, name_prefix, data, save: bool = True, **kwargs):
        data = _obj_to_dict(data)
        for name, value in data.items():
            self.dump(f"{name_prefix}_{name}", value, save=save, **kwargs)

    def dump(self, name: str, value, save: bool = True, **kwargs) -> None:
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
    ) -> None:
        self._ensure_http_server()

        if not self._config.enable:
            return

        tags = dict(name=name, **extra_kwargs, **self._global_ctx)
        if (f := self._config.filter) is not None and re.search(
            f, _format_tags(tags)
        ) is None:
            return

        if not (enable_value or enable_curr_grad or enable_future_grad):
            return

        if self._forward_pass_id < 1:
            print("Dump without on_forward_pass_start()")

        value = _materialize_value(value)

        if enable_value:
            self._dump_single(
                tag=value_tag,
                tags=tags,
                value=value,
                save=save,
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
            )

        if enable_future_grad:
            self._register_dump_grad_hook(
                name=name,
                tensor=value,
                extra_kwargs=extra_kwargs,
                save=save,
            )

    def _register_dump_grad_hook(
        self,
        *,
        name: str,
        tensor,
        extra_kwargs: dict,
        save: bool,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if not tensor.requires_grad:
            return

        captured_forward_pass_id = self._forward_pass_id
        captured_tags = dict(name=f"grad__{name}", **deepcopy(extra_kwargs))

        def grad_hook(grad: torch.Tensor) -> None:
            self._dump_single(
                tag="Dumper.Grad",
                tags=captured_tags,
                value=grad,
                save=save,
                forward_pass_id=captured_forward_pass_id,
            )

        tensor.register_hook(grad_hook)

    def _dump_single(
        self,
        *,
        tag: str,
        tags: dict,
        value,
        save: bool,
        forward_pass_id: Optional[int] = None,
    ) -> None:
        self._ensure_partial_name()
        self._dump_index += 1

        rank = _get_rank()
        full_kwargs = dict(
            forward_pass_id=(
                forward_pass_id
                if forward_pass_id is not None
                else self._forward_pass_id
            ),
            rank=rank,
            dump_index=self._dump_index,
            **tags,
        )
        full_filename = _format_tags(full_kwargs) + ".pt"
        path = (
            Path(self._config.dir)
            / f"sglang_dump_{self._config.partial_name}"
            / full_filename
        )

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

        capturing = self._captured_output_data is not None
        if save and (self._config.enable_output_file or capturing):
            output_data = {
                "value": value.data if isinstance(value, torch.nn.Parameter) else value,
                "meta": dict(**full_kwargs, **self._static_meta),
            }

            if capturing:
                output_data["value"] = _deepcopy_or_clone(output_data["value"])
                self._captured_output_data[tags["name"]] = output_data
            else:
                if not self._cleanup_previous_handled:
                    self._cleanup_previous_handled = True
                    _cleanup_old_dumps(Path(self._config.dir))

                path.parent.mkdir(parents=True, exist_ok=True)
                _torch_save(output_data, str(path))

    @cached_property
    def _static_meta(self) -> dict:
        return _compute_static_meta()


def _torch_save(value, path: str):
    try:
        try:
            return torch.save(value, path)
        except RuntimeError as e:
            if "not pickleable" in str(e):
                # Some parameter subclasses with extra fields are not pickleable
                if isinstance(value, torch.nn.Parameter):
                    print(f"[Dumper] Observe error={e} and try pickling value.data")
                    return _torch_save(value.data, path)
            raise
    except Exception as e:
        print(f"[Dumper] Observe error={e} when saving data, skip the tensor")


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


def _get_partial_name(timeout_seconds: int = 60):
    rank = _get_rank()
    object_list = [str(time.time()) if rank == 0 else None]

    if dist.is_initialized():
        _collective_with_timeout(
            lambda: dist.broadcast_object_list(object_list, device="cuda"),
            operation_name="broadcast_object_list in _get_partial_name",
            timeout_seconds=timeout_seconds,
        )

    return object_list[0]


def _cleanup_old_dumps(base_dir: Path) -> None:
    import shutil

    if _get_rank() == 0:
        for entry in base_dir.glob("sglang_dump_*"):
            if entry.is_dir():
                shutil.rmtree(entry)
                print(f"[Dumper] Cleaned up {entry}")

    if dist.is_initialized():
        dist.barrier()


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

    if x := _collect_sglang_parallel_info():
        result["sglang_parallel_info"] = x
    if x := _collect_megatron_parallel_info():
        result["megatron_parallel_info"] = x

    return result


def _collect_sglang_parallel_info():
    info = {}

    try:
        from sglang.srt.distributed import (
            get_moe_expert_parallel_rank,
            get_moe_expert_parallel_world_size,
            get_moe_tensor_parallel_rank,
            get_moe_tensor_parallel_world_size,
            get_pipeline_model_parallel_rank,
            get_pipeline_model_parallel_world_size,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        info["tp_rank"] = get_tensor_model_parallel_rank()
        info["tp_size"] = get_tensor_model_parallel_world_size()
        info["pp_rank"] = get_pipeline_model_parallel_rank()
        info["pp_size"] = get_pipeline_model_parallel_world_size()
        info["moe_ep_rank"] = get_moe_expert_parallel_rank()
        info["moe_ep_size"] = get_moe_expert_parallel_world_size()
        info["moe_tp_rank"] = get_moe_tensor_parallel_rank()
        info["moe_tp_size"] = get_moe_tensor_parallel_world_size()
    except (ImportError, AttributeError, AssertionError):
        info["distributed_error"] = True

    try:
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            get_attention_dp_size,
            get_attention_tp_rank,
            get_attention_tp_size,
            get_local_attention_dp_rank,
            get_local_attention_dp_size,
            is_dp_attention_enabled,
        )

        info["enable_dp_attention"] = is_dp_attention_enabled()
        info["attn_tp_rank"] = get_attention_tp_rank()
        info["attn_tp_size"] = get_attention_tp_size()
        info["attn_dp_rank"] = get_attention_dp_rank()
        info["attn_dp_size"] = get_attention_dp_size()
        info["local_attn_dp_rank"] = get_local_attention_dp_rank()
        info["local_attn_dp_size"] = get_local_attention_dp_size()
    except (ImportError, AttributeError, AssertionError):
        info["dp_attention_error"] = True

    return info


def _collect_megatron_parallel_info():
    info = {}

    try:
        from megatron.core import parallel_state as mpu

        info["tp_rank"] = mpu.get_tensor_model_parallel_rank()
        info["tp_size"] = mpu.get_tensor_model_parallel_world_size()
        info["pp_rank"] = mpu.get_pipeline_model_parallel_rank()
        info["pp_size"] = mpu.get_pipeline_model_parallel_world_size()
        info["dp_rank"] = mpu.get_data_parallel_rank()
        info["dp_size"] = mpu.get_data_parallel_world_size()
        info["cp_rank"] = mpu.get_context_parallel_rank()
        info["cp_size"] = mpu.get_context_parallel_world_size()
        info["vpp_rank"] = mpu.get_virtual_pipeline_model_parallel_rank()
        info["vpp_size"] = mpu.get_virtual_pipeline_model_parallel_world_size()
        info["ep_rank"] = mpu.get_expert_model_parallel_rank()
        info["ep_size"] = mpu.get_expert_model_parallel_world_size()
        info["etp_rank"] = mpu.get_expert_tensor_parallel_rank()
        info["etp_size"] = mpu.get_expert_tensor_parallel_world_size()
        info["edp_rank"] = mpu.get_expert_data_parallel_rank()
        info["edp_size"] = mpu.get_expert_data_parallel_world_size()
        info["tcp_rank"] = mpu.get_tensor_and_context_parallel_rank()
        info["tcp_size"] = mpu.get_tensor_and_context_parallel_world_size()
        info["etmp_rank"] = mpu.get_expert_tensor_and_model_parallel_rank()
        info["etmp_size"] = mpu.get_expert_tensor_and_model_parallel_world_size()
        info["tp_src_rank"] = mpu.get_tensor_model_parallel_src_rank()
        info["mp_src_rank"] = mpu.get_model_parallel_src_rank()
        info["dp_src_rank"] = mpu.get_data_parallel_src_rank()
    except (ImportError, AttributeError, AssertionError):
        info["megatron_error"] = True

    return info


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
                result = target._handle_http_control_request(
                    method=method, body=req_body
                )
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
    handler, base_port: int, timeout_seconds: int = 60
) -> Optional["_ZmqRpcBroadcast"]:
    """A general-purpose minimal RPC to support broadcasting executions to multi processes"""
    import zmq

    rank = _get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    port = base_port + rank
    local_addr = f"tcp://{_get_local_ip_by_remote()}:{port}"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")

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


class _LocalOnlyBroadcast(_RpcBroadcastBase):
    """Calls methods directly on the local dumper, wrapping the result in a list."""

    def __init__(self, dumper: "_Dumper"):
        self._dumper = dumper

    def __getattr__(self, method_name: str):
        def call(*args, **kwargs):
            return [getattr(self._dumper, method_name)(*args, **kwargs)]

        return call


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


def get_int_env_var(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


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


# -------------------------------------- singleton ------------------------------------------


dumper = _Dumper(config=_DumperConfig.from_env())


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

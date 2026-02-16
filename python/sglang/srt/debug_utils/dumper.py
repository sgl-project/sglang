import json
import os
import re
import socket
import threading
import time
from copy import deepcopy
from functools import cached_property
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

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

    Alternatively, disable at startup and enable via HTTP:
    1. `python ...`
    2. `curl -X POST http://localhost:40000/dumper -d '{"enable": true}'`

    Related: `sglang.srt.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(
        self,
        *,
        enable: bool,
        base_dir: Path,
        filter: Optional[str] = None,
        enable_write_file: bool = True,
        enable_value: bool = True,
        enable_grad: bool = False,
        enable_model_value: bool = True,
        enable_model_grad: bool = True,
        partial_name: Optional[str] = None,
        enable_http_server: bool = True,
    ):
        # Config
        self._enable = enable
        # TODO (1) support filtering kv instead of name only (2) allow HTTP req change it
        self._filter = filter
        self._base_dir = base_dir
        self._enable_write_file = enable_write_file
        self._enable_value = enable_value
        self._enable_grad = enable_grad
        self._enable_model_value = enable_model_value
        self._enable_model_grad = enable_model_grad

        # States
        self._partial_name = partial_name
        self._dump_index = 0
        self._forward_pass_id = 0
        self._global_ctx = {}
        self._override_enable = None
        self._http_server_handled = not enable_http_server

    @classmethod
    def from_env(cls) -> "_Dumper":
        return cls(
            enable=get_bool_env_var("SGLANG_DUMPER_ENABLE", "0"),
            base_dir=Path(_get_str_env_var("SGLANG_DUMPER_DIR", "/tmp")),
            filter=_get_str_env_var("SGLANG_DUMPER_FILTER"),
            enable_write_file=get_bool_env_var("SGLANG_DUMPER_WRITE_FILE", "1"),
            enable_value=get_bool_env_var("SGLANG_DUMPER_ENABLE_VALUE", "1"),
            enable_grad=get_bool_env_var("SGLANG_DUMPER_ENABLE_GRAD", "0"),
            enable_model_value=get_bool_env_var(
                "SGLANG_DUMPER_ENABLE_MODEL_VALUE", "1"
            ),
            enable_model_grad=get_bool_env_var("SGLANG_DUMPER_ENABLE_MODEL_GRAD", "1"),
            partial_name=_get_str_env_var("SGLANG_DUMPER_PARTIAL_NAME"),
            enable_http_server=get_bool_env_var(
                "SGLANG_ENABLE_DUMPER_HTTP_SERVER", "1"
            ),
        )

    def on_forward_pass_start(self):
        """This should be called on all ranks."""

        # Even if SGLANG_DUMPER_ENABLE=0, users may want to use HTTP endpoint to enable it
        self._ensure_http_server()

        if not self._enable:
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
        _start_maybe_http_server(self)

    def _ensure_partial_name(self):
        if self._partial_name is None:
            self._partial_name = _get_partial_name()
            print(f"[Dumper] Choose partial_name={self._partial_name}")

    def set_ctx(self, **kwargs):
        """
        Example:

        dumper.override_enable(self.layer_id <= 3)
        dumper.set_ctx(layer_id=self.layer_id)
        ...
        dumper.set_ctx(layer_id=None)
        """
        self._global_ctx = {
            k: v for k, v in (self._global_ctx | kwargs).items() if v is not None
        }

    def override_enable(self, value: bool):
        self._override_enable = value

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
            enable_value=self._enable_value,
            enable_curr_grad=False,
            enable_future_grad=self._enable_grad,
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
                enable_value=self._enable_model_value,
                enable_curr_grad=self._enable_model_grad,
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

        if not (self._enable and (self._override_enable is not False)):
            return
        if (f := self._filter) is not None and re.search(f, name) is None:
            return
        if not (enable_value or enable_curr_grad or enable_future_grad):
            return

        if self._forward_pass_id < 1:
            print("Dump without on_forward_pass_start()")

        value = _materialize_value(value)

        if enable_value:
            self._dump_single(
                tag=value_tag,
                name=name,
                value=value,
                extra_kwargs=extra_kwargs,
                save=save,
            )

        if (
            enable_curr_grad
            and isinstance(value, torch.Tensor)
            and (g := value.grad) is not None
        ):
            self._dump_single(
                tag=grad_tag,
                name=f"grad__{name}",
                value=g,
                extra_kwargs=extra_kwargs,
                save=save,
            )

        if enable_future_grad:
            self._register_dump_grad_hook(
                name=name,
                tensor=value,
                save=save,
                **extra_kwargs,
            )

    def _register_dump_grad_hook(
        self, *, name: str, tensor, save: bool, **kwargs
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if not tensor.requires_grad:
            return

        captured_forward_pass_id = self._forward_pass_id
        captured_extra = deepcopy(dict(**kwargs))

        def grad_hook(grad: torch.Tensor) -> None:
            self._dump_single(
                tag="Dumper.Grad",
                name=f"grad__{name}",
                value=grad,
                extra_kwargs=captured_extra,
                save=save,
                forward_pass_id=captured_forward_pass_id,
            )

        tensor.register_hook(grad_hook)

    def _dump_single(
        self,
        *,
        tag: str,
        name: str,
        value,
        extra_kwargs: dict,
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
            name=name,
            dump_index=self._dump_index,
            **extra_kwargs,
            **self._global_ctx,
        )
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs.items()) + ".pt"
        path = self._base_dir / f"sglang_dump_{self._partial_name}" / full_filename

        print(
            f"[{tag}] [{rank}, {time.time()}] {path} "
            f"type={type(value)} "
            f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
            f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
            f"device={value.device if isinstance(value, torch.Tensor) else None} "
            f"id={id(value)} "
            f"sample_value={get_truncated_value(value)}"
        )

        if self._enable_write_file and save:
            path.parent.mkdir(parents=True, exist_ok=True)
            output_data = {
                "value": value.data if isinstance(value, torch.nn.Parameter) else value,
                "meta": dict(**full_kwargs, **self._static_meta),
            }
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


def _get_partial_name():
    rank = _get_rank()
    object_list = [str(time.time()) if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(object_list, device="cuda")
    return object_list[0]


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


def _start_maybe_http_server(dumper):
    http_port = get_int_env_var("SGLANG_DUMPER_SERVER_PORT", 40000)
    zmq_base_port = get_int_env_var("SGLANG_DUMPER_ZMQ_BASE_PORT", 16800)
    if http_port <= 0:
        return

    local_handler = _DumperRpcHandler(dumper)
    rpc_handles = _create_zmq_rpc_handles(local_handler, base_port=zmq_base_port)

    if _get_rank() == 0:
        handler_class = _make_dumper_http_handler(rpc_handles=rpc_handles)
        server = HTTPServer(("0.0.0.0", http_port), handler_class)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[Dumper] HTTP server started on port {http_port}")


def _make_dumper_http_handler(rpc_handles):
    class _DumperHTTPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/dumper":
                try:
                    self._handle_endpoint_dumper()
                    self.send_response(200)
                    self.end_headers()
                except Exception as e:
                    self.send_error(400, str(e))
            else:
                self.send_error(404)

        def _get_request_body(self):
            content_length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(content_length))

        def _handle_endpoint_dumper(self):
            data = self._get_request_body()
            print(f"[Dumper#{_get_rank()}] Handle HTTP endpoint {data=}")
            for rpc_handle in rpc_handles:
                rpc_handle.set_enable(data["enable"])

    return _DumperHTTPHandler


class _DumperRpcHandler:
    def __init__(self, dumper):
        self._dumper = dumper

    def set_enable(self, enable: bool):
        print(f"[DumperRpcHandler] set_enable {enable=}")
        self._dumper._enable = enable


# -------------------------------------- zmq rpc ------------------------------------------


def _create_zmq_rpc_handles(handler, base_port: int) -> Optional[List["_ZmqRpcHandle"]]:
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
        dist.all_gather_object(all_addresses, local_addr)
    else:
        all_addresses = [local_addr]
    print(f"[Dumper.ZmqRpc] rank={rank} all_addresses={all_addresses}")

    if rank == 0:
        handles = []
        for i, addr in enumerate(all_addresses):
            req_socket = ctx.socket(zmq.REQ)
            req_socket.connect(addr)
            handles.append(_ZmqRpcHandle(req_socket, debug_name=f"rank-{i}"))
        return handles
    else:
        return None


class _ZmqRpcHandle:
    """Proxy object to call remote handler methods via ZMQ."""

    def __init__(self, socket, debug_name):
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


# --------------------------------- copied code (avoid dependency) --------------------------------------


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()
    truthy_values = ("true", "1")
    return value in truthy_values


def _get_str_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value


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


dumper = _Dumper.from_env()


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

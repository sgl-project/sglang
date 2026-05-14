import json
import os
import re
import socket
import threading
import time
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

    Disable at startup and enable via HTTP:
    1. `SGLANG_DUMPER_ENABLE=0 python ...`
    2. `curl -X POST http://localhost:40000/dumper -d '{"enable": true}'`

    Related: `sglang.srt.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(self):
        # Do not import `sglang` to make this file standalone
        self._enable = bool(int(os.environ.get("SGLANG_DUMPER_ENABLE", "1")))
        # TODO (1) support filtering kv instead of name only (2) allow HTTP req change it
        self._filter = os.environ.get("SGLANG_DUMPER_FILTER")
        self._base_dir = Path(os.environ.get("SGLANG_DUMPER_DIR", "/tmp"))
        self._enable_write_file = bool(
            int(os.environ.get("SGLANG_DUMPER_WRITE_FILE", "1"))
        )
        self._partial_name: Optional[str] = None
        self._dump_index = 0
        self._forward_pass_id = 0
        self._global_ctx = {}
        self._override_enable = None
        self._http_server_handled = False

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

    def dump(self, name, value, save: bool = True, **kwargs):
        self._ensure_http_server()

        if not (self._enable and (self._override_enable is not False)):
            return
        if (f := self._filter) is not None and re.search(f, name) is None:
            return

        if self._forward_pass_id < 1:
            print("Dump without on_forward_pass_start()")
        self._ensure_partial_name()
        self._dump_index += 1

        rank = _get_rank()
        full_kwargs = dict(
            forward_pass_id=self._forward_pass_id,
            rank=rank,
            name=name,
            dump_index=self._dump_index,
            **kwargs,
            **self._global_ctx,
        )
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs.items()) + ".pt"
        path = self._base_dir / f"sglang_dump_{self._partial_name}" / full_filename

        sample_value = get_truncated_value(value)

        print(
            f"[Dumper] [{rank}, {time.time()}] {path} "
            f"type={type(value)} "
            f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
            f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
            f"device={value.device if isinstance(value, torch.Tensor) else None} "
            f"id={id(value)} "
            f"sample_value={sample_value}"
        )

        if self._enable_write_file and save:
            path.parent.mkdir(parents=True, exist_ok=True)
            _torch_save(value, str(path))


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


# -------------------------------------- http control server ------------------------------------------


def _start_maybe_http_server(dumper):
    http_port = int(os.environ.get("SGLANG_DUMPER_SERVER_PORT", "40000"))
    zmq_base_port = int(os.environ.get("SGLANG_DUMPER_ZMQ_BASE_PORT", "16800"))
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


dumper = _Dumper()


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

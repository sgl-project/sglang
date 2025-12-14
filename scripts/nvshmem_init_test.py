#!/usr/bin/env python3
"""
Minimal NVSHMEM bootstrap bridge tester.

Usage:
  # 1. Start the HTTP bootstrap server (on any accessible node)
  python scripts/nvshmem_init_test.py server --host 0.0.0.0 --port 8997

  # 2. Pairwise Mode (Default) - 4 independent pairs across 2 nodes
  # Node 1 (Prefill):
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr <IP> ... \
    scripts/nvshmem_init_test.py prefill --bootstrap <SERVER_IP>:8997
  # Node 2 (Decode):
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr <IP> ... \
    scripts/nvshmem_init_test.py decode --bootstrap <SERVER_IP>:8997

  # 3. Full Mesh Mode - Single 8-GPU world
  # Node 1 (Prefill, Ranks 0-3):
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 ... \
    scripts/nvshmem_init_test.py prefill --bootstrap <SERVER_IP>:8997 --nranks 8 --rank 0 --pair-id global
  # Node 2 (Decode, Ranks 4-7):
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 ... \
    scripts/nvshmem_init_test.py decode --bootstrap <SERVER_IP>:8997 --nranks 8 --rank 4 --pair-id global

  Note: In Full Mesh mode, you must manually manage the 'rank' argument if not using a script wrapper that maps LOCAL_RANK to global rank.
  The script automatically uses LOCAL_RANK for device index.


  python scripts/nvshmem_init_test.py server --host 10.192.2.60 --port 8997

#pairwise: 
torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr 10.192.2.60 --master_port 12345 \
    scripts/nvshmem_init_test.py prefill --bootstrap 10.192.2.60:8997

torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr 10.192.2.60 --master_port 12345 \
    scripts/nvshmem_init_test.py decode --bootstrap 10.192.2.60:8997

#fullmesh:
torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr 10.192.2.60 --master_port 12345 \
    scripts/nvshmem_init_test.py prefill  --bootstrap 10.192.2.60:8997 \
    --nranks 8 --rank 0  --pair-id global

torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr 10.192.2.60 --master_port 12345 \
    scripts/nvshmem_init_test.py decode  --bootstrap 10.192.2.60:8997 \
    --nranks 8 --rank 4  --pair-id global
"""

from __future__ import annotations

import argparse
import base64
import pickle
import json
import logging
import sys
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

import requests
try:
    import nvshmem  # type: ignore
    import nvshmem.core  # type: ignore
    from nvshmem import bindings as nvshmem_bindings  # type: ignore
    from nvshmem.core.interop import torch as nvshmem_torch  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"nvshmem import failed: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"torch import failed: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    from cuda.core.experimental import Device as CudaDevice  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"cuda.core.experimental import failed: {exc}", file=sys.stderr)
    sys.exit(1)


LOG = logging.getLogger("nvshmem_init_test")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

UID_STORE: dict[str, Optional[str]] = {"default": None}

# Query NVSHMEM for the unique id type up front so we can serialize later.
_NVSHMEM_UID_TYPE = type(nvshmem.core.get_unique_id())
class BootstrapHandler(BaseHTTPRequestHandler):
    def _extract_pair(self) -> str:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query or "")
        return str(params.get("pair_id", ["default"])[0])

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-type", content_type)
        self.end_headers()

    def do_PUT(self):  # noqa: N802 (stdlib signature)
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        if parsed.path != "/nvshmem_uid":
            self._set_headers(404)
            self.wfile.write(b'{"error":"unknown path"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length))
        pair_id = str(data.get("pair_id", "default"))
        UID_STORE[pair_id] = data.get("uid")
        LOG.info("Published NVSHMEM UID (pair=%s)", pair_id)
        self._set_headers(200)
        self.wfile.write(b'{"status":"ok"}')

    def do_GET(self):  # noqa: N802 (stdlib signature)
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        if parsed.path != "/nvshmem_uid":
            self._set_headers(404)
            self.wfile.write(b'{"error":"unknown path"}')
            return
        pair_id = self._extract_pair()
        if UID_STORE.get(pair_id) is None:
            self._set_headers(404)
            self.wfile.write(b'{"error":"uid not set"}')
            return
        self._set_headers(200)
        self.wfile.write(
            json.dumps({"uid": UID_STORE[pair_id]}).encode("ascii")
        )

    def log_message(self, format: str, *args):  # noqa: A003 - BaseHTTPRequestHandler signature
        LOG.debug("HTTP %s - %s", self.client_address[0], format % args)


def run_server(host: str, port: int):
    LOG.info("Starting bootstrap server on %s:%s", host, port)
    server = HTTPServer((host, port), BootstrapHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _select_device_indices() -> Tuple[int, int]:
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    return local_rank, local_rank


def _init_device(device_index: int) -> Tuple[torch.device, Optional[CudaDevice]]:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    torch_device = torch.device("cuda", device_index)
    try:
        nv_dev = CudaDevice(device_index)
        nv_dev.set_current()
    except Exception:
        nv_dev = None
    return torch_device, nv_dev


def _encode_uid(raw_uid) -> bytes:
    if not isinstance(raw_uid, _NVSHMEM_UID_TYPE):
        raise TypeError(f"Unexpected unique ID type {type(raw_uid)}")
    return base64.b64encode(pickle.dumps(raw_uid)).decode("ascii")


def _prefill_flow(bootstrap: str, device_index: int, pair_id: str, rank: int, nranks: int):
    torch_device, nv_dev = _init_device(device_index)
    LOG.info("Prefill using torch device %s", torch_device)

    if rank == 0:
        raw_uid = nvshmem.core.get_unique_id()
        LOG.info("Prefill raw UID repr: %r", raw_uid)
        uid_b64 = _encode_uid(raw_uid)
        payload = {"uid": uid_b64, "pair_id": pair_id}
        resp = requests.put(f"http://{bootstrap}/nvshmem_uid", json=payload, timeout=5)
        resp.raise_for_status()
        LOG.info("Published UID to %s (pair=%s)", bootstrap, pair_id)
        uid_obj = raw_uid
    else:
        # If we are part of a larger world (nranks > 2) and not rank 0, we must fetch the UID
        # from the bootstrap server (published by rank 0).
        uid_obj = _fetch_uid(bootstrap, pair_id)

    init_kwargs = dict(uid=uid_obj, rank=rank, nranks=nranks, initializer_method="uid")
    if nv_dev is not None:
        init_kwargs["device"] = nv_dev
    
    LOG.info("Prefill: Calling nvshmem.core.init with rank=%s nranks=%s...", rank, nranks)
    nvshmem.core.init(**init_kwargs)  # type: ignore[arg-type]
    LOG.info("Prefill: nvshmem.core.init returned for rank=%s", rank)

    if nvshmem_bindings is not None:
        LOG.info("Prefill: Calling barrier_all for rank=%s...", rank)
        nvshmem_bindings.barrier_all()
        LOG.info("Prefill: barrier_all returned for rank=%s", rank)
    LOG.info("NVSHMEM initialized (role=prefill rank=%s nranks=%s).", rank, nranks)
    _allocate_demo_tensor(torch_device)


def _fetch_uid(bootstrap: str, pair_id: str, timeout: float = 60.0, interval: float = 1.0):
    uid_obj = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://{bootstrap}/nvshmem_uid?pair_id={pair_id}", timeout=5)
            if resp.status_code == 200:
                uid_obj = pickle.loads(base64.b64decode(resp.json()["uid"]))
                LOG.info("Fetched raw UID repr: %r", uid_obj)
                break
        except Exception:
            pass
        LOG.info("Waiting for UID from %s (pair=%s)...", bootstrap, pair_id)
        time.sleep(interval)
    
    if uid_obj is None:
        raise RuntimeError("Timed out waiting for NVSHMEM UID.")
    return uid_obj


def _decode_flow(bootstrap: str, device_index: int, timeout: float, interval: float, pair_id: str, rank: int, nranks: int):
    torch_device, nv_dev = _init_device(device_index)
    LOG.info("Decode using torch device %s", torch_device)

    uid_obj = _fetch_uid(bootstrap, pair_id, timeout, interval)

    init_kwargs = dict(uid=uid_obj, rank=rank, nranks=nranks, initializer_method="uid")
    if nv_dev is not None:
        init_kwargs["device"] = nv_dev
    
    LOG.info("Decode: Calling nvshmem.core.init with rank=%s nranks=%s...", rank, nranks)
    nvshmem.core.init(**init_kwargs)  # type: ignore[arg-type]
    LOG.info("Decode: nvshmem.core.init returned for rank=%s", rank)

    if nvshmem_bindings is not None:
        LOG.info("Decode: Calling barrier_all for rank=%s...", rank)
        nvshmem_bindings.barrier_all()
        LOG.info("Decode: barrier_all returned for rank=%s", rank)
    LOG.info("NVSHMEM initialized (role=decode rank=%s nranks=%s).", rank, nranks)
    _allocate_demo_tensor(torch_device)


def _allocate_demo_tensor(torch_device: torch.device):
    factory = nvshmem_torch.tensor
    tensor = factory((4,), dtype=torch.float32)
    tensor.fill_(torch.cuda.current_device() if torch.cuda.is_available() else -1)
    LOG.info("Allocated NVSHMEM tensor on %s with values %s", torch_device, tensor)


def parse_args():
    LOG.info("sys.argv: %s", sys.argv)
    parser = argparse.ArgumentParser(description="NVSHMEM init bridge tester.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Detect torchrun / env vars
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    node_rank = int(os.getenv("GROUP_RANK", os.getenv("NODE_RANK", "0")))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    # Simple assumption: 4 GPUs per node for auto-rank calculation
    global_rank = node_rank * 4 + local_rank
    
    # Smart defaults based on environment
    if world_size > 2:
        default_nranks = world_size
        default_pair = "global"
    else:
        default_nranks = 2
        default_pair = f"pair_{local_rank}"

    server = subparsers.add_parser("server", help="Run bootstrap HTTP server.")
    server.add_argument("--host", type=str, default="127.0.0.1")
    server.add_argument("--port", type=int, default=8997)

    prefill = subparsers.add_parser("prefill", help="Run prefill NVSHMEM init.")
    prefill.add_argument("--bootstrap", type=str, default="127.0.0.1:8997")
    prefill.add_argument("--device-index", type=int, default=local_rank)
    prefill.add_argument("--pair-id", type=str, default=default_pair)
    prefill.add_argument("--nranks", type=int, default=default_nranks)
    prefill.add_argument("--rank", type=int, default=global_rank if global_rank < 4 else 0) # Default logic: if we are node 0, we are likely prefill ranks 0-3

    decode = subparsers.add_parser("decode", help="Run decode NVSHMEM init.")
    decode.add_argument("--bootstrap", type=str, default="127.0.0.1:8997")
    decode.add_argument("--device-index", type=int, default=local_rank)
    decode.add_argument("--timeout", type=float, default=60.0)
    decode.add_argument("--interval", type=float, default=1.0)
    decode.add_argument("--pair-id", type=str, default=default_pair)
    decode.add_argument("--nranks", type=int, default=default_nranks)
    decode.add_argument("--rank", type=int, default=global_rank if global_rank >= 4 else 1) # Default logic: if we are node 1, we are likely decode ranks 4-7

    args = parser.parse_args()
    
    # Override rank if running under torchrun to ensure unique ranks per process
    if os.getenv("LOCAL_RANK") is not None:
        args.rank = global_rank
        # Also ensure device index matches local rank
        args.device_index = local_rank

    return args


def main():
    args = parse_args()
    if args.mode == "server":
        run_server(args.host, args.port)
    elif args.mode == "prefill":
        _prefill_flow(args.bootstrap, args.device_index, args.pair_id, args.rank, args.nranks)
    elif args.mode == "decode":
        _decode_flow(args.bootstrap, args.device_index, args.timeout, args.interval, args.pair_id, args.rank, args.nranks)
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full-mesh NVSHMEM transfer tester.

It bootstraps NVSHMEM across two nodes (prefill + decode) and drives a
pairwise transfer: GPU0 on node 0 (tp0 for prefill) writes to GPU0 on node 1
(tp0 for decode), GPU1 talks to GPU1, etc. The script assumes a full-mesh
initialization where all GPUs take part in a single NVSHMEM world.

Usage:
  # 1. Start the HTTP bootstrap server (on any accessible node)
  python scripts/nvshmem_transfer_test.py server --host 0.0.0.0 --port 8997

  # 2. Launch prefill ranks on node 0 (LOCAL_RANK drives GPU selection)
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr <IP> ... \
    scripts/nvshmem_transfer_test.py prefill --bootstrap <SERVER_IP>:8997 \
    --nranks 8 --pair-id global --tensor-len 128

  # 3. Launch decode ranks on node 1
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr <IP> ... \
    scripts/nvshmem_transfer_test.py decode --bootstrap <SERVER_IP>:8997 \
    --nranks 8 --pair-id global --tensor-len 128

Set WORLD_SIZE, RANK, NODE_RANK, and LOCAL_RANK via torchrun or mpirun so
each process knows which GPU/rank it owns. The script automatically maps
LOCAL_RANK -> CUDA device unless overridden.
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
    from nvshmem.core.interop.torch import get_peer_tensor  # type: ignore
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


LOG = logging.getLogger("nvshmem_transfer_test")
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


def _quiet_stream():
    if nvshmem_bindings is None:
        return
    try:
        stream = torch.cuda.current_stream().cuda_stream
    except Exception:
        return
    try:
        nvshmem_bindings.quiet_on_stream(stream)
    except Exception:
        pass


def _compute_partner_rank(role: str, rank: int, nranks: int) -> int:
    if nranks % 2 != 0:
        raise ValueError(f"Pairwise transfer requires an even number of ranks (got {nranks}).")
    half = nranks // 2
    if role == "prefill":
        if rank >= half:
            raise ValueError(f"Prefill rank must be in [0, {half}), got rank={rank}.")
        return rank + half
    if role == "decode":
        if rank < half:
            raise ValueError(f"Decode rank must be in [{half}, {nranks}), got rank={rank}.")
        return rank - half
    raise ValueError(f"Unknown role {role}")


def _wait_for_flag(flag_tensor, expected: int, timeout: float, desc: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = int(flag_tensor[0].item())
        if value == expected:
            return value
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {desc} (expected flag={expected}).")


def _run_pairwise_transfer(
    role: str,
    rank: int,
    nranks: int,
    tensor_len: int,
    transfer_timeout: float,
):
    partner = _compute_partner_rank(role, rank, nranks)
    mailbox = nvshmem_torch.tensor((tensor_len,), dtype=torch.int32)
    flag = nvshmem_torch.tensor((1,), dtype=torch.int32)
    mailbox.fill_(rank)
    flag.zero_()
    LOG.info(
        "[role=%s rank=%d] Partner rank=%d tensor_len=%d timeout=%.1fs",
        role,
        rank,
        partner,
        tensor_len,
        transfer_timeout,
    )

    if nvshmem_bindings is not None:
        nvshmem_bindings.barrier_all()

    try:
        if role == "prefill":
            payload = (
                torch.arange(tensor_len, dtype=torch.int32, device=mailbox.device) + rank * 1000
            )
            peer_mailbox = get_peer_tensor(mailbox, partner)
            peer_flag = get_peer_tensor(flag, partner)
            peer_mailbox.copy_(payload)
            peer_flag[0] = 1
            #_quiet_stream()
            LOG.info(
                "[role=prefill rank=%d] sent payload min=%d max=%d",
                rank,
                int(payload.min()),
                int(payload.max()),
            )
            # Leave the local flag set to 1 so decode can observe it before we reset.
            #flag.zero_()
            _wait_for_flag(flag, 2, transfer_timeout, f"decode ack from rank {partner}")
            LOG.info("[role=prefill rank=%d] received ack from rank %d", rank, partner)
            flag.zero_()
        else:
            _wait_for_flag(flag, 1, transfer_timeout, f"prefill payload from rank {partner}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            values = mailbox.clone().cpu().tolist()
            LOG.info(
                "[role=decode rank=%d] received payload[:4]=%s",
                rank,
                values[: min(4, len(values))],
            )
            flag[0] = 0
            peer_flag = get_peer_tensor(flag, partner)
            peer_flag[0] = 2
            #_quiet_stream()
            LOG.info("[role=decode rank=%d] sent ack to rank %d", rank, partner)
    except TimeoutError as exc:
        LOG.error("[role=%s rank=%d] %s", role, rank, exc)
        raise RuntimeError(f"{role} rank {rank} timed out during transfer") from exc
    finally:
        if nvshmem_bindings is not None:
            try:
                nvshmem_bindings.barrier_all()
            except Exception:
                pass
        for tensor in (mailbox, flag):
            try:
                nvshmem.core.free_tensor(tensor)
            except Exception:
                pass


def _prefill_flow(
    bootstrap: str,
    device_index: int,
    pair_id: str,
    rank: int,
    nranks: int,
    tensor_len: int,
    transfer_timeout: float,
):
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
    try:
        _run_pairwise_transfer("prefill", rank, nranks, tensor_len, transfer_timeout)
    finally:
        try:
            nvshmem.core.finalize()
        except Exception:
            pass


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


def _decode_flow(
    bootstrap: str,
    device_index: int,
    timeout: float,
    interval: float,
    pair_id: str,
    rank: int,
    nranks: int,
    tensor_len: int,
    transfer_timeout: float,
):
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
    try:
        _run_pairwise_transfer("decode", rank, nranks, tensor_len, transfer_timeout)
    finally:
        try:
            nvshmem.core.finalize()
        except Exception:
            pass


def parse_args():
    LOG.info("sys.argv: %s", sys.argv)
    parser = argparse.ArgumentParser(description="Full-mesh NVSHMEM transfer tester.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Detect torchrun / env vars
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    node_rank = int(os.getenv("GROUP_RANK", os.getenv("NODE_RANK", "0")))
    world_size = int(os.getenv("WORLD_SIZE", "2"))
    per_node = int(os.getenv("LOCAL_WORLD_SIZE", os.getenv("NPROC_PER_NODE", "4")))

    # Fallback global rank estimate if RANK is not exported (common for manual runs)
    guessed_global_rank = node_rank * per_node + local_rank
    global_rank = int(os.getenv("RANK", str(guessed_global_rank)))

    default_nranks = max(2, world_size)
    default_pair = "global"
    half_world = max(1, default_nranks // 2)

    server = subparsers.add_parser("server", help="Run bootstrap HTTP server.")
    server.add_argument("--host", type=str, default="127.0.0.1")
    server.add_argument("--port", type=int, default=8997)

    prefill = subparsers.add_parser("prefill", help="Run prefill NVSHMEM init.")
    prefill.add_argument("--bootstrap", type=str, default="127.0.0.1:8997")
    prefill.add_argument("--device-index", type=int, default=local_rank)
    prefill.add_argument("--pair-id", type=str, default=default_pair)
    prefill.add_argument("--nranks", type=int, default=default_nranks)
    prefill.add_argument(
        "--rank",
        type=int,
        default=global_rank if global_rank < half_world else 0,
        help="Global NVSHMEM rank for this process (prefill ranks reside in the first half).",
    )
    prefill.add_argument(
        "--tensor-len",
        type=int,
        default=128,
        help="Number of int32 elements transferred per pair.",
    )
    prefill.add_argument(
        "--transfer-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for NVSHMEM flags/acks during the transfer test.",
    )

    decode = subparsers.add_parser("decode", help="Run decode NVSHMEM init.")
    decode.add_argument("--bootstrap", type=str, default="127.0.0.1:8997")
    decode.add_argument("--device-index", type=int, default=local_rank)
    decode.add_argument("--timeout", type=float, default=60.0)
    decode.add_argument("--interval", type=float, default=1.0)
    decode.add_argument("--pair-id", type=str, default=default_pair)
    decode.add_argument("--nranks", type=int, default=default_nranks)
    decode.add_argument(
        "--rank",
        type=int,
        default=global_rank if global_rank >= half_world else half_world,
        help="Global NVSHMEM rank for this process (decode ranks reside in the second half).",
    )
    decode.add_argument(
        "--tensor-len",
        type=int,
        default=128,
        help="Number of int32 elements expected from the paired prefill GPU.",
    )
    decode.add_argument(
        "--transfer-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for NVSHMEM flags/acks during the transfer test.",
    )

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
        _prefill_flow(
            args.bootstrap,
            args.device_index,
            args.pair_id,
            args.rank,
            args.nranks,
            args.tensor_len,
            args.transfer_timeout,
        )
    elif args.mode == "decode":
        _decode_flow(
            args.bootstrap,
            args.device_index,
            args.timeout,
            args.interval,
            args.pair_id,
            args.rank,
            args.nranks,
            args.tensor_len,
            args.transfer_timeout,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()

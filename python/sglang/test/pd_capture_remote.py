"""Supervised external P fixture for cross-node RDMA capture tests.

Only observation files and immutable snapshot references use shared storage.
Serving KV/metadata and snapshot tensors travel through Mooncake.
"""

import argparse
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase, free_port
from sglang.test.training_capture_rdma import RDMACaptureRuntimeBase, load_setup


class ExternalPrefill:
    def __init__(self, url):
        self.url = url

    def poll(self):
        # A timeout is an observation failure, never evidence of process exit.
        requests.get(self.url + "/server_info", timeout=10).raise_for_status()


class CrossNodePDCaptureRuntimeBase(RDMACaptureRuntimeBase):
    transfer_protocol = "rdma"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.peer = json.loads(
            Path(os.environ["TRAINING_CAPTURE_PD_PREFILL"]).read_text()
        )
        if (
            cls.peer["hostname"] == socket.gethostname()
            or cls.peer["protocol"] != "rdma"
            or cls.peer["store_endpoint"] != cls.store_setup["master_server_addr"]
            or cls.peer["model"] != cls.model
        ):
            raise ValueError("External P must match this test on another RDMA node")
        cls.prefill_host = cls.peer["host"]
        cls.decode_host = cls.store_setup["local_hostname"]
        cls.ib_device = cls.store_setup["rdma_devices"]
        cls.prefill = ExternalPrefill(cls.peer["url"])
        cls.prefill.poll()

    def new_bootstrap_port(self):
        return self.peer["bootstrap_port"]

    def reference_paths(self, root):
        return super().reference_paths(root) + sorted(
            Path(self.peer["source_root"]).rglob("*.pt")
        )

    def launch(self, role, *args, **kwargs):
        if role == "prefill":
            self.prefill.poll()
            return self.prefill, self.prefill.url
        return super().launch(role, *args, **kwargs)

    @staticmethod
    def stop_process(process):
        if isinstance(process, ExternalPrefill):
            # This fixture is owned by its separate foreground supervisor.
            return
        PDCaptureRuntimeBase.stop_process(process)


def serve_prefill(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    fixture = PDCaptureRuntimeBase()
    fixture.model = os.environ.get("TRAINING_CAPTURE_TEST_MODEL", "Qwen/Qwen3-0.6B")
    fixture.prefill_host = args.host
    fixture.transfer_protocol = "rdma"
    fixture.ib_device = args.devices
    fixture.bootstrap_port = free_port(args.host)
    fixture.store_setup = load_setup(args.setup)
    # P owns no Catalog reservation and must never connect to this endpoint.
    fixture.catalog = SimpleNamespace(endpoint="http://127.0.0.1:1")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        process, url = fixture.launch(
            "prefill", root, replay=True, tp_size=1, pp_size=1
        )
        ready = {
            "pid": os.getpid(),
            "server_pid": process.pid,
            "hostname": socket.gethostname(),
            "host": args.host,
            "url": url,
            "protocol": fixture.transfer_protocol,
            "bootstrap_port": fixture.bootstrap_port,
            "source_root": str(root / "prefill"),
            "model": fixture.model,
            "store_endpoint": fixture.store_setup["master_server_addr"],
        }
        Path(args.ready).write_text(json.dumps(ready))
        print(json.dumps({"pd_rdma_prefill_ready": ready}), flush=True)
        deadline = time.monotonic() + args.lifetime
        while not stop.wait(0.5) and time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("External P exited before fixture shutdown")
    finally:
        if not fixture.doCleanups():
            raise RuntimeError("External P cleanup failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--lifetime", type=float, default=1800)
    serve_prefill(parser.parse_args())

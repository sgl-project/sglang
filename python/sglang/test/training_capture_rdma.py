"""Dedicated remote RDMA Store fixture and independent post-producer readback."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import torch

from sglang.srt.training_capture.config import StoreSetup
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    digest_bytes,
    tensor_bytes,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase
from sglang.test.training_capture_utils import read_snapshot


def load_setup(path):
    setup = msgspec.json.decode(Path(path).read_bytes(), type=StoreSetup)
    if setup.protocol != "rdma" or setup.global_segment_size != 0:
        raise ValueError("RDMA clients must use rdma with no local storage segment")
    return msgspec.to_builtins(setup)


class RDMACaptureRuntimeBase(PDCaptureRuntimeBase):
    @classmethod
    def start_store(cls):
        cls.setup_path = os.environ["TRAINING_CAPTURE_RDMA_SETUP"]
        cls.store_setup = load_setup(cls.setup_path)

    def setUp(self):
        super().setUp()
        self.servers = []

    def launch(self, *args, **kwargs):
        process, url = super().launch(*args, **kwargs)
        self.servers.append(process)
        return process, url

    def exercise_remote(self, *, replay, draft_kind=None):
        self.exercise(replay=replay, draft_kind=draft_kind)
        for process in reversed(self.servers):
            self.stop_process(process)
            self.assertIsNotNone(process.poll())
        publications = self.root / "publications.json"
        payload = json.dumps(list(self.catalog.publications.values()))
        publications.write_text(payload)
        # A supervised external P can outlive the test driver. Retain only
        # manifest references for another readback after that fixture exits.
        if destination := os.environ.get("TRAINING_CAPTURE_RDMA_PUBLICATIONS"):
            Path(destination).write_text(payload)
        output = self.root / "readback.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.test.training_capture_rdma",
                "read",
                "--setup",
                self.setup_path,
                "--publications",
                str(publications),
                "--output",
                str(output),
                "--batch",
            ],
            check=True,
            timeout=120,
        )
        snapshots = json.loads(output.read_text())["snapshots"]
        self.assertEqual(len(snapshots), len(self.catalog.publications))
        self.assertEqual(
            {(row["sample_id"], row["generation_id"]) for row in snapshots},
            {
                (row["sample_id"], row["generation_id"])
                for row in self.catalog.publications.values()
            },
        )
        self.assertTrue(all(row["tensor_bytes"] > 0 for row in snapshots))
        self.assertTrue(all(row["native_read_batches"] == 1 for row in snapshots))


def free_port(host):
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def serve(args):
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    port = free_port(args.host)
    master = subprocess.Popen(
        [
            "mooncake_master",
            f"--rpc_port={port}",
            f"--metrics_port={free_port(args.host)}",
            f"--http_metadata_server_port={free_port(args.host)}",
        ]
    )
    segment = None
    try:
        deadline = time.monotonic() + 20
        while True:
            if master.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Remote Mooncake master failed to start")
            try:
                with socket.create_connection((args.host, port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        setup = msgspec.to_builtins(
            StoreSetup(
                local_hostname=args.host,
                master_server_addr=f"{args.host}:{port}",
                protocol="rdma",
                rdma_devices=args.devices,
            )
        )
        segment = MooncakeSnapshotStore.connect(
            setup | {"global_segment_size": args.segment_mib << 20}
        )
        # Native SDK initialization installs signal handlers. Restore the
        # fixture handler so shutdown also reaps our separate Master process.
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        ready = {
            "pid": os.getpid(),
            "master_pid": master.pid,
            "hostname": socket.gethostname(),
            "setup": setup,
            "segment_bytes": args.segment_mib << 20,
        }
        Path(args.ready).write_text(json.dumps(ready))
        print(json.dumps({"rdma_store_ready": ready}), flush=True)
        stop.wait(args.lifetime)
    finally:
        try:
            if segment is not None:
                segment.close()
        finally:
            # The wheel's CLI may wrap the actual Master binary. Leave the
            # wrapper alive long enough to reap it before stopping the parent.
            kill_process_tree(master.pid, include_parent=False, wait_timeout=10)
            master.terminate()
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait(timeout=10)


def probe(args):
    setup = load_setup(args.setup)
    if args.batch:
        return batch_probe(setup)
    key = "rdma-probe/" + uuid.uuid4().hex
    source = torch.arange(1 << 20, dtype=torch.uint8).pin_memory()
    digest = digest_bytes(tensor_bytes(source))
    writer = MooncakeSnapshotStore.connect(setup)
    try:
        writer.register(source)
        writer.put_registered(key, source, digest)
        source.zero_()
    finally:
        writer.close()
    reader = MooncakeSnapshotStore.connect(setup)
    try:
        value = reader.get_tensor(key, list(source.shape), source.dtype, digest)
        print(
            json.dumps(
                {"rdma_probe_bytes": value.numel(), "sha256": digest, "key": key}
            ),
            flush=True,
        )
        # The completed read still has a server-side lease. Keep normal lease
        # protection while the probe's retention owner waits to delete its key.
        deadline = time.monotonic() + 15
        while (rc := reader.client.remove(key, force=False)) == -706:
            if time.monotonic() >= deadline:
                raise TimeoutError("Probe read lease did not expire")
            time.sleep(0.05)
        if rc != 0:
            raise TransportError(f"Probe cleanup failed: status={rc}")
    finally:
        reader.close()


def batch_probe(setup):
    prefix = "rdma-batch-probe/" + uuid.uuid4().hex
    source = torch.arange(4 << 20, dtype=torch.uint8).pin_memory()
    views = [source[: 1 << 20], source[1 << 20 : 3 << 20], source[3 << 20 :]]
    objects = [
        (f"{prefix}/{index}", tensor, digest_bytes(tensor_bytes(tensor)))
        for index, tensor in enumerate(views)
    ]
    descriptors = [
        (key, list(tensor.shape), tensor.dtype, digest)
        for key, tensor, digest in objects
    ]
    writer = MooncakeSnapshotStore.connect(setup)
    writer.client = MagicMock(wraps=writer.client)
    try:
        writer.register(source)
        writer.put_registered_batch(objects[:1])
        writer.put_registered_batch(objects)
        writer.put_registered_batch(objects)
        assert writer.client.batch_put_from.call_count == 2
        assert writer.client.batch_get_into.call_count == 2
        assert [
            call.args[0] for call in writer.client.batch_put_from.call_args_list
        ] == [[objects[0][0]], [objects[1][0], objects[2][0]]]
        assert all(
            call.args[3].with_hard_pin
            for call in writer.client.batch_put_from.call_args_list
        )
        assert not writer.quarantined
    finally:
        writer.close()
    source.zero_()
    reader = MooncakeSnapshotStore.connect(setup)
    reader.client = MagicMock(wraps=reader.client)
    try:
        outputs = reader.get_tensors(descriptors)
        assert sum(out.nbytes for out in outputs) == source.nbytes
        assert reader.client.batch_get_into.call_count == 1
        assert not reader.registered and not reader.quarantined
        # A missing key must not hide successful completions in the same RPC.
        missing = (prefix + "/missing", [256], torch.uint8, digest_bytes(bytes(256)))
        try:
            reader.get_tensors([descriptors[0], missing, descriptors[2]])
        except TransportError:
            pass
        else:
            raise AssertionError("missing batch object was accepted")
        assert len(reader.quarantined) == 1
        assert set(reader.registered) == reader.quarantined
        assert sum(tensor.nbytes for tensor in reader.registered.values()) == 256
        retained = set(reader.quarantined)
        wrong_digest = (*descriptors[0][:3], "0" * 64)
        try:
            reader.get_tensors([wrong_digest, descriptors[1]])
        except ContractError:
            pass
        else:
            raise AssertionError("incorrect batch digest was accepted")
        assert set(reader.registered) == retained == reader.quarantined
        reader.verify_tensors(descriptors)
        assert reader.client.batch_get_into.call_count == 4
        # Delete only this probe's keys after their normal server read leases end.
        deadline = time.monotonic() + 15
        for key, _, _, _ in descriptors:
            while (rc := reader.client.remove(key, force=False)) == -706:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Batch probe read leases did not expire")
                time.sleep(0.05)
            if rc != 0:
                raise TransportError(f"Batch probe cleanup failed: status={rc}")
        result = {
            "payload_bytes": source.nbytes,
            "objects": len(objects),
            "native_write_batches": writer.client.batch_put_from.call_count,
            "immutable_retry_read_batches": writer.client.batch_get_into.call_count,
            "reader_batches": reader.client.batch_get_into.call_count,
            "partial_failure_quarantined_bytes": 256,
            "source_closed_and_overwritten": True,
            "keys_removed_without_force": len(objects),
        }
    finally:
        reader.close()
    assert not reader.registered and not reader.quarantined
    print(json.dumps({"rdma_batch_probe": result}), flush=True)


def read(args):
    setup = load_setup(args.setup)
    publications = json.loads(Path(args.publications).read_text())
    reader = MooncakeSnapshotStore.connect(setup)
    if args.batch:
        reader.client = MagicMock(wraps=reader.client)
    try:
        rows = []
        for publication in publications:
            before = reader.client.batch_get_into.call_count if args.batch else 0
            manifest, _ = read_snapshot(reader, publication, batch=args.batch)
            calls = (
                reader.client.batch_get_into.call_count - before if args.batch else 0
            )
            if args.batch:
                assert calls == 1
                assert not reader.registered and not reader.quarantined
            rows.append(
                {
                    "sample_id": manifest.sample_id,
                    "generation_id": manifest.generation_id,
                    "capture_mode": manifest.provenance.capture_mode,
                    "objects": len(manifest.objects),
                    "tensor_bytes": manifest.total_tensor_bytes,
                    "native_read_batches": calls,
                }
            )
        result = {"hostname": socket.gethostname(), "snapshots": rows}
        Path(args.output).write_text(json.dumps(result))
        print(json.dumps({"rdma_readback": result}), flush=True)
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serving = commands.add_parser("serve")
    serving.add_argument("--host", required=True)
    serving.add_argument("--devices", default="")
    serving.add_argument("--ready", required=True)
    serving.add_argument("--segment-mib", type=int, default=256)
    serving.add_argument("--lifetime", type=float, default=1800)
    probing = commands.add_parser("probe")
    probing.add_argument("--setup", required=True)
    probing.add_argument("--batch", action="store_true")
    reading = commands.add_parser("read")
    reading.add_argument("--setup", required=True)
    reading.add_argument("--publications", required=True)
    reading.add_argument("--output", required=True)
    reading.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"serve": serve, "probe": probe, "read": read}[args.command](args)


if __name__ == "__main__":
    main()

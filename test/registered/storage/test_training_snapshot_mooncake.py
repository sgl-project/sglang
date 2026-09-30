"""Cross-process TCP roundtrip of the complete training tensor wire format.

This checks real Store transport, not Catalog publication or serving capture.
The producer owns the data segment while a separate consumer reads it.
"""

import argparse
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import torch
from sglang.srt.training_capture.catalog import CaptureLease
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    DTYPES,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.training_capture.snapshot_writer import (
    PublicationJournal,
    SnapshotWriter,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import make_snapshot

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def connect(master, *, segment_bytes):
    return MooncakeSnapshotStore.connect(
        {
            "local_hostname": f"127.0.0.1:{free_port()}",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": segment_bytes,
            "local_buffer_size": 16 << 20,
            "protocol": "tcp",
            "rdma_devices": "",
            "master_server_addr": master,
        }
    )


def read_sample(master, key, size, digest):
    store = connect(master, segment_bytes=0)
    try:
        manifest_buffer = store.get_tensor(key, [size], torch.uint8, digest)
        manifest = decode_manifest(bytes(tensor_bytes(manifest_buffer)))
        tensors = {
            obj.key: store.get_tensor(obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256)
            for obj in manifest.objects
        }
        validate_tensors(manifest, tensors)
        print(
            json.dumps(
                {
                    "sample_id": manifest.sample_id,
                    "objects": len(tensors),
                    "validated_tensor_bytes": manifest.total_tensor_bytes,
                }
            ),
            flush=True,
        )
    finally:
        store.close()


@unittest.skipUnless(
    shutil.which("mooncake_master") and importlib.util.find_spec("mooncake"),
    "Mooncake master and SDK required",
)
class TestTrainingSnapshotMooncake(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = None
        cls.master_log = tempfile.TemporaryFile()
        port = free_port()
        cls.master_address = f"127.0.0.1:{port}"
        cls.master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={port}",
                f"--metrics_port={free_port()}",
                f"--http_metadata_server_port={free_port()}",
                "--default_kv_lease_ttl=100ms",
            ],
            stdout=cls.master_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if cls.master.poll() is not None:
                cls.master_log.seek(0)
                raise RuntimeError(cls.master_log.read().decode(errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("Mooncake master startup timed out")

    @classmethod
    def tearDownClass(cls):
        if cls.master is not None:
            cls.master.terminate()
            try:
                cls.master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.master.kill()
                cls.master.wait()
        cls.master_log.close()

    def test_manifest_last_and_registered_arena_cross_process_read(self):
        manifest, tensors = make_snapshot(response_length=4)
        store = connect(self.master_address, segment_bytes=64 << 20)
        pool = None
        slot = None
        complete = False
        journal = None
        journal_directory = tempfile.TemporaryDirectory()
        try:
            pool = HostBufferPool(
                kv=manifest.kv,
                max_tokens=8,
                slots=1,
                max_bytes=2 << 20,
                registrar=store,
                pin_memory=False,
            )
            slot = pool.acquire()
            packed = {}
            for obj in manifest.objects:
                target = slot.tensors[obj.name]
                if obj.kind == "kv":
                    target = target[obj.token_range[0] : obj.token_range[1]]
                else:
                    target = target[: obj.shape[0]]
                target.copy_(tensors[obj.key])
                packed[obj.key] = target
            manifest_data = canonical_bytes(manifest)
            key = manifest.key_prefix + "manifest"
            digest = digest_bytes(manifest_data)
            catalog = MagicMock()
            catalog.seal.return_value = {"state": "PREPARED", "manifest_sha256": digest}
            catalog.publish.return_value = {
                "state": "AVAILABLE",
                "publication_id": "test-publication",
                "catalog_cursor": "test-cursor",
            }
            lease = CaptureLease(
                capture_id="test-capture",
                fencing_token=1,
                dataset_id=manifest.dataset_id,
                sample_id=manifest.sample_id,
                generation_id=manifest.generation_id,
                expires_in_seconds=120,
                renew_after_seconds=20,
            )
            journal = PublicationJournal(journal_directory.name)
            writer = SnapshotWriter(store, catalog, journal)
            writer.write(manifest, packed, slot.manifest_buffer, lease)
            self.assertFalse(list(journal.pending()))
            # A separate process has no access to producer tensor pointers.
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--reader",
                    "--master",
                    self.master_address,
                    "--manifest-key",
                    key,
                    "--manifest-size",
                    str(len(manifest_data)),
                    "--manifest-sha256",
                    digest,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(
                f'"validated_tensor_bytes": {manifest.total_tensor_bytes}',
                result.stdout,
            )
            print(result.stdout, flush=True)
            # Same bytes are an idempotent retry, including the manifest.
            store.put_registered(
                key, slot.manifest_buffer[: len(manifest_data)], digest
            )
            complete = True
        finally:
            if slot is not None:
                pool.release(slot, transfer_complete=complete)
            if pool is not None and complete:
                pool.close()
            store.close()
            if journal is not None:
                journal.close()
            journal_directory.cleanup()


if __name__ == "__main__":
    if "--reader" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--reader", action="store_true")
        parser.add_argument("--master", required=True)
        parser.add_argument("--manifest-key", required=True)
        parser.add_argument("--manifest-size", type=int, required=True)
        parser.add_argument("--manifest-sha256", required=True)
        args = parser.parse_args()
        read_sample(
            args.master, args.manifest_key, args.manifest_size, args.manifest_sha256
        )
    else:
        unittest.main()

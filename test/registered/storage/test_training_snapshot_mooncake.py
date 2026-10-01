"""Real Store transport, owner-local writes and fenced manifest-last publication.

Catalog calls use a test double. This does not exercise distributed inference.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from sglang.srt.training_capture.catalog import CaptureLease, HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    DTYPES,
    ContractError,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.snapshot_writer import (
    OwnerWriteReceipt,
    PublicationJournal,
    SnapshotWriter,
)
from sglang.srt.training_capture.startup import (
    CaptureStartupError,
    coordinate_resource_startup,
)
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_partition import make_partitioned_snapshot
from sglang.test.training_capture_utils import (
    make_kv_spec,
    make_snapshot,
    read_snapshot,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


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


def prepare_store_rank(rank, root, master, catalog):
    """Real Store clients; synthetic local KV, no model or CUDA collectives."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    kv = make_kv_spec()
    base, _ = make_snapshot()
    layout = plan_capture_layout(kv, tp_size=4, pp_layer_ranges=[(0, 4)], aux_tp_rank=1)
    partition = layout.partitions[rank]
    config = CaptureConfig(
        dataset_id="resource-startup",
        model_id="fixture",
        producer_revision="test",
        selected_layer_ids=kv.selected_layer_ids,
        catalog_endpoint=catalog,
        journal_directory=str(Path(root) / "journal"),
        store=StoreSetup(
            local_hostname=f"127.0.0.1:{free_port()}", master_server_addr=master
        ),
        max_sample_tokens=8,
        max_inflight_samples=1,
        max_host_bytes=2 << 20,
    )
    source = MagicMock(spec=MHATokenToKVPool)
    source.is_quantized_kv_cache = source.use_hnd = False
    source.page_size, source.start_layer, source.layer_num = kv.source_page_size, 0, 4
    keys, values = {}, {}
    for layer in partition.local_layers(kv):
        keys[layer.layer_id] = torch.zeros(
            16, layer.num_kv_heads, layer.key_head_dim, dtype=torch.bfloat16
        )
        values[layer.layer_id] = torch.zeros(
            16, layer.num_kv_heads, layer.value_head_dim, dtype=torch.bfloat16
        )
    source.get_key_buffer.side_effect = keys.__getitem__
    source.get_value_buffer.side_effect = values.__getitem__
    results = []
    try:
        for reject in (True, False):
            local = (
                msgspec.structs.replace(config, max_host_bytes=1)
                if reject and rank == 2
                else config
            )
            prepared = []

            def prepare(local=local, prepared=prepared):
                resources = CaptureResources.prepare(
                    config=local,
                    kv=kv,
                    partition=partition,
                    source_pool=source,
                    pin_memory=False,
                )
                prepared.append(resources)
                return resources

            try:
                resource = coordinate_resource_startup(
                    group=dist.group.WORLD,
                    build_policy=lambda: (config.startup_policy, layout),
                    prepare_local=prepare,
                    timeout_seconds=15,
                )
            except CaptureStartupError as error:
                results.append(
                    {"phase": error.phase, "failed_ranks": error.failed_ranks}
                )
            else:
                control = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))
                try:
                    row = {"phase": "ready", "active": partition.active}
                    allocator = CaptureCohortAllocator(
                        group=control,
                        layout=layout,
                        config=config,
                        teacher=base.teacher,
                        kv=kv,
                        resources=resource,
                        timeout_seconds=10,
                    )
                    cohort = allocator.reserve()
                    assert cohort is not None
                    row.update(
                        capture_id=cohort.lease.capture_id,
                        reserved_bytes=cohort.reserved_bytes,
                        local_bytes=resource.pool.allocated_bytes
                        if partition.active
                        else 0,
                    )
                    if partition.active:
                        slot = cohort.slot
                        complete = False
                        try:
                            payload = slot.storage[:64]
                            payload.fill_(rank + 1)
                            key = f"resource-startup/{Path(root).name}/{rank}"
                            digest = digest_bytes(tensor_bytes(payload))
                            resource.store.put_registered(key, payload, digest)
                            complete = True
                            row.update(key=key, digest=digest)
                        finally:
                            resource.pool.release(slot, transfer_complete=complete)
                    dist.barrier()
                    if partition.include_aux:
                        resource.catalog.fail(
                            cohort.lease, "transport_fixture_finished"
                        )
                    results.append(row)
                finally:
                    dist.destroy_process_group(control)
                    resource.close()
            assert all(resource.closed for resource in prepared)
            assert all(
                resource.store is None or not resource.store.registered
                for resource in prepared
            )
            dist.barrier()
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


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

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo required")
    def test_collective_resource_rollback_then_real_registered_store_write(self):
        store = connect(self.master_address, segment_bytes=64 << 20)
        catalog = TestCaptureCatalog()
        workers = []
        try:
            with tempfile.TemporaryDirectory() as root:
                context = mp.get_context("spawn")
                workers = [
                    context.Process(
                        target=prepare_store_rank,
                        args=(rank, root, self.master_address, catalog.endpoint),
                    )
                    for rank in range(4)
                ]
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 100
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                captures, budgets, local_bytes = set(), set(), 0
                for rank in range(4):
                    failure, ready = json.loads(
                        (Path(root) / f"rank-{rank}.json").read_bytes()
                    )
                    self.assertEqual(
                        failure, {"phase": "resources", "failed_ranks": [2]}
                    )
                    self.assertEqual(ready["phase"], "ready")
                    self.assertEqual(ready["active"], rank != 3)
                    captures.add(ready["capture_id"])
                    budgets.add(ready["reserved_bytes"])
                    local_bytes += ready["local_bytes"]
                    if ready["active"]:
                        payload = store.get_tensor(
                            ready["key"], [64], torch.uint8, ready["digest"]
                        )
                        self.assertEqual(payload.tolist(), [rank + 1] * 64)
                self.assertEqual(captures, set(catalog.captures))
                self.assertEqual(len(captures), 1)
                self.assertEqual(budgets, {local_bytes})
                record = next(iter(catalog.captures.values()))
                self.assertEqual(record["begin"]["reserved_bytes"], local_bytes)
                self.assertEqual(
                    set(record["begin"]["owners"]),
                    {"dp0-pp0-tp0", "dp0-pp0-tp1", "dp0-pp0-tp2"},
                )
                self.assertEqual(record["state"], "FAILED")
                self.assertEqual(record["reason"], "transport_fixture_finished")
                self.assertFalse(catalog.errors)
                self.assertFalse(catalog.publications)
                PublicationJournal(str(Path(root) / "journal")).close()
                print(
                    "Resource startup: 4 ranks, rollback, one capture cohort, "
                    "3 registered writes readable after exit",
                    flush=True,
                )
        finally:
            for worker in workers:
                if worker.pid is not None and worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=5)
            catalog.close()
            store.close()

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

    def test_independent_owner_processes_publish_only_after_all_receipts(self):
        manifest, tensors = make_partitioned_snapshot()
        # The independent storage segment outlives both owner processes.
        store = connect(self.master_address, segment_bytes=64 << 20)
        catalog_server = TestCaptureCatalog()
        journal = None
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            try:
                catalog = HTTPCaptureCatalog(catalog_server.endpoint)
                lease = catalog.begin(
                    {
                        "dataset_id": manifest.dataset_id,
                        "sample_id": manifest.sample_id,
                        "generation_id": manifest.generation_id,
                        "owners": manifest.topology.owners,
                        "idempotency_key": "partition-integration",
                    }
                )
                data = canonical_bytes(manifest)
                (directory / "manifest.json").write_bytes(data)
                (directory / "lease.json").write_bytes(canonical_bytes(lease))
                journal = PublicationJournal(str(directory / "publisher"))
                writer = SnapshotWriter(store, catalog, journal)
                manifest_buffer = torch.empty(1 << 20, dtype=torch.uint8)
                store.register(manifest_buffer)
                receipts = []
                for owner in manifest.topology.owners:
                    path = directory / (owner + ".safetensors")
                    receipt_path = directory / (owner + ".json")
                    save_file(
                        {
                            obj.key: tensors[obj.key]
                            for obj in manifest.objects
                            if obj.owner_id == owner
                        },
                        str(path),
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "sglang.test.training_capture_partition",
                            "--master",
                            self.master_address,
                            "--catalog",
                            catalog_server.endpoint,
                            "--manifest",
                            str(directory / "manifest.json"),
                            "--lease",
                            str(directory / "lease.json"),
                            "--tensors",
                            str(path),
                            "--owner",
                            owner,
                            "--receipt",
                            str(receipt_path),
                            "--journal",
                            str(directory / (owner + "-journal")),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    receipts.append(
                        msgspec.json.decode(
                            receipt_path.read_bytes(), type=OwnerWriteReceipt
                        )
                    )
                    self.assertFalse(catalog_server.publications)
                    if len(receipts) < len(manifest.topology.owners):
                        with self.assertRaisesRegex(ContractError, "missing owner"):
                            writer.publish_partitions(
                                manifest, receipts, manifest_buffer, lease
                            )
                        self.assertFalse(list(journal.pending()))
                stale = [
                    msgspec.structs.replace(
                        receipts[0], fencing_token=lease.fencing_token + 1
                    ),
                    *receipts[1:],
                ]
                with self.assertRaisesRegex(ContractError, "stale"):
                    writer.publish_partitions(manifest, stale, manifest_buffer, lease)
                writer.publish_partitions(
                    manifest, list(reversed(receipts)), manifest_buffer, lease
                )
                publication = catalog_server.wait_publications(1)[0]
                _, actual = read_snapshot(store, publication)
                expected_manifest, expected = make_snapshot(response_length=4)
                for obj in expected_manifest.objects:
                    value = actual[obj.name]
                    if obj.kind == "kv":
                        value = value[obj.token_range[0] : obj.token_range[1]]
                    torch.testing.assert_close(value, expected[obj.key], rtol=0, atol=0)
                result = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--reader",
                        "--master",
                        self.master_address,
                        "--manifest-key",
                        publication["manifest_key"],
                        "--manifest-size",
                        str(publication["manifest_nbytes"]),
                        "--manifest-sha256",
                        publication["manifest_sha256"],
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(catalog_server.errors)
                self.assertFalse(list(journal.pending()))
                print(
                    json.dumps(
                        {
                            "partition_publication": {
                                "owners": manifest.topology.owners,
                                "aux_owner": manifest.topology.aux_owner,
                                "writer_processes": len(receipts),
                                "writers_exited_before_publish": True,
                                "objects": len(manifest.objects),
                                "tensor_bytes": manifest.total_tensor_bytes,
                                "missing_owner_rejected": True,
                                "stale_receipt_rejected": True,
                                "logical_head_reassembly_exact": True,
                                "independent_reader_passed": True,
                            }
                        }
                    ),
                    flush=True,
                )
            finally:
                store.close()
                if journal is not None:
                    journal.close()
                catalog_server.close()


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

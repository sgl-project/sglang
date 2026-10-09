"""Cross-rank metadata agreement and actor lifetime under deterministic failures."""

import json
import multiprocessing as mp
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_exchange import SnapshotOffer, agree_snapshot
from sglang.srt.training_capture.cohort_service import CaptureCohortService
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
    digest_bytes,
)
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.snapshot_writer import OwnerWriteReceipt
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_partition import prepare_cohort_partition
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

_REQUEST = "ab" * 32
_EXECUTION = "cd" * 32
_CASES = (
    "delayed_owners",
    "inactive_ingress",
    "execution",
    "token_ledger",
    "metadata",
    "contract",
    "owner",
    "fence",
    "invalid_json",
    "oversized",
    "build_failure",
    "allocation_failure",
    "parse_failure",
    "result_disagreement",
    "exchange_expiry",
    "cancel_during_exchange",
    "receipt_digest",
    "receipt_owner",
    "receipt_fence",
    "receipt_missing",
    "inactive_receipt",
    "submission_failure",
)


def expect_contract(call):
    try:
        call()
    except ContractError:
        return
    raise AssertionError("expected capture contract rejection")


def run_exchange_case(rank, root, endpoint, case):
    base, _ = make_snapshot()
    inactive_ingress = case == "inactive_ingress"
    layout = plan_capture_layout(
        base.kv,
        tp_size=2 if inactive_ingress else 4,
        pp_layer_ranges=[(0, 1), (1, 4)] if inactive_ingress else [(0, 4)],
        aux_tp_rank=1,
    )
    partition = layout.partitions[rank]
    config = CaptureConfig(
        dataset_id=case,
        model_id="fixture",
        producer_revision="test",
        selected_layer_ids=base.kv.selected_layer_ids,
        catalog_endpoint=endpoint,
        journal_directory=str(Path(root) / f"journal-{rank}"),
        store=StoreSetup(
            local_hostname=f"rank-{rank}", master_server_addr="localhost:1"
        ),
        max_sample_tokens=8,
        max_inflight_samples=1,
        max_host_bytes=2 << 20,
    )
    resources = CaptureResources()
    resources.catalog = HTTPCaptureCatalog(endpoint)
    if partition.active:
        resources.pool = HostBufferPool(
            kv=base.kv,
            max_tokens=8,
            slots=1,
            max_bytes=config.max_host_bytes,
            registrar=Registrar(),
            pin_memory=False,
            partition=partition,
        )
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
    allocator = CaptureCohortAllocator(
        group=group,
        layout=layout,
        config=config,
        teacher=base.teacher,
        kv=base.kv,
        resources=resources,
        timeout_seconds=15,
    )
    service = CaptureCohortService(allocator)
    service._cycle()
    service._cycle()
    tickets = [service.claim(_REQUEST)]
    dist.broadcast_object_list(tickets, src=0)
    handle = service.bind(tickets[0], _REQUEST)
    assert handle is not None
    metadata, prepared, _ = prepare_cohort_partition(handle.cohort, layout, rank)

    def submit():
        service.submit_snapshot(
            handle,
            execution_sha256="ef" * 32
            if case == "execution" and rank == 3
            else _EXECUTION,
            prepared=prepared,
            metadata=metadata if partition.include_aux else None,
        )

    if case == "submission_failure" and rank == 2:
        try:
            service.submit_snapshot(
                handle, execution_sha256="invalid", prepared=prepared
            )
        except msgspec.ValidationError:
            pass
        else:
            raise AssertionError("malformed submission was accepted")
    elif case != "delayed_owners" or rank != 2:
        submit()

    if case == "delayed_owners":
        service._cycle()
        assert service.get_manifest(handle) is None
        if rank == 2:
            submit()

    if rank == 2 and case in ("token_ledger", "metadata", "owner", "fence", "contract"):
        offer = msgspec.json.decode(handle.snapshot_payload, type=SnapshotOffer)
        if case in ("token_ledger", "metadata"):
            field = "token_ids_sha256" if case == "token_ledger" else "metadata_sha256"
            offer = msgspec.structs.replace(
                offer,
                partition=msgspec.structs.replace(
                    offer.partition, **{field: "00" * 32}
                ),
            )
        elif case == "owner":
            offer = msgspec.structs.replace(
                offer, owner_id=layout.partitions[0].owner_id
            )
        elif case == "fence":
            offer = msgspec.structs.replace(offer, fencing_token=2)
        else:
            # A consistent descriptor hash cannot authorize a different teacher.
            metadata = msgspec.structs.replace(metadata, contract_id="foreign-contract")
            offer = msgspec.structs.replace(
                offer,
                partition=msgspec.structs.replace(
                    offer.partition,
                    metadata_sha256=digest_bytes(canonical_bytes(metadata)),
                ),
            )
        handle.snapshot_payload = canonical_bytes(offer)
    if case == "contract" and partition.include_aux:
        offer = msgspec.json.decode(handle.snapshot_payload, type=SnapshotOffer)
        metadata = msgspec.structs.replace(metadata, contract_id="foreign-contract")
        handle.snapshot_payload = canonical_bytes(
            msgspec.structs.replace(offer, metadata=metadata)
        )
    if rank == 2 and case in ("invalid_json", "oversized"):
        handle.snapshot_payload = (
            b"{"
            if case == "invalid_json"
            else b"x" * (config.manifest_buffer_bytes + 4097)
        )

    with ExitStack() as stack:
        if rank == 2 and case == "allocation_failure":
            stack.enter_context(
                patch(
                    "sglang.srt.training_capture.cohort.torch.empty_like",
                    side_effect=MemoryError("injected control allocation failure"),
                )
            )
        if rank == 2 and case == "build_failure":
            original_exchange = allocator.exchange

            def exchange(*args, **kwargs):
                def build():
                    raise MemoryError("injected encoder failure")

                return original_exchange(*args, **{**kwargs, "build_local": build})

            stack.enter_context(patch.object(allocator, "exchange", exchange))
        if rank == 2 and case in (
            "parse_failure",
            "result_disagreement",
            "exchange_expiry",
            "cancel_during_exchange",
        ):

            def validate(*args, **kwargs):
                if case == "parse_failure":
                    raise ValueError("injected parser failure")
                result = agree_snapshot(*args, **kwargs)
                if case == "result_disagreement":
                    return result + b" "
                if case == "exchange_expiry":
                    # Affect only the final readiness check, after the state vote.
                    stack.enter_context(
                        patch(
                            "sglang.srt.training_capture.cohort.time.monotonic",
                            return_value=handle.cohort.deadline + 1,
                        )
                    )
                if case == "cancel_during_exchange":
                    service.cancel(handle.ticket, "cancelled_during_validation")
                return result

            stack.enter_context(
                patch(
                    "sglang.srt.training_capture.cohort_service.agree_snapshot",
                    validate,
                )
            )
        service._cycle()

    good = (
        case in ("delayed_owners", "inactive_ingress")
        or case.startswith("receipt_")
        or case == "inactive_receipt"
    )
    result = {"case": case, "capture_id": handle.cohort.lease.capture_id}
    if good:
        manifest = service.get_manifest(handle)
        assert manifest is not None
        fingerprint = digest_bytes(canonical_bytes(manifest))
        result["manifest_sha256"] = fingerprint
        # Mutating a returned nested container cannot alter the voted manifest.
        manifest.objects.clear()
        assert service.get_manifest(handle).objects
        assert service.get_receipts(handle) is None
        expect_contract(
            lambda: service.finish(
                handle,
                outcome="published" if partition.include_aux else "stored",
                transfer_complete=True,
            )
        ) if partition.active else None
        if partition.active:
            receipt = OwnerWriteReceipt(
                capture_id=handle.cohort.lease.capture_id,
                fencing_token=handle.cohort.lease.fencing_token,
                owner_id=partition.owner_id,
                manifest_sha256=fingerprint,
            )
            if rank != 2:
                service.submit_receipt(handle, receipt)
        service._cycle()
        assert service.get_receipts(handle) is None
        if rank == 2:
            service.submit_receipt(handle, receipt)
            changes = {
                "receipt_digest": {"manifest_sha256": "00" * 32},
                "receipt_owner": {"owner_id": layout.partitions[0].owner_id},
                "receipt_fence": {"fencing_token": 2},
            }.get(case)
            if changes:
                handle.receipt_payload = canonical_bytes(
                    msgspec.structs.replace(receipt, **changes)
                )
            elif case == "receipt_missing":
                handle.receipt_payload = b"null"
        if case == "inactive_receipt" and rank == 3:
            # An inactive rank cannot enter an owner receipt through the actor API.
            expect_contract(
                lambda: service.submit_receipt(
                    handle,
                    OwnerWriteReceipt(
                        capture_id=handle.cohort.lease.capture_id,
                        fencing_token=1,
                        owner_id=partition.owner_id,
                        manifest_sha256=fingerprint,
                    ),
                )
            )
        service._cycle()
        if case in ("delayed_owners", "inactive_ingress"):
            receipts = service.get_receipts(handle)
            assert {item.owner_id for item in receipts} == set(layout.topology.owners)
            assert all(item.manifest_sha256 == fingerprint for item in receipts)
            result["receipt_count"] = len(receipts)
        else:
            assert service.status(handle)[1] is not None
            assert handle.receipts_payload is None
    else:
        if case == "cancel_during_exchange":
            service._cycle()
        assert service.status(handle)[1] is not None
        expect_contract(lambda: service.get_manifest(handle))
        result["failure"] = service.status(handle)[1]
        if case not in ("submission_failure", "cancel_during_exchange"):
            assert result["failure"].startswith("snapshot_exchange_failed:")
    assert service.error is None and not allocator.poisoned
    assert resources.pool is None or resources.pool.stats()["filling"] == 1
    service.close(timeout=0)
    service.finish(handle, outcome="failed", transfer_complete=True)
    assert service._cycle()
    assert service.close(timeout=0)
    if resources.pool is not None:
        assert resources.pool.stats()["free"] == 1
        resources.pool.close()
    dist.destroy_process_group(group)
    dist.barrier()
    return result


def exchange_worker(rank, root, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        results = [run_exchange_case(rank, root, endpoint, case) for case in _CASES]
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


class TestCaptureCohortExchange(CustomTestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_metadata_consensus_preserves_actor_ownership(self):
        """A local failure cannot expose a partial manifest or recycle live buffers."""
        catalog = TestCaptureCatalog()
        self.addCleanup(catalog.close)
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(
                    target=exchange_worker, args=(rank, root, catalog.endpoint)
                )
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 150
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                results = [
                    json.loads((Path(root) / f"rank-{rank}.json").read_bytes())
                    for rank in range(4)
                ]
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)
        for rows in zip(*results, strict=True):
            with self.subTest(case=rows[0]["case"]):
                self.assertEqual(len({row["capture_id"] for row in rows}), 1)
                self.assertEqual(
                    catalog.captures[rows[0]["capture_id"]]["state"], "FAILED"
                )
                if "manifest_sha256" in rows[0]:
                    self.assertEqual(len({row["manifest_sha256"] for row in rows}), 1)
                if "receipt_count" in rows[0]:
                    self.assertTrue(all(row["receipt_count"] > 1 for row in rows))
        self.assertFalse(catalog.errors)
        self.assertFalse(catalog.publications)


if __name__ == "__main__":
    unittest.main()

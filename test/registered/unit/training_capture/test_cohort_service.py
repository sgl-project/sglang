"""Distributed lease and transfer ownership across request/control interleavings."""

import json
import multiprocessing as mp
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.training_capture.catalog import CatalogUnavailable, HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_service import CaptureCohortService
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.protocol import ContractError, canonical_bytes
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

_FINGERPRINT = "ab" * 32


class LifecycleCatalog:
    def __init__(self, endpoint, case):
        self.client = HTTPCaptureCatalog(endpoint)
        self.case = case
        self.begins = self.heartbeats = self.fails = 0
        self.heartbeat_entered = threading.Event()
        self.heartbeat_release = threading.Event()

    def begin(self, payload):
        self.begins += 1
        return self.client.begin(payload)

    def heartbeat(self, lease):
        self.heartbeats += 1
        if self.case == "background":
            self.heartbeat_entered.set()
            assert self.heartbeat_release.wait(15), "foreground did not release Catalog"
        if self.case == "renew_failure":
            raise CatalogUnavailable("injected renewal outage")
        value = self.client.heartbeat(lease)
        if self.case == "renew_fence":
            value = msgspec.structs.replace(value, fencing_token=2)
        return value

    def fail(self, lease, reason):
        self.fails += 1
        if self.case == "fail_uncertain":
            raise CatalogUnavailable("injected ambiguous failure response")
        return self.client.fail(lease, reason)


class ObservedService(CaptureCohortService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.installed = threading.Event()

    def _cycle(self):
        done = super()._cycle()
        with self.lock:
            if any(handle.available for handle in self.records.values()):
                self.installed.set()
        return done


def prepare(rank, root, endpoint, case):
    base, _ = make_snapshot()
    ranges = [(0, 1), (1, 4)] if case == "inactive_leader" else [(0, 4)]
    layout = plan_capture_layout(
        base.kv,
        tp_size=2 if len(ranges) == 2 else 4,
        pp_layer_ranges=ranges,
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
        max_inflight_samples=2 if case == "multiple" else 1,
        max_host_bytes=4 << 20,
    )
    resources = CaptureResources()
    resources.catalog = LifecycleCatalog(endpoint, case)
    if partition.active:
        resources.pool = HostBufferPool(
            kv=base.kv,
            max_tokens=8,
            slots=config.max_inflight_samples,
            max_bytes=config.max_host_bytes,
            registrar=Registrar(),
            pin_memory=False,
            partition=partition,
        )
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
    with torch.device("meta"):
        allocator = CaptureCohortAllocator(
            group=group,
            layout=layout,
            config=config,
            teacher=base.teacher,
            kv=base.kv,
            resources=resources,
            timeout_seconds=15,
        )
        service = ObservedService(allocator, poll_seconds=0.01)
    assert service.frame.device.type == "cpu"
    return service, resources, group


def claim(service, rank):
    ticket = service.claim(_FINGERPRINT)
    if rank != 0:
        assert ticket is None
    values = [ticket]
    dist.broadcast_object_list(values, src=0)
    assert values[0] is not None
    return values[0]


def finish_failed(service, handle, *, transfer_complete=True):
    if handle is not None:
        service.finish(handle, outcome="failed", transfer_complete=transfer_complete)


def mark_renewal_due(service):
    with service.lock:
        handle = next(iter(service.records.values()))
        handle.cohort = msgspec.structs.replace(
            handle.cohort, renew_at=time.monotonic() - 1
        )
        service.wake.set()


def run_case(rank, root, endpoint, case):
    service, resources, group = prepare(rank, root, endpoint, case)
    pool, client = resources.pool, resources.catalog
    if case == "empty_stop":
        held = pool.acquire() if rank == 2 else None
        build = service._build_frame

        def stop_after_snapshot():
            value = build()
            if rank == 0:
                service.stopping = True
            return value

        with patch.object(service, "_build_frame", stop_after_snapshot):
            assert not service._cycle(), "stop must be voted before a rank leaves"
        assert service._cycle()
        if held is not None:
            pool.release(held, transfer_complete=True)
        assert service.close(timeout=0)
        if pool is not None:
            pool.close()
        dist.destroy_process_group(group)
        dist.barrier()
        return {"case": case, "begins": client.begins}
    if case == "background":
        service.start()
        assert service.installed.wait(15)
    else:
        assert not service._cycle()
        assert service.claim(_FINGERPRINT) is None, "not globally installed yet"
        assert not service._cycle()
    ticket = claim(service, rank)
    handle = None
    if case in ("unbound_cancel", "operator_abort_unbound") or (
        case in ("late_bind", "operator_abort_mixed") and rank == 0
    ):
        pass
    elif case == "identity" and rank == 2:
        assert service.bind(ticket, "cd" * 32) is None
    else:
        stale = msgspec.structs.replace(ticket, fencing_token=2)
        assert service.bind(stale, _FINGERPRINT) is None
        handle = service.bind(ticket, _FINGERPRINT)
        assert handle is not None
        assert service.bind(ticket, _FINGERPRINT) is None
        rejected = False
        try:
            service.finish(
                handle,
                outcome="stored" if service.partition.include_aux else "published",
                transfer_complete=True,
            )
        except ContractError:
            rejected = True
        assert rejected, "only aux may publish, and aux cannot finish as stored"
    assert service.claim("cd" * 32) is None
    result = {"case": case, "capture_id": ticket.capture_id}

    if case in ("operator_abort_unbound", "operator_abort_mixed"):
        if rank == 0:
            service.set_admission_ready(False)
            service.cancel_unbound("operator_aborted")
        assert not service._cycle()
        if handle is not None:
            assert service.status(handle)[1] is not None
            assert pool is None or pool.stats()["filling"] == 1
        service.set_admission_ready(True)
        assert service.bind(ticket, _FINGERPRINT) is None
        finish_failed(service, handle)
        service.stopping = True
        assert service._cycle()
        assert service.close(timeout=0)
    elif case == "multiple":
        assert not service._cycle()
        second_ticket = claim(service, rank)
        second = service.bind(second_ticket, _FINGERPRINT)
        assert second is not None
        finish_failed(service, handle)
        assert not service._cycle()
        assert not service._cycle()
        assert service.status(second)[1] is None
        assert service.status(second)[0].lease.capture_id == second_ticket.capture_id
        assert pool is None or pool.stats()["filling"] == 2
        assert not service._cycle()
        third_ticket = claim(service, rank)
        captures = [
            ticket.capture_id,
            second_ticket.capture_id,
            third_ticket.capture_id,
        ]
        assert len(set(captures)) == 3
        if rank == 0:
            assert service.cancel(third_ticket, "unused_replacement")
        finish_failed(service, second)
        if rank == 3:
            service.stopping = True
        assert not service._cycle()
        assert service._cycle()
        assert service.close(timeout=0)
        result["captures"] = captures
    elif case == "background":
        if rank == 2:
            mark_renewal_due(service)
        # All foreground actors wait for the SAME blocked heartbeat; control
        # workers are now in HTTP/collectives on the dedicated group.
        entered = [
            client.heartbeat_entered.wait(15) if rank == service.aux_rank else None
        ]
        dist.broadcast_object_list(entered, src=service.aux_rank)
        assert entered == [True]
        inference = torch.tensor(rank)
        dist.all_reduce(inference)
        assert inference.item() == 6
        service.fail(handle, "request_cancelled")
        cohort, reason = service.status(handle)
        assert reason == "request_cancelled"
        assert cohort.slot is None or cohort.slot.state == "filling"
        assert not service.close(timeout=0)
        finish_failed(service, handle)
        assert pool is None or pool.stats()["filling"] == 1
        dist.barrier()
        client.heartbeat_release.set()
        assert service.close(timeout=15)
        assert service.error is None
        result["inference_sum"] = inference.item()
    elif case in ("control_failure", "frame_shape", "ledger_mismatch", "state_flag"):
        build = service._build_frame

        def corrupt_frame():
            if rank == 2 and case == "control_failure":
                raise MemoryError("control snapshot failed")
            ledger, frame, outputs = build()
            if rank == 2:
                if case == "frame_shape":
                    outputs = [item[:, :9].contiguous() for item in outputs]
                elif case == "ledger_mismatch":
                    ledger = []
                elif case == "state_flag":
                    frame[1, 0] = 7
            return ledger, frame, outputs

        with patch.object(service, "_build_frame", corrupt_frame):
            service._run()
        assert service.error is not None
        assert service.status(handle)[1] == "control_failed"
        assert pool is None or pool.stats()["filling"] == 1
        assert service in service._retained
        finish_failed(service, handle, transfer_complete=rank != 2)
        result["safe_close"] = service.close(timeout=0)
        assert result["safe_close"] == (rank != 2)
        assert not service.records
    else:
        if case in ("renew", "renew_failure", "renew_fence"):
            if rank == 2:
                mark_renewal_due(service)
            service._cycle()
            assert pool is None or pool.stats()["filling"] == 1
            if case == "renew":
                cohort, reason = service.status(handle)
                assert reason is None and cohort.renew_at > time.monotonic()
            else:
                assert service.status(handle)[1].startswith("renewal_failed:")
        if case == "capture_timeout" and rank == 2:
            handle.claimed_at = (
                time.monotonic() - service.allocator.config.max_capture_seconds - 1
            )
        if case == "expiry" and rank == 2:
            handle.cohort = msgspec.structs.replace(
                handle.cohort, deadline=time.monotonic() - 1
            )
        if case == "unbound_cancel":
            if rank == 0:
                stale = msgspec.structs.replace(ticket, fencing_token=2)
                assert not service.cancel(stale, "cancelled")
                assert service.cancel(ticket, "cancelled")
            assert not service._cycle()
            assert service.bind(ticket, _FINGERPRINT) is None
            service.stopping = True
            assert service._cycle()
            assert not service.cancel(ticket, "late_cancellation")
        elif case in ("published", "inactive_draining"):
            # Only the service ownership protocol is under test here. The
            # separate Store publication suite verifies the writer's outcome.
            if case != "inactive_draining" or rank != 3:
                service.finish(
                    handle,
                    outcome="published" if service.partition.include_aux else "stored",
                    transfer_complete=True,
                )
            if rank == 3:
                service.stopping = True
            if case == "inactive_draining":
                assert not service._cycle()
                if rank == 3:
                    service.finish(handle, outcome="stored", transfer_complete=True)
            assert service._cycle()
        elif case == "late_bind":
            if handle is not None:
                finish_failed(service, handle)
            if rank == 0:
                captured, proceed = threading.Event(), threading.Event()
                build = service._build_frame

                def pause_after_snapshot():
                    frame = build()
                    captured.set()
                    assert proceed.wait(10)
                    return frame

                with (
                    patch.object(service, "_build_frame", pause_after_snapshot),
                    ThreadPoolExecutor(max_workers=1) as executor,
                ):
                    pending = executor.submit(service._cycle)
                    assert captured.wait(10)
                    handle = service.bind(ticket, _FINGERPRINT)
                    assert handle is not None
                    proceed.set()
                    assert not pending.result(timeout=15)
                assert pool.stats()["filling"] == 1, "late binding must retain its slot"
                assert service.status(handle)[1] is not None
                finish_failed(service, handle)
            else:
                assert not service._cycle()
            service.stopping = True
            assert service._cycle()
        else:
            if case not in ("identity", "capture_timeout", "expiry") and rank == 2:
                service.fail(handle, "peer_cancelled")
            assert not service._cycle()
            if handle is not None:
                assert service.status(handle)[1] is not None
            assert pool is None or pool.stats()["filling"] == 1
            delayed_rank = 2 if case == "inactive_leader" else 0
            if rank != delayed_rank:
                finish_failed(service, handle)
            service.stopping = True
            assert not service._cycle(), "an owner has not acknowledged its transfer"
            assert pool is None or pool.stats()["filling"] == 1
            if rank == delayed_rank:
                finish_failed(service, handle)
            if case == "fail_uncertain":
                service._run()
                assert service.error is not None
            else:
                assert service._cycle()
        assert service.close(timeout=0)
    result.update(
        begins=client.begins,
        heartbeats=client.heartbeats,
        fails=client.fails,
        filling=pool.stats()["filling"] if pool else 0,
        quarantined=pool.stats()["quarantined"] if pool else 0,
        error=type(service.error).__name__ if service.error is not None else None,
    )
    if pool is not None and not pool.stats()["quarantined"]:
        pool.close()
    dist.destroy_process_group(group)
    dist.barrier()
    return result


def lifecycle_worker(rank, root, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    cases = (
        "renew",
        "renew_failure",
        "renew_fence",
        "identity",
        "capture_timeout",
        "expiry",
        "late_bind",
        "unbound_cancel",
        "operator_abort_unbound",
        "operator_abort_mixed",
        "published",
        "inactive_draining",
        "control_failure",
        "frame_shape",
        "ledger_mismatch",
        "state_flag",
        "fail_uncertain",
        "inactive_leader",
        "multiple",
        "empty_stop",
        "background",
    )
    try:
        results = [run_case(rank, root, endpoint, case) for case in cases]
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


class TestCaptureCohortService(CustomTestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_failure_and_shutdown_wait_for_transfer_ownership(self):
        """Expiry/cancel cannot recycle storage still owned by a request actor.

        Includes a deterministic bind after its control snapshot and before
        peer cancellation is merged, plus foreground work during a blocked RPC.
        """
        catalog = TestCaptureCatalog()
        self.addCleanup(catalog.close)
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(
                    target=lifecycle_worker, args=(rank, root, catalog.endpoint)
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
            case = rows[0]["case"]
            with self.subTest(case=case):
                if case == "empty_stop":
                    self.assertTrue(all(row["begins"] == 0 for row in rows))
                    continue
                aux = 3 if case == "inactive_leader" else 1
                self.assertEqual(len({row["capture_id"] for row in rows}), 1)
                record = catalog.captures[rows[0]["capture_id"]]
                self.assertEqual(
                    [row["begins"] for row in rows],
                    [
                        (3 if case == "multiple" else 1) * int(rank == aux)
                        for rank in range(4)
                    ],
                )
                self.assertTrue(all(row["filling"] == 0 for row in rows))
                failed_control = case in (
                    "control_failure",
                    "frame_shape",
                    "ledger_mismatch",
                    "state_flag",
                )
                self.assertEqual(
                    [row["quarantined"] for row in rows],
                    [0, 0, 1, 0] if failed_control else [0] * 4,
                )
                expected_state = (
                    "CAPTURING"
                    if failed_control
                    or case in ("published", "inactive_draining", "fail_uncertain")
                    else "FAILED"
                )
                self.assertEqual(record["state"], expected_state)
                if case == "multiple":
                    self.assertTrue(
                        all(row["captures"] == rows[0]["captures"] for row in rows)
                    )
                    self.assertTrue(
                        all(
                            catalog.captures[capture]["state"] == "FAILED"
                            for capture in rows[0]["captures"]
                        )
                    )
                self.assertEqual(
                    sum(row["fails"] for row in rows),
                    (3 if case == "multiple" else 1)
                    * int(
                        not failed_control
                        and case not in ("published", "inactive_draining")
                    ),
                )
                self.assertEqual(
                    sum(row["heartbeats"] for row in rows),
                    int(
                        case in ("renew", "renew_failure", "renew_fence", "background")
                    ),
                )
                self.assertTrue(
                    all(
                        row["heartbeats"] == row["fails"] == 0
                        for rank, row in enumerate(rows)
                        if rank != aux
                    )
                )
                if case == "background":
                    self.assertTrue(all(row["inference_sum"] == 6 for row in rows))
        self.assertFalse(catalog.errors)
        self.assertFalse(catalog.publications)


if __name__ == "__main__":
    unittest.main()

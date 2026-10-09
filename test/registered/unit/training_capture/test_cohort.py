"""All-owner capacity, lease identity and rollback before distributed admission."""

import json
import multiprocessing as mp
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.training_capture.catalog import (
    CaptureLease,
    CatalogUnavailable,
    HTTPCaptureCatalog,
)
from sglang.srt.training_capture.cohort import (
    CaptureCohortAllocator,
    CaptureCohortError,
)
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.protocol import ContractError, canonical_bytes
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


class FaultCatalog:
    def __init__(self, endpoint, case):
        self.client = HTTPCaptureCatalog(endpoint)
        self.case, self.begins, self.fails = case, 0, 0

    def begin(self, payload):
        self.begins += 1
        if self.case == "begin_failure":
            raise CatalogUnavailable("injected outage")
        lease = self.client.begin(payload)
        if self.case == "lost_response":
            raise CatalogUnavailable("response lost after begin")
        if self.case == "wrong_identity":
            return msgspec.structs.replace(lease, sample_id="another-sample")
        if self.case == "expired":
            time.sleep(0.05)
            return msgspec.structs.replace(
                lease, expires_in_seconds=0.02, renew_after_seconds=0.01
            )
        return lease

    def fail(self, lease, reason):
        self.fails += 1
        if self.case == "cleanup_failure":
            raise CatalogUnavailable("failure acknowledgement lost")
        return self.client.fail(lease, reason)


def cohort_worker(rank, root, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    base, _ = make_snapshot()
    cases = (
        "replicated",
        "pp_aux_only",
        "root_inactive",
        "backpressure",
        "acquire_failure",
        "policy",
        "begin_failure",
        "lost_response",
        "wrong_identity",
        "lease_encoding",
        "control_copy",
        "expired",
        "ready_expiry",
        "decode_failure",
        "lease_disagreement",
        "cleanup_failure",
        "round_disagreement",
        "late_validator",
    )
    rows = []
    try:
        for case in cases:
            pp = case in ("pp_aux_only", "root_inactive")
            ranges = [(0, 4), (4, 6)] if case == "pp_aux_only" else [(0, 1), (1, 4)]
            layout = plan_capture_layout(
                base.kv,
                tp_size=2 if pp else 4,
                pp_layer_ranges=ranges if pp else [(0, 4)],
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
                max_host_bytes=(2 << 20) + rank,
                sample_ratio=0.5 if case == "policy" and rank == 2 else 1.0,
            )
            resources = CaptureResources()
            client = FaultCatalog(endpoint, case)
            resources.catalog = client
            pool = resources.pool = (
                HostBufferPool(
                    kv=base.kv,
                    max_tokens=8,
                    slots=1,
                    max_bytes=config.max_host_bytes,
                    registrar=Registrar(),
                    pin_memory=False,
                    partition=partition,
                )
                if partition.active
                else None
            )
            group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
            with torch.device("meta"):
                allocator = CaptureCohortAllocator(
                    group=group,
                    layout=layout,
                    config=config,
                    teacher=base.teacher,
                    kv=base.kv,
                    resources=resources,
                    timeout_seconds=0.5 if case == "late_validator" else 10,
                )
            if case == "round_disagreement" and rank == 3:
                allocator.round = 1
            held = pool.acquire() if case == "backpressure" and rank == 2 else None
            cohort = None
            with ExitStack() as stack:
                if case == "lease_encoding" and partition.include_aux:
                    encode = canonical_bytes

                    def oversized(value, encode=encode):
                        return (
                            b"x" * 4097
                            if isinstance(value, CaptureLease)
                            else encode(value)
                        )

                    stack.enter_context(
                        patch(
                            "sglang.srt.training_capture.cohort.canonical_bytes",
                            oversized,
                        )
                    )
                if case == "control_copy" and partition.include_aux:
                    stack.enter_context(
                        patch(
                            "sglang.srt.training_capture.cohort.torch.frombuffer",
                            side_effect=MemoryError("control copy"),
                        )
                    )
                if case == "ready_expiry" and rank == 3:
                    vote = allocator._vote
                    clock = time.monotonic

                    def expiring_vote(
                        phase, *args, vote=vote, clock=clock, stack=stack, **kwargs
                    ):
                        result = vote(phase, *args, **kwargs)
                        if phase == "validation":
                            stack.enter_context(
                                patch(
                                    "sglang.srt.training_capture.cohort.time.monotonic",
                                    side_effect=lambda: clock() + 60,
                                )
                            )
                        return result

                    stack.enter_context(patch.object(allocator, "_vote", expiring_vote))
                if case == "acquire_failure" and rank == 2:
                    stack.enter_context(
                        patch.object(
                            pool, "acquire", side_effect=RuntimeError("allocation")
                        )
                    )
                decode = allocator._decode_lease

                def fault_decode(*args, case=case, decode=decode, **kwargs):
                    if case in ("decode_failure", "cleanup_failure") and rank == 2:
                        raise ValueError("invalid local lease")
                    if case == "late_validator" and rank == 3:
                        time.sleep(2)
                    value = decode(*args, **kwargs)
                    if case == "lease_disagreement" and rank == 2:
                        return msgspec.structs.replace(
                            value,
                            lease=msgspec.structs.replace(value.lease, fencing_token=2),
                        )
                    return value

                stack.enter_context(
                    patch.object(allocator, "_decode_lease", fault_decode)
                )
                try:
                    if case == "replicated":
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            pending = executor.submit(allocator.reserve)
                            inference = torch.tensor(rank)
                            dist.all_reduce(inference, group=dist.group.WORLD)
                            cohort = pending.result(timeout=20)
                    else:
                        cohort = allocator.reserve()
                    row = {"case": case, "phase": "ready" if cohort else "backpressure"}
                    if cohort:
                        row.update(
                            lease=msgspec.to_builtins(cohort.lease),
                            slot=cohort.slot is not None,
                            reserved_bytes=cohort.reserved_bytes,
                            renew_remaining=cohort.renew_at - time.monotonic(),
                        )
                    if case == "replicated":
                        row["inference_sum"] = inference.item()
                except CaptureCohortError as error:
                    row = {
                        "case": case,
                        "phase": error.phase,
                        "failed_ranks": error.failed_ranks,
                    }
                row.update(
                    begins=client.begins,
                    fails=client.fails,
                    filling=pool.stats()["filling"] if pool else 0,
                    local_bytes=pool.allocated_bytes if pool else 0,
                    poisoned=allocator.poisoned,
                )
                if allocator.poisoned:
                    try:
                        allocator.reserve()
                    except ContractError:
                        row["reuse_rejected"] = True
                rows.append(row)
            if case == "backpressure":
                if held is not None:
                    pool.release(held, transfer_complete=True)
                    held = None
                cohort = allocator.reserve()
                assert cohort is not None
                row["retry_capture_id"] = cohort.lease.capture_id
                row["retry_round"] = allocator.round
            if cohort is not None:
                if cohort.slot is not None:
                    pool.release(cohort.slot, transfer_complete=True)
                if partition.include_aux:
                    client.client.fail(cohort.lease, "test_finished")
            if held is not None:
                pool.release(held, transfer_complete=True)
            if pool is not None:
                pool.close()
            dist.destroy_process_group(group)
            dist.barrier()
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(rows))
    finally:
        dist.destroy_process_group()


class TestCaptureCohort(CustomTestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_all_owner_capacity_and_identity_precede_any_request_binding(self):
        """One rank's capacity/failure must never admit a partial-owner cohort."""
        catalog = TestCaptureCatalog()
        self.addCleanup(catalog.close)
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(
                    target=cohort_worker, args=(rank, root, catalog.endpoint)
                )
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 120
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
        failures = {
            "acquire_failure": ("slots", [2]),
            "policy": ("policy_agreement", [0, 1, 2, 3]),
            "begin_failure": ("lease", [1]),
            "lost_response": ("lease", [1]),
            "wrong_identity": ("lease", [1]),
            "lease_encoding": ("lease", [1]),
            "control_copy": ("lease", [1]),
            "expired": ("validation", [0, 1, 2, 3]),
            "ready_expiry": ("ready", [3]),
            "decode_failure": ("validation", [2]),
            "lease_disagreement": ("lease_agreement", [0, 1, 2, 3]),
            "cleanup_failure": ("rollback", [1]),
            "round_disagreement": ("protocol", [0, 1, 2, 3]),
            "late_validator": ("transport", []),
        }
        for rows in zip(*results, strict=True):
            case = rows[0]["case"]
            with self.subTest(case=case):
                captures = [
                    record
                    for record in catalog.captures.values()
                    if record["begin"]["dataset_id"] == case
                ]
                aux = 3 if case in ("pp_aux_only", "root_inactive") else 1
                self.assertTrue(
                    all(
                        row["begins"] == 0
                        for rank, row in enumerate(rows)
                        if rank != aux
                    )
                )
                if case in failures:
                    phase, ranks = failures[case]
                    self.assertEqual([row["phase"] for row in rows], [phase] * 4)
                    self.assertEqual([row["failed_ranks"] for row in rows], [ranks] * 4)
                    self.assertTrue(all(row["filling"] == 0 for row in rows))
                elif case == "backpressure":
                    self.assertEqual(
                        [row["phase"] for row in rows], ["backpressure"] * 4
                    )
                    self.assertEqual([row["filling"] for row in rows], [0, 0, 1, 0])
                    self.assertTrue(all(row["begins"] == 0 for row in rows))
                    self.assertEqual(len({row["retry_capture_id"] for row in rows}), 1)
                    self.assertTrue(all(row["retry_round"] == 2 for row in rows))
                else:
                    self.assertEqual([row["phase"] for row in rows], ["ready"] * 4)
                    self.assertTrue(
                        all(row["lease"] == rows[0]["lease"] for row in rows)
                    )
                    total = sum(row["local_bytes"] for row in rows)
                    self.assertTrue(all(row["reserved_bytes"] == total for row in rows))
                    self.assertTrue(all(row["renew_remaining"] > 0 for row in rows))
                    self.assertEqual(len(captures), 1)
                    self.assertEqual(captures[0]["begin"]["reserved_bytes"], total)
                    self.assertEqual(
                        [row["slot"] for row in rows],
                        [True, True, True, False]
                        if case == "replicated"
                        else [True, True, False, True]
                        if case == "pp_aux_only"
                        else [False, False, True, True],
                    )
                    if case == "replicated":
                        self.assertTrue(all(row["inference_sum"] == 6 for row in rows))
                if case in (
                    "acquire_failure",
                    "policy",
                    "begin_failure",
                    "round_disagreement",
                ):
                    self.assertFalse(captures)
                else:
                    self.assertEqual(len(captures), 1)
                    self.assertFalse(captures[0]["written"])
                    expected = (
                        "CAPTURING"
                        if case
                        in ("lost_response", "wrong_identity", "cleanup_failure")
                        else "FAILED"
                    )
                    self.assertEqual(captures[0]["state"], expected)
                if case in ("round_disagreement", "late_validator"):
                    self.assertTrue(
                        all(row["poisoned"] and row["reuse_rejected"] for row in rows)
                    )
        self.assertFalse(catalog.errors)
        self.assertFalse(catalog.publications)


if __name__ == "__main__":
    unittest.main()

"""Serving factory startup must roll back every rank before capture work begins."""

import faulthandler
import json
import multiprocessing as mp
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.runtime_context import get_context
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import canonical_bytes
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.startup import CaptureStartupError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_partition import synthetic_rank_contract
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_snapshot,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def startup_worker(rank, root, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=15),
    )
    cases = (
        "tp_pp",
        "replicated",
        "inactive_ingress",
        "policy",
        "resources",
        "cohort-writer",
        "capture-cohorts",
        "capture-handoff",
        "tp_pp",
    )
    results, group_names = [], set()
    try:
        for index, case in enumerate(cases):
            faulthandler.dump_traceback_later(25)
            print(f"startup rank={rank} case={index}:{case} begin", flush=True)
            base, _ = make_snapshot(response_length=4)
            tp_size = 4 if case == "replicated" else 2
            ranges = (
                [(0, 4)]
                if tp_size == 4
                else (
                    [(0, 1), (1, 4)] if case == "inactive_ingress" else [(0, 2), (2, 4)]
                )
            )
            config = CaptureConfig(
                dataset_id=f"factory-{index}",
                model_id=base.teacher.model_id,
                producer_revision="test",
                selected_layer_ids=base.kv.selected_layer_ids,
                storage_chunk_tokens=base.kv.storage_chunk_tokens,
                catalog_endpoint=endpoint,
                journal_directory=str(Path(root) / f"journal-{index}-{rank}"),
                store=StoreSetup(
                    local_hostname=f"rank-{rank}", master_server_addr="localhost:1"
                ),
                max_sample_tokens=8,
                max_inflight_samples=1,
                max_host_bytes=2 << 20,
                sample_ratio=0.5 if case == "policy" and rank == 1 else 1.0,
            )
            config_path = Path(root) / f"config-{rank}.json"
            config_path.write_bytes(msgspec.json.encode(config))
            contract = synthetic_rank_contract(
                base, rank, tp_size=tp_size, pp_layer_ranges=ranges
            )
            resources, groups, destroyed = [], [], []
            new_group, destroy_group, start_thread = (
                dist.new_group,
                dist.destroy_process_group,
                threading.Thread.start,
            )

            def track_group(*args, new_group=new_group, groups=groups, **kwargs):
                group = new_group(*args, **kwargs)
                groups.append(group)
                assert group.group_name not in group_names, (
                    "reused rendezvous namespace"
                )
                group_names.add(group.group_name)
                return group

            def track_destroy(
                group=None, destroy_group=destroy_group, destroyed=destroyed
            ):
                assert all(
                    not thread.is_alive()
                    for thread in threading.enumerate()
                    if thread.name
                    in ("capture-cohorts", "cohort-writer", "capture-handoff")
                )
                destroy_group(group)
                destroyed.append(group)

            def prepare(
                *, config, kv, partition, source_pool, resources=resources, case=case
            ):
                value = CaptureResources()
                resources.append(value)
                if partition.active:
                    value.store = MooncakeSnapshotStore(
                        BufferStore(), FakeReplicateConfig()
                    )
                    value.catalog = HTTPCaptureCatalog(endpoint)
                    value._allocate(config, kv, partition, pin_memory=False)
                    # Factory lifecycle only: storage is CPU, no forward is run.
                    if partition.heads:
                        value.exporter = SimpleNamespace(device=torch.device("cuda"))
                if case == "resources" and rank == 1:
                    value.close()
                    raise RuntimeError("rank-local preparation failed")
                return value

            def fail_thread(thread, case=case, start_thread=start_thread):
                if rank == 1 and thread.name == case:
                    raise RuntimeError("thread creation failed")
                return start_thread(thread)

            coordinator = None
            with (
                get_context().override_server_args(speculative_algorithm=None),
                patch(
                    "sglang.srt.training_capture.coordinator.bind_rank_target_contract",
                    return_value=contract,
                ),
                patch.object(CaptureResources, "prepare", side_effect=prepare),
                patch.object(dist, "new_group", side_effect=track_group),
                patch.object(dist, "destroy_process_group", side_effect=track_destroy),
                patch.object(threading.Thread, "start", fail_thread),
            ):
                try:
                    coordinator = CaptureCoordinator.create(
                        config_path=str(config_path),
                        model=None,
                        model_config=None,
                        tokenizer_path=None,
                        pool=None,
                        req_to_token=None,
                        startup_group=dist.group.WORLD,
                        tp_rank=rank % tp_size,
                        tp_size=tp_size,
                        pp_rank=rank // tp_size,
                        pp_size=len(ranges),
                    )
                except CaptureStartupError as error:
                    row = {
                        "case": case,
                        "phase": error.phase,
                        "failed_ranks": error.failed_ranks,
                    }
                else:
                    deadline = time.monotonic() + 10
                    while not coordinator.stats()["states"].get("available", 0):
                        assert coordinator.service.error is None
                        assert coordinator.writer_actor.error is None
                        if time.monotonic() >= deadline:
                            raise TimeoutError(str(coordinator.stats()))
                        time.sleep(0.01)
                    assert coordinator.request_router is not None
                    assert coordinator.control_group is not dist.group.WORLD
                    assert coordinator.activation.is_set()
                    row = {
                        "case": case,
                        "phase": "ready",
                        "active": coordinator.partition.active,
                    }
                    dist.barrier()
                finally:
                    if coordinator is not None:
                        assert coordinator.close()
                assert groups == destroyed
                assert all(value.closed for value in resources)
                row["groups"] = len(groups)
                results.append(row)
                print(
                    f"startup rank={rank} case={index}:{case} closed phase={row['phase']}",
                    flush=True,
                )
            dist.barrier()
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class TestCohortCaptureStartup(CustomTestCase):
    def test_factory_owns_control_group_and_rolls_back_rank_local_failures(self):
        catalog = TestCaptureCatalog()
        workers = []
        try:
            with tempfile.TemporaryDirectory() as root:
                context = mp.get_context("spawn")
                workers = [
                    context.Process(
                        target=startup_worker, args=(rank, root, catalog.endpoint)
                    )
                    for rank in range(4)
                ]
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 100
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                results = [
                    json.loads((Path(root) / f"rank-{rank}.json").read_bytes())
                    for rank in range(4)
                ]
                for rows in results:
                    self.assertEqual(
                        [row["phase"] for row in rows],
                        [
                            "ready",
                            "ready",
                            "ready",
                            "policy_agreement",
                            "resources",
                            "activation",
                            "activation",
                            "activation",
                            "ready",
                        ],
                    )
                    self.assertEqual(
                        [row["groups"] for row in rows], [1, 1, 1, 0, 1, 1, 1, 1, 1]
                    )
                    for row in rows[4:8]:
                        self.assertEqual(row["failed_ranks"], [1])
                self.assertFalse(results[3][1]["active"])
                self.assertFalse(results[0][2]["active"])
                self.assertFalse(catalog.errors)
                self.assertFalse(catalog.publications)
                datasets = {
                    entry["begin"]["dataset_id"] for entry in catalog.captures.values()
                }
                self.assertEqual(
                    datasets, {"factory-0", "factory-1", "factory-2", "factory-8"}
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


if __name__ == "__main__":
    unittest.main()

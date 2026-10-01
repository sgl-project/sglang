"""Real Gloo startup failures must not strand peers in another collective phase."""

import json
import multiprocessing as mp
import os
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
from sglang.srt.training_capture import startup
from sglang.srt.training_capture.identity import LocalLayerContract, RankTargetContract
from sglang.srt.training_capture.protocol import (
    TeacherIdentity,
    canonical_bytes,
    digest_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import make_kv_spec

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def rank_contract(rank, *, tp_size, pp_size):
    kv = make_kv_spec()
    pp_rank, tp_rank = divmod(rank, tp_size)
    start, end = pp_rank * (4 // pp_size), (pp_rank + 1) * (4 // pp_size)
    first = tp_rank // (tp_size // 2)
    return RankTargetContract(
        teacher=TeacherIdentity(
            model_id="startup-fixture",
            weights_revision="weights",
            adapter_revision=None,
            tokenizer_revision="tokenizer",
            fingerprint_sha256="1" * 64,
            vocab_size=256,
            output_transform="identity",
        ),
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_rank=0,
        pp_layer_range=(start, end),
        num_attention_layers=4,
        selected_layer_ids=kv.selected_layer_ids,
        dtype=kv.dtype,
        source_page_size=kv.source_page_size,
        storage_chunk_tokens=kv.storage_chunk_tokens,
        layers=[
            LocalLayerContract(
                geometry=layer,
                head_range=(first, first + 1),
                rope_config=kv.rope_config,
                source_k_norm=kv.source_k_norm,
            )
            for layer in kv.layers
            if start <= layer.layer_id < end
        ],
    )


FAILURES = {
    "policy_local_error": ("policy", [1]),
    "policy_disagreement": ("policy_agreement", [0, 1, 2, 3]),
    "binding": ("binding", [1]),
    "record_limit": ("binding", [1]),
    "aggregate_limit": ("allocation", [1]),
    "allocation": ("allocation", [1]),
    "bad_schema": ("validation", [0, 1, 2, 3]),
    "identity": ("validation", [0, 1, 2, 3]),
    "origin": ("validation", [0, 1, 2, 3]),
    "topology": ("validation", [0, 1, 2, 3]),
    "local_validation": ("validation", [1]),
    "digest": ("agreement", [0, 1, 2, 3]),
    "version": ("protocol", [0, 1, 2, 3]),
}


def startup_worker(rank, root, timeout_case):
    torch.set_num_threads(1)
    results = []
    destination = Path(root) / f"rank-{rank}.json"
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    scenarios = (
        ["validation_timeout" if timeout_case == "validation" else "timeout"]
        if timeout_case
        else [
            "policy_local_error",
            "policy_disagreement",
            "policy_ready",
            "pp",
            "success",
        ]
    )
    if not timeout_case:
        for case in FAILURES:
            scenarios.extend((case, "success"))
        scenarios.append("peer_exit")
    try:
        for case in scenarios:
            tp_size, pp_size = (2, 2) if case == "pp" else (4, 1)
            contract = rank_contract(rank, tp_size=tp_size, pp_size=pp_size)
            if case == "identity" and rank == 1:
                contract = msgspec.structs.replace(
                    contract,
                    teacher=msgspec.structs.replace(
                        contract.teacher, weights_revision="other"
                    ),
                )
            if case == "origin":
                contract = rank_contract(
                    (rank + 1) % 4, tp_size=tp_size, pp_size=pp_size
                )

            def build_local(case=case, contract=contract):
                if case == "binding" and rank == 1:
                    raise OSError("local artifact cannot be read")
                if case == "timeout" and rank == 3:
                    time.sleep(5)
                return contract

            with ExitStack() as patches:
                if case == "validation_timeout" and rank == 3:
                    assemble = startup.assemble_target_contract

                    def slow_validation(*args, assemble=assemble, **kwargs):
                        time.sleep(5)
                        return assemble(*args, **kwargs)

                    patches.enter_context(
                        patch.object(
                            startup, "assemble_target_contract", slow_validation
                        )
                    )
                if rank == 1:
                    if case == "record_limit":
                        patches.enter_context(
                            patch.object(startup, "MAX_RECORD_BYTES", 1)
                        )
                    elif case == "aggregate_limit":
                        patches.enter_context(
                            patch.object(startup, "MAX_EXCHANGE_BYTES", 1)
                        )
                    elif case == "allocation":
                        allocate = torch.empty_like

                        def allocate_payload(value, *args, allocate=allocate, **kwargs):
                            if value.dtype == torch.uint8:
                                raise MemoryError(
                                    "simulated receive allocation failure"
                                )
                            return allocate(value, *args, **kwargs)

                        patches.enter_context(
                            patch.object(torch, "empty_like", allocate_payload)
                        )
                    elif case == "bad_schema":
                        patches.enter_context(
                            patch.object(startup, "canonical_bytes", return_value=b"{")
                        )
                    elif case == "local_validation":
                        patches.enter_context(
                            patch.object(
                                startup,
                                "assemble_target_contract",
                                side_effect=RuntimeError("local validation failure"),
                            )
                        )
                    elif case == "digest":
                        patches.enter_context(
                            patch.object(startup, "digest_bytes", return_value="0" * 64)
                        )
                    elif case == "version":
                        patches.enter_context(
                            patch.object(
                                startup,
                                "PROTOCOL_VERSION",
                                startup.PROTOCOL_VERSION + 1,
                            )
                        )
                dist.barrier()
                if case == "peer_exit" and rank == 3:
                    results.append({"case": case, "phase": "exited"})
                    destination.write_bytes(canonical_bytes(results))
                    os._exit(0)
                started = time.monotonic()
                try:
                    if case.startswith("policy_"):

                        def policy(case=case):
                            if case == "policy_local_error" and rank == 1:
                                raise ValueError("draft checkpoint is invalid")
                            return {
                                "weights": "other"
                                if case == "policy_disagreement" and rank == 2
                                else "common"
                            }

                        startup.coordinate_policy_startup(
                            group=dist.group.WORLD,
                            build_policy=policy,
                            timeout_seconds=10,
                        )
                    value = startup.coordinate_target_startup(
                        group=dist.group.WORLD,
                        build_local=build_local,
                        tp_size=2 if case == "topology" and rank == 1 else tp_size,
                        pp_size=2 if case == "topology" and rank == 1 else pp_size,
                        aux_tp_rank=1,
                        timeout_seconds=0.5 if timeout_case else 10,
                    )
                    results.append(
                        {
                            "case": case,
                            "phase": "ready",
                            "digest": digest_bytes(canonical_bytes(value)),
                            "owners": value[2].topology.owners,
                        }
                    )
                except startup.CaptureStartupError as error:
                    results.append(
                        {
                            "case": case,
                            "phase": error.phase,
                            "failed_ranks": list(error.failed_ranks),
                        }
                    )
                results[-1]["seconds"] = time.monotonic() - started
                destination.write_bytes(canonical_bytes(results))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class TestCaptureStartup(CustomTestCase):
    def run_workers(self, *, timeout_case=False):
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=startup_worker, args=(rank, root, timeout_case))
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 100
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                return [
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

    def test_rank_failures_preserve_collective_order_and_peer_exit_is_bounded(self):
        results = self.run_workers()
        for rows in zip(*results, strict=True):
            case = rows[0]["case"]
            with self.subTest(case=case):
                if case in ("pp", "success", "policy_ready"):
                    self.assertTrue(all(row["phase"] == "ready" for row in rows), rows)
                    self.assertEqual(len({row["digest"] for row in rows}), 1)
                    if case == "success":
                        self.assertNotIn("dp0-pp0-tp3", rows[0]["owners"])
                elif case == "peer_exit":
                    self.assertEqual(
                        [row["phase"] for row in rows], ["transport"] * 3 + ["exited"]
                    )
                    self.assertTrue(all(row["seconds"] < 15 for row in rows[:3]))
                else:
                    phase, ranks = FAILURES[case]
                    self.assertEqual([row["phase"] for row in rows], [phase] * 4)
                    self.assertEqual([row["failed_ranks"] for row in rows], [ranks] * 4)

    def test_slow_binding_times_out_without_returning_a_partial_contract(self):
        rows = [result[0] for result in self.run_workers(timeout_case=True)]
        self.assertEqual([row["phase"] for row in rows], ["transport"] * 4)
        self.assertTrue(all(row["seconds"] < 4 for row in rows[:3]), rows)
        self.assertLess(rows[3]["seconds"], 10)

    def test_slow_validator_cannot_return_after_peers_time_out(self):
        rows = [result[0] for result in self.run_workers(timeout_case="validation")]
        self.assertEqual([row["phase"] for row in rows], ["transport"] * 4)
        self.assertTrue(all(row["seconds"] < 4 for row in rows[:3]), rows)
        self.assertLess(rows[3]["seconds"], 10)

    def test_configuration_failure_occurs_inside_the_startup_vote(self):
        from sglang.srt.training_capture.coordinator import CaptureCoordinator

        def exchange(*, build_local, **kwargs):
            with self.assertRaises(FileNotFoundError):
                build_local()
            raise startup.CaptureStartupError("binding", [0])

        with (
            tempfile.TemporaryDirectory() as root,
            patch(
                "sglang.srt.training_capture.coordinator.coordinate_target_startup",
                side_effect=exchange,
            ) as vote,
            patch(
                "sglang.srt.training_capture.resources.MooncakeSnapshotStore.connect"
            ) as connect,
            self.assertRaises(startup.CaptureStartupError),
        ):
            try:
                CaptureCoordinator.create(
                    config_path=str(Path(root) / "missing.json"),
                    model=None,
                    model_config=None,
                    tokenizer_path=None,
                    pool=None,
                    req_to_token=None,
                    startup_group=object(),
                )
            finally:
                vote.assert_called_once()
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()

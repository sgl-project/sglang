"""Regression tests for PP-stage-local EPLB communication ownership."""

from __future__ import annotations

import multiprocessing
import os
import queue
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.multiprocessing as mp

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.eplb.control import ExpertDistributionControl
from sglang.srt.eplb.expert_distribution import (
    _DequeCollection,
    _StatAccumulator,
)
from sglang.srt.eplb.expert_location import (
    _compute_logical_to_all_physical_map,
    compute_logical_to_rank_dispatch_physical_map,
)
from sglang.srt.eplb.expert_location_updater import (
    _update_expert_weights,
    update_expert_weights_single_layer,
)
from sglang.srt.eplb.stage_domain import EPLBStageDomain
from sglang.srt.managers.io_struct import (
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
)
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_parallel,
    get_resources,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    publish_build_topology,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _run_stage_domain_worker(
    rank: int,
    master_port: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Build PP2/EP2 groups and exercise the real recorder collectives."""

    ps.set_custom_all_reduce(False)

    def _cpu_init_model_parallel_group(
        *args, _orig=ps.init_model_parallel_group, **kwargs
    ):
        kwargs.setdefault("use_pynccl", False)
        kwargs.setdefault("use_custom_allreduce", False)
        return _orig(*args, **kwargs)

    try:
        with patch.object(
            ps, "init_model_parallel_group", _cpu_init_model_parallel_group
        ):
            os.environ.update(
                RANK=str(rank),
                WORLD_SIZE="4",
                MASTER_ADDR="localhost",
                MASTER_PORT=str(master_port),
                LOCAL_SIZE="4",
            )
            init_distributed_environment(
                world_size=4,
                rank=rank,
                distributed_init_method="env://",
                local_rank=rank,
                backend="gloo",
            )
            publish_build_topology(
                tp_size=2,
                pp_size=2,
                ep_size=2,
                world_rank=rank,
                device="cpu",
                eplb_min_rebalancing_utilization_threshold=0.95,
            )
            initialize_model_parallel(backend="gloo")

            domain = EPLBStageDomain.from_runtime()

            accumulator = _StatAccumulator.__new__(_StatAccumulator)
            accumulator._expert_location_metadata = SimpleNamespace(
                num_layers=1,
                num_logical_experts=1,
                ep_size=2,
                physical_to_logical_map=torch.tensor([[0, 0]], dtype=torch.int32),
            )
            accumulator._rank = domain.ep_rank
            accumulator._stage_domain = domain
            accumulator._enable = True
            accumulator.window_sizes = (10,)
            accumulator._history = _DequeCollection(maxlens=(10,))
            accumulator._reset_server_log_history = True
            accumulator._handle_metric_eplb_heatmap = lambda _: None
            accumulator._first_dump = False

            # PP0 should report a perfectly balanced [1, 1] load. PP1 should
            # report [30, 0]. A WORLD reduction would mix them into [31, 1]
            # and make both stage results wrong.
            stage_physical_counts = {
                0: [1, 0],
                1: [0, 1],
                2: [10, 0],
                3: [20, 0],
            }
            metric_outputs = {}
            accumulator._append_utilization_rate(
                forward_pass_id=1,
                single_pass_global_physical_count=torch.tensor(
                    [stage_physical_counts[rank]], dtype=torch.int32
                ),
                outputs=metric_outputs,
            )

            # The stat DUMP has a different sentinel so the test independently
            # guards the all-reduce used by manual recording.
            dump_value = {0: 1, 1: 2, 2: 10, 3: 20}[rank]
            accumulator._global_physical_count_of_buffered_step = SimpleNamespace(
                get_all=lambda: torch.tensor([[[dump_value, 0]]], dtype=torch.int32)
            )
            dump_output = accumulator.dump(output_mode="object")

            expert_weight = torch.tensor([[100 + 50 * domain.pp_rank + rank]])
            temp_buffer = torch.empty_like(expert_weight)
            update_expert_weights_single_layer(
                routed_experts_weights=[expert_weight],
                temp_buffers=[temp_buffer],
                old_physical_to_logical_map=[0, 1],
                new_physical_to_logical_map=[1, 0],
                num_local_physical_experts=1,
                num_gpu_per_node=domain.ep_size,
                rank=domain.ep_rank,
                world_size=domain.ep_size,
                rank_to_node=domain.node_ids,
                peer_ranks=domain.global_ranks,
                p2p_group=domain.device_group,
            )

            class Recorder:
                recording = False

                def start_record(self):
                    if rank == 3:
                        raise OSError("stage1 EP1 recorder failure")
                    self.recording = True

                def stop_record(self):
                    self.recording = False

            recorder = Recorder()
            get_resources().expert_distribution_recorder = recorder
            control = ExpertDistributionControl(tokenizer_ipc_name="unused")
            replies = []

            def send_result(result, req):
                replies.append(
                    msgspec.json.decode(
                        msgspec.json.encode(result), type=ExpertDistributionReqOutput
                    )
                )

            with get_exec().moe.override(expert_distribution_recorder_mode="stat"):
                control.handle(
                    ExpertDistributionReq(
                        action=ExpertDistributionReqType.START_RECORD,
                        operation_id="gloo:1",
                        http_worker_ipc="worker-A",
                    ),
                    send_result=send_result,
                )
                start_result = replies[-1] if replies else None
                control.handle(
                    ExpertDistributionReq(
                        action=ExpertDistributionReqType.STOP_RECORD,
                        operation_id="gloo:2",
                        http_worker_ipc="worker-A",
                    ),
                    send_result=send_result,
                )
                stop_result = replies[-1] if replies else None
                control.handle(
                    ExpertDistributionReq(
                        action=ExpertDistributionReqType.START_RECORD,
                        operation_id="gloo:1",
                        http_worker_ipc="worker-B",
                    ),
                    send_result=send_result,
                )
                if domain.is_root:
                    assert len(replies) == 3
                    assert replies[-1].http_worker_ipc == "worker-B"
                    assert replies[-1].success == start_result.success
                    assert start_result.http_worker_ipc == "worker-A"
                    assert control._completed["gloo:1"][1].http_worker_ipc is None
                else:
                    assert not replies

            metric = metric_outputs.get("metrics")
            result_queue.put(
                {
                    "rank": rank,
                    "pp_rank": domain.pp_rank,
                    "ep_rank": domain.ep_rank,
                    "global_ranks": domain.global_ranks,
                    "root": domain.root_global_rank,
                    "is_root": domain.is_root,
                    "balancedness": (
                        None if metric is None else metric.eplb_balancedness.item()
                    ),
                    "dump_count": dump_output["logical_count"].item(),
                    "average_utilization": dump_output[
                        "average_utilization_rate_over_window"
                    ],
                    "expert_weight": expert_weight.item(),
                    "control_start_success": start_result.success
                    if start_result
                    else None,
                    "control_stop_success": stop_result.success
                    if stop_result
                    else None,
                    "control_error": start_result.message if start_result else None,
                    "recording_after_stop": recorder.recording,
                }
            )
    except Exception:
        traceback.print_exc()
        os._exit(2)
    finally:
        try:
            destroy_model_parallel()
        except Exception:
            pass
        try:
            destroy_distributed_environment()
        except Exception:
            pass


class TestEPLBStageDomain(CustomTestCase):
    def test_inactive_layers_do_not_dilute_stage_balancedness(self):
        """Zero rows from other stages must not make a skewed stage look balanced."""
        accumulator = _StatAccumulator.__new__(_StatAccumulator)
        accumulator._expert_location_metadata = SimpleNamespace(ep_size=2)
        accumulator._stage_domain = SimpleNamespace(
            is_root=True, root_global_rank=0, device_group=None
        )
        accumulator._handle_metric_eplb_heatmap = lambda _: None
        accumulator._reset_server_log_history = False
        with (
            get_context().override_server_args(
                device="cpu",
                eplb_min_rebalancing_utilization_threshold=1.0,
                expert_balancedness_report_mode="prometheus",
            ),
            patch("torch.distributed.reduce"),
        ):
            for counts, expected in (
                ([[30, 0]], 0.5),
                ([[0, 0], [30, 0], [0, 0]], 0.5),
                ([[0, 0], [0, 0]], 1.0),
            ):
                with self.subTest(counts=counts):
                    outputs = {}
                    accumulator._append_utilization_rate(
                        1, torch.tensor(counts, dtype=torch.int32), outputs
                    )
                    self.assertAlmostEqual(
                        outputs["metrics"].eplb_balancedness.item(), expected, places=5
                    )

    def test_replica_locality_uses_stage_nodes(self):
        # Four launch nodes, but this stage's EP4 owns only nodes 2 and 3.
        group = SimpleNamespace(
            ranks=(4, 5, 6, 7),
            world_size=4,
            rank=4,
            rank_in_group=0,
            first_rank=4,
            device_group=None,
            cpu_group=None,
        )
        physical = torch.tensor([[0, 1, 2, 1, 2, 0, 1, 0]])
        with (
            get_context().override_server_args(
                device="cpu",
                moe_a2a_backend="deepep",
                tp_size=4,
                pp_size=2,
                ep_size=4,
                nnodes=4,
            ),
            get_parallel().override(
                pp_rank=1,
                moe_ep_group=group,
            ),
        ):
            collapsed = _compute_logical_to_all_physical_map(
                physical,
                num_logical_experts=3,
                ep_size=4,
                moe_ep_rank=0,
            )
            self.assertEqual(collapsed[0, 2][collapsed[0, 2] >= 0].tolist(), [2])
            candidates = torch.tensor([[[0, 5, 7], [1, 3, 6], [2, 4, -1]]])
            for seed in range(4):
                dispatch = compute_logical_to_rank_dispatch_physical_map(
                    candidates,
                    ep_size=4,
                    num_physical_experts=8,
                    ep_rank=0,
                    seed=seed,
                )
                self.assertEqual(dispatch[0, 2].item(), 2)

    def test_foreign_layer_rejected_before_moving_owned_weights(self):
        """A later foreign layer must not leave earlier owned layers migrated."""
        weight = torch.tensor([[10], [20]])
        domain = EPLBStageDomain(
            pp_rank=0,
            global_rank=0,
            ep_rank=0,
            ep_size=1,
            global_ranks=(0,),
            node_ids=(0,),
            root_global_rank=0,
            device_group=None,
            cpu_group=None,
        )
        old = SimpleNamespace(
            ep_size=1,
            num_local_physical_experts=2,
            physical_to_logical_map_cpu=torch.tensor([[0, 1], [0, 1]]),
        )
        new = SimpleNamespace(
            ep_size=1,
            physical_to_logical_map_cpu=torch.tensor([[1, 0], [1, 0]]),
        )
        with self.assertRaisesRegex(ValueError, "stage-owned"):
            _update_expert_weights(
                routed_experts_weights_of_layer={0: [weight]},
                old_expert_location_metadata=old,
                new_expert_location_metadata=new,
                update_layer_ids=[0, 1],
                nnodes=1,
                rank=0,
                stage_domain=domain,
            )
        torch.testing.assert_close(weight, torch.tensor([[10], [20]]))

    def test_solver_rejects_nonuniform_or_interleaved_node_ownership(self):
        """Hierarchical packing assumes equal, contiguous EP slots per node."""
        for ranks in [(0, 1, 2, 4), (0, 2, 1, 3)]:
            with self.subTest(ranks=ranks):
                group = SimpleNamespace(
                    ranks=ranks,
                    world_size=4,
                    rank=ranks[0],
                    rank_in_group=0,
                    first_rank=ranks[0],
                    device_group=None,
                    cpu_group=None,
                )
                with get_parallel().override(
                    moe_ep_group=group,
                    moe_ep_size=4,
                    tp_size=4,
                    pp_size=2,
                    pp_rank=0,
                    nnodes=4,
                ):
                    with self.assertRaisesRegex(RuntimeError, "node ownership"):
                        EPLBStageDomain.from_runtime()

    def test_stage_uses_only_its_nodes_in_multinode_launch(self):
        """A stage wholly on node 1 must solve as one node, with global peers."""
        group = SimpleNamespace(
            ranks=(4, 5, 6, 7),
            world_size=4,
            rank=6,
            rank_in_group=2,
            first_rank=4,
            device_group=None,
            cpu_group=None,
        )
        with get_parallel().override(
            moe_ep_group=group, moe_ep_size=4, tp_size=4, pp_size=2, pp_rank=1, nnodes=2
        ):
            domain = EPLBStageDomain.from_runtime()
        self.assertEqual(domain.num_nodes, 1)
        self.assertEqual(domain.node_ids, (1, 1, 1, 1))
        self.assertEqual(domain.global_peer(2), 6)

    def test_pp_stages_reduce_independently(self):
        """PP stages must not mix recorder statistics or assume global root 0."""

        mp.set_start_method("spawn", force=True)
        master_port = find_available_port(23456)
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()

        old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "99"
        try:
            processes = [
                ctx.Process(
                    target=_run_stage_domain_worker,
                    args=(rank, master_port, result_queue),
                )
                for rank in range(4)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            self.assertEqual([process.exitcode for process in processes], [0] * 4)
        finally:
            if old_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices

        results = []
        for _ in range(4):
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty as exc:
                self.fail(f"missing child result: {exc}")

        by_rank = {result["rank"]: result for result in results}
        self.assertEqual(by_rank[0]["global_ranks"], (0, 1))
        self.assertEqual(by_rank[2]["global_ranks"], (2, 3))
        self.assertEqual(by_rank[0]["root"], 0)
        self.assertEqual(by_rank[2]["root"], 2)
        self.assertTrue(by_rank[0]["is_root"])
        self.assertTrue(by_rank[2]["is_root"])
        self.assertEqual([by_rank[i]["dump_count"] for i in range(4)], [3, 3, 30, 30])
        for rank, expected in enumerate([1.0, 1.0, 0.5, 0.5]):
            self.assertAlmostEqual(
                by_rank[rank]["average_utilization"], expected, places=5
            )
        self.assertEqual(
            [by_rank[i]["expert_weight"] for i in range(4)],
            [101, 100, 153, 152],
        )
        self.assertAlmostEqual(by_rank[0]["balancedness"], 1.0, places=5)
        self.assertAlmostEqual(by_rank[2]["balancedness"], 0.5, places=5)
        self.assertIsNone(by_rank[1]["balancedness"])
        self.assertIsNone(by_rank[3]["balancedness"])
        self.assertEqual(
            [by_rank[i]["control_start_success"] for i in range(4)],
            [True, None, False, None],
        )
        self.assertEqual(
            [by_rank[i]["control_stop_success"] for i in range(4)],
            [True, None, True, None],
        )
        for result in by_rank.values():
            self.assertFalse(result["recording_after_stop"])
        self.assertIn("EP1", by_rank[2]["control_error"])
        self.assertIn("PP1", by_rank[2]["control_error"])


if __name__ == "__main__":
    unittest.main()

"""Dependency-free checks of EPLB's PP communication and layer contracts.

Execute the actual source definitions with small distributed/tensor doubles;
this does not validate torch kernels or a real HCCL communicator.
"""

import ast
import logging
import math
import unittest
from abc import ABC
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "python/sglang/srt"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class TensorDouble:
    shape = (2, 1)
    nbytes = 4

    def to(self, *args, **kwargs):
        return self

    def item(self):
        return 0.5

    def sum(self):
        return self

    def copy_(self, other):
        return self

    def __getitem__(self, key):
        return self

    def tolist(self):
        return list(range(8))


class NumberDouble(float):
    def item(self):
        return float(self)


class VectorDouble:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return NumberDouble(sum(self.values))

    def __gt__(self, value):
        return VectorDouble([item > value for item in self.values])


class CountMatrixDouble:
    def __init__(self, rows):
        self.rows = rows

    def to(self, *args, **kwargs):
        return self

    def sum(self, dim=None):
        sums = [sum(row) for row in self.rows]
        return VectorDouble(sums) if dim == -1 else NumberDouble(sum(sums))


class DistributedDouble:
    ReduceOp = SimpleNamespace(SUM="sum")

    def __init__(self, global_rank, active_ranks):
        self.global_rank = global_rank
        self.active_ranks = active_ranks
        self.calls = []
        self.ops = []
        self.isend = Mock()
        self.irecv = Mock()

    def get_world_size(self, group=None):
        return len(group) if group is not None else 16

    def get_rank(self, group=None):
        return group.index(self.global_rank) if group is not None else self.global_rank

    def get_global_rank(self, group, rank):
        return group[rank]

    def get_group_rank(self, group, rank):
        return group.index(rank)

    def collective(self, kind, tensor, group=None, **kwargs):
        participants = group if group is not None else tuple(range(16))
        if participants != self.active_ranks:
            raise RuntimeError(f"{kind} waits for other PP stages: {participants}")
        self.calls.append((kind, group, kwargs))

    def reduce(self, tensor, **kwargs):
        self.collective("reduce", tensor, **kwargs)

    def all_reduce(self, tensor, **kwargs):
        self.collective("all_reduce", tensor, **kwargs)

    def broadcast(self, tensor, **kwargs):
        self.collective("broadcast", tensor, **kwargs)

    def p2p_op(self, **kwargs):
        op = SimpleNamespace(**kwargs)
        self.ops.append(op)
        return op

    def batch_isend_irecv(self, ops):
        return [SimpleNamespace(wait=lambda: None) for _ in ops]


class TestEplbPPContract(unittest.TestCase):
    def recorder(self, pp_size=4, stage=1, enabled=True, threshold=1.0, rows=None):
        ranks = (
            tuple(range(stage * 4, stage * 4 + 4)) if pp_size > 1 else tuple(range(16))
        )
        dist = DistributedDouble(ranks[0], ranks)
        parallel = SimpleNamespace(
            pp_size=pp_size,
            pp_rank=stage,
            tp_group=SimpleNamespace(device_group=ranks, ranks=list(ranks)),
        )
        config = SimpleNamespace(
            eplb_min_rebalancing_utilization_threshold=threshold,
            expert_distribution_recorder_buffer_size=50,
        )
        history = SimpleNamespace(mean=lambda: {1000: 0.5}, append=lambda value: None)
        tensor = TensorDouble()
        counts = CountMatrixDouble(
            rows if rows is not None else [[4, 0, 0, 0]] + [[0, 0, 0, 0]] * 3
        )

        def make_tensor(*args, device=None, **kwargs):
            self.assertEqual(device, "npu")
            return tensor

        namespace = dict(
            ABC=ABC,
            torch=SimpleNamespace(
                distributed=dist,
                int32="int32",
                float32="float32",
                mean=lambda value: NumberDouble(sum(value.values) / len(value.values)),
                where=lambda mask, values, zero: VectorDouble(
                    [
                        value if active else zero
                        for active, value in zip(mask.values, values.values)
                    ]
                ),
                tensor=make_tensor,
                empty=make_tensor,
                get_device_module=lambda: SimpleNamespace(empty_cache=lambda: None),
            ),
            get_parallel=lambda: parallel,
            get_exec=lambda: SimpleNamespace(moe=config),
            get_device_namespace=lambda: SimpleNamespace(device="npu"),
            reports_expert_balancedness=lambda: enabled,
            logs_expert_balancedness_to_server_log=lambda: True,
            compute_gpu_physical_count=lambda value, **kwargs: counts,
            compute_utilization_rate=lambda value: VectorDouble(
                [(sum(row) / len(row) + 1e-5) / (max(row) + 1e-5) for row in value.rows]
            ),
            _convert_global_physical_count_to_logical_count=lambda *args, **kwargs: (
                tensor
            ),
            _Buffer=SimpleNamespace(
                init_new=lambda **kwargs: SimpleNamespace(get_all=lambda: tensor)
            ),
            _DequeCollection=lambda **kwargs: history,
            EPLB_BALANCEDNESS_WINDOW_SIZES=(1000,),
            resolve_collector_class=lambda *args: lambda *args: None,
            STAT_LOGGER_ROLE_EXPERT_DISPATCH=None,
            ExpertDispatchCollector=None,
            ExpertDistributionMetrics=lambda **kwargs: kwargs,
            math=math,
        )
        load_definitions(
            SRT / "eplb/expert_distribution.py",
            ["_Accumulator", "_UtilizationRateAccumulatorMixin", "_StatAccumulator"],
            namespace,
        )
        metadata = SimpleNamespace(
            num_layers=8,
            num_physical_experts=8,
            num_logical_experts=8,
            ep_size=4,
            physical_to_logical_map=tensor,
        )
        accumulator = namespace["_StatAccumulator"](metadata, rank=0)
        accumulator._handle_metric_eplb_heatmap = lambda value: None
        return accumulator, dist

    def test_metrics_use_stage_group_and_stage_root(self):
        for stage in range(4):
            with self.subTest(stage=stage):
                accumulator, dist = self.recorder(stage=stage)
                output = {}
                accumulator._append_utilization_rate(1, TensorDouble(), output)
                self.assertEqual(dist.calls[0][2]["dst"], stage * 4)
                self.assertIn("metrics", output)

    def test_stat_dump_uses_stage_group_even_without_metrics(self):
        accumulator, dist = self.recorder(enabled=False)
        self.assertIn("logical_count", accumulator.dump("object"))
        self.assertEqual(dist.calls[0][0], "all_reduce")

    def test_threshold_broadcast_uses_stage_root_and_npu_device(self):
        accumulator, dist = self.recorder(threshold=0.8)
        self.assertEqual(accumulator._get_global_average_utilization_rate(), 0.5)
        self.assertEqual(dist.calls[0][2]["src"], 4)

    def test_pp1_preserves_default_collectives(self):
        accumulator, dist = self.recorder(pp_size=1, stage=0)
        accumulator._append_utilization_rate(1, TensorDouble(), {})
        accumulator.dump("object")
        self.assertTrue(all(group is None for _, group, _ in dist.calls))

    def test_metrics_exclude_nonlocal_zero_layers(self):
        accumulator, _ = self.recorder()
        output = {}
        accumulator._append_utilization_rate(1, TensorDouble(), output)
        self.assertAlmostEqual(output["metrics"]["eplb_balancedness"], 0.25, places=4)

    def test_fully_idle_metrics_are_balanced(self):
        accumulator, _ = self.recorder(rows=[[0, 0, 0, 0]] * 4)
        output = {}
        accumulator._append_utilization_rate(1, TensorDouble(), output)
        self.assertEqual(output["metrics"]["eplb_balancedness"], 1.0)

    def test_weight_migration_maps_stage_peers_to_global_ranks(self):
        group = (4, 5, 6, 7)
        dist = DistributedDouble(4, group)
        namespace = dict(
            torch=SimpleNamespace(distributed=dist),
            P2POp=dist.p2p_op,
            ElasticEPStateManager=SimpleNamespace(instance=lambda: None),
            envs=SimpleNamespace(
                SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE=SimpleNamespace(get=lambda: 32)
            ),
            _LOG_INPUT=False,
            logger=logging.getLogger(__name__),
        )
        load_definitions(
            SRT / "eplb/expert_location_updater.py",
            [
                "update_expert_weights_single_layer",
                "_ChunkUtils",
                "_deduplicate_ordered",
            ],
            namespace,
        )
        namespace["update_expert_weights_single_layer"](
            [TensorDouble()],
            [TensorDouble()],
            list(range(8)),
            [2, 1, 0, 3, 4, 5, 6, 7],
            num_local_physical_experts=2,
            num_gpu_per_node=4,
            rank=0,
            world_size=4,
            group=group,
        )
        self.assertEqual(len(dist.ops), 2)
        self.assertTrue(all(op.peer == 5 and op.group == group for op in dist.ops))

    def test_updater_propagates_stage_group_and_world_size(self):
        for pp_size in (1, 4):
            with self.subTest(pp_size=pp_size):
                group = (4, 5, 6, 7)
                dist = DistributedDouble(4, group)
                metadata = SimpleNamespace(
                    num_local_physical_experts=2,
                    physical_to_logical_map_cpu=TensorDouble(),
                    update=Mock(),
                )
                migrate = Mock()
                namespace = dict(
                    torch=SimpleNamespace(distributed=dist),
                    get_parallel=lambda: SimpleNamespace(
                        pp_size=pp_size,
                        tp_group=SimpleNamespace(device_group=group),
                    ),
                    get_global_expert_location_metadata=lambda: metadata,
                    get_bool_env_var=lambda key: False,
                    create_temp_buffers=lambda tensors: [],
                    update_expert_weights_single_layer=migrate,
                )
                load_definitions(
                    SRT / "eplb/expert_location_updater.py",
                    [
                        "ExpertLocationUpdater",
                        "_update_expert_weights",
                        "_update_expert_weights_raw",
                    ],
                    namespace,
                )
                updater = namespace["ExpertLocationUpdater"]()
                updater._first_execution = False
                updater.update({2: [TensorDouble()]}, metadata, [2], nnodes=1, rank=0)
                kwargs = migrate.call_args.kwargs
                self.assertEqual(kwargs["group"], group if pp_size > 1 else None)
                self.assertEqual(kwargs["world_size"], 4 if pp_size > 1 else 16)
                self.assertEqual(kwargs["num_gpu_per_node"], kwargs["world_size"])
                metadata.update.assert_called_once_with(metadata, update_layer_ids=[2])

    def test_glm_weight_export_ignores_other_stage_placeholders(self):
        class MoE:
            def get_moe_weights(self):
                return ["layer2_weights"]

        model = SimpleNamespace(
            start_layer=2,
            end_layer=3,
            layers=[object(), object(), SimpleNamespace(mlp=MoE()), object()],
        )
        tree = ast.parse((SRT / "models/glm5_next.py").read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "_routed_experts_weights_of_layer"
                for target in node.targets
            )
        )
        lazy = eval(
            compile(ast.Expression(assignment.value), "glm_weight_export", "eval"),
            dict(
                self=SimpleNamespace(model=model),
                LazyValue=lambda fn: fn,
                Glm5NextMoE=MoE,
            ),
        )
        self.assertEqual(lazy(), {2: ["layer2_weights"]})


if __name__ == "__main__":
    unittest.main()

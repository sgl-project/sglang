# SPDX-License-Identifier: Apache-2.0

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    maybe_enable_ipc_weight_cache,
)
from sglang.srt.weight_cache.daemon import (
    WeightCacheDaemon,
    WeightCacheDaemonArgs,
    launch_weight_cache_daemons,
)
from sglang.srt.weight_cache.mooncake_weight_adapter import (
    _parallel_axes,
    build_mooncake_weight_manifests,
)
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
    SourceWeightsManifest,
    WeightHeterogeneousTransferError,
    WeightParallelLayout,
    _initialize_weight_transfer_engine,
    fetch_source_weights_manifest,
    register_source_weights_manifest,
    transfer_weights_from_source_daemons,
    validate_weight_heterogeneous_transfer_configuration,
)
from sglang.srt.weight_cache.weight_load_recorder import (
    WeightLoadRecorder,
    WeightLoadRecordingError,
    capture_weight_load_plan,
    record_target_weight_load_plan,
)
from sglang.srt.weight_cache.weight_manifest_server import WeightManifestServer
from sglang.srt.weight_cache.weight_runtime_manifest import (
    IMMUTABLE_WEIGHT_GENERATION,
    ImmutableWeightRuntimeManifestBuilder,
    WeightManifestError,
    WeightParallelTopology,
    model_identity_from_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _daemon_server_args(**overrides):
    values = {
        "model_path": "/models/demo",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "moe_dp_size": 1,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "attn_cp_size": 1,
        "moe_dense_tp_size": 1,
        "moe_a2a_backend": "none",
        "deepep_mode": "auto",
        "load_format": "auto",
        "dtype": "auto",
        "quantization": None,
        "model_loader_extra_config": {},
        "trust_remote_code": False,
        "revision": None,
        "nnodes": 1,
        "node_rank": 0,
        "base_gpu_id": 0,
        "gpu_id_step": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestWeightHeterogeneousTransfer(unittest.TestCase):
    def test_source_capture_isolated_to_daemon_loader_instance(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))
                self.postprocessed = False

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight)

        class Loader:
            @staticmethod
            def load_weights_and_postprocess(model, weights, target_device):
                del target_device
                model.load_weights(weights)
                model.postprocessed = True

            def load_model(self):
                model = Model()
                self.load_weights_and_postprocess(
                    model,
                    (("weight", torch.ones(2, 2)),),
                    torch.device("cpu"),
                )
                return model

        loader = Loader()
        unrelated_loader = Loader()
        native_load_and_postprocess = Loader.load_weights_and_postprocess

        with capture_weight_load_plan(loader) as capture:
            self.assertIs(
                unrelated_loader.load_weights_and_postprocess,
                native_load_and_postprocess,
            )
            model = loader.load_model()

        self.assertTrue(model.postprocessed)
        self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))
        self.assertNotIn("load_weights", vars(model))
        self.assertNotIn("load_weights_and_postprocess", vars(loader))
        self.assertIs(loader.load_weights_and_postprocess, native_load_and_postprocess)
        self.assertEqual(capture.plan.views[0].tensor_id, "weight")

    def test_model_identity_ignores_local_path_and_runtime_metadata(self):
        class Config:
            def __init__(self, path, hidden_size=128):
                self.model_type = "demo"
                self.path = path
                self.hidden_size = hidden_size

            def to_dict(self):
                return {
                    "model_type": self.model_type,
                    "hidden_size": self.hidden_size,
                    "_name_or_path": self.path,
                    "_commit_hash": "local-commit",
                    "transformers_version": "local-version",
                    "text_config": {
                        "hidden_size": self.hidden_size,
                        "layer_types": {0: "attention"},
                    },
                }

        source = model_identity_from_config(Config("/source/model"))
        target = model_identity_from_config(Config("/target/model"))
        incompatible = model_identity_from_config(Config("/target/model", 256))

        self.assertEqual(source, target)
        self.assertNotEqual(source, incompatible)

    def test_transfer_engine_uses_normal_mooncake_endpoint_discovery(self):
        engine = MagicMock()
        engine.initialize.return_value = 0
        engine.get_rpc_port.return_value = 12345

        with (
            patch("mooncake.engine.TransferEngine", return_value=engine),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.get_local_ip_auto",
                return_value="10.0.0.8",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.envs.MOONCAKE_PROTOCOL.get",
                return_value="rdma",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.envs.MOONCAKE_DEVICE.get",
                return_value="mlx5_0",
            ),
        ):
            actual_engine, endpoint = _initialize_weight_transfer_engine()

        self.assertIs(actual_engine, engine)
        self.assertEqual(endpoint, "10.0.0.8:12345")
        engine.initialize.assert_called_once_with(
            "10.0.0.8", "P2PHANDSHAKE", "rdma", "mlx5_0"
        )

    def test_daemon_manifest_adapts_to_immutable_runtime_binding(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight)

        model = Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            model,
            (("weight", torch.arange(4).reshape(2, 2).float()),),
            execute_writes=True,
        )
        manifest = ImmutableWeightRuntimeManifestBuilder(
            model=model,
            load_plan=recorder.build_plan(),
            topology=WeightParallelTopology(),
            allowed_devices=("cpu",),
        ).build(
            model_id="model",
            revision="revision",
            instance_id="worker",
            worker_id="worker",
            endpoint="127.0.0.1:1",
        )

        self.assertEqual(manifest.generation, IMMUTABLE_WEIGHT_GENERATION)
        self.assertFalse(hasattr(manifest, "lease_id"))
        placement, bindings = build_mooncake_weight_manifests(
            (manifest,),
            placement_set_id="test",
            tp_size=1,
            pp_size=1,
            ep_size=1,
        )
        self.assertEqual(bindings[0].generation, IMMUTABLE_WEIGHT_GENERATION)
        self.assertEqual(bindings[0].placement_id, placement.placement_id)
        self.assertEqual(bindings[0].lease_id, "immutable:worker")

    def test_recorder_replays_column_and_row_parallel_native_loaders(self):
        class ShardedModel(torch.nn.Module):
            def __init__(self, *, dim, rank, size):
                super().__init__()
                self.dim = dim
                self.rank = rank
                self.size = size
                shape = [4, 6]
                shape[dim] //= size
                self.weight = torch.nn.Parameter(torch.full(shape, -1.0))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    shard = loaded_weight.shape[self.dim] // self.size
                    loaded_weight = loaded_weight.narrow(
                        self.dim, self.rank * shard, shard
                    )
                    self.weight.data.copy_(loaded_weight)

        for dim, expected_offset in ((0, (2, 0)), (1, (0, 3))):
            source = ShardedModel(dim=dim, rank=0, size=1)
            source_recorder = WeightLoadRecorder()
            source_recorder.record_model_load(
                source,
                (("projection.weight", torch.arange(24).reshape(4, 6).float()),),
                execute_writes=True,
            )
            target = ShardedModel(dim=dim, rank=1, size=2)
            before = target.weight.detach().clone()
            target_plan = record_target_weight_load_plan(
                target, source_recorder.build_plan().logical_weights
            )
            self.assertTrue(torch.equal(target.weight, before))
            self.assertEqual(target_plan.views[0].global_offset, expected_offset)
            self.assertEqual(
                target_plan.views[0].local_shape,
                tuple(target.weight.shape),
            )

    def test_recorder_tracks_fused_split_and_grouped_loaders(self):
        class FusedModel(torch.nn.Module):
            def __init__(self, *, rank=0, size=1):
                super().__init__()
                self.rank = rank
                self.size = size
                self.weight = torch.nn.Parameter(torch.empty(6 // size, 2))

            def load_weights(self, weights):
                for name, loaded_weight in weights:
                    shard = loaded_weight.shape[0] // self.size
                    source = loaded_weight.narrow(0, self.rank * shard, shard)
                    cursor = 0 if name.startswith("q_a_proj") else 2 // self.size
                    destination = self.weight.data.narrow(0, cursor, shard)
                    destination.copy_(source)

        source = FusedModel()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source,
            (
                ("q_a_proj.weight", torch.ones(2, 2)),
                ("kv_a_proj_with_mqa.weight", torch.ones(4, 2)),
            ),
            execute_writes=True,
        )
        target = FusedModel(rank=1, size=2)
        plan = record_target_weight_load_plan(
            target, recorder.build_plan().logical_weights
        )
        self.assertEqual(
            tuple(
                (view.tensor_id, view.global_offset, view.byte_offset)
                for view in plan.views
            ),
            (
                ("q_a_proj.weight", (1, 0), 0),
                ("kv_a_proj_with_mqa.weight", (2, 0), 8),
            ),
        )

        class GroupedModel(torch.nn.Module):
            def __init__(self, rank):
                super().__init__()
                self.rank = rank
                self.weight = torch.nn.Parameter(torch.empty(6, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(
                        loaded_weight.narrow(0, self.rank * 6, 6)
                    )

        grouped = GroupedModel(rank=1)
        grouped_recorder = WeightLoadRecorder()
        grouped_recorder.record_model_load(
            grouped,
            (("in_proj_qkvz.weight", torch.empty(12, 2)),),
            execute_writes=False,
        )
        grouped_view = grouped_recorder.build_plan().views[0]
        self.assertEqual(grouped_view.global_offset, (6, 0))
        self.assertEqual(grouped_view.local_shape, (6, 2))

    def test_recorder_tracks_concat_then_shard_in_threaded_loader(self):
        import concurrent.futures

        class ConcatenatedModel(torch.nn.Module):
            def __init__(self, *, rank, size):
                super().__init__()
                self.rank = rank
                self.size = size
                self.weight = torch.nn.Parameter(torch.empty(6 // size, 2))

                def weight_loader(param, loaded_weight):
                    shard = loaded_weight.shape[0] // self.size
                    param.data.copy_(
                        loaded_weight.narrow(0, self.rank * shard, shard)
                    )

                self.weight.weight_loader = weight_loader

            def load_weights(self, weights):
                parts = {name: tensor for name, tensor in weights}
                fused = torch.cat((parts["q.weight"], parts["kv.weight"]), dim=0)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(
                        self.weight.weight_loader,
                        self.weight,
                        fused,
                    ).result()

        source = ConcatenatedModel(rank=0, size=1)
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source,
            (
                ("q.weight", torch.empty(4, 2)),
                ("kv.weight", torch.empty(2, 2)),
            ),
            execute_writes=True,
        )
        target = ConcatenatedModel(rank=1, size=2)
        before = target.weight.detach().clone()
        plan = record_target_weight_load_plan(
            target, recorder.build_plan().logical_weights
        )

        self.assertTrue(torch.equal(target.weight, before))
        self.assertEqual(
            tuple(
                (
                    view.tensor_id,
                    view.global_offset,
                    view.local_shape,
                    view.byte_offset,
                )
                for view in plan.views
            ),
            (
                ("q.weight", (3, 0), (1, 2), 0),
                ("kv.weight", (0, 0), (2, 2), 8),
            ),
        )
        manifest = ImmutableWeightRuntimeManifestBuilder(
            model=target,
            load_plan=plan,
            topology=WeightParallelTopology(tp_rank=1, tp_size=2),
            allowed_devices=("cpu",),
        ).build(
            model_id="model",
            revision="revision",
            instance_id="worker",
            worker_id="worker",
            endpoint="127.0.0.1:1",
        )
        self.assertEqual(sum(tensor.nbytes for tensor in manifest.tensors), 24)

    def test_recorder_tracks_default_loader_in_thread_pool(self):
        import concurrent.futures

        from sglang.srt.model_loader.weight_utils import default_weight_loader

        class ThreadedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))

            def load_weights(self, weights):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            default_weight_loader,
                            self.weight,
                            loaded_weight,
                        )
                        for _, loaded_weight in weights
                    ]
                    for future in futures:
                        future.result()

        native_submit = concurrent.futures.ThreadPoolExecutor.submit
        source = ThreadedModel()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source,
            (("weight", torch.ones(2, 2)),),
            execute_writes=True,
        )

        self.assertIs(concurrent.futures.ThreadPoolExecutor.submit, native_submit)
        self.assertTrue(torch.equal(source.weight, torch.ones(2, 2)))
        self.assertEqual(recorder.build_plan().views[0].tensor_id, "weight")

        target = ThreadedModel()
        before = target.weight.detach().clone()
        target_plan = record_target_weight_load_plan(
            target, recorder.build_plan().logical_weights
        )

        self.assertIs(concurrent.futures.ThreadPoolExecutor.submit, native_submit)
        self.assertTrue(torch.equal(target.weight, before))
        self.assertEqual(target_plan.views[0].tensor_id, "weight")

    def test_recorder_tracks_transposed_moe_expert_storage(self):
        class ExpertModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w13_weight = torch.nn.Parameter(torch.empty(2, 3, 2))

                def weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    *,
                    shard_id,
                    expert_id,
                ):
                    del weight_name, shard_id
                    param.data[expert_id].copy_(loaded_weight.transpose(0, 1))

                self.w13_weight.weight_loader = weight_loader

            def load_weights(self, weights):
                for name, loaded_weight in weights:
                    expert_id = int(name.split(".experts.")[1].split(".")[0])
                    self.w13_weight.weight_loader(
                        self.w13_weight,
                        loaded_weight,
                        name,
                        shard_id="w1",
                        expert_id=expert_id,
                    )

        model = ExpertModel()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            model,
            (
                ("layers.0.experts.0.gate_proj.weight", torch.empty(2, 3)),
                ("layers.0.experts.1.gate_proj.weight", torch.empty(2, 3)),
            ),
            execute_writes=False,
        )
        plan = recorder.build_plan()
        self.assertEqual(tuple(view.expert_id for view in plan.views), (0, 1))
        self.assertTrue(all(view.global_shape == (3, 2) for view in plan.views))
        self.assertTrue(
            all("permute(1,0)" in view.layout_fingerprint for view in plan.views)
        )

    def test_recorder_rejects_value_transform(self):
        class ValueTransformModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight + 1)

        with self.assertRaisesRegex(
            WeightLoadRecordingError, "unsupported loader operation"
        ):
            recorder = WeightLoadRecorder()
            recorder.record_model_load(
                ValueTransformModel(),
                (("weight", torch.empty(2, 2)),),
                execute_writes=False,
            )

    def test_manifest_rejects_parameter_not_covered_by_native_loader(self):
        class IncompleteModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2))
                self.unused = torch.nn.Parameter(torch.empty(2, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight)

        model = IncompleteModel()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            model,
            (("weight", torch.empty(2, 2)),),
            execute_writes=True,
        )
        with self.assertRaisesRegex(
            WeightManifestError,
            "native weight-load recorder did not cover parameter: unused",
        ):
            ImmutableWeightRuntimeManifestBuilder(
                model=model,
                load_plan=recorder.build_plan(),
                topology=WeightParallelTopology(),
                allowed_devices=("cpu",),
            ).build(
                model_id="model",
                revision="revision",
                instance_id="worker",
                worker_id="worker",
                endpoint="127.0.0.1:1",
            )

    def test_weight_parallel_layout_supports_attention_dp_pp_and_ep(self):
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=8,
            dp_size=2,
            ep_size=4,
            pp_size=2,
            enable_dp_attention=True,
            quantization=None,
        )
        parallel_layout = WeightParallelLayout(
            tp_size=4, dp_size=2, pp_size=2, ep_size=2
        )
        self.assertEqual(parallel_layout.world_size, 8)
        self.assertEqual(parallel_layout.dp_size, 2)
        self.assertEqual(
            tuple(parallel_layout.iter_ranks()),
            tuple((pp, tp) for pp in range(2) for tp in range(4)),
        )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "without enable_dp_attention"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=False,
                quantization=None,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "must be divisible by dp_size"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=3,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=True,
                quantization=None,
            )
        with self.assertRaises(WeightHeterogeneousTransferError):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                dp_size=1,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=False,
                quantization="fp8",
            )

    def test_local_expert_axis_is_not_misclassified_as_tp(self):
        from mooncake.reshard.weight import ReplicatedAxis, SplitAxis

        tensor = {"shard_dims": (0, 2), "expert_id": None}
        self.assertEqual(
            _parallel_axes(tensor, tp_size=2, pp_size=1, ep_size=1),
            (SplitAxis("tp", dim=2),),
        )
        self.assertEqual(
            _parallel_axes(tensor, tp_size=4, pp_size=1, ep_size=2),
            (
                SplitAxis("ep", dim=0),
                SplitAxis("tp", dim=2),
            ),
        )
        self.assertEqual(
            _parallel_axes(
                {
                    "shard_dims": (0,),
                    "global_shape": (32, 8),
                    "local_shape": (32, 8),
                    "expert_id": None,
                },
                tp_size=2,
                pp_size=1,
                ep_size=1,
            ),
            (ReplicatedAxis("tp"),),
        )

    def test_internal_fragments_replicated_by_dp_attention_are_not_tp_shards(self):
        from mooncake.reshard.weight import ReplicatedAxis

        inventories = []
        for tp_rank in range(2):
            tensors = []
            storage_address = 100_000 + tp_rank * 1_000
            for part in range(3):
                tensors.append(
                    {
                        "fragment_id": f"rank{tp_rank}:part{part}",
                        "tensor_id": "model.layers.0.linear_attn.conv1d.weight",
                        "runtime_name": "runtime.conv1d.weight",
                        "global_shape": (6, 1, 4),
                        "global_offset": (part * 2, 0, 0),
                        "local_shape": (2, 1, 4),
                        "dtype": "bfloat16",
                        "itemsize": 2,
                        "shard_dims": (0,),
                        "layer_id": 0,
                        "expert_id": None,
                        "layout_fingerprint": "recorded-slices",
                        "address": storage_address + part * 16,
                        "nbytes": 16,
                        "byte_offset": part * 16,
                        "stride": (4, 4, 1),
                        "storage_offset": part * 8,
                        "device": f"cuda:{tp_rank}",
                        "worker_id": f"worker-{tp_rank}",
                        "endpoint": f"127.0.0.1:{1000 + tp_rank}",
                        "rank": {"dp": 0, "tp": tp_rank, "pp": 0, "ep": 0},
                    }
                )
            inventories.append(
                {
                    "model_id": "model",
                    "revision": "revision",
                    "instance_id": f"worker-{tp_rank}",
                    "generation": IMMUTABLE_WEIGHT_GENERATION,
                    "tensors": tensors,
                }
            )

        placement, _ = build_mooncake_weight_manifests(
            inventories,
            placement_set_id="target",
            tp_size=2,
            pp_size=1,
            ep_size=1,
        )

        descriptors = {
            tensor.tensor_id: tensor
            for part in placement.parts
            for tensor in part.tensors
        }
        descriptor = descriptors[
            "model.layers.0.linear_attn.conv1d.weight"
        ]
        self.assertEqual(descriptor.parallel_axes, (ReplicatedAxis("tp"),))
        self.assertEqual(descriptor.shard_dims, ())

    def test_manifest_server_aggregates_source_ranks(self):
        server = object.__new__(WeightManifestServer)
        server.expected_rank_count = 2
        server._node_id = None
        server._parallel_layout = None
        server._model_identity = None
        server._rank_manifests = {}
        server._lock = threading.Lock()

        for global_rank, gpu_id in ((0, 2), (1, 3)):
            server.register_weight_manifest(
                {
                    "node_id": "source-node",
                    "global_rank": global_rank,
                    "gpu_id": gpu_id,
                    "parallel_layout": {
                        "tp_size": 2,
                        "dp_size": 2,
                        "pp_size": 1,
                        "ep_size": 1,
                    },
                    "runtime_inventory": {
                        "model_id": "model",
                        "revision": "revision",
                        "rank": global_rank,
                    },
                    "content_checksums": [{"rank": global_rank}],
                }
            )

        source = SourceWeightsManifest.from_wire(server.get_source_weights_manifest())
        self.assertEqual(source.node_id, "source-node")
        self.assertEqual(source.gpu_ids, (2, 3))
        self.assertEqual(source.parallel_layout.dp_size, 2)
        self.assertEqual(
            tuple(item["rank"] for item in source.runtime_inventories), (0, 1)
        )

    def test_tcp_manifest_registry_round_trip(self):
        server = WeightManifestServer(
            host="127.0.0.1", port=0, expected_rank_count=1
        )
        registry_url = f"tcp://127.0.0.1:{server.port}"
        manifest_state = MagicMock()
        manifest_state.runtime_inventory_for_wire.return_value = {
            "model_id": "model",
            "revision": "revision",
            "rank": 0,
        }
        manifest_state.source_content_checksums = ({"rank": 0},)
        try:
            register_source_weights_manifest(
                registry_url,
                global_rank=0,
                gpu_id=2,
                tp_size=1,
                dp_size=1,
                pp_size=1,
                ep_size=1,
                manifest_state=manifest_state,
                timeout=2,
            )
            actual = fetch_source_weights_manifest(registry_url, timeout=2)
        finally:
            server.close()

        self.assertEqual(actual.gpu_ids, (2,))
        self.assertEqual(actual.parallel_layout, WeightParallelLayout(tp_size=1))
        self.assertEqual(
            actual.runtime_inventories,
            ({"model_id": "model", "revision": "revision", "rank": 0},),
        )
        self.assertEqual(actual.content_checksum_groups, (({"rank": 0},),))

    def test_target_rejects_overlap_after_fetching_source_manifest(self):
        source = SourceWeightsManifest(
            node_id="shared-node",
            parallel_layout=WeightParallelLayout(1),
            gpu_ids=(0,),
            runtime_inventories=({"rank": 0},),
            content_checksum_groups=(({"rank": 0},),),
        )

        def all_gather(output, _local):
            output[:] = [0]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_source_weights_manifest",
                return_value=source,
            ) as fetch,
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.get_local_ip_auto",
                return_value="shared-node",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source"
            ) as pull,
        ):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError, "overlapping ranks"
            ):
                transfer_weights_from_source_daemons(
                    target_manifest_state=MagicMock(),
                    model=MagicMock(),
                    gpu_id=0,
                    registry_url="tcp://source:31999",
                )

        fetch.assert_called_once_with("tcp://source:31999")
        pull.assert_not_called()

    def test_target_allows_same_gpu_id_on_another_node(self):
        source = SourceWeightsManifest(
            node_id="source-node",
            parallel_layout=WeightParallelLayout(1),
            gpu_ids=(0,),
            runtime_inventories=({"rank": 0},),
            content_checksum_groups=(({"rank": 0},),),
        )

        def all_gather(output, _local):
            output[:] = [0]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_source_weights_manifest",
                return_value=source,
            ),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.get_local_ip_auto",
                return_value="target-node",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source",
                return_value={},
            ) as pull,
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.rebuild_transferred_weight_state"
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.validate_transferred_weights",
                return_value={},
            ),
        ):
            transfer_weights_from_source_daemons(
                target_manifest_state=MagicMock(),
                model=MagicMock(),
                gpu_id=0,
                registry_url="tcp://source:31999",
            )

        pull.assert_called_once()

    def test_heterogeneous_socket_paths_use_physical_gpu_rank(self):
        daemon = WeightCacheDaemon(
            server_args=_daemon_server_args(tp_size=2),
            gpu_id=4,
            tp_rank=0,
            pp_rank=0,
            daemon_args=WeightCacheDaemonArgs(
                enable_weight_heterogeneous_copy=True,
                weight_heterogeneous_transfer_registry_url="tcp://source:31999",
                weight_cache_socket_rank=4,
            ),
        )
        self.assertEqual(daemon.socket_path, get_socket_path(4))
        self.assertEqual(daemon.ready_path, get_ready_path(4))

    @patch(
        "sglang.srt.model_executor.model_runner_components.load_model_utils.get_model",
        return_value=SimpleNamespace(weight_cache_mode="client"),
    )
    def test_client_uses_physical_socket_without_heterogeneous_server_args(self, _):
        client_load_config = SimpleNamespace(
            load_format="auto", weight_cache_socket=None
        )
        maybe_enable_ipc_weight_cache(
            load_config=client_load_config,
            server_args=SimpleNamespace(weight_cache_mode="client"),
            tp_size=2,
            pp_rank=0,
            tp_rank=0,
            gpu_id=4,
        )
        self.assertEqual(client_load_config.weight_cache_socket, get_socket_path(4))

        daemon_load_config = SimpleNamespace(
            load_format="auto", weight_cache_socket=None
        )
        maybe_enable_ipc_weight_cache(
            load_config=daemon_load_config,
            server_args=SimpleNamespace(weight_cache_mode="daemon"),
            tp_size=2,
            pp_rank=0,
            tp_rank=0,
            gpu_id=4,
        )
        self.assertEqual(daemon_load_config.weight_cache_socket, get_socket_path(0))

    def test_source_and_target_gates_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            WeightCacheDaemon(
                server_args=_daemon_server_args(),
                gpu_id=0,
                tp_rank=0,
                pp_rank=0,
                daemon_args=WeightCacheDaemonArgs(
                    enable_weight_heterogeneous_copy=True,
                    weight_heterogeneous_copy=True,
                    weight_heterogeneous_transfer_registry_url="tcp://source:31999",
                ),
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            launch_weight_cache_daemons(
                _daemon_server_args(),
                daemon_args=WeightCacheDaemonArgs(
                    enable_weight_heterogeneous_copy=True,
                    weight_heterogeneous_copy=True,
                ),
            )

    def test_daemon_socket_does_not_serve_weight_manifests(self):
        daemon = object.__new__(WeightCacheDaemon)
        responses = []
        with (
            patch(
                "sglang.srt.weight_cache.daemon.recv_msg",
                return_value={"type": "query_weight_manifest"},
            ),
            patch(
                "sglang.srt.weight_cache.daemon.send_msg",
                side_effect=lambda _conn, response: responses.append(response),
            ),
        ):
            daemon._handle_connection(MagicMock())

        self.assertEqual(
            responses,
            [
                {
                    "status": "error",
                    "message": "Unknown request type: query_weight_manifest",
                }
            ],
        )

    def test_target_launcher_passes_manifest_registry_url(self):
        process = MagicMock(pid=123, exitcode=0)
        process.is_alive.return_value = False

        with (
            patch(
                "sglang.srt.weight_cache.daemon.cleanup_stale_daemon_files"
            ) as cleanup,
            patch("sglang.srt.weight_cache.daemon.os.path.exists", return_value=True),
            patch(
                "sglang.srt.weight_cache.daemon.spawn_weight_cache_daemon",
                return_value=process,
            ) as spawn,
        ):
            with self.assertRaisesRegex(RuntimeError, "exited with code 0"):
                launch_weight_cache_daemons(
                    _daemon_server_args(base_gpu_id=4),
                    timeout=1,
                    daemon_args=WeightCacheDaemonArgs(
                        weight_heterogeneous_copy=True,
                        weight_heterogeneous_transfer_source_ip="source",
                        weight_heterogeneous_transfer_source_port=31999,
                    ),
                )

        cleanup.assert_called_once_with(4, force=False)
        spawn.assert_called_once()
        call = spawn.call_args
        self.assertEqual(call.kwargs["socket_rank"], 4)
        self.assertEqual(call.kwargs["gpu_id"], 4)
        child_args = call.kwargs["daemon_args"]
        self.assertTrue(child_args.weight_heterogeneous_copy)
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_registry_url,
            "tcp://source:31999",
        )


if __name__ == "__main__":
    unittest.main()

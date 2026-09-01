# SPDX-License-Identifier: Apache-2.0

import argparse
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from sglang.srt.weight_cache.daemon import (
    WeightCacheDaemon,
    WeightCacheDaemonArgs,
    _prepare_weight_heterogeneous_transfer,
    launch_weight_cache_daemons,
)
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.srt.weight_cache.mooncake_weight_adapter import (
    _parallel_axes,
    build_mooncake_placement_and_bindings,
)
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
    SourceDaemonGroup,
    WeightHeterogeneousTransferError,
    WeightParallelLayout,
    _all_gather_rank_local_phase,
    _initialize_weight_transfer_engine,
    fetch_weight_manifest,
    parse_source_daemon_group,
    register_weight_manifest,
    transfer_weights_from_source_daemons,
    validate_weight_heterogeneous_transfer_configuration,
)
from sglang.srt.weight_cache.weight_load_recorder import (
    WeightLoadRecorder,
    WeightLoadRecordingError,
    _names_an_expert_slot,
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
    def test_daemon_exports_exact_plain_derived_tensor_for_ipc_client(self):
        class DerivedLayer(torch.nn.Module):
            def __init__(self, derived):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.w_kc = derived
                self.w_scale = 1.0
                self.use_deep_gemm_bmm = derived is not None

        class Model(torch.nn.Module):
            def __init__(self, derived):
                super().__init__()
                self.layer = DerivedLayer(derived)

        derived = torch.arange(24).reshape(2, 3, 4).transpose(1, 2)
        daemon = object.__new__(WeightCacheDaemon)
        daemon.model = Model(derived)
        daemon.gpu_id = 0
        daemon.state_entries = {}

        captured = {}
        backend = MagicMock()
        backend.name = "test"

        def prepare_export(state_tensors):
            captured.update(state_tensors)
            return {
                name: {"handle": b"", "is_param": is_param}
                for name, (_, is_param) in state_tensors.items()
            }

        backend.prepare_export.side_effect = prepare_export
        with patch(
            "sglang.srt.weight_cache.daemon.choose_daemon_transport_backend",
            return_value=backend,
        ):
            daemon._export_state()

        exported, is_param = captured["layer.w_kc"]
        self.assertFalse(is_param)
        self.assertEqual(exported.data_ptr(), derived.data_ptr())
        self.assertEqual(exported.stride(), derived.stride())
        self.assertTrue(daemon.state_entries["layer.w_kc"]["is_plain_tensor"])
        self.assertNotIn("layer.w_scale", captured)
        self.assertEqual(
            daemon.state_entries["layer.w_scale"]["value"], 1.0
        )
        self.assertTrue(
            daemon.state_entries["layer.use_deep_gemm_bmm"]["value"]
        )

        client = Model(None)
        imported = torch.full(derived.shape, 7, dtype=derived.dtype)
        IpcModelLoader._set_plain_module_tensor(
            client, "layer.w_kc", imported
        )
        self.assertIs(client.layer.w_kc, imported)
        self.assertNotIn("w_kc", client.layer._parameters)
        self.assertNotIn("w_kc", client.layer._buffers)
        imported_scale = torch.tensor(3.0)
        IpcModelLoader._set_plain_module_tensor(
            client, "layer.w_scale", imported_scale
        )
        self.assertIs(client.layer.w_scale, imported_scale)
        IpcModelLoader._set_plain_module_value(
            client, "layer.use_deep_gemm_bmm", True
        )
        self.assertTrue(client.layer.use_deep_gemm_bmm)

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
        placement, bindings = build_mooncake_placement_and_bindings(
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

    def test_recorder_records_scalar_broadcast_fill(self):
        # default_weight_loader broadcasts single-element weights with
        # fill_(loaded_weight.item()), which carries no tensor provenance.
        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.empty(1))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.scale.data.fill_(loaded_weight.item())

        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            ScalarModel(),
            (("scale", torch.tensor([2.5])),),
            execute_writes=True,
        )
        plan = recorder.build_plan()
        self.assertEqual(len(plan.views), 1)
        self.assertEqual(plan.views[0].tensor_id, "scale")

    def test_recorder_tracks_transpose_shorthand(self):
        # torch exposes Tensor.t() as aten::t rather than decomposing it.
        class TransposeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(3, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight.t().contiguous())

        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            TransposeModel(),
            (("weight", torch.empty(2, 3)),),
            execute_writes=True,
        )
        plan = recorder.build_plan()
        self.assertIn("permute(1,0)", plan.views[0].layout_fingerprint)

    def test_recorder_allows_value_only_in_place_operation(self):
        # Reshard records where bytes live, so a mutation that leaves geometry
        # untouched cannot invalidate a recorded position.
        class ScaleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight.mul_(2.0))

        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            ScaleModel(),
            (("weight", torch.ones(2, 2)),),
            execute_writes=True,
        )
        self.assertEqual(len(recorder.build_plan().views), 1)

    def test_recorder_rejects_geometry_changing_in_place_operation(self):
        class TransposeInPlaceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(4))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    loaded_weight.transpose_(0, 1)
                    self.weight.data.copy_(loaded_weight.reshape(-1))

        with self.assertRaisesRegex(
            WeightLoadRecordingError, "geometry-changing in-place operation"
        ):
            WeightLoadRecorder().record_model_load(
                TransposeInPlaceModel(),
                (("weight", torch.empty(2, 2)),),
                execute_writes=True,
            )

    def test_recorder_records_dtype_cast_as_layout_op(self):
        class CastModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty(2, 2, dtype=torch.float16)
                )

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight.to(torch.float16))

        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            CastModel(),
            (("weight", torch.empty(2, 2, dtype=torch.float32)),),
            execute_writes=True,
        )
        self.assertIn(
            "cast(float32->float16)",
            recorder.build_plan().views[0].layout_fingerprint,
        )

    def test_expert_slot_matching_ignores_layer_index(self):
        for tensor_id, expert_id, expected in (
            ("layers.3.mlp.experts.7.w1.weight", 3, False),
            ("layers.3.mlp.experts.7.w1.weight", 7, True),
            ("layers.7.mlp.experts.7.w1.weight", 7, True),
            ("layers.3.mlp.shared_expert.gate_proj.weight", 4, False),
            ("experts.0.w1.weight", 0, True),
        ):
            with self.subTest(tensor_id=tensor_id, expert_id=expert_id):
                self.assertEqual(
                    _names_an_expert_slot(tensor_id, expert_id), expected
                )

    def test_recorder_rejects_index_after_shape_changing_view(self):
        class ReshapeSelectModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(3, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    selected = loaded_weight.reshape(3, 2, 4).select(2, 0)
                    self.weight.data.copy_(selected)

        with self.assertRaisesRegex(
            WeightLoadRecordingError, "select after shape-changing view"
        ):
            WeightLoadRecorder().record_model_load(
                ReshapeSelectModel(),
                (("weight", torch.empty(2, 3, 4)),),
                execute_writes=False,
            )

    def test_runtime_manifest_includes_checkpoint_loaded_buffer(self):
        class BufferModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.register_buffer("weight_scale", torch.zeros(2))

            def load_weights(self, weights):
                destinations = {
                    "weight": self.weight.data,
                    "weight_scale": self.weight_scale,
                }
                for name, loaded_weight in weights:
                    destinations[name].copy_(loaded_weight)

        source = BufferModel()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source,
            (
                ("weight", torch.ones(2)),
                ("weight_scale", torch.full((2,), 2.0)),
            ),
            execute_writes=True,
        )
        source_plan = recorder.build_plan()

        target = BufferModel()
        target_plan = record_target_weight_load_plan(
            target, source_plan.logical_weights
        )
        self.assertTrue(torch.equal(target.weight_scale, torch.zeros(2)))

        def build_manifest(model, plan, worker):
            return ImmutableWeightRuntimeManifestBuilder(
                model=model,
                load_plan=plan,
                topology=WeightParallelTopology(),
                allowed_devices=("cpu",),
            ).build(
                model_id="model",
                revision="revision",
                instance_id=worker,
                worker_id=worker,
                endpoint="127.0.0.1:1",
            )

        for manifest in (
            build_manifest(source, source_plan, "source"),
            build_manifest(target, target_plan, "target"),
        ):
            self.assertEqual(
                {tensor.runtime_name for tensor in manifest.tensors},
                {"weight", "weight_scale"},
            )

        source.weight_scale = source.weight_scale.clone()
        with self.assertRaisesRegex(
            WeightManifestError, "recorded tensor storage changed"
        ):
            build_manifest(
                source,
                source_plan,
                "replaced-buffer",
            )

    def test_plan_rejects_parameter_not_covered_by_native_loader(self):
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
        # build_plan owns this guarantee so an uncovered parameter cannot reach
        # any consumer; the manifest builder repeats the check as a second net.
        with self.assertRaisesRegex(
            WeightLoadRecordingError,
            "no recorded write covers parameter: unused",
        ):
            recorder.build_plan()

    def test_weight_parallel_layout_supports_ep_and_rejects_unsupported_dp(self):
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=2,
            dp_size=2,
            ep_size=1,
            pp_size=2,
            enable_dp_attention=True,
            quantization=None,
        )
        parallel_layout = WeightParallelLayout(tp_size=2, dp_size=2, pp_size=2)
        self.assertEqual(parallel_layout.world_size, 4)
        self.assertEqual(parallel_layout.dp_size, 2)
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "attention-DP partial replication"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=8,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=True,
                quantization=None,
            )
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=4,
            dp_size=1,
            ep_size=2,
            pp_size=1,
            enable_dp_attention=False,
            quantization=None,
        )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "moe_dp_size=2"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=4,
                dp_size=1,
                ep_size=2,
                pp_size=1,
                enable_dp_attention=False,
                moe_dp_size=2,
                quantization=None,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "attn_cp_size=2"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=4,
                dp_size=1,
                ep_size=2,
                pp_size=1,
                enable_dp_attention=False,
                attn_cp_size=2,
                quantization=None,
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

    def test_parallel_axes_use_inferred_process_semantics(self):
        from mooncake.reshard.weight import ReplicatedAxis, SplitAxis

        tensor = {"tensor_id": "tensor", "expert_id": None}
        self.assertEqual(
            _parallel_axes(
                tensor,
                tp_size=2,
                pp_size=1,
                ep_size=1,
                tp_split_dims=(2,),
            ),
            (SplitAxis("tp", dim=2),),
        )
        self.assertEqual(
            _parallel_axes(
                tensor,
                tp_size=2,
                pp_size=1,
                ep_size=1,
                tp_replicated=True,
            ),
            (ReplicatedAxis("tp"),),
        )

    def test_internal_fragments_replicated_by_dp_attention_are_not_tp_shards(self):
        from mooncake.reshard.weight import ReplicatedAxis

        manifests = []
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
            manifests.append(
                {
                    "model_id": "model",
                    "revision": "revision",
                    "instance_id": f"worker-{tp_rank}",
                    "generation": IMMUTABLE_WEIGHT_GENERATION,
                    "tensors": tensors,
                }
            )

        placement, _ = build_mooncake_placement_and_bindings(
            manifests,
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

    def test_ep_moe_tp_manifest_reshards_to_different_ep_layout(self):
        from math import prod

        from mooncake.reshard.weight import (
            OwnershipAxis,
            ReplicatedAxis,
            SplitAxis,
            plan_placement_transfer,
        )

        def contiguous_stride(shape):
            stride = 1
            result = []
            for extent in reversed(shape):
                result.append(stride)
                stride *= extent
            return tuple(reversed(result))

        def make_manifests(*, ep_size, moe_tp_size, prefix, address_base):
            manifests = []
            num_experts = 4
            experts_per_rank = num_experts // ep_size
            for ep_rank in range(ep_size):
                for tp_rank in range(moe_tp_size):
                    tensors = []

                    def add_tensor(
                        tensor_id,
                        global_shape,
                        global_offset,
                        local_shape,
                        shard_dims,
                        *,
                        expert_id=None,
                    ):
                        tensor_index = len(tensors)
                        address = (
                            address_base
                            + (ep_rank * moe_tp_size + tp_rank) * 0x100000
                            + tensor_index * 0x10000
                        )
                        tensors.append(
                            {
                                "fragment_id": (
                                    f"{prefix}:e{ep_rank}:t{tp_rank}:"
                                    f"{tensor_index}:{tensor_id}"
                                ),
                                "tensor_id": tensor_id,
                                "runtime_name": tensor_id,
                                "global_shape": global_shape,
                                "global_offset": global_offset,
                                "local_shape": local_shape,
                                "dtype": "bfloat16",
                                "itemsize": 2,
                                "shard_dims": shard_dims,
                                "layer_id": 0,
                                "expert_id": expert_id,
                                "layout_fingerprint": "ep-moe-tp-test",
                                "address": address,
                                "nbytes": prod(local_shape) * 2,
                                "byte_offset": 0,
                                "stride": contiguous_stride(local_shape),
                                "storage_offset": 0,
                                "device": f"cuda:{ep_rank * moe_tp_size + tp_rank}",
                                "worker_id": f"{prefix}-e{ep_rank}-t{tp_rank}",
                                "endpoint": f"127.0.0.1:{2000 + ep_rank * moe_tp_size + tp_rank}",
                                "rank": {
                                    "dp": 0,
                                    "tp": tp_rank,
                                    "pp": 0,
                                    "ep": ep_rank,
                                },
                            }
                        )

                    global_tp_rank = ep_rank * moe_tp_size + tp_rank
                    global_tp_size = ep_size * moe_tp_size
                    for internal_part in range(2):
                        add_tensor(
                            "model.layers.0.dense.weight",
                            (4, 8),
                            (
                                internal_part * 2,
                                global_tp_rank * (8 // global_tp_size),
                            ),
                            (2, 8 // global_tp_size),
                            (0, 1),
                        )
                    add_tensor(
                        "model.layers.0.experts.weight",
                        (num_experts, 8),
                        (ep_rank * experts_per_rank, tp_rank * (8 // moe_tp_size)),
                        (experts_per_rank, 8 // moe_tp_size),
                        tuple(dim for dim, size in enumerate((ep_size, moe_tp_size)) if size > 1),
                    )
                    first_expert = ep_rank * experts_per_rank
                    for expert_id in range(
                        first_expert, first_expert + experts_per_rank
                    ):
                        add_tensor(
                            f"model.layers.0.experts.{expert_id}.weight",
                            (2, 8),
                            (0, tp_rank * (8 // moe_tp_size)),
                            (2, 8 // moe_tp_size),
                            (1,) if moe_tp_size > 1 else (),
                            expert_id=expert_id,
                        )
                        for internal_part in range(2):
                            add_tensor(
                                f"model.layers.0.experts.{expert_id}.replicated",
                                (4, 4),
                                (internal_part * 2, 0),
                                (2, 4),
                                (0,),
                                expert_id=expert_id,
                            )

                    manifests.append(
                        {
                            "model_id": "model",
                            "revision": "revision",
                            "instance_id": f"{prefix}-e{ep_rank}-t{tp_rank}",
                            "generation": IMMUTABLE_WEIGHT_GENERATION,
                            "tensors": tensors,
                        }
                    )
            return tuple(manifests)

        source_manifests = make_manifests(
            ep_size=2,
            moe_tp_size=2,
            prefix="source",
            address_base=0x1000000,
        )
        target_manifests = make_manifests(
            ep_size=4,
            moe_tp_size=1,
            prefix="target",
            address_base=0x2000000,
        )
        source, _ = build_mooncake_placement_and_bindings(
            source_manifests,
            placement_set_id="source",
            tp_size=4,
            pp_size=1,
            ep_size=2,
        )
        target, _ = build_mooncake_placement_and_bindings(
            target_manifests,
            placement_set_id="target",
            tp_size=4,
            pp_size=1,
            ep_size=4,
        )

        descriptors = {
            tensor.tensor_id: tensor
            for part in source.parts
            for tensor in part.tensors
        }
        self.assertEqual(
            descriptors["model.layers.0.dense.weight"].parallel_axes,
            (SplitAxis("ep", dim=1), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors["model.layers.0.experts.weight"].parallel_axes,
            (SplitAxis("ep", dim=0), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors["model.layers.0.experts.0.weight"].parallel_axes,
            (OwnershipAxis("ep"), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors[
                "model.layers.0.experts.0.replicated"
            ].parallel_axes,
            (OwnershipAxis("ep"), ReplicatedAxis("tp")),
        )
        logical_plan = plan_placement_transfer(source, target)
        self.assertTrue(logical_plan.operations)
        reverse_plan = plan_placement_transfer(target, source)
        self.assertTrue(reverse_plan.operations)

    def test_manifest_server_aggregates_source_ranks(self):
        server = object.__new__(WeightManifestServer)
        server.expected_rank_count = 2
        server._node_id = None
        server._parallel_layout = None
        server._model_identity = None
        server._rank_manifests = {}
        server._lock = threading.Lock()

        for global_rank, device_uuid in ((0, "GPU-2"), (1, "GPU-3")):
            server.register_weight_manifest(
                {
                    "node_id": "source-node",
                    "global_rank": global_rank,
                    "device_uuid": device_uuid,
                    "parallel_layout": {
                        "tp_size": 2,
                        "dp_size": 2,
                        "pp_size": 1,
                        "ep_size": 1,
                    },
                    "runtime_manifest": {
                        "model_id": "model",
                        "revision": "revision",
                        "rank": global_rank,
                    },
                }
            )

        source = parse_source_daemon_group(server.get_weight_manifest())
        self.assertEqual(source.device_uuids, ("GPU-2", "GPU-3"))
        self.assertEqual(source.parallel_layout.dp_size, 2)
        self.assertEqual(
            tuple(item["rank"] for item in source.runtime_manifests), (0, 1)
        )

    def test_tcp_manifest_registry_round_trip(self):
        server = WeightManifestServer(
            host="127.0.0.1", port=0, expected_rank_count=1
        )
        registry_url = f"tcp://127.0.0.1:{server.port}"
        session = MagicMock()
        session.parallel_layout = WeightParallelLayout(tp_size=1)
        session.runtime_manifest_for_wire.return_value = {
            "model_id": "model",
            "revision": "revision",
            "rank": 0,
        }
        try:
            register_weight_manifest(
                registry_url,
                global_rank=0,
                device_uuid="GPU-2",
                session=session,
                timeout=2,
            )
            actual = fetch_weight_manifest(registry_url, timeout=2)
        finally:
            server.close()

        self.assertEqual(actual.device_uuids, ("GPU-2",))
        self.assertEqual(actual.parallel_layout, WeightParallelLayout(tp_size=1))
        self.assertEqual(
            actual.runtime_manifests,
            ({"model_id": "model", "revision": "revision", "rank": 0},),
        )

    def test_target_rejects_overlapping_device_uuid(self):
        source = SourceDaemonGroup(
            parallel_layout=WeightParallelLayout(1),
            device_uuids=("GPU-shared",),
            runtime_manifests=({"rank": 0},),
        )

        def all_gather(output, local):
            output[:] = [local]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_weight_manifest",
                return_value=source,
            ) as fetch,
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source"
            ) as pull,
        ):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError,
                "overlapping device UUIDs",
            ):
                transfer_weights_from_source_daemons(
                    target_session=MagicMock(),
                    model=MagicMock(),
                    device_uuid="GPU-shared",
                    registry_url="tcp://source:31999",
                )

        fetch.assert_called_once_with("tcp://source:31999")
        pull.assert_not_called()

    def test_target_allows_distinct_device_uuid(self):
        source = SourceDaemonGroup(
            parallel_layout=WeightParallelLayout(1),
            device_uuids=("GPU-source",),
            runtime_manifests=({"rank": 0},),
        )

        def all_gather(output, local):
            output[:] = [local]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_weight_manifest",
                return_value=source,
            ),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source",
                return_value={},
            ) as pull,
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.rebuild_transferred_weight_state"
            ),
        ):
            transfer_weights_from_source_daemons(
                target_session=MagicMock(),
                model=MagicMock(),
                device_uuid="GPU-target",
                registry_url="tcp://source:31999",
            )

        pull.assert_called_once()

    def test_rank_local_phase_propagates_failure_to_every_rank(self):
        def all_gather(output, local):
            output[:] = [local]

        def fail():
            raise RuntimeError("rank-local boom")

        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
        ):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError,
                "pull failed on target rank.*rank 0.*rank-local boom",
            ):
                _all_gather_rank_local_phase("pull", fail)

    @patch(
        "sglang.srt.weight_cache.daemon.current_platform.get_device_uuid",
        return_value="GPU-test-4",
    )
    def test_socket_paths_use_physical_device_uuid(self, get_device_uuid):
        for mode in ("source", "target", None):
            with self.subTest(mode=mode):
                daemon_args = WeightCacheDaemonArgs()
                if mode is not None:
                    daemon_args = WeightCacheDaemonArgs(
                        weight_heterogeneous_transfer_mode=mode,
                        weight_heterogeneous_transfer_host="host",
                        weight_heterogeneous_transfer_registry_url=(
                            "tcp://host:31999"
                        ),
                    )
                daemon = WeightCacheDaemon(
                    server_args=_daemon_server_args(tp_size=2),
                    gpu_id=4,
                    tp_rank=0,
                    pp_rank=0,
                    daemon_args=daemon_args,
                )
                self.assertEqual(
                    daemon.socket_path, get_socket_path("GPU-test-4")
                )
                self.assertEqual(
                    daemon.ready_path, get_ready_path("GPU-test-4")
                )

        self.assertEqual(get_device_uuid.call_count, 3)
        get_device_uuid.assert_any_call(4)

    def test_weight_heterogeneous_transfer_mode_contract(self):
        disabled = WeightCacheDaemonArgs()
        source = WeightCacheDaemonArgs(
            weight_heterogeneous_transfer_mode="source",
            weight_heterogeneous_transfer_host="0.0.0.0",
        )
        target = WeightCacheDaemonArgs(
            weight_heterogeneous_transfer_mode="target",
            weight_heterogeneous_transfer_host="source",
        )
        self.assertIsNone(disabled.weight_heterogeneous_transfer_mode)
        self.assertEqual(source.weight_heterogeneous_transfer_mode, "source")
        self.assertEqual(target.weight_heterogeneous_transfer_mode, "target")

        invalid_cases = (
            (
                {
                    "weight_heterogeneous_transfer_mode": "invalid",
                    "weight_heterogeneous_transfer_host": "host",
                },
                "mode must be",
            ),
            (
                {
                    "weight_heterogeneous_transfer_host": "host",
                },
                "require an active mode",
            ),
            (
                {
                    "weight_heterogeneous_transfer_mode": "source",
                },
                "requires a host",
            ),
            (
                {
                    "weight_heterogeneous_transfer_mode": "target",
                    "weight_heterogeneous_transfer_host": "",
                },
                "requires a host",
            ),
            (
                {
                    "weight_heterogeneous_transfer_mode": "target",
                    "weight_heterogeneous_transfer_host": "source",
                    "weight_heterogeneous_transfer_port": 0,
                },
                "port must be positive",
            ),
        )
        for kwargs, error in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, error):
                    WeightCacheDaemonArgs(**kwargs)

    def test_weight_heterogeneous_transfer_cli_args(self):
        parser = argparse.ArgumentParser()
        WeightCacheDaemonArgs.add_cli_args(parser)

        disabled = WeightCacheDaemonArgs.from_cli_args(parser.parse_args([]))
        source = WeightCacheDaemonArgs.from_cli_args(
            parser.parse_args(
                [
                    "--weight-heterogeneous-transfer-mode",
                    "source",
                    "--weight-heterogeneous-transfer-host",
                    "0.0.0.0",
                ]
            )
        )
        target = WeightCacheDaemonArgs.from_cli_args(
            parser.parse_args(
                [
                    "--weight-heterogeneous-transfer-mode",
                    "target",
                    "--weight-heterogeneous-transfer-host",
                    "source",
                    "--weight-heterogeneous-transfer-port",
                    "31999",
                ]
            )
        )
        self.assertIsNone(disabled.weight_heterogeneous_transfer_mode)
        self.assertEqual(source.weight_heterogeneous_transfer_mode, "source")
        self.assertEqual(target.weight_heterogeneous_transfer_mode, "target")

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

    def test_source_registry_url_uses_actual_bind_host_and_port(self):
        cases = (
            ("0.0.0.0", "tcp://127.0.0.1:43210"),
            ("::", "tcp://[::1]:43210"),
            ("10.0.0.8", "tcp://10.0.0.8:43210"),
            ("::1", "tcp://[::1]:43210"),
        )
        for bind_host, expected_url in cases:
            with self.subTest(bind_host=bind_host):
                manifest_server = MagicMock()
                manifest_server.port = 43210
                with patch(
                    "sglang.srt.weight_cache.weight_manifest_server.WeightManifestServer",
                    return_value=manifest_server,
                ):
                    actual_server, registry_url = (
                        _prepare_weight_heterogeneous_transfer(
                            _daemon_server_args(),
                            WeightCacheDaemonArgs(
                                weight_heterogeneous_transfer_mode="source",
                                weight_heterogeneous_transfer_host=bind_host,
                                weight_heterogeneous_transfer_port=31999,
                            ),
                            expected_rank_count=1,
                        )
                    )
                self.assertIs(actual_server, manifest_server)
                self.assertEqual(registry_url, expected_url)

    def test_resolved_registry_url_does_not_change_mode(self):
        for mode in ("source", "target"):
            with self.subTest(mode=mode):
                daemon_args = WeightCacheDaemonArgs(
                    weight_heterogeneous_transfer_mode=mode,
                    weight_heterogeneous_transfer_host="host",
                    weight_heterogeneous_transfer_registry_url=(
                        "tcp://resolved:31999"
                    ),
                )
                manifest_server, registry_url = (
                    _prepare_weight_heterogeneous_transfer(
                        _daemon_server_args(),
                        daemon_args,
                        expected_rank_count=1,
                    )
                )
                self.assertIsNone(manifest_server)
                self.assertEqual(registry_url, "tcp://resolved:31999")
                self.assertEqual(
                    daemon_args.weight_heterogeneous_transfer_mode,
                    mode,
                )

    def test_launcher_cleans_up_after_partial_spawn_failure(self):
        manifest_server = MagicMock()
        process = MagicMock(pid=123)
        process.is_alive.side_effect = (True, True)

        with (
            patch(
                "sglang.srt.weight_cache.daemon._prepare_weight_heterogeneous_transfer",
                return_value=(manifest_server, "tcp://source:31999"),
            ),
            patch("sglang.srt.weight_cache.daemon.cleanup_stale_daemon_files"),
            patch(
                "sglang.srt.weight_cache.daemon.spawn_weight_cache_daemon",
                side_effect=(process, RuntimeError("second spawn failed")),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "second spawn failed"):
                launch_weight_cache_daemons(
                    _daemon_server_args(tp_size=2),
                    dist_init_method="tcp://127.0.0.1:12345",
                    daemon_args=WeightCacheDaemonArgs(
                        weight_heterogeneous_transfer_mode="source",
                        weight_heterogeneous_transfer_host="0.0.0.0",
                    ),
                )

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.join.call_args_list,
            [call(timeout=5), call()],
        )
        manifest_server.close.assert_called_once_with()

    def test_target_launcher_passes_manifest_registry_url(self):
        process = MagicMock(pid=123, exitcode=0)
        process.is_alive.return_value = False

        with (
            patch(
                "sglang.srt.weight_cache.daemon.cleanup_stale_daemon_files"
            ) as cleanup,
            patch(
                "sglang.srt.weight_cache.daemon.current_platform.get_device_uuid",
                return_value="GPU-test-4",
            ),
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
                        weight_heterogeneous_transfer_mode="target",
                        weight_heterogeneous_transfer_host="source",
                        weight_heterogeneous_transfer_port=31999,
                    ),
                )

        cleanup.assert_called_once_with("GPU-test-4", force=False)
        spawn.assert_called_once()
        call = spawn.call_args
        self.assertEqual(call.kwargs["gpu_id"], 4)
        child_args = call.kwargs["daemon_args"]
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_mode,
            "target",
        )
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_host,
            "source",
        )
        self.assertEqual(child_args.weight_heterogeneous_transfer_port, 31999)
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_registry_url,
            "tcp://source:31999",
        )


if __name__ == "__main__":
    unittest.main()

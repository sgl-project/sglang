# SPDX-License-Identifier: Apache-2.0
"""Unit and contract tests for weight-cache heterogeneous transfer.

Cover recorder placement, manifests, Mooncake planning, the TCP registry, and
daemon/IPC coordination using small Torch modules and mocked transfer calls.
"""

import unittest

import torch

from sglang.srt.weight_cache.weight_load_recorder import (
    WeightLoadRecorder,
    WeightLoadRecordingError,
    _names_an_expert_slot,
    capture_weight_load_plan,
    record_target_weight_load_plan,
)
from sglang.srt.weight_cache.weight_runtime_manifest import (
    ImmutableWeightRuntimeManifestBuilder,
    WeightManifestError,
    WeightParallelTopology,
    model_identity_from_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestWeightLoadRecorder(unittest.TestCase):
    def test_recorder_allows_reloading_scalar_checkpoint_values(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.zeros(1))

            def load_weights(self, weights):
                for _, loaded in weights:
                    self.scale.data.copy_(loaded)

        source = Model()
        recorder = WeightLoadRecorder()
        for value in (2.0, 3.0):
            recorder.record_model_load(
                source, (("scale", torch.tensor([value])),), execute_writes=True
            )
        plan = recorder.build_plan()
        self.assertEqual(source.scale.item(), 3.0)
        self.assertEqual(plan.logical_weights[0].scalar_value, 3.0)
        target = Model()
        replay = record_target_weight_load_plan(target, plan.logical_weights)
        self.assertEqual(len(replay.views), 1)
        self.assertEqual(target.scale.item(), 0.0)

    def test_recorder_negative_slice_bounds_match_checkpoint_bytes(self):
        class SliceModel(torch.nn.Module):
            def __init__(self, selection, concatenate):
                super().__init__()
                self.selection = selection
                self.concatenate = concatenate
                self.weight = torch.nn.Parameter(
                    torch.zeros_like(torch.empty(6, 2)[selection])
                )

            def load_weights(self, weights):
                parts = dict(weights)
                loaded = (
                    torch.cat((parts["a"], parts["b"]))
                    if self.concatenate
                    else parts["weight"]
                )
                self.weight.data.copy_(loaded[self.selection])

        full = torch.arange(12).reshape(6, 2).float()
        for concatenate in (False, True):
            weights = (
                {"a": full[:3], "b": full[3:]} if concatenate else {"weight": full}
            )
            for selection in (slice(-2, None), slice(None, -2), slice(-4, -1)):
                with self.subTest(concatenate=concatenate, selection=selection):
                    source = SliceModel(selection, concatenate)
                    recorder = WeightLoadRecorder()
                    recorder.record_model_load(
                        source, weights.items(), execute_writes=True
                    )
                    source_plan = recorder.build_plan()
                    reconstructed = torch.empty_like(source.weight).flatten()
                    for view in source_plan.views:
                        box = tuple(
                            slice(offset, offset + extent)
                            for offset, extent in zip(
                                view.global_offset, view.local_shape
                            )
                        )
                        values = weights[view.tensor_id][box].flatten()
                        begin = view.byte_offset // source.weight.element_size()
                        reconstructed[begin : begin + values.numel()].copy_(values)
                    self.assertTrue(
                        torch.equal(
                            reconstructed.reshape_as(source.weight), full[selection]
                        )
                    )
                    target = SliceModel(selection, concatenate)
                    target_plan = record_target_weight_load_plan(
                        target, source_plan.logical_weights
                    )
                    self.assertTrue(
                        torch.equal(target.weight, torch.zeros_like(target.weight))
                    )
                    self.assertEqual(
                        [
                            (v.tensor_id, v.global_offset, v.local_shape, v.byte_offset)
                            for v in target_plan.views
                        ],
                        [
                            (v.tensor_id, v.global_offset, v.local_shape, v.byte_offset)
                            for v in source_plan.views
                        ],
                    )

    def test_recorder_covers_distinct_parameters_sharing_storage(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))
                self.alias = torch.nn.Parameter(self.weight.data)

            def load_weights(self, weights):
                for _, loaded in weights:
                    self.weight.data.copy_(loaded)

        source = Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source, (("weight", torch.ones(2, 2)),), execute_writes=True
        )
        plan = recorder.build_plan()
        self.assertEqual(len(plan.views), 1)
        self.assertEqual(plan.views[0].parameter_names, ("alias", "weight"))
        target = Model()
        target_plan = record_target_weight_load_plan(target, plan.logical_weights)
        self.assertEqual(len(target_plan.views), 1)
        self.assertTrue(torch.equal(target.alias, torch.zeros(2, 2)))
        manifest = ImmutableWeightRuntimeManifestBuilder(
            model=source,
            load_plan=plan,
            topology=WeightParallelTopology(),
            allowed_devices=("cpu",),
        ).build(
            model_id="alias-model",
            revision="test",
            instance_id="source",
            worker_id="source",
            endpoint="127.0.0.1:1",
        )
        self.assertEqual(len(manifest.tensors), 1)
        source.alias.data = source.alias.data.clone()
        with self.assertRaisesRegex(WeightLoadRecordingError, "no recorded write"):
            recorder.build_plan()

    def test_recorder_concat_cast_matches_implicit_copy_cast(self):
        class Model(torch.nn.Module):
            def __init__(self, explicit_cast):
                super().__init__()
                self.explicit_cast = explicit_cast
                self.weight = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.float16))

            def load_weights(self, weights):
                parts = dict(weights)
                fused = torch.cat((parts["a"], parts["b"]))
                if self.explicit_cast:
                    fused = fused.to(torch.float16)
                self.weight.data.copy_(fused)

        weights = {"a": torch.full((2, 2), 1.25), "b": torch.full((2, 2), -2.5)}
        signatures = []
        for explicit_cast in (False, True):
            with self.subTest(explicit_cast=explicit_cast):
                source = Model(explicit_cast)
                recorder = WeightLoadRecorder()
                recorder.record_model_load(source, weights.items(), execute_writes=True)
                plan = recorder.build_plan()
                self.assertTrue(
                    torch.equal(
                        source.weight, torch.cat(tuple(weights.values())).half()
                    )
                )
                target = Model(explicit_cast)
                replay = record_target_weight_load_plan(target, plan.logical_weights)
                self.assertTrue(
                    torch.equal(target.weight, torch.zeros_like(target.weight))
                )
                signature = [
                    (v.tensor_id, v.byte_offset, v.layout_fingerprint)
                    for v in plan.views
                ]
                self.assertEqual(
                    signature,
                    [
                        (v.tensor_id, v.byte_offset, v.layout_fingerprint)
                        for v in replay.views
                    ],
                )
                signatures.append(signature)
                manifest = ImmutableWeightRuntimeManifestBuilder(
                    model=source,
                    load_plan=plan,
                    topology=WeightParallelTopology(),
                    allowed_devices=("cpu",),
                ).build(
                    model_id="cast-model",
                    revision="test",
                    instance_id="source",
                    worker_id="source",
                    endpoint="127.0.0.1:1",
                )
                self.assertEqual(sum(t.nbytes for t in manifest.tensors), 16)
                self.assertEqual({t.dtype for t in manifest.tensors}, {"float16"})
        self.assertEqual(*signatures)

    def test_recorder_preserves_empty_slice_copy_as_noop(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(6, 2))

            def load_weights(self, weights):
                for _, loaded in weights:
                    self.weight.data[:0].copy_(loaded[-1:-4])
                    self.weight.data.copy_(loaded)

        source = Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source, (("weight", torch.ones(6, 2)),), execute_writes=True
        )
        plan = recorder.build_plan()
        self.assertEqual(len(plan.views), 1)
        target = Model()
        replay = record_target_weight_load_plan(target, plan.logical_weights)
        self.assertEqual(len(replay.views), 1)
        self.assertTrue(torch.equal(source.weight, torch.ones(6, 2)))
        self.assertTrue(torch.equal(target.weight, torch.zeros(6, 2)))

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
                    self.weight.data.copy_(loaded_weight.narrow(0, self.rank * 6, 6))

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
                    param.data.copy_(loaded_weight.narrow(0, self.rank * shard, shard))

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
                self.weight = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.float16))

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
                self.assertEqual(_names_an_expert_slot(tensor_id, expert_id), expected)

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


if __name__ == "__main__":
    unittest.main()

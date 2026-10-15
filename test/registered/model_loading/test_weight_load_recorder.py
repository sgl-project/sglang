# SPDX-License-Identifier: Apache-2.0
"""Unit and contract tests for weight-cache heterogeneous transfer.

Cover recorder placement, manifests, Mooncake planning, the TCP registry, and
daemon/IPC coordination using small Torch modules and mocked transfer calls.
"""

import unittest
from unittest.mock import patch

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


def _manifest(model, plan, *, worker="source", topology=None, endpoint="127.0.0.1:1"):
    return ImmutableWeightRuntimeManifestBuilder(
        model=model,
        load_plan=plan,
        topology=topology or WeightParallelTopology(),
        allowed_devices=(model.weight.device.type,),
    ).build(
        model_id="model",
        revision="test",
        instance_id=worker,
        worker_id=worker,
        endpoint=endpoint,
    )


def _placements(plan):
    return [
        (
            v.tensor_id,
            v.global_shape,
            v.global_offset,
            v.local_shape,
            v.byte_offset,
            v.layout_fingerprint,
        )
        for v in plan.views
    ]


class TestWeightLoadRecorder(unittest.TestCase):
    def test_manifest_rejects_native_interleaved_swiglu_postprocessing(self):
        from types import SimpleNamespace

        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w13_weight = torch.nn.Parameter(
                    torch.zeros(1, 6, 4, dtype=torch.bfloat16)
                )
                self.quant_method = object.__new__(UnquantizedFusedMoEMethod)
                torch.nn.Module.__init__(self.quant_method)
                self.quant_method.w13_swiglu_interleaved = False
                self.quant_method._aiter_runner = None
                self.quant_method.with_bias = False
                self.quant_method.runner = SimpleNamespace(
                    runner_backend=SimpleNamespace(is_triton=lambda: True)
                )
                self.moe_runner_config = SimpleNamespace(
                    activation="silu",
                    is_gated=True,
                    gemm1_alpha=None,
                    gemm1_clamp_limit=None,
                    swiglu_limit=None,
                    apply_router_weight_on_input=False,
                )

            @property
            def weight(self):
                return self.w13_weight

            def load_weights(self, weights):
                for _, loaded in weights:
                    self.w13_weight.data.copy_(loaded)

        model = Model()
        recorder = WeightLoadRecorder()
        checkpoint = torch.arange(24, dtype=torch.bfloat16).reshape(1, 6, 4)
        recorder.record_model_load(model, (("w13", checkpoint),), execute_writes=True)
        plan = recorder.build_plan()
        _manifest(model, plan)
        address = model.weight.data_ptr()
        module = "sglang.srt.layers.quantization.unquant"
        with (
            patch(f"{module}._is_cuda", True),
            patch(
                f"{module}.envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.get",
                return_value=True,
            ),
            patch(
                f"{module}.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch(
                f"{module}.get_lora",
                return_value=SimpleNamespace(enable_lora=False, lora_paths=None),
            ),
            patch(
                f"{module}.get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
            ),
        ):
            model.quant_method._maybe_interleave_w13_for_fused_swiglu(model)
        self.assertEqual(model.weight.data_ptr(), address)
        self.assertFalse(torch.equal(model.weight, checkpoint))
        with self.assertRaisesRegex(WeightManifestError, "interleaved SwiGLU"):
            _manifest(model, plan)

    def test_constant_initialization_keeps_strict_coverage_and_replay(self):
        class Model(torch.nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full((4,), -1.0))
                self.mode = mode

            def load_weights(self, weights):
                for _, loaded in weights:
                    self.weight.data[:2].copy_(loaded)
                    if self.mode == "gap":
                        self.weight.data[3:].fill_(7)
                    elif self.mode == "overlap":
                        self.weight.data[1:].fill_(7)
                    elif self.mode == "strided":
                        self.weight.data[::2].fill_(7)
                    else:
                        self.weight.data[2:].fill_(7)

        for mode in ("gap", "overlap", "strided", "valid"):
            with self.subTest(mode=mode):
                model = Model(mode)
                recorder = WeightLoadRecorder()
                if mode != "valid":
                    with self.assertRaises(WeightLoadRecordingError):
                        recorder.record_model_load(
                            model, (("w", torch.ones(2)),), execute_writes=True
                        )
                        recorder.build_plan()
                    continue
                recorder.record_model_load(
                    model, (("w", torch.ones(2)),), execute_writes=True
                )
                plan = recorder.build_plan()
                target = Model(mode)
                replay = record_target_weight_load_plan(target, plan.logical_weights)
                self.assertTrue(torch.equal(target.weight, torch.full((4,), -1.0)))
                _manifest(target, replay)
                replay.initialize_local_regions()
                self.assertEqual(target.weight.tolist(), [-1.0, -1.0, 7.0, 7.0])

    def test_existing_plain_concat_scalar_mutation_remains_supported(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))

            def load_weights(self, weights):
                values = dict(weights)
                self.weight.data.copy_(torch.cat((values["a"], values["b"])).mul_(2))

        source, target = Model(), Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source,
            (("a", torch.tensor([[1.0, 2.0]])), ("b", torch.tensor([[3.0, 4.0]]))),
            execute_writes=True,
        )
        plan = recorder.build_plan()
        replay = record_target_weight_load_plan(target, plan.logical_weights)
        self.assertTrue(
            torch.equal(source.weight, torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
        )
        self.assertEqual(
            [(v.tensor_id, v.byte_offset) for v in plan.views],
            [(v.tensor_id, v.byte_offset) for v in replay.views],
        )
        self.assertEqual(torch.count_nonzero(target.weight).item(), 0)

    def test_long_same_axis_concat_is_flat_and_replayable(self):
        count = 1100

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(count))

            def load_weights(self, inputs):
                value = None
                for _, tensor in inputs:
                    value = tensor if value is None else torch.cat((value, tensor))
                self.weight.data.copy_(value)

        source, target = Model(), Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            source,
            ((f"w{i:04d}", torch.tensor([float(i)])) for i in range(count)),
            execute_writes=True,
        )
        plan = recorder.build_plan()
        replay = record_target_weight_load_plan(target, plan.logical_weights)
        torch.testing.assert_close(
            source.weight, torch.arange(count).float(), rtol=0, atol=0
        )
        expected = [(f"w{i:04d}", 4 * i) for i in range(count)]
        self.assertEqual([(v.tensor_id, v.byte_offset) for v in plan.views], expected)
        self.assertEqual([(v.tensor_id, v.byte_offset) for v in replay.views], expected)
        self.assertEqual(torch.count_nonzero(target.weight).item(), 0)

    def test_composite_and_static_index_mappings(self):
        weights = {
            "a": torch.arange(24).reshape(6, 4).float(),
            "b": torch.arange(24, 48).reshape(6, 4).float(),
        }
        cases = {
            "inner_concat": lambda w: torch.cat((w["a"], w["b"]), dim=1),
            "concat_transpose": lambda w: torch.cat((w["a"], w["b"])).t(),
            "concat_reshape_select": lambda w: (
                torch.cat((w["a"], w["b"])).reshape(3, 4, 4).select(1, 0)
            ),
            "view_concat": lambda w: torch.cat(
                (w["a"].reshape(3, 8)[:, ::2], w["b"].reshape(3, 8)[:, ::2])
            ),
            "stack": lambda w: torch.stack((w["a"], w["b"])),
            "stack_view": lambda w: (
                torch.stack((w["a"], w["b"]), dim=1)
                .transpose(0, 1)
                .contiguous()
                .reshape(3, 4, 4)
            ),
            "static_index": lambda w: w["a"].index_select(0, torch.tensor([4, 1, 5])),
            "index_columns": lambda w: w["a"].index_select(
                1, torch.tensor([2, 0, 3, 1])
            ),
            "index_after_concat": lambda w: (
                torch.cat((w["a"], w["b"])).index_select(0, torch.tensor([8, 1, 9])).t()
            ),
        }
        for name, transform in cases.items():
            with self.subTest(operation=name):
                self._check_mapping(weights, transform)

    def test_static_index_rejects_unavailable_or_repeated_indices(self):
        class Model(torch.nn.Module):
            def __init__(self, duplicate):
                super().__init__()
                self.duplicate = duplicate
                self.weight = torch.nn.Parameter(torch.zeros(2, 4))

            def load_weights(self, inputs):
                for _, value in inputs:
                    indices = torch.tensor(
                        [0, 0] if self.duplicate else [0, 1], device=value.device
                    )
                    self.weight.data.copy_(value.index_select(0, indices))

        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            Model(False), (("a", torch.ones(3, 4)),), execute_writes=True
        )
        with self.assertRaisesRegex(WeightLoadRecordingError, "static CPU indices"):
            record_target_weight_load_plan(
                Model(False), recorder.build_plan().logical_weights
            )
        with self.assertRaisesRegex(
            WeightLoadRecordingError, "unique in-range indices"
        ):
            WeightLoadRecorder().record_model_load(
                Model(True), (("a", torch.ones(3, 4)),), execute_writes=True
            )

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
        full = torch.arange(12).reshape(6, 2).float()
        for concatenate in (False, True):
            weights = (
                {"a": full[:3], "b": full[3:]} if concatenate else {"weight": full}
            )
            for selection in (slice(-2, None), slice(None, -2), slice(-4, -1)):
                with self.subTest(concatenate=concatenate, selection=selection):

                    def transform(parts):
                        loaded = (
                            torch.cat((parts["a"], parts["b"]))
                            if concatenate
                            else parts["weight"]
                        )
                        return loaded[selection]

                    self._check_mapping(weights, transform, expected=full[selection])

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
        manifest = _manifest(source, plan)
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
                self._assert_replay(plan, target)
                signatures.append(_placements(plan))
                manifest = _manifest(source, plan)
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
        manifest = _manifest(
            target,
            plan,
            worker="worker",
            topology=WeightParallelTopology(tp_rank=1, tp_size=2),
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

    def _check_composed_view(self, weight, transform):
        return self._check_mapping({"weight": weight}, lambda w: transform(w["weight"]))

    def _assert_replay(self, plan, target):
        replay = record_target_weight_load_plan(target, plan.logical_weights)
        self.assertEqual(_placements(plan), _placements(replay))
        self.assertEqual(torch.count_nonzero(target.weight).item(), 0)
        return replay

    def _check_mapping(self, weights, transform, expected=None):
        if expected is None:
            expected = transform(weights)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.zeros_like(expected.contiguous())
                )

            def load_weights(self, inputs):
                self.weight.data.copy_(transform(dict(inputs)))

        source, target = Model(), Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(source, weights.items(), execute_writes=True)
        plan = recorder.build_plan()
        torch.testing.assert_close(source.weight, expected, rtol=0, atol=0)

        # Reconstruct using checkpoint values and ordinary Torch indexing, without
        # reusing the recorder's mapping algorithm as the expected-result oracle.
        reconstructed = torch.empty_like(source.weight).flatten()
        for view in plan.views:
            box = tuple(
                slice(o, o + n) for o, n in zip(view.global_offset, view.local_shape)
            )
            values = weights[view.tensor_id][box].flatten()
            begin = view.byte_offset // source.weight.element_size()
            reconstructed[begin : begin + values.numel()].copy_(values)
        torch.testing.assert_close(
            reconstructed.reshape_as(expected), expected, rtol=0, atol=0
        )
        self._assert_replay(plan, target)
        manifest = _manifest(source, plan)
        self.assertEqual(
            sum(t.nbytes for t in manifest.tensors),
            source.weight.numel() * source.weight.element_size(),
        )
        return plan

    def test_recorder_tracks_composed_views(self):
        weight = torch.arange(24).reshape(2, 3, 4).float()
        transforms = {
            "reshape_select": lambda w: w.reshape(3, 2, 4).select(2, 0),
            "reshape_slice": lambda w: w.reshape(6, 4)[1:5:2, 1:],
            "reshape_transpose": lambda w: w.reshape(3, 8).t(),
            "materialize_then_slice": lambda w: w.reshape(3, 8).t().contiguous()[1:6:2],
            "materialize_then_reshape": lambda w: (
                w.reshape(3, 8).t().contiguous().reshape(2, 4, 3)[1]
            ),
            "clone_then_cast": lambda w: w.reshape(6, 4)[::2].clone().half(),
            "slice_then_reshape": lambda w: w[1:].reshape(4, 3)[::2],
            "reshape_unbind": lambda w: w.reshape(3, 2, 4).unbind(1)[1],
            "reshape_diagonal": lambda w: w.reshape(6, 4).diagonal(offset=1),
            "nonzero_storage_offset": lambda w: w.reshape(6, 4)[1:5:2].transpose(0, 1),
        }
        for name, transform in transforms.items():
            with self.subTest(transform=name):
                self._check_composed_view(weight, transform)

    def test_recorder_composed_views_keep_large_runs_compact(self):
        weight = torch.arange(8 * 32768).reshape(8, 32768).float()
        plan = self._check_composed_view(
            weight, lambda w: w.reshape(4, 2, 32768).select(1, 1)
        )
        self.assertEqual(len(plan.views), 4)

    def test_indexing_ops_keep_compact_logical_regions(self):
        weight = torch.arange(28).reshape(7, 4).float()
        transforms = {
            "slice": lambda w: w[-5:-1],
            "narrow": lambda w: w.narrow(-2, -5, 4),
            "select": lambda w: w.select(-1, -2),
            "split": lambda w: w.split(3, dim=-2)[1],
            "split_sizes": lambda w: w.split((1, 4, 2), dim=0)[1],
            "unbind": lambda w: w.unbind(-1)[2],
        }
        for name, transform in transforms.items():
            with self.subTest(operation=name):
                plan = self._check_composed_view(weight, transform)
                self.assertEqual(len(plan.views), 1)
        self._check_composed_view(
            torch.arange(6).reshape(2, 1, 3).float(), lambda w: w.select(1, 0)
        )

    def test_casts_preserve_all_provenance_kinds(self):
        class Model(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
                self.weight = torch.nn.Parameter(
                    torch.zeros(4 if kind == "composite" else 2, 4, dtype=torch.float16)
                )

            def load_weights(self, weights):
                tensors = dict(weights)
                if self.kind == "composite":
                    value = torch.cat((tensors["a"], tensors["b"]))
                elif self.kind == "view":
                    value = tensors["a"].reshape(4, 2).t()
                else:
                    value = tensors["a"].clone()
                self.weight.data.copy_(value.float())

        weights = {
            "a": torch.arange(8, dtype=torch.float64).reshape(2, 4),
            "b": torch.arange(8, 16, dtype=torch.float64).reshape(2, 4),
        }
        for kind in ("logical", "composite", "view"):
            with self.subTest(kind=kind):
                source, reference, target = Model(kind), Model(kind), Model(kind)
                reference.load_weights(weights.items())
                recorder = WeightLoadRecorder()
                recorder.record_model_load(source, weights.items(), execute_writes=True)
                plan = recorder.build_plan()
                self._assert_replay(plan, target)
                torch.testing.assert_close(
                    source.weight, reference.weight, rtol=0, atol=0
                )
                for view in plan.views:
                    self.assertIn(
                        "cast(float64->float32);cast(float32->float16)",
                        view.layout_fingerprint,
                    )

    def test_recorder_composed_view_with_noncontiguous_checkpoint(self):
        weight = torch.arange(48).reshape(6, 8).t()[1:7].float()
        self._check_composed_view(weight, lambda w: w.reshape(3, 12)[::2, 1::3])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_recorder_composed_views_on_cuda(self):
        weight = torch.arange(24, device="cuda").reshape(2, 3, 4).float()
        self._check_composed_view(
            weight, lambda w: w.reshape(3, 8).t().contiguous().reshape(2, 4, 3)[1]
        )

    def test_recorder_composed_view_rejects_reads_from_input_holes(self):
        with self.assertRaisesRegex(WeightLoadRecordingError, "tracked input"):
            self._check_composed_view(
                torch.arange(24).reshape(2, 3, 4).float(),
                lambda w: w.reshape(6, 4)[1:5:2].as_strided((1, 4), (4, 1), 8),
            )

    def test_recorder_rejects_mutation_of_composed_view(self):
        def transform(w):
            selected = w.reshape(3, 8).t().clone()
            selected.zero_()
            return selected

        with self.assertRaisesRegex(WeightLoadRecordingError, "mutable composed view"):
            self._check_composed_view(
                torch.arange(24).reshape(2, 3, 4).float(), transform
            )

    def test_recorder_does_not_treat_lazy_negation_as_a_view(self):
        with self.assertRaisesRegex(
            WeightLoadRecordingError, "unsupported loader operation"
        ):
            self._check_composed_view(
                torch.arange(24).reshape(2, 3, 4).float(),
                lambda w: torch._neg_view(w.reshape(3, 8).t()),
            )

    def test_recorder_bounds_irregular_view_fragments(self):
        with (
            patch(
                "sglang.srt.weight_cache.weight_load_recorder._MAX_VIEW_FRAGMENTS", 4
            ),
            self.assertRaisesRegex(WeightLoadRecordingError, "exceeds 4 fragments"),
        ):
            self._check_composed_view(
                torch.arange(24).reshape(2, 3, 4).float(),
                lambda w: w.reshape(3, 8).t(),
            )

    def test_recorder_rejects_unknown_value_op_after_composed_view(self):
        with self.assertRaisesRegex(
            WeightLoadRecordingError, "unsupported loader operation: aten::add"
        ):
            self._check_composed_view(
                torch.arange(24).reshape(2, 3, 4).float(),
                lambda w: w.reshape(3, 8).t() + 1,
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

        for manifest in (
            _manifest(source, source_plan),
            _manifest(target, target_plan, worker="target"),
        ):
            self.assertEqual(
                {tensor.runtime_name for tensor in manifest.tensors},
                {"weight", "weight_scale"},
            )

        source.weight_scale = source.weight_scale.clone()
        with self.assertRaisesRegex(
            WeightManifestError, "recorded tensor storage changed"
        ):
            _manifest(source, source_plan, worker="replaced-buffer")

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

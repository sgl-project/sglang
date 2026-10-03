"""Common-loader checkpoint metadata capture; no device or serving imports."""

import ast
import gc
import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_root = Path(__file__).resolve().parents[4] / "python/sglang/srt"
_spec = importlib.util.spec_from_file_location(
    "gpu_delta_checkpoint_under_test", _root / "weight_sync/gpu_delta_checkpoint.py"
)
checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checkpoint)


class TensorMetadata:
    def __init__(self, shape=(2, 3), dtype="torch.bfloat16"):
        self.shape = shape
        self.dtype = dtype


class Model:
    def load_weights(self, weights, is_nextn=False, *, token=None):
        self.names = [name for name, _ in weights]
        self.options = (is_nextn, token)
        return token


def common_load_weights_only():
    # Exercise the production boundary without importing unrelated CUDA loaders.
    tree = ast.parse((_root / "model_loader/loader.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DefaultModelLoader"
    )
    method = next(
        node for node in cls.body if getattr(node, "name", "") == "load_weights_only"
    )
    method.decorator_list = []
    namespace = {"is_cuda_alike": lambda: False}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "common_loader", "exec"),
        namespace,
    )
    return namespace[method.name]


def layout_class():
    name = "gpu_delta_checkpoint_layout_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, _root / "weight_sync/gpu_delta_layout.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].GpuDeltaLayout


class TestCanonicalWeightObserver(unittest.TestCase):
    def test_source_metadata_precedes_conversion_and_preserves_arguments(self):
        source = TensorMetadata()
        seen = []

        class ConvertingModel(Model):
            def load_weights(self, weights, is_nextn=False, *, token=None):
                for name, tensor in weights:
                    seen.append(tensor)
                    tensor.shape = (6,)
                    tensor.dtype = "torch.float32"
                self.options = (is_nextn, token)
                return token

        model = ConvertingModel()
        checkpoint.install_canonical_weight_observer(model)
        result = object()
        self.assertIs(
            model.load_weights([("source.alias", source)], False, token=result), result
        )
        self.assertEqual(model.options, (False, result))
        self.assertIs(seen[0], source)
        self.assertEqual(
            model._gpu_delta_canonical_inventory,
            {"source.alias": {"shape": [2, 3], "dtype": "BF16"}},
        )
        self.assertTrue(model._gpu_delta_metadata_complete)

    def test_installation_and_iteration_are_lazy(self):
        consumed = []

        def weights():
            consumed.append("first")
            yield "first", TensorMetadata()
            consumed.append("second")
            yield "second", TensorMetadata()

        class PartialModel(Model):
            def load_weights(self, weights, **kwargs):
                self.first = next(iter(weights))[0]
                return "partial"

        model = PartialModel()
        checkpoint.install_canonical_weight_observer(model)
        self.assertFalse(hasattr(model, "_gpu_delta_canonical_inventory"))
        self.assertEqual(consumed, [])
        self.assertEqual(model.load_weights(weights()), "partial")
        self.assertEqual(consumed, ["first"])
        self.assertFalse(model._gpu_delta_metadata_complete)

    def test_installation_is_idempotent_and_direct_reload_invalidates_generation(self):
        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        first_wrapper = model.load_weights
        checkpoint.install_canonical_weight_observer(model)
        self.assertIs(model.load_weights, first_wrapper)
        model.load_weights([("first", TensorMetadata())])
        first_inventory = model._gpu_delta_canonical_inventory
        self.assertEqual(model._gpu_delta_load_generation, 1)
        model.load_weights([("second", TensorMetadata((4,), "torch.uint8"))])
        self.assertEqual(model._gpu_delta_load_generation, 2)
        self.assertIsNot(model._gpu_delta_canonical_inventory, first_inventory)
        self.assertEqual(set(first_inventory), {"first"})
        self.assertEqual(set(model._gpu_delta_canonical_inventory), {"second"})

    def test_failed_load_is_incomplete_even_after_exhausting_source(self):
        class FailingModel(Model):
            def load_weights(self, weights):
                list(weights)
                raise RuntimeError("post-load failure")

        model = FailingModel()
        checkpoint.install_canonical_weight_observer(model)
        with self.assertRaisesRegex(RuntimeError, "post-load failure"):
            model.load_weights([("weight", TensorMetadata())])
        self.assertEqual(model._gpu_delta_load_generation, 1)
        self.assertFalse(model._gpu_delta_metadata_complete)
        self.assertEqual(set(model._gpu_delta_canonical_inventory), {"weight"})

    def test_failure_before_consuming_source_still_invalidates_generation(self):
        class FailingModel(Model):
            def load_weights(self, weights):
                raise RuntimeError("before iterator")

        model = FailingModel()
        model._gpu_delta_load_generation = 3
        checkpoint.install_canonical_weight_observer(model)
        with self.assertRaisesRegex(RuntimeError, "before iterator"):
            model.load_weights([])
        self.assertEqual(model._gpu_delta_load_generation, 4)
        self.assertFalse(model._gpu_delta_metadata_complete)

    def test_iterator_failure_remains_original_failure(self):
        def weights():
            yield "weight", TensorMetadata()
            raise OSError("source read failed")

        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        with self.assertRaisesRegex(OSError, "source read failed"):
            model.load_weights(weights())
        self.assertFalse(model._gpu_delta_metadata_complete)

    def test_unsupported_dtype_and_duplicate_names_do_not_break_normal_load(self):
        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        model.load_weights(
            [
                ("same", TensorMetadata(dtype="torch.float64")),
                ("same", TensorMetadata(dtype="torch.float64")),
            ]
        )
        self.assertEqual(model.names, ["same", "same"])
        self.assertTrue(model._gpu_delta_duplicate_source_names)
        self.assertTrue(model._gpu_delta_metadata_complete)
        self.assertEqual(
            model._gpu_delta_canonical_inventory["same"]["dtype"], "torch.float64"
        )

    def test_no_source_tensor_is_retained(self):
        references = []

        def weights():
            tensor = TensorMetadata()
            references.append(weakref.ref(tensor))
            yield "weight", tensor

        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        model.load_weights(weights())
        gc.collect()
        self.assertIsNone(references[0]())

    def test_ignored_and_alias_source_names_remain_in_inventory(self):
        names = [
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.9.mtp.weight",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
            "model.layers.0.mlp.experts.0.gate_proj.input_scale",
        ]
        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        model.load_weights((name, TensorMetadata()) for name in names)
        self.assertEqual(list(model._gpu_delta_canonical_inventory), names)

    def test_explicit_nextn_calls_leave_target_inventory_unchanged(self):
        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        model.load_weights([("target", TensorMetadata())])
        original_inventory = model._gpu_delta_canonical_inventory
        for args, kwargs in [((True,), {}), ((), {"is_nextn": True})]:
            model.load_weights([("draft", TensorMetadata())], *args, **kwargs)
            self.assertIs(model._gpu_delta_canonical_inventory, original_inventory)
            self.assertEqual(model._gpu_delta_load_generation, 1)
            self.assertTrue(model.options[0])

    def test_draft_exclusion_survives_leaving_build_scope(self):
        model = Model()
        original = model.load_weights
        checkpoint.install_canonical_weight_observer(model, is_draft=True)
        checkpoint.install_canonical_weight_observer(model, is_draft=False)
        self.assertEqual(model.load_weights, original)
        model.load_weights([("draft", TensorMetadata())])
        self.assertFalse(hasattr(model, "_gpu_delta_canonical_inventory"))

    def test_common_loader_installs_before_loading_and_skips_draft(self):
        load = common_load_weights_only()
        for is_draft in [False, True]:
            runtime = types.ModuleType("sglang.srt.runtime_context")
            runtime.get_flags = lambda: types.SimpleNamespace(
                moe=types.SimpleNamespace(in_speculative_scope=is_draft)
            )
            model = Model()
            with patch.dict(
                sys.modules,
                {
                    runtime.__name__: runtime,
                    "sglang.srt.weight_sync.gpu_delta_checkpoint": checkpoint,
                },
            ):
                load(model, [("raw", TensorMetadata())], None)
            self.assertEqual(model.names, ["raw"])
            self.assertEqual(
                hasattr(model, "_gpu_delta_canonical_inventory"), not is_draft
            )

    def test_layout_rejects_partial_or_failed_checkpoint_metadata(self):
        class PartialModel(Model):
            def load_weights(self, weights):
                next(iter(weights))

        model = PartialModel()
        checkpoint.install_canonical_weight_observer(model)
        model.load_weights([("weight", TensorMetadata())])
        with self.assertRaisesRegex(ValueError, "canonical startup metadata"):
            layout_class()(model)

    def test_layout_rejects_metadata_without_completed_loader_observation(self):
        model = Model()
        model._gpu_delta_canonical_inventory = {
            "weight": {"shape": [2, 3], "dtype": "BF16"}
        }
        with self.assertRaisesRegex(ValueError, "canonical startup metadata"):
            layout_class()(model)

    def test_captured_dtypes_match_the_physical_layout_contract(self):
        cls = layout_class()
        physical_dtypes = sys.modules[cls.__module__]._DTYPES
        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        model.load_weights(
            (str(dtype), TensorMetadata(dtype=dtype)) for dtype in physical_dtypes
        )
        for dtype, expected in physical_dtypes.items():
            self.assertEqual(
                model._gpu_delta_canonical_inventory[str(dtype)]["dtype"], expected
            )

    def test_layout_detects_reload_even_without_tensor_pointer_change(self):
        model = Model()
        checkpoint.install_canonical_weight_observer(model)
        tensor = TensorMetadata()
        model.load_weights([("weight", tensor)])
        cls = layout_class()
        admitted = object.__new__(cls)
        admitted.model = model
        admitted.generation = model._gpu_delta_load_generation
        model.load_weights([("weight", tensor)])
        with self.assertRaisesRegex(RuntimeError, "ordinary reload invalidated"):
            admitted.check_identity()


if __name__ == "__main__":
    unittest.main()

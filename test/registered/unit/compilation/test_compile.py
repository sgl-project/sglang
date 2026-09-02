"""Unit tests for IntermediateTensors and helpers in srt/compilation/compile.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from typing import Optional

import torch

from sglang.srt.compilation.compile import (
    IntermediateTensors,
    _infer_dynamic_arg_dims_from_annotations,
    _normalize_dims,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# IntermediateTensors
# ---------------------------------------------------------------------------


class TestIntermediateTensorsGetitem(CustomTestCase):
    def test_string_key_returns_tensor(self):
        t = torch.zeros(4)
        it = IntermediateTensors({"hidden": t})
        self.assertIs(it["hidden"], t)

    def test_string_key_missing_raises_key_error(self):
        it = IntermediateTensors({"hidden": torch.zeros(4)})
        with self.assertRaises(KeyError):
            _ = it["nonexistent"]

    def test_slice_returns_new_intermediate_tensors(self):
        h = torch.arange(8, dtype=torch.float32)
        it = IntermediateTensors({"h": h})
        sliced = it[1:4]
        self.assertIsInstance(sliced, IntermediateTensors)

    def test_slice_applies_to_each_tensor(self):
        h = torch.arange(8, dtype=torch.float32)
        r = torch.arange(8, dtype=torch.float32) * 2
        it = IntermediateTensors({"h": h, "r": r})
        sliced = it[2:5]
        self.assertEqual(sliced["h"].tolist(), h[2:5].tolist())
        self.assertEqual(sliced["r"].tolist(), r[2:5].tolist())

    def test_slice_shares_storage_with_original(self):
        # IntermediateTensors.__getitem__ uses v[key] which returns a tensor
        # view, not a copy. Mutating the slice mutates the underlying storage.
        h = torch.zeros(6)
        it = IntermediateTensors({"h": h})
        sliced = it[0:3]
        sliced["h"][0] = 99.0
        self.assertEqual(it["h"][0].item(), 99.0)


class TestIntermediateTensorsSetitem(CustomTestCase):
    def test_setitem_stores_new_tensor(self):
        it = IntermediateTensors({"a": torch.zeros(3)})
        new_t = torch.ones(3)
        it["a"] = new_t
        self.assertIs(it.tensors["a"], new_t)

    def test_setitem_adds_new_key(self):
        it = IntermediateTensors({})
        t = torch.zeros(2)
        it["z"] = t
        self.assertIs(it["z"], t)


class TestIntermediateTensorsLenAndItems(CustomTestCase):
    def test_len_returns_number_of_tensors(self):
        it = IntermediateTensors({"a": torch.zeros(1), "b": torch.zeros(1)})
        self.assertEqual(len(it), 2)

    def test_len_of_empty_is_zero(self):
        it = IntermediateTensors({})
        self.assertEqual(len(it), 0)

    def test_items_returns_key_tensor_pairs(self):
        ta = torch.zeros(2)
        it = IntermediateTensors({"a": ta})
        pairs = list(it.items())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "a")
        self.assertIs(pairs[0][1], ta)

    def test_items_multiple_tensors(self):
        ta = torch.zeros(2)
        tb = torch.ones(3)
        it = IntermediateTensors({"a": ta, "b": tb})
        result = dict(it.items())
        self.assertIs(result["a"], ta)
        self.assertIs(result["b"], tb)


# ---------------------------------------------------------------------------
# _normalize_dims
# ---------------------------------------------------------------------------


class TestNormalizeDims(CustomTestCase):
    def test_scalar_non_negative_wraps_in_list(self):
        self.assertEqual(_normalize_dims(0, 4), [0])

    def test_scalar_positive_preserved(self):
        self.assertEqual(_normalize_dims(2, 4), [2])

    def test_scalar_negative_resolved_relative_to_ndim(self):
        self.assertEqual(_normalize_dims(-1, 4), [3])

    def test_scalar_negative_one_with_ndim_3(self):
        self.assertEqual(_normalize_dims(-1, 3), [2])

    def test_list_positive_dims_unchanged(self):
        self.assertEqual(_normalize_dims([0, 1, 2], 4), [0, 1, 2])

    def test_list_with_negative_dim_resolved(self):
        self.assertEqual(_normalize_dims([0, -1], 3), [0, 2])

    def test_list_all_negative_resolved(self):
        self.assertEqual(_normalize_dims([-2, -1], 4), [2, 3])

    def test_empty_list_returns_empty(self):
        self.assertEqual(_normalize_dims([], 3), [])


# ---------------------------------------------------------------------------
# _infer_dynamic_arg_dims_from_annotations
# ---------------------------------------------------------------------------


class TestInferDynamicArgDims(CustomTestCase):
    def test_torch_tensor_annotation_detected(self):
        def fn(x: torch.Tensor, y: int):
            pass

        result = _infer_dynamic_arg_dims_from_annotations(fn)
        self.assertIn("x", result)
        self.assertEqual(result["x"], 0)
        self.assertNotIn("y", result)

    def test_optional_torch_tensor_annotation_detected(self):
        def fn(x: Optional[torch.Tensor]):
            pass

        result = _infer_dynamic_arg_dims_from_annotations(fn)
        self.assertIn("x", result)
        self.assertEqual(result["x"], 0)

    def test_string_annotation_torch_tensor_detected(self):
        # Simulate annotations stored as strings (from __future__ import annotations).
        def fn(x):
            pass

        # Manually assign a string annotation as if from __future__ imports.
        fn.__annotations__ = {"x": "torch.Tensor"}
        result = _infer_dynamic_arg_dims_from_annotations(fn)
        self.assertIn("x", result)

    def test_no_tensor_params_raises_value_error(self):
        def fn(x: int, y: str):
            pass

        with self.assertRaises(ValueError) as ctx:
            _infer_dynamic_arg_dims_from_annotations(fn)
        self.assertIn("dynamic_arg_dims", str(ctx.exception))

    def test_no_annotated_params_raises_value_error(self):
        def fn(x, y):
            pass

        with self.assertRaises(ValueError):
            _infer_dynamic_arg_dims_from_annotations(fn)

    def test_multiple_tensor_params_all_detected(self):
        def fn(a: torch.Tensor, b: torch.Tensor, n: int):
            pass

        result = _infer_dynamic_arg_dims_from_annotations(fn)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertNotIn("n", result)
        self.assertEqual(result["a"], 0)
        self.assertEqual(result["b"], 0)

    def test_returns_dict_with_zero_values(self):
        def fn(x: torch.Tensor):
            pass

        result = _infer_dynamic_arg_dims_from_annotations(fn)
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertEqual(v, 0)


if __name__ == "__main__":
    unittest.main()

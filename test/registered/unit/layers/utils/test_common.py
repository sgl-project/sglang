# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC

"""Unit tests for srt/layers/utils/common.py — CPU-only, no server."""

from sglang.test.ci.ci_register import register_cpu_ci

# Register CPU CI with estimated time and suite
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
import torch
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.layers.utils.common import (
    get_layer_id,
    pad_or_narrow_weight,
    is_strict_contiguous,
    strict_contiguous,
    copy_or_rebind_param,
    alias_or_bind_derived_param,
    PPMissingLayer,
)
from sglang.test.test_utils import CustomTestCase


class TestLayersCommonUtils(CustomTestCase):

    def test_get_layer_id(self):
        # Test standard matches
        self.assertEqual(get_layer_id("model.layers.10.self_attn.qkv_proj.weight"), 10)
        self.assertEqual(get_layer_id("layers.0.weight"), 0)
        self.assertEqual(get_layer_id("model.layers.102.self_attn.qkv_proj.weight"), 102)
        # Test no match
        self.assertIsNone(get_layer_id("model.embed_tokens.weight"))
        self.assertIsNone(get_layer_id("model.layers.abc.weight"))

    def test_pad_or_narrow_weight_padding(self):
        # Test padding when remaining weight size is smaller than shard_size
        loaded_weight = torch.ones(3, 4)  # shape: [3, 4]
        # Narrow at dim=1 from start_idx=2, shard_size=6
        # valid_size = 4 - 2 = 2. We need 6, so pad 4 zeros.
        result = pad_or_narrow_weight(
            loaded_weight, input_dim=1, start_idx=2, shard_size=6
        )
        
        self.assertEqual(result.shape, (3, 6))
        self.assertTrue(torch.all(result[:, :2] == 1.0))
        self.assertTrue(torch.all(result[:, 2:] == 0.0))

    def test_pad_or_narrow_weight_all_padding(self):
        # Test when start_idx is out of bounds (all padding)
        loaded_weight = torch.ones(3, 4)
        result = pad_or_narrow_weight(
            loaded_weight, input_dim=1, start_idx=5, shard_size=4
        )
        
        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(torch.all(result == 0.0))

    def test_is_strict_contiguous(self):
        x = torch.randn(3, 4, 5)
        self.assertTrue(is_strict_contiguous(x))
        
        # Transposed tensor is not strictly contiguous
        y = x.transpose(0, 1)
        self.assertFalse(is_strict_contiguous(y))

        # Slice can also be non-contiguous depending on strides
        z = x[:, 1:3, :]
        # Slicing dim=1 with step 1 still keeps it technically contiguous if strides match,
        # but let's force non-contiguity via step
        z_step = x[:, ::2, :]
        self.assertFalse(is_strict_contiguous(z_step))

    def test_strict_contiguous(self):
        x = torch.randn(3, 4, 5)
        # Should return identity if already contiguous
        self.assertIs(strict_contiguous(x), x)

        # Should return a clone if not contiguous
        y = x.transpose(0, 1)
        res = strict_contiguous(y)
        self.assertIsNot(res, y)
        self.assertTrue(is_strict_contiguous(res))
        self.assertTrue(torch.equal(res, y))

    def test_copy_or_rebind_param_new(self):
        module = nn.Module()
        val = torch.randn(2, 3)
        copy_or_rebind_param(module, "test_param", val)
        
        self.assertTrue(hasattr(module, "test_param"))
        self.assertTrue(isinstance(module.test_param, Parameter))
        self.assertTrue(torch.equal(module.test_param.data, val))
        self.assertFalse(module.test_param.requires_grad)

    def test_copy_or_rebind_param_inplace_copy(self):
        module = nn.Module()
        # Initialize parameter
        orig_val = torch.zeros(2, 3)
        module.test_param = Parameter(orig_val, requires_grad=False)
        orig_id = id(module.test_param)

        # Copy new value of same shape/dtype
        new_val = torch.ones(2, 3)
        copy_or_rebind_param(module, "test_param", new_val)

        # Identity should be preserved (inplace copy)
        self.assertEqual(id(module.test_param), orig_id)
        self.assertTrue(torch.equal(module.test_param.data, new_val))

    def test_copy_or_rebind_param_rebind(self):
        module = nn.Module()
        # Initialize parameter
        orig_val = torch.zeros(2, 3)
        module.test_param = Parameter(orig_val, requires_grad=False)
        orig_id = id(module.test_param)

        # Rebind with different shape
        new_val = torch.ones(4, 5)
        copy_or_rebind_param(module, "test_param", new_val)

        # Identity should remain stable even on shape mismatch, but shape should update
        self.assertEqual(id(module.test_param), orig_id)
        self.assertEqual(module.test_param.shape, (4, 5))
        self.assertTrue(torch.equal(module.test_param.data, new_val))


    def test_alias_or_bind_derived_param_alias(self):
        module = nn.Module()
        # Source parameter of shape [4, 4]
        module.src = Parameter(torch.zeros(4, 4), requires_grad=False)
        src_id = id(module.src)

        # Derived value of shape [1, 4] (broadcast-compatible to [4, 4])
        derived_val = torch.ones(1, 4)
        alias_or_bind_derived_param(module, "src", "derived", derived_val)

        # Derived should be an alias of src (same identity)
        self.assertIs(module.derived, module.src)
        self.assertEqual(id(module.derived), src_id)
        # Source data should be filled with broadcasted values
        expected = torch.ones(4, 4)
        self.assertTrue(torch.equal(module.src.data, expected))

    def test_alias_or_bind_derived_param_fallback(self):
        module = nn.Module()
        # Source parameter of shape [4, 4]
        module.src = Parameter(torch.zeros(4, 4), requires_grad=False)
        src_id = id(module.src)

        # Derived value of shape [3, 3] (not broadcast-compatible to [4, 4])
        derived_val = torch.ones(3, 3)
        alias_or_bind_derived_param(module, "src", "derived", derived_val)

        # Derived should be a separate parameter (fallback)
        self.assertIsNot(module.derived, module.src)
        self.assertNotEqual(id(module.derived), src_id)
        self.assertTrue(torch.equal(module.derived.data, derived_val))
        # Source should remain untouched
        self.assertTrue(torch.all(module.src.data == 0.0))

    def test_pp_missing_layer(self):
        # Test returning tuple
        layer_tuple = PPMissingLayer(return_tuple=True)
        x = torch.tensor([42.0])
        self.assertEqual(layer_tuple(x), (x,))

        # Test returning single value
        layer_single = PPMissingLayer(return_tuple=False)
        self.assertEqual(layer_single(x), x)

        # Test kwargs fallback
        self.assertEqual(layer_single(y=x), x)


if __name__ == "__main__":
    unittest.main()

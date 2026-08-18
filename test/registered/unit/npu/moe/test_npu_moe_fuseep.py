"""
Unit tests for sglang.srt.hardware_backend.npu.moe.fuseep.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

# Mock NPU-only and heavy modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sglang.srt.distributed",
    "sglang.srt.environ",
    "sglang.srt.hardware_backend.npu.utils",
    "sglang.srt.layers",
    "sglang.srt.layers.moe",
    "sglang.srt.layers.moe.token_dispatcher",
    "sglang.srt.layers.moe.token_dispatcher.deepep",
    "sglang.srt.layers.moe.utils",
    "sglang.srt.runtime_context",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.moe.fuseep import (
    _PARAMS_BYTES,
    _get_fuseep_buffer,
    _permute_w13_weight_scale,
    _release_weight_cache,
    _reshape_w13_weight,
    _scale_from_float_to_int64,
    forward_fuseep,
    process_fuseep_weights,
)

# torch_npu adds .npu() to tensors; mock it as identity for CPU testing.
if not hasattr(torch.Tensor, "npu"):
    torch.Tensor.npu = lambda self: self

_MODULE = "sglang.srt.hardware_backend.npu.moe.fuseep"


# =============================================================================
# _PARAMS_BYTES
# =============================================================================
class TestParamsBytes(unittest.TestCase):
    def test_value(self):
        self.assertEqual(_PARAMS_BYTES, 2)


# =============================================================================
# _permute_w13_weight_scale  (pure torch)
# =============================================================================
class TestPermuteW13WeightScale(unittest.TestCase):
    def test_shape_preserved(self):
        w = torch.randn(4, 8)
        out = _permute_w13_weight_scale(w, tile_n=4)
        self.assertEqual(out.shape, w.shape)

    def test_values(self):
        w = torch.arange(8.0)
        out = _permute_w13_weight_scale(w, tile_n=4)

        # w.reshape(2, 2, 2).permute(1, 0, 2).reshape(8)
        expected = w.reshape(2, 2, 2).permute(1, 0, 2).reshape(8)
        self.assertTrue(torch.equal(out, expected))

    def test_multiple_leading_dims(self):
        w = torch.randn(2, 3, 16)
        out = _permute_w13_weight_scale(w, tile_n=8)
        self.assertEqual(out.shape, (2, 3, 16))

    def test_odd_tile_n_raises(self):
        with self.assertRaises(ValueError):
            _permute_w13_weight_scale(torch.randn(4, 6), tile_n=3)

    def test_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            _permute_w13_weight_scale(torch.randn(4, 7), tile_n=4)


# =============================================================================
# _reshape_w13_weight  (pure torch)
# =============================================================================
class TestReshapeW13Weight(unittest.TestCase):
    def test_shape_preserved(self):
        w = torch.randn(2, 128, 8)
        out = _reshape_w13_weight(w, dim=1)
        self.assertEqual(out.shape, w.shape)

    def test_values(self):
        w = torch.randn(4, 8)
        out = _reshape_w13_weight(w, dim=1, chunk_size=2)

        # reshape(4, 2, 2, 2).transpose(1, 2).contiguous().view(4, 8)
        expected = (
            w.view(4, 2, 2, 2).transpose(1, 2).contiguous().view(4, 8)
        )
        self.assertTrue(torch.equal(out, expected))

    def test_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            _reshape_w13_weight(torch.randn(4, 7), dim=1, chunk_size=4)

    def test_negative_dim(self):
        w = torch.randn(2, 8, 128)
        out_neg = _reshape_w13_weight(w, dim=-1)
        out_pos = _reshape_w13_weight(w, dim=2)
        self.assertTrue(torch.equal(out_neg, out_pos))

    def test_custom_chunk_size(self):
        w = torch.randn(4, 16)
        out = _reshape_w13_weight(w, dim=1, chunk_size=4)
        self.assertEqual(out.shape, (4, 16))


# =============================================================================
# _release_weight_cache  (pure torch)
# =============================================================================
class TestReleaseWeightCache(unittest.TestCase):
    def test_transposes_dims_1_and_2(self):
        w = torch.randn(2, 3, 4, 5)
        out = _release_weight_cache(w)
        self.assertEqual(out.shape, (2, 4, 3, 5))

    def test_output_contiguous(self):
        w = torch.randn(2, 3, 4)
        out = _release_weight_cache(w)
        self.assertTrue(out.is_contiguous())

    def test_values(self):
        w = torch.randn(2, 3, 4)
        expected = w.transpose(1, 2).contiguous().clone()
        out = _release_weight_cache(w)
        self.assertTrue(torch.equal(out, expected))

    def test_returns_new_tensor(self):
        w = torch.randn(2, 3, 4)
        out = _release_weight_cache(w)
        self.assertIsNot(out, w)


# =============================================================================
# _scale_from_float_to_int64  (numpy conversion)
# =============================================================================
class TestScaleFromFloatToInt64(unittest.TestCase):
    def test_returns_parameter(self):
        scale = torch.tensor([1.0], dtype=torch.float32)
        out = _scale_from_float_to_int64(scale)
        self.assertIsInstance(out, torch.nn.Parameter)

    def test_requires_grad_false(self):
        scale = torch.tensor([1.0], dtype=torch.float32)
        out = _scale_from_float_to_int64(scale)
        self.assertFalse(out.requires_grad)

    def test_dtype_int64(self):
        scale = torch.tensor([1.0], dtype=torch.float32)
        out = _scale_from_float_to_int64(scale)
        self.assertEqual(out.dtype, torch.int64)

    def test_shape_flattened_to_1d(self):
        """numpy frombuffer flattens; output is always 1D."""
        scale = torch.randn(3, 4, dtype=torch.float32)
        out = _scale_from_float_to_int64(scale)
        self.assertEqual(out.shape, (12,))

    def test_values(self):
        scale = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = _scale_from_float_to_int64(scale)
        # Reinterpret float32 bytes as int32, cast to int64
        import numpy as np

        expected = np.frombuffer(
            scale.numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
        self.assertTrue(torch.equal(out, torch.from_numpy(expected)))


# =============================================================================
# _get_fuseep_buffer  (mocked)
# =============================================================================
class TestGetFuseepBuffer(unittest.TestCase):
    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}.DeepEPMode")
    @patch(f"{_MODULE}.get_moe_ep_group")
    @patch(f"{_MODULE}.DeepEPBuffer")
    def test_calls_set_dispatch_mode(
        self, mock_buf, mock_ep_group, mock_mode, mock_envs
    ):
        layer = SimpleNamespace(hidden_size=128, num_experts=4)
        _get_fuseep_buffer(layer)
        mock_buf.set_dispatch_mode_as_low_latency.assert_called_once()

    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}.DeepEPMode")
    @patch(f"{_MODULE}.get_moe_ep_group")
    @patch(f"{_MODULE}.DeepEPBuffer")
    def test_calls_get_deepep_buffer(
        self, mock_buf, mock_ep_group, mock_mode, mock_envs
    ):
        mock_envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get.return_value = 128
        layer = SimpleNamespace(hidden_size=256, num_experts=8)
        _get_fuseep_buffer(layer)
        mock_buf.get_deepep_buffer.assert_called_once()

    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}.DeepEPMode")
    @patch(f"{_MODULE}.get_moe_ep_group")
    @patch(f"{_MODULE}.DeepEPBuffer")
    def test_get_deepep_buffer_args(
        self, mock_buf, mock_ep_group, mock_mode, mock_envs
    ):
        mock_envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get.return_value = 128
        mock_ep_group.return_value = SimpleNamespace(
            device_group="device_group_obj"
        )
        layer = SimpleNamespace(hidden_size=256, num_experts=8)
        _get_fuseep_buffer(layer)

        args = mock_buf.get_deepep_buffer.call_args.args
        self.assertEqual(args[0], "device_group_obj")
        self.assertEqual(args[1], 256)
        self.assertEqual(args[2], _PARAMS_BYTES)
        self.assertIs(args[3], mock_mode.LOW_LATENCY)
        self.assertEqual(args[4], 128)
        self.assertEqual(args[5], 8)

    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}.DeepEPMode")
    @patch(f"{_MODULE}.get_moe_ep_group")
    @patch(f"{_MODULE}.DeepEPBuffer")
    def test_returns_buffer(
        self, mock_buf, mock_ep_group, mock_mode, mock_envs
    ):
        expected = MagicMock(name="buffer")
        mock_buf.get_deepep_buffer.return_value = expected
        layer = SimpleNamespace(hidden_size=128, num_experts=4)
        result = _get_fuseep_buffer(layer)
        self.assertIs(result, expected)


# =============================================================================
# forward_fuseep  (mocked)
# =============================================================================
class TestForwardFuseep(unittest.TestCase):
    def _make_layer(self):
        layer = SimpleNamespace(
            hidden_size=128,
            num_experts=4,
            w13_weight=torch.randn(4, 8, 128),
            w13_weight_scale=torch.randn(4, 8, 1),
            w2_weight=torch.randn(4, 128, 8),
            w2_weight_scale=torch.randn(4, 128, 1),
        )
        return layer

    def _make_topk_output(self):
        return SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.tensor([[0.5, 0.5]]),
        )

    @patch(f"{_MODULE}.get_exec")
    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}._get_fuseep_buffer")
    def test_calls_fused_deep_moe(
        self, mock_get_buf, mock_envs, mock_get_exec
    ):
        mock_envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get.return_value = 128
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        buf = MagicMock()
        buf.fused_deep_moe.return_value = (torch.randn(2, 128), None)
        mock_get_buf.return_value = buf

        layer = self._make_layer()
        topk = self._make_topk_output()
        forward_fuseep(layer, torch.randn(2, 128), topk)

        buf.fused_deep_moe.assert_called_once()

    @patch(f"{_MODULE}.get_exec")
    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}._get_fuseep_buffer")
    def test_fused_deep_moe_kwargs(
        self, mock_get_buf, mock_envs, mock_get_exec
    ):
        mock_envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get.return_value = 128
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        buf = MagicMock()
        buf.fused_deep_moe.return_value = (torch.randn(2, 128), None)
        mock_get_buf.return_value = buf

        layer = self._make_layer()
        hidden = torch.randn(2, 128)
        topk = self._make_topk_output()
        forward_fuseep(layer, hidden, topk)

        args, kwargs = buf.fused_deep_moe.call_args
        self.assertIs(args[0], hidden)
        self.assertIs(kwargs["topk_idx"], topk.topk_ids)
        self.assertIs(kwargs["topk_weights"], topk.topk_weights)
        self.assertIs(kwargs["gmm1_permuted_weight"], layer.w13_weight)
        self.assertIs(kwargs["gmm1_permuted_weight_scale"], layer.w13_weight_scale)
        self.assertIs(kwargs["gmm2_weight"], layer.w2_weight)
        self.assertIs(kwargs["gmm2_weight_scale"], layer.w2_weight_scale)
        self.assertEqual(kwargs["num_max_dispatch_tokens_per_rank"], 128)
        self.assertEqual(kwargs["num_experts"], 4)
        self.assertEqual(kwargs["fuse_mode"], 1)

    @patch(f"{_MODULE}.get_exec")
    @patch(f"{_MODULE}.envs")
    @patch(f"{_MODULE}._get_fuseep_buffer")
    def test_returns_first_element(
        self, mock_get_buf, mock_envs, mock_get_exec
    ):
        mock_envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get.return_value = 128
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        expected = torch.randn(2, 128)
        buf = MagicMock()
        buf.fused_deep_moe.return_value = (expected, "discarded")
        mock_get_buf.return_value = buf

        layer = self._make_layer()
        out = forward_fuseep(layer, torch.randn(2, 128), self._make_topk_output())
        self.assertIs(out, expected)


# =============================================================================
# process_fuseep_weights  (mocked)
# =============================================================================
def _make_real_layer(prefix="w13", with_offset=False):
    """Create a layer with real tensor attributes (not MagicMock)."""
    layer = SimpleNamespace()
    if prefix == "w13":
        layer.w13_weight = torch.nn.Parameter(torch.randn(4, 128, 8))
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.randn(4, 128, 1)
        )
        if with_offset:
            layer.w13_weight_offset = torch.nn.Parameter(
                torch.randn(4, 128, 1)
            )
    else:
        layer.w2_weight = torch.nn.Parameter(torch.randn(4, 8, 128))
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.randn(4, 128, 1)
        )
        if with_offset:
            layer.w2_weight_offset = torch.nn.Parameter(
                torch.randn(4, 128, 1)
            )
    return layer


class TestProcessFuseepWeightsMode1(unittest.TestCase):
    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w13_calls_npu_format_cast(
        self, mock_get_exec, mock_cast
    ):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13")
        process_fuseep_weights(layer, "w13")
        mock_cast.assert_called()

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w13_scale_becomes_float32_parameter(
        self, mock_get_exec, mock_cast
    ):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13")
        process_fuseep_weights(layer, "w13")
        self.assertIsInstance(layer.w13_weight_scale, torch.nn.Parameter)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w13_weight_data_set(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13")
        process_fuseep_weights(layer, "w13")
        self.assertIsNotNone(layer.w13_weight.data)

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w2_calls_npu_format_cast(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w2")
        process_fuseep_weights(layer, "w2")
        mock_cast.assert_called()

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w2_scale_becomes_float32_parameter(
        self, mock_get_exec, mock_cast
    ):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w2")
        process_fuseep_weights(layer, "w2")
        self.assertIsInstance(layer.w2_weight_scale, torch.nn.Parameter)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)


class TestProcessFuseepWeightsMode2(unittest.TestCase):
    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w13_weight_transposed(self, mock_get_exec, mock_cast):
        """_release_weight_cache transposes dims 1 and 2."""
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=2)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13")  # shape (4, 128, 8)
        process_fuseep_weights(layer, "w13")
        # transposed -> (4, 8, 128)
        self.assertEqual(layer.w13_weight.data.shape[1], 8)
        self.assertEqual(layer.w13_weight.data.shape[2], 128)

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w13_scale_becomes_int64(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=2)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13")
        process_fuseep_weights(layer, "w13")
        self.assertIsInstance(layer.w13_weight_scale, torch.nn.Parameter)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.int64)

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w2_weight_transposed(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=2)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w2")  # shape (4, 8, 128)
        process_fuseep_weights(layer, "w2")
        # transposed -> (4, 128, 8)
        self.assertEqual(layer.w2_weight.data.shape[1], 128)
        self.assertEqual(layer.w2_weight.data.shape[2], 8)

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_w2_scale_becomes_int64(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=2)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w2")
        process_fuseep_weights(layer, "w2")
        self.assertIsInstance(layer.w2_weight_scale, torch.nn.Parameter)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.int64)


class TestProcessFuseepWeightsOffset(unittest.TestCase):
    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_offset_squeezed_when_present(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13", with_offset=True)
        process_fuseep_weights(layer, "w13")
        result = layer.w13_weight_offset
        self.assertIsInstance(result, torch.nn.Parameter)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.shape[-1], 128)

    @patch(f"{_MODULE}.npu_format_cast")
    @patch(f"{_MODULE}.get_exec")
    def test_no_offset_attr_skips(self, mock_get_exec, mock_cast):
        mock_get_exec.return_value = SimpleNamespace(
            moe=SimpleNamespace(fuseep_mode=1)
        )
        mock_cast.side_effect = lambda x: x
        layer = _make_real_layer("w13", with_offset=False)
        process_fuseep_weights(layer, "w13")
        self.assertFalse(hasattr(layer, "w13_weight_offset"))


if __name__ == "__main__":
    unittest.main()

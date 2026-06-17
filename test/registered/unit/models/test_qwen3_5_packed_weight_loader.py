"""
Unit tests for Qwen3_5GatedDeltaNet._make_packed_weight_loader.

Validates that per-tensor FP8 scales (scalar or single-element tensors)
are broadcast to every logical shard, while normal multi-element weights
are split correctly.

Regression test for https://github.com/sgl-project/sglang/issues/23051
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.test.test_utils import CustomTestCase


class MockQuantConfig:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name


def _write_safetensors_header(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def _make_mock_module(output_sizes):
    """Create a lightweight mock module with the attributes needed by the loader."""
    return SimpleNamespace(output_sizes=output_sizes)


def _make_per_tensor_scale_param(num_shards):
    """Create a PerTensorScaleParameter pre-allocated for `num_shards` scales.

    PerTensorScaleParameter requires a weight_loader callable;
    we supply a no-op since the packed loader wraps it anyway.
    """
    return PerTensorScaleParameter(
        data=torch.zeros(num_shards),
        weight_loader=lambda *args, **kwargs: None,
    )


class TestMakePackedWeightLoader(CustomTestCase):
    """Tests for _make_packed_weight_loader broadcast / split logic."""

    # ------------------------------------------------------------------ #
    #  Per-tensor scale broadcast                                         #
    # ------------------------------------------------------------------ #

    def test_scalar_weight_broadcast(self):
        """A 0-d scalar should be broadcast (via .view(-1)) to every shard."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])
        param = _make_per_tensor_scale_param(num_shards=4)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scalar = torch.tensor(0.5)  # shape=[]
        loader(param, scalar, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for shard_id, chunk in calls:
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.5, places=5)

    def test_single_element_tensor_broadcast(self):
        """A [1]-shaped tensor (e.g. per-tensor weight_scale) should be
        broadcast to every logical shard."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])
        param = _make_per_tensor_scale_param(num_shards=4)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([0.25])  # shape=[1]
        loader(param, scale, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for idx, (shard_id, chunk) in enumerate(calls):
            self.assertEqual(shard_id, idx)
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.25, places=5)

    def test_broadcast_with_two_shards(self):
        """Broadcast for in_proj_ba style (2 shards: b, a)."""
        module = _make_mock_module(output_sizes=[16, 16])
        param = _make_per_tensor_scale_param(num_shards=2)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([0.1])
        loader(param, scale, loaded_shard_id=(0, 1))

        self.assertEqual(len(calls), 2)
        for shard_id, chunk in calls:
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.1, places=5)

    # ------------------------------------------------------------------ #
    #  Normal weight split                                                #
    # ------------------------------------------------------------------ #

    def test_normal_weight_split(self):
        """Multi-element weights should be split by output_sizes, not broadcast."""
        module = _make_mock_module(output_sizes=[128, 128, 64])
        param = MagicMock()
        param.output_dim = 0

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        # Simulate a checkpoint weight that covers shard 0, 1, 2
        weight = torch.randn(128 + 128 + 64, 256)
        loader(param, weight, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1].shape[0], 128)
        self.assertEqual(calls[1][1].shape[0], 128)
        self.assertEqual(calls[2][1].shape[0], 64)

    # ------------------------------------------------------------------ #
    #  Passthrough for non-tuple shard_id                                 #
    # ------------------------------------------------------------------ #

    def test_int_shard_id_passthrough(self):
        """An int shard_id should bypass the tuple logic entirely."""
        module = _make_mock_module(output_sizes=[128, 128, 64, 64])

        calls = []

        def original_loader(p, loaded_weight, shard_id):
            calls.append(("original", shard_id))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        weight = torch.randn(128, 256)
        loader(MagicMock(), weight, loaded_shard_id=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("original", 2))

    def test_none_shard_id_passthrough(self):
        """None shard_id should pass through to the original loader."""
        module = _make_mock_module(output_sizes=[128])

        calls = []

        def original_loader(p, loaded_weight, shard_id):
            calls.append(("original", shard_id))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        weight = torch.randn(128, 256)
        loader(MagicMock(), weight, loaded_shard_id=None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("original", None))

    # ------------------------------------------------------------------ #
    #  Edge case: nested single-element tensors                           #
    # ------------------------------------------------------------------ #

    def test_nested_single_element_tensor_broadcast(self):
        """A [[value]] shaped tensor (numel==1, ndim==2) should also broadcast."""
        module = _make_mock_module(output_sizes=[128, 128, 64])
        param = _make_per_tensor_scale_param(num_shards=3)

        calls = []

        def original_loader(p, chunk, shard_id):
            calls.append((shard_id, chunk.clone()))

        loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, original_loader
        )

        scale = torch.tensor([[0.75]])  # shape=[1,1], numel==1
        loader(param, scale, loaded_shard_id=(0, 1, 2))

        self.assertEqual(len(calls), 3)
        for shard_id, chunk in calls:
            # .view(-1) should flatten to [1]
            self.assertEqual(chunk.shape, torch.Size([1]))
            self.assertAlmostEqual(chunk.item(), 0.75, places=5)


class TestQwen35ModelOptFp4Loading(CustomTestCase):
    def setUp(self):
        qwen3_5._get_modelopt_fp4_submodule_policy.cache_clear()

    def test_modelopt_fp4_policy_uses_layer_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_safetensors_header(
                Path(tmpdir) / "model.safetensors",
                {
                    "model.layers.0.self_attn.q_proj.weight": {"dtype": "U8"},
                    "model.layers.0.self_attn.k_proj.weight": {"dtype": "U8"},
                    "model.layers.0.self_attn.v_proj.weight": {"dtype": "U8"},
                    "model.layers.1.self_attn.q_proj.weight": {"dtype": "BF16"},
                    "model.layers.1.self_attn.k_proj.weight": {"dtype": "BF16"},
                    "model.layers.1.self_attn.v_proj.weight": {"dtype": "BF16"},
                },
            )
            config = SimpleNamespace(_name_or_path=tmpdir)
            modelopt = MockQuantConfig("modelopt_fp4")
            awq = MockQuantConfig("awq")

            self.assertIs(
                qwen3_5._resolve_modelopt_fp4_submodule_quant_config(
                    config, modelopt, "attention", 0
                ),
                modelopt,
            )
            self.assertIsNone(
                qwen3_5._resolve_modelopt_fp4_submodule_quant_config(
                    config, modelopt, "attention", 1
                )
            )
            self.assertIs(
                qwen3_5._resolve_modelopt_fp4_submodule_quant_config(
                    config, awq, "attention", 0
                ),
                awq,
            )

    def test_modelopt_fp4_policy_scans_safetensors_without_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_safetensors_header(
                Path(tmpdir) / "part-00001.safetensors",
                {
                    "model.layers.0.linear_attn.in_proj_b.weight": {"dtype": "U8"},
                    "model.layers.0.linear_attn.in_proj_a.weight": {"dtype": "U8"},
                },
            )
            config = SimpleNamespace(_name_or_path=tmpdir)
            modelopt = MockQuantConfig("modelopt_fp4")

            self.assertIs(
                qwen3_5._resolve_modelopt_fp4_submodule_quant_config(
                    config, modelopt, "linear_attn", 0
                ),
                modelopt,
            )

    def test_modelopt_fp4_policy_rejects_mixed_layer_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_safetensors_header(
                Path(tmpdir) / "model.safetensors",
                {
                    "model.layers.0.linear_attn.in_proj_b.weight": {"dtype": "BF16"},
                    "model.layers.0.linear_attn.in_proj_a.weight": {"dtype": "U8"},
                },
            )
            config = SimpleNamespace(_name_or_path=tmpdir)
            modelopt = MockQuantConfig("modelopt_fp4")

            with self.assertRaisesRegex(ValueError, "Unsupported mixed"):
                qwen3_5._resolve_modelopt_fp4_submodule_quant_config(
                    config, modelopt, "linear_attn", 0
                )

    def test_modelopt_fp4_mtp_packed_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_safetensors_header(
                Path(tmpdir) / "model.safetensors",
                {
                    "mtp.layers.0.self_attn.q_proj.weight": {"dtype": "U8"},
                },
            )
            config = SimpleNamespace(_name_or_path=tmpdir)
            modelopt = MockQuantConfig("modelopt_fp4")

            with self.assertRaisesRegex(ValueError, "packed MTP weights"):
                qwen3_5._resolve_modelopt_fp4_mtp_quant_config(config, modelopt)

    def test_modelopt_fp4_weight_load_squeezes_leading_singleton(self):
        param = ModelWeightParameter(
            data=torch.empty(4, 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *args, **kwargs: None,
        )
        param.is_modelopt_fp4_weight = True
        loaded_weight = torch.arange(16, dtype=torch.uint8).reshape(1, 8, 2)

        param.load_column_parallel_weight(loaded_weight, tp_rank=1)

        torch.testing.assert_close(param.data, loaded_weight.squeeze(0)[4:8])

    def test_conv1d_loader_accepts_checkpoint_singleton_dim(self):
        calls = []

        def loader(param, loaded_weight):
            del param
            calls.append(loaded_weight)

        wrapped_loader = Qwen3_5GatedDeltaNet._make_conv1d_weight_loader(loader)

        param = SimpleNamespace(data=torch.empty(5120, 4))
        loaded_weight = torch.empty(5120, 1, 4)
        wrapped_loader(param, loaded_weight)
        self.assertEqual(calls.pop().shape, torch.Size([5120, 4]))

        param = SimpleNamespace(data=torch.empty(5120, 1, 4))
        loaded_weight = torch.empty(5120, 4)
        wrapped_loader(param, loaded_weight)
        self.assertEqual(calls.pop().shape, torch.Size([5120, 1, 4]))


if __name__ == "__main__":
    unittest.main()

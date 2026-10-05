import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend
from sglang.srt.models.minicpm import MiniCPMAttention, MiniCPMSALAForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _IndexerModel(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_future_proj = nn.Module()
        layer.self_attn.q_future_proj.weight = nn.Parameter(torch.empty(2, 3))
        layer.self_attn.q_curr_proj = nn.Module()
        layer.self_attn.q_curr_proj.weight = nn.Parameter(torch.empty(2, 3))
        self.model.layers.append(layer)
        self.config = SimpleNamespace(
            sparda_enabled=True,
            sparda_indexer_path=str(checkpoint_path),
        )
        self._sparda_indexer_loaded = False


class TestMiniCPMSparda(CustomTestCase):
    def test_forecast_selection_is_reused_across_requests_in_one_forward(self):
        class _DecodeMode:
            def is_decode_or_idle(self):
                return True

        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.sparda_enabled = True
        backend.num_kv_heads = 1
        backend.head_dim = 4
        backend.forward_metadata = SimpleNamespace(sparse_bs_list=[0, 1])
        topk = torch.tensor([[[0, 1], [2, 3]]])
        calls = []

        def get_topk_for_sparse(**kwargs):
            calls.append(kwargs)
            return topk

        backend.get_topk_for_sparse = get_topk_for_sparse
        forward_batch = SimpleNamespace(
            batch_size=2,
            forward_mode=_DecodeMode(),
            model_specific_states={
                "sparda_request_generations": (4, 9),
                "sparda_request_context": (
                    SimpleNamespace(cache_salt="salt-a"),
                    SimpleNamespace(cache_salt="salt-b"),
                ),
            },
            seq_lens_cpu=[128, 256],
            rids=["request-a", "request-b"],
        )
        forecast = torch.zeros((2, 1, 4))

        first = backend.predict_sparda_blocks(forecast, forward_batch, 0, 3)
        second = backend.predict_sparda_blocks(forecast, forward_batch, 1, 3)

        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [2, 3])
        self.assertEqual(len(calls), 1)

        forward_batch.model_specific_states["sparda_request_generations"] = (4, 10)
        backend.predict_sparda_blocks(forecast, forward_batch, 1, 3)
        self.assertEqual(len(calls), 2)

    def test_forecast_projection_preserves_local_gqa_layout_under_tp(self):
        with get_parallel().override(tp_size=2, tp_rank=1):
            attention = MiniCPMAttention(
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                attn_use_rope=False,
                sparda_enabled=True,
            )
            forecast = attention._project_forecast(
                positions=torch.arange(3),
                hidden_states=torch.randn(3, 16),
            )

        self.assertEqual(tuple(attention.q_future_proj.weight.shape), (8, 16))
        self.assertEqual(tuple(attention.q_curr_proj.weight.shape), (8, 16))
        self.assertEqual(tuple(forecast.shape), (3, 1, 4))

    def test_current_selector_is_only_created_for_the_first_layer(self):
        with get_parallel().override(tp_size=1, tp_rank=0):
            attention = MiniCPMAttention(
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                layer_id=1,
                attn_use_rope=False,
                sparda_enabled=True,
            )

        self.assertIsNone(attention.q_curr_proj)

    def test_forecast_projection_applies_rope_before_reshaping(self):
        class _FlattenedRope(nn.Module):
            def forward(self, positions, query, key):
                if query.ndim != 2 or key.ndim != 2:
                    raise AssertionError("Forecast RoPE input must be flattened")
                return query, key

        with get_parallel().override(tp_size=1, tp_rank=0):
            attention = MiniCPMAttention(
                hidden_size=16,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                max_position_embeddings=32,
                attn_use_rope=True,
                sparda_enabled=True,
            )
            attention.rotary_emb = _FlattenedRope()
            with torch.no_grad():
                attention.q_future_proj.weight.fill_(0.01)
            forecast = attention._project_forecast(
                positions=torch.arange(3),
                hidden_states=torch.randn(3, 16),
            )

        self.assertEqual(tuple(forecast.shape), (3, 2, 4))
        self.assertTrue(torch.isfinite(forecast).all())

    def test_sparda_indexer_loads_nested_state_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "indexer.pt"
            torch.save(
                {
                    "state_dict": {
                        "model.layers.0.self_attn.q_future_proj.weight": torch.ones(
                            2, 3
                        ),
                        "model.layers.0.self_attn.q_curr_proj.weight": torch.ones(2, 3),
                    }
                },
                checkpoint_path,
            )
            model = _IndexerModel(checkpoint_path)

            MiniCPMSALAForCausalLM._load_sparda_indexer_weights(model)

            self.assertTrue(model._sparda_indexer_loaded)
            self.assertTrue(
                torch.equal(
                    model.model.layers[0].self_attn.q_future_proj.weight,
                    torch.ones(2, 3),
                )
            )

    def test_sparda_indexer_loads_training_metadata_with_safe_globals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "indexer.pt"
            torch.save(
                {
                    "step": 2,
                    "rng_state": np.array([1, 2], dtype=np.uint32),
                    "model_state_dict": {
                        "model.layers.0.self_attn.q_future_proj.weight": torch.ones(
                            2, 3
                        ),
                        "model.layers.0.self_attn.q_curr_proj.weight": torch.ones(2, 3),
                    },
                },
                checkpoint_path,
            )
            model = _IndexerModel(checkpoint_path)

            MiniCPMSALAForCausalLM._load_sparda_indexer_weights(model)

            self.assertTrue(model._sparda_indexer_loaded)

    def test_sparda_indexer_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "indexer.pt"
            torch.save(
                {
                    "state_dict": {
                        "model.layers.0.self_attn.q_future_proj.weight": torch.ones(
                            3, 3
                        ),
                        "model.layers.0.self_attn.q_curr_proj.weight": torch.ones(2, 3),
                    }
                },
                checkpoint_path,
            )
            model = _IndexerModel(checkpoint_path)

            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                MiniCPMSALAForCausalLM._load_sparda_indexer_weights(model)

    def test_sparda_indexer_rejects_missing_forecast_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "indexer.pt"
            torch.save({"state_dict": {}}, checkpoint_path)
            model = _IndexerModel(checkpoint_path)

            with self.assertRaisesRegex(ValueError, "missing Forecast weights"):
                MiniCPMSALAForCausalLM._load_sparda_indexer_weights(model)


if __name__ == "__main__":
    unittest.main()

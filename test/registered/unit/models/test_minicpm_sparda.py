import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

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
        self.model.layers.append(layer)
        self.config = SimpleNamespace(
            sparda_enabled=True,
            sparda_indexer_path=str(checkpoint_path),
        )
        self._sparda_indexer_loaded = False


class TestMiniCPMSparda(CustomTestCase):
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
        self.assertEqual(tuple(forecast.shape), (3, 1, 4))

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
                        )
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

    def test_sparda_indexer_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "indexer.pt"
            torch.save(
                {
                    "state_dict": {
                        "model.layers.0.self_attn.q_future_proj.weight": torch.ones(
                            3, 3
                        )
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

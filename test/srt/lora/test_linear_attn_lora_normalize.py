"""Unit tests for the linear-attention LoRA weight normalization."""

import unittest
from unittest.mock import MagicMock

import torch


class TestNormalizeInProjQkvz(unittest.TestCase):
    def _adapter(self):
        from sglang.srt.lora.lora import LoRAAdapter

        adapter = MagicMock(spec=LoRAAdapter)
        adapter.lora_backend = MagicMock()
        adapter.lora_backend.name = "triton"
        adapter.normalize_in_proj_qkvz = (
            LoRAAdapter.normalize_in_proj_qkvz.__get__(adapter, LoRAAdapter)
        )
        return adapter

    def test_tinker_4_tensor_split_is_fused(self):
        adapter = self._adapter()

        rank = 4
        hidden = 8
        key_dim = 6
        value_dim = 10
        weights = {
            "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_A.weight": (
                torch.full((rank, hidden), 1.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_k.lora_A.weight": (
                torch.full((rank, hidden), 2.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_A.weight": (
                torch.full((rank, hidden), 3.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_A.weight": (
                torch.full((rank, hidden), 4.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_q.lora_B.weight": (
                torch.full((key_dim, rank), 1.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_k.lora_B.weight": (
                torch.full((key_dim, rank), 2.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_v.lora_B.weight": (
                torch.full((value_dim, rank), 3.0)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_B.weight": (
                torch.full((value_dim, rank), 4.0)
            ),
        }

        adapter.normalize_in_proj_qkvz(list(weights.keys()), weights)

        for split_name in ("in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"):
            for ab in ("lora_A", "lora_B"):
                key = (
                    f"base_model.model.model.layers.0.linear_attn."
                    f"{split_name}.{ab}.weight"
                )
                self.assertNotIn(key, weights)

        fused_a = weights[
            "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight"
        ]
        fused_b = weights[
            "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_B.weight"
        ]
        self.assertEqual(fused_a.shape, (rank * 4, hidden))
        self.assertTrue(torch.equal(fused_a[:rank], torch.full((rank, hidden), 1.0)))
        self.assertTrue(
            torch.equal(fused_a[rank : rank * 2], torch.full((rank, hidden), 2.0))
        )
        self.assertTrue(
            torch.equal(fused_a[rank * 2 : rank * 3], torch.full((rank, hidden), 3.0))
        )
        self.assertTrue(
            torch.equal(fused_a[rank * 3 :], torch.full((rank, hidden), 4.0))
        )
        self.assertEqual(fused_b.shape, (2 * key_dim + 2 * value_dim, rank))
        self.assertTrue(torch.equal(fused_b[:key_dim], torch.full((key_dim, rank), 1.0)))
        self.assertTrue(
            torch.equal(
                fused_b[key_dim : 2 * key_dim], torch.full((key_dim, rank), 2.0)
            )
        )
        self.assertTrue(
            torch.equal(
                fused_b[2 * key_dim : 2 * key_dim + value_dim],
                torch.full((value_dim, rank), 3.0),
            )
        )
        self.assertTrue(
            torch.equal(
                fused_b[2 * key_dim + value_dim :],
                torch.full((value_dim, rank), 4.0),
            )
        )

    def test_already_fused_is_passthrough(self):
        adapter = self._adapter()

        weights = {
            "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight": (
                torch.ones(4, 8)
            ),
            "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_B.weight": (
                torch.ones(20, 4)
            ),
        }
        snapshot = {k: v.clone() for k, v in weights.items()}

        adapter.normalize_in_proj_qkvz(list(weights.keys()), weights)

        self.assertEqual(set(weights.keys()), set(snapshot.keys()))
        for key, value in snapshot.items():
            self.assertTrue(torch.equal(weights[key], value))

    def test_unrelated_weights_are_passthrough(self):
        adapter = self._adapter()

        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                torch.ones(4, 8)
            ),
        }

        adapter.normalize_in_proj_qkvz(list(weights.keys()), weights)
        self.assertEqual(
            set(weights.keys()),
            {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"},
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for LoRAAdapter weight normalization in srt/lora/lora.py.

Covers ``_normalize_in_proj_qkvz``, which stacks GDN (GatedDeltaNet)
input-projection LoRA weights into the fused ``in_proj_qkvz`` layout used by
the LoRA memory pool. Three adapter formats are exercised:

1. 2-way split (native HF/PEFT layout for Qwen3.5/3.6): ``in_proj_qkv``
   (fused q+k+v) + ``in_proj_z`` — previously unhandled and the root cause of
   the startup crash in issue #30168.
2. 4-way split: ``in_proj_q/k/v/z``.
3. Already-merged: a single ``in_proj_qkvz`` pair.

The tests are hermetic (CPU-only): ``LoRAAdapter`` is instantiated via
``__new__`` and only the fields ``_normalize_in_proj_qkvz`` reads are
populated, following test_mem_pool_ep_unit.py.

Dims follow the issue's Qwen3.6-35B-A3B repro: hidden=2048, per-head key
dim total 2048, value dim total 4096 → fused q|k|v|z output 12288.

Usage:
    python -m pytest test/registered/unit/lora/test_lora.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import types
import unittest

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.test.test_utils import CustomTestCase

HIDDEN = 2048
KEY_DIM = 2048  # q and k rows each
VALUE_DIM = 4096  # v and z rows each
QKV_ROWS = KEY_DIM * 2 + VALUE_DIM  # 8192
FUSED_ROWS = KEY_DIM * 2 + VALUE_DIM * 2  # 12288
RANK = 4
PREFIX = "base_model.model.model.layers.0.linear_attn"


def _make_adapter() -> LoRAAdapter:
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter.base_hf_config = types.SimpleNamespace()
    # Fake base model exposing only what the zero-synthesis helper needs.
    fake_base_model = types.SimpleNamespace(
        get_hidden_dim=lambda module_name, layer_idx: (HIDDEN, FUSED_ROWS)
    )
    # Same bypass as LoRAAdapter.__init__ (plain reference, not a submodule).
    object.__setattr__(adapter, "base_model", fake_base_model)
    return adapter


def _pair(module: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    return {
        f"{PREFIX}.{module}.lora_A.weight": a,
        f"{PREFIX}.{module}.lora_B.weight": b,
    }


def _qkvz_key(kind: str) -> str:
    return f"{PREFIX}.in_proj_qkvz.lora_{kind}.weight"


class TestNormalizeInProjQkvz(CustomTestCase):
    def setUp(self):
        self.adapter = _make_adapter()

    def test_two_way_split(self):
        """in_proj_qkv + in_proj_z (issue #30168) fuse into in_proj_qkvz."""
        a_qkv = torch.randn(RANK, HIDDEN)
        b_qkv = torch.randn(QKV_ROWS, RANK)
        a_z = torch.randn(RANK, HIDDEN)
        b_z = torch.randn(VALUE_DIM, RANK)
        weights = {
            **_pair("in_proj_qkv", a_qkv, b_qkv),
            **_pair("in_proj_z", a_z, b_z),
        }

        self.adapter._normalize_in_proj_qkvz(weights)

        self.assertEqual(set(weights), {_qkvz_key("A"), _qkvz_key("B")})
        a = weights[_qkvz_key("A")]
        b = weights[_qkvz_key("B")]
        self.assertEqual(a.shape, (4 * RANK, HIDDEN))
        # The shared qkv A covers the q, k, v slots; z has its own slot.
        for slot in range(3):
            torch.testing.assert_close(
                a[slot * RANK : (slot + 1) * RANK], a_qkv, rtol=0, atol=0
            )
        torch.testing.assert_close(a[3 * RANK :], a_z, rtol=0, atol=0)
        self.assertEqual(b.shape, (FUSED_ROWS, RANK))
        torch.testing.assert_close(b, torch.cat((b_qkv, b_z), 0), rtol=0, atol=0)

    def test_two_way_split_z_first_in_dict(self):
        """Fusion is insertion-order independent."""
        a_qkv = torch.randn(RANK, HIDDEN)
        b_qkv = torch.randn(QKV_ROWS, RANK)
        a_z = torch.randn(RANK, HIDDEN)
        b_z = torch.randn(VALUE_DIM, RANK)
        weights = {
            **_pair("in_proj_z", a_z, b_z),
            **_pair("in_proj_qkv", a_qkv, b_qkv),
        }

        self.adapter._normalize_in_proj_qkvz(weights)

        self.assertEqual(set(weights), {_qkvz_key("A"), _qkvz_key("B")})
        torch.testing.assert_close(
            weights[_qkvz_key("B")], torch.cat((b_qkv, b_z), 0), rtol=0, atol=0
        )

    def test_qkv_only_synthesizes_zero_z(self):
        a_qkv = torch.randn(RANK, HIDDEN)
        b_qkv = torch.randn(QKV_ROWS, RANK)
        weights = _pair("in_proj_qkv", a_qkv, b_qkv)

        self.adapter._normalize_in_proj_qkvz(weights)

        a = weights[_qkvz_key("A")]
        b = weights[_qkvz_key("B")]
        self.assertEqual(a.shape, (4 * RANK, HIDDEN))
        self.assertTrue(torch.all(a[3 * RANK :] == 0))
        self.assertEqual(b.shape, (FUSED_ROWS, RANK))
        torch.testing.assert_close(b[:QKV_ROWS], b_qkv, rtol=0, atol=0)
        self.assertTrue(torch.all(b[QKV_ROWS:] == 0))

    def test_z_only_synthesizes_zero_qkv(self):
        a_z = torch.randn(RANK, HIDDEN)
        b_z = torch.randn(VALUE_DIM, RANK)
        weights = _pair("in_proj_z", a_z, b_z)

        self.adapter._normalize_in_proj_qkvz(weights)

        a = weights[_qkvz_key("A")]
        b = weights[_qkvz_key("B")]
        self.assertEqual(a.shape, (4 * RANK, HIDDEN))
        self.assertTrue(torch.all(a[: 3 * RANK] == 0))
        torch.testing.assert_close(a[3 * RANK :], a_z, rtol=0, atol=0)
        self.assertEqual(b.shape, (FUSED_ROWS, RANK))
        self.assertTrue(torch.all(b[:QKV_ROWS] == 0))
        torch.testing.assert_close(b[QKV_ROWS:], b_z, rtol=0, atol=0)

    def test_four_way_split(self):
        a_parts = {m: torch.randn(RANK, HIDDEN) for m in ("q", "k", "v", "z")}
        b_rows = {"q": KEY_DIM, "k": KEY_DIM, "v": VALUE_DIM, "z": VALUE_DIM}
        b_parts = {m: torch.randn(rows, RANK) for m, rows in b_rows.items()}
        weights = {}
        for m in ("q", "k", "v", "z"):
            weights.update(_pair(f"in_proj_{m}", a_parts[m], b_parts[m]))

        self.adapter._normalize_in_proj_qkvz(weights)

        self.assertEqual(set(weights), {_qkvz_key("A"), _qkvz_key("B")})
        torch.testing.assert_close(
            weights[_qkvz_key("A")],
            torch.cat([a_parts[m] for m in ("q", "k", "v", "z")], 0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            weights[_qkvz_key("B")],
            torch.cat([b_parts[m] for m in ("q", "k", "v", "z")], 0),
            rtol=0,
            atol=0,
        )

    def test_already_merged(self):
        a = torch.randn(RANK, HIDDEN)
        b = torch.randn(FUSED_ROWS, RANK)
        weights = _pair("in_proj_qkvz", a, b)

        self.adapter._normalize_in_proj_qkvz(weights)

        # A is repeated exactly once across the 4 slots; B is untouched.
        self.assertEqual(weights[_qkvz_key("A")].shape, (4 * RANK, HIDDEN))
        torch.testing.assert_close(
            weights[_qkvz_key("A")], a.repeat(4, 1), rtol=0, atol=0
        )
        torch.testing.assert_close(weights[_qkvz_key("B")], b, rtol=0, atol=0)

    def test_unrelated_weights_untouched(self):
        weights = _pair("q_proj", torch.randn(RANK, HIDDEN), torch.randn(64, RANK))
        original = dict(weights)

        self.adapter._normalize_in_proj_qkvz(weights)

        self.assertEqual(set(weights), set(original))
        for name, tensor in original.items():
            torch.testing.assert_close(weights[name], tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

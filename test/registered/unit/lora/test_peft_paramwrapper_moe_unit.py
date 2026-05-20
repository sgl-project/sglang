"""Unit tests for ``LoRAAdapter._normalize_peft_paramwrapper_moe``.

Covers PEFT 0.18+ ``target_parameters`` (``ParamWrapper``) → SGLang 3D-per-expert
rewrite: name resolution (inner/outer slot → first/last listed param), reshape
semantics (c=1 and c=2 stacking, B-axis layout per PEFT source), gating against
non-PEFT-ParamWrapper adapters.

Usage:
    python test/registered/unit/lora/test_peft_paramwrapper_moe_unit.py
"""

from sglang.test.ci.ci_register import register_cuda_ci

# CPU-only unit test; no CUDA needed.
register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")

import types
import unittest

import torch

from sglang.srt.lora.lora import LoRAAdapter

# Standard LFM2-MoE-style adapter_config target_parameters (in user order).
_TPARAMS = ["experts.gate_up_proj", "experts.down_proj"]


def _make_adapter(
    target_parameters,
    *,
    rank: int = 8,
    num_experts: int = 32,
):
    """Build a bare LoRAAdapter-like stub with just the fields the normalizer reads.

    We bypass ``__init__`` so the test doesn't need to stand up a base model
    or a real LoRAConfig instance — both are heavy. The normalizer only reads
    ``self.config.hf_config["target_parameters"]``, ``self.config.r``, and
    ``self.base_hf_config.num_experts``.
    """
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter.config = types.SimpleNamespace(
        hf_config={"target_parameters": target_parameters},
        r=rank,
    )
    adapter.base_hf_config = types.SimpleNamespace(num_experts=num_experts)
    return adapter


class TestPeftParamWrapperRename(unittest.TestCase):
    """Name resolution: inner/outer slot → first/last listed target_parameter."""

    def test_inner_base_layer_maps_to_first_target(self):
        adapter = _make_adapter(_TPARAMS)
        name = (
            "base_model.model.model.layers.5.feed_forward.experts.base_layer"
            ".lora_A.default.weight"
        )
        weights = {name: torch.zeros(32 * 8, 2048, dtype=torch.bfloat16)}
        adapter._normalize_peft_paramwrapper_moe(weights)
        new_keys = list(weights.keys())
        self.assertEqual(len(new_keys), 1)
        self.assertTrue(new_keys[0].endswith(".experts.gate_up_proj.lora_A.weight"))

    def test_outer_no_base_layer_maps_to_last_target(self):
        adapter = _make_adapter(_TPARAMS)
        name = "x.layers.5.mlp.experts.lora_B.default.weight"
        weights = {name: torch.zeros(2048, 8 * 32, dtype=torch.bfloat16)}
        adapter._normalize_peft_paramwrapper_moe(weights)
        new_keys = list(weights.keys())
        self.assertEqual(len(new_keys), 1)
        self.assertTrue(new_keys[0].endswith(".experts.down_proj.lora_B.weight"))

    def test_user_supplied_order_is_honored(self):
        """If the user lists target_parameters reversed, the slot mapping flips."""
        adapter = _make_adapter(list(reversed(_TPARAMS)))
        inner = "x.experts.base_layer.lora_A.default.weight"
        outer = "x.experts.lora_A.default.weight"
        weights = {
            inner: torch.zeros(32 * 8, 1792, dtype=torch.bfloat16),
            outer: torch.zeros(32 * 8, 2048, dtype=torch.bfloat16),
        }
        adapter._normalize_peft_paramwrapper_moe(weights)
        keys = set(weights.keys())
        self.assertIn("x.experts.down_proj.lora_A.weight", keys)
        self.assertIn("x.experts.gate_up_proj.lora_A.weight", keys)


class TestPeftParamWrapperReshape(unittest.TestCase):
    """Reshape semantics — flat 2D → 3D-per-expert with c-tiling for stacked targets."""

    NUM_EXPERTS = 32
    RANK = 8
    HIDDEN = 2048
    MOE_INTERMEDIATE = 1792

    def test_gate_up_proj_lora_A_tiles_for_stacked_c2(self):
        adapter = _make_adapter(_TPARAMS)
        name = "x.experts.base_layer.lora_A.default.weight"
        weights = {
            name: torch.randn(
                self.NUM_EXPERTS * self.RANK, self.HIDDEN, dtype=torch.bfloat16
            )
        }
        adapter._normalize_peft_paramwrapper_moe(weights)
        out = next(iter(weights.values()))
        # c=2 → rank dim doubled; the two halves identical (PEFT's
        # fused-parameter LoRA shared across gate/up shards).
        self.assertEqual(
            tuple(out.shape), (self.NUM_EXPERTS, 2 * self.RANK, self.HIDDEN)
        )
        self.assertTrue(torch.equal(out[:, : self.RANK, :], out[:, self.RANK :, :]))

    def test_down_proj_lora_A_unchanged_for_c1(self):
        adapter = _make_adapter(_TPARAMS)
        name = "x.experts.lora_A.default.weight"
        weights = {
            name: torch.randn(
                self.NUM_EXPERTS * self.RANK,
                self.MOE_INTERMEDIATE,
                dtype=torch.bfloat16,
            )
        }
        adapter._normalize_peft_paramwrapper_moe(weights)
        out = next(iter(weights.values()))
        self.assertEqual(
            tuple(out.shape),
            (self.NUM_EXPERTS, self.RANK, self.MOE_INTERMEDIATE),
        )

    def test_lora_B_reshape_and_permute_preserves_per_expert(self):
        """Round-trip B: build PEFT's layout (experts FAST-varying inside r*E
        per ``B.reshape(out, rank, num_experts)`` in peft/tuners/lora/layer.py)
        then verify the reshape recovers each expert's per-expert ``(out, r)``.
        """
        adapter = _make_adapter(_TPARAMS, rank=4, num_experts=5)
        out_dim, rank, E = 64, 4, 5
        per_expert = [
            torch.full((out_dim, rank), float(i + 1), dtype=torch.bfloat16)
            for i in range(E)
        ]
        # Experts FAST-varying — matches PEFT's saved layout.
        flat = torch.stack(per_expert, dim=2).reshape(out_dim, rank * E)
        name = "x.experts.lora_B.default.weight"
        weights = {name: flat}
        adapter._normalize_peft_paramwrapper_moe(weights)
        out = next(iter(weights.values()))
        self.assertEqual(tuple(out.shape), (E, out_dim, rank))
        for i in range(E):
            self.assertTrue(torch.equal(out[i], per_expert[i]))


class TestPeftParamWrapperGating(unittest.TestCase):
    """The normalizer must be inert for non-PEFT-ParamWrapper adapters."""

    def test_no_target_parameters_is_noop(self):
        adapter = _make_adapter([])
        weights = {
            "x.q_proj.lora_A.weight": torch.zeros(8, 2048),
            "x.experts.0.gate_proj.lora_A.weight": torch.zeros(8, 2048),
        }
        snap = {k: v.clone() for k, v in weights.items()}
        adapter._normalize_peft_paramwrapper_moe(weights)
        self.assertEqual(set(weights.keys()), set(snap.keys()))
        for k, v in snap.items():
            self.assertTrue(torch.equal(weights[k], v))

    def test_target_parameters_none_is_noop(self):
        """Older PEFT writes ``"target_parameters": null`` — should be a no-op."""
        adapter = LoRAAdapter.__new__(LoRAAdapter)
        adapter.config = types.SimpleNamespace(
            hf_config={"target_parameters": None}, r=8
        )
        adapter.base_hf_config = types.SimpleNamespace(num_experts=32)
        weights = {"x.q_proj.lora_A.weight": torch.zeros(8, 2048)}
        adapter._normalize_peft_paramwrapper_moe(weights)
        self.assertIn("x.q_proj.lora_A.weight", weights)

    def test_per_expert_names_not_renamed(self):
        """Mixtral-classic per-expert names (``experts.0.gate_proj.lora_A.weight``)
        must NOT match the regex — they take SGLang's existing per-expert path."""
        adapter = _make_adapter(_TPARAMS)
        name = "x.layers.5.block_sparse_moe.experts.0.gate_proj.lora_A.weight"
        weights = {name: torch.zeros(8, 2048)}
        adapter._normalize_peft_paramwrapper_moe(weights)
        self.assertIn(name, weights)

    def test_already_converted_3d_names_not_renamed(self):
        """Already-converted names (``experts.gate_up_proj.lora_A.weight``)
        must NOT match (no ``base_layer`` slot, and a leaf module between
        ``experts.`` and ``lora_``)."""
        adapter = _make_adapter(_TPARAMS)
        name = "x.experts.gate_up_proj.lora_A.weight"
        weights = {name: torch.zeros(32, 8, 2048)}  # already 3D
        adapter._normalize_peft_paramwrapper_moe(weights)
        self.assertIn(name, weights)

    def test_non_moe_names_untouched(self):
        adapter = _make_adapter(_TPARAMS)
        weights = {
            "x.q_proj.lora_A.weight": torch.zeros(8, 2048),
            "x.gate_up_proj.lora_B.weight": torch.zeros(2048, 8),
        }
        snap = {k: v.clone() for k, v in weights.items()}
        adapter._normalize_peft_paramwrapper_moe(weights)
        self.assertEqual(set(weights.keys()), set(snap.keys()))

    def test_more_than_two_target_parameters_rejected(self):
        adapter = _make_adapter(_TPARAMS + ["experts.gate"])
        weights = {"x.experts.base_layer.lora_A.weight": torch.zeros(32 * 8, 2048)}
        with self.assertRaises(ValueError):
            adapter._normalize_peft_paramwrapper_moe(weights)


if __name__ == "__main__":
    unittest.main()

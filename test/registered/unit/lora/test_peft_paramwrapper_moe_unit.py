"""Unit tests for ``LoRAAdapter._normalize_peft_paramwrapper_moe``.

Covers PEFT 0.18+ ``target_parameters`` (``ParamWrapper``) → SGLang 3D-per-expert
rewrite: shape-based slot resolution (PEFT's inner/outer mapping comes from the
HF module class's attribute-definition order, NOT the user's
``target_parameters`` list order), reshape semantics (c=1 and c=2 stacking,
B-axis layout per PEFT source), gating against non-PEFT-ParamWrapper adapters.

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
    hidden_size: int = 2048,
    moe_intermediate_size: int = 1792,
    intermediate_size: int = 7168,
):
    """Build a bare LoRAAdapter-like stub with just the fields the normalizer reads.

    We bypass ``__init__`` so the test doesn't need to stand up a base model
    or a real LoRAConfig instance — both are heavy. The normalizer only reads
    ``self.config.hf_config["target_parameters"]``, ``self.config.r``, and
    the MoE dims off ``self.base_hf_config``.
    """
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter.config = types.SimpleNamespace(
        hf_config={"target_parameters": target_parameters},
        r=rank,
    )
    adapter.base_hf_config = types.SimpleNamespace(
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        intermediate_size=intermediate_size,
    )
    return adapter


def _ab_pair(prefix: str, *, inner: bool, rank: int, num_experts: int, in_dim: int, out_dim: int):
    """Build {A_name: tensor, B_name: tensor} in PEFT's saved layout for one
    wrapper slot — inner uses ``base_layer``, outer doesn't."""
    middle = "base_layer." if inner else ""
    a_name = f"{prefix}.experts.{middle}lora_A.weight"
    b_name = f"{prefix}.experts.{middle}lora_B.weight"
    return {
        a_name: torch.zeros(num_experts * rank, in_dim, dtype=torch.bfloat16),
        b_name: torch.zeros(out_dim, rank * num_experts, dtype=torch.bfloat16),
    }


class TestPeftParamWrapperRename(unittest.TestCase):
    """Slot → leaf resolution is shape-based, NOT user-order-based."""

    HIDDEN = 2048
    MOE_INTER = 1792
    NUM_EXPERTS = 32
    RANK = 8

    def test_inner_with_hidden_in_dim_maps_to_gate_up_proj(self):
        """Inner ``base_layer`` slot with A-in == hidden → gate_up_proj
        (matches both LFM2-MoE and Qwen3-MoE PEFT save layouts)."""
        adapter = _make_adapter(_TPARAMS)
        weights = _ab_pair(
            "x",
            inner=True,
            rank=self.RANK,
            num_experts=self.NUM_EXPERTS,
            in_dim=self.HIDDEN,
            out_dim=2 * self.MOE_INTER,
        )
        adapter._normalize_peft_paramwrapper_moe(weights)
        keys = set(weights.keys())
        self.assertIn("x.experts.gate_up_proj.lora_A.weight", keys)
        self.assertIn("x.experts.gate_up_proj.lora_B.weight", keys)

    def test_outer_with_moe_intermediate_in_dim_maps_to_down_proj(self):
        """Outer slot with A-in == moe_intermediate_size → down_proj."""
        adapter = _make_adapter(_TPARAMS)
        weights = _ab_pair(
            "x",
            inner=False,
            rank=self.RANK,
            num_experts=self.NUM_EXPERTS,
            in_dim=self.MOE_INTER,
            out_dim=self.HIDDEN,
        )
        adapter._normalize_peft_paramwrapper_moe(weights)
        keys = set(weights.keys())
        self.assertIn("x.experts.down_proj.lora_A.weight", keys)
        self.assertIn("x.experts.down_proj.lora_B.weight", keys)

    def test_user_supplied_order_does_not_affect_mapping(self):
        """Regression: PEFT writes inner=gate_up_proj / outer=down_proj
        regardless of the user's ``target_parameters`` list order — both LFM2
        (``[gate_up, down]``) and Qwen3 (``[down, gate_up]``) demo adapters
        ship with the same on-disk slot layout. We must resolve by shape, not
        by index, so the same adapter loads the same way under either order.
        """
        weights_a = {}
        weights_a.update(
            _ab_pair(
                "x",
                inner=True,
                rank=self.RANK,
                num_experts=self.NUM_EXPERTS,
                in_dim=self.HIDDEN,
                out_dim=2 * self.MOE_INTER,
            )
        )
        weights_a.update(
            _ab_pair(
                "x",
                inner=False,
                rank=self.RANK,
                num_experts=self.NUM_EXPERTS,
                in_dim=self.MOE_INTER,
                out_dim=self.HIDDEN,
            )
        )
        weights_b = {k: v.clone() for k, v in weights_a.items()}

        adapter_lfm = _make_adapter(_TPARAMS)
        adapter_qwen = _make_adapter(list(reversed(_TPARAMS)))
        adapter_lfm._normalize_peft_paramwrapper_moe(weights_a)
        adapter_qwen._normalize_peft_paramwrapper_moe(weights_b)

        self.assertEqual(set(weights_a.keys()), set(weights_b.keys()))
        self.assertIn("x.experts.gate_up_proj.lora_A.weight", weights_a)
        self.assertIn("x.experts.down_proj.lora_A.weight", weights_a)


class TestPeftParamWrapperReshape(unittest.TestCase):
    """Reshape semantics — flat 2D → 3D-per-expert with c-tiling for stacked targets."""

    NUM_EXPERTS = 32
    RANK = 8
    HIDDEN = 2048
    MOE_INTERMEDIATE = 1792

    def test_gate_up_proj_lora_A_tiles_for_stacked_c2(self):
        adapter = _make_adapter(_TPARAMS)
        weights = _ab_pair(
            "x",
            inner=True,
            rank=self.RANK,
            num_experts=self.NUM_EXPERTS,
            in_dim=self.HIDDEN,
            out_dim=2 * self.MOE_INTERMEDIATE,
        )
        a_in = next(
            v
            for k, v in weights.items()
            if k.endswith(".base_layer.lora_A.weight")
        )
        # Replace with randn so the tile-equality check is meaningful.
        a_in.copy_(torch.randn_like(a_in))
        adapter._normalize_peft_paramwrapper_moe(weights)
        out = weights["x.experts.gate_up_proj.lora_A.weight"]
        # c=2 → rank dim doubled; the two halves identical (PEFT's
        # fused-parameter LoRA shared across gate/up shards).
        self.assertEqual(
            tuple(out.shape), (self.NUM_EXPERTS, 2 * self.RANK, self.HIDDEN)
        )
        self.assertTrue(torch.equal(out[:, : self.RANK, :], out[:, self.RANK :, :]))

    def test_down_proj_lora_A_unchanged_for_c1(self):
        adapter = _make_adapter(_TPARAMS)
        weights = _ab_pair(
            "x",
            inner=False,
            rank=self.RANK,
            num_experts=self.NUM_EXPERTS,
            in_dim=self.MOE_INTERMEDIATE,
            out_dim=self.HIDDEN,
        )
        adapter._normalize_peft_paramwrapper_moe(weights)
        out = weights["x.experts.down_proj.lora_A.weight"]
        self.assertEqual(
            tuple(out.shape),
            (self.NUM_EXPERTS, self.RANK, self.MOE_INTERMEDIATE),
        )

    def test_lora_B_reshape_and_permute_preserves_per_expert(self):
        """Round-trip B: build PEFT's layout (experts FAST-varying inside r*E
        per ``B.reshape(out, rank, num_experts)`` in peft/tuners/lora/layer.py)
        then verify the reshape recovers each expert's per-expert ``(out, r)``.

        We pair the B with a matching A (with in_dim == moe_intermediate to
        select the down_proj branch).
        """
        adapter = _make_adapter(
            _TPARAMS,
            rank=4,
            num_experts=5,
            hidden_size=64,
            moe_intermediate_size=37,
            intermediate_size=128,
        )
        out_dim, rank, E = 64, 4, 5
        per_expert = [
            torch.full((out_dim, rank), float(i + 1), dtype=torch.bfloat16)
            for i in range(E)
        ]
        # Experts FAST-varying — matches PEFT's saved layout.
        flat = torch.stack(per_expert, dim=2).reshape(out_dim, rank * E)
        a_in_dim = 37  # moe_intermediate_size → down_proj branch
        weights = {
            "x.experts.lora_A.weight": torch.zeros(
                E * rank, a_in_dim, dtype=torch.bfloat16
            ),
            "x.experts.lora_B.weight": flat,
        }
        adapter._normalize_peft_paramwrapper_moe(weights)
        out = weights["x.experts.down_proj.lora_B.weight"]
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
        adapter.base_hf_config = types.SimpleNamespace(
            num_experts=32,
            hidden_size=2048,
            moe_intermediate_size=1792,
            intermediate_size=7168,
        )
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

    def test_orphan_lora_B_without_matching_A_skipped(self):
        """If lora_B has no sibling lora_A (shape-resolve cannot run),
        leave the key alone for the loader to flag."""
        adapter = _make_adapter(_TPARAMS)
        name = "x.experts.lora_B.weight"
        weights = {name: torch.zeros(2048, 8 * 32, dtype=torch.bfloat16)}
        adapter._normalize_peft_paramwrapper_moe(weights)
        self.assertIn(name, weights)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for granitemoe_load_split_experts.

Compressed-tensors checkpoints (llmcompressor) store MoE experts one tensor per
expert per projection -- `experts.<id>.{gate,up,down}_proj.{weight,weight_scale}`
-- while the unquantized HF checkpoint packs them into `input_linear` /
`output_linear`. Only the packed layout was recognised, so every split expert
tensor fell through to a `logger.warning(...not found in params_dict)` and was
silently dropped: the server started normally and then emitted garbage
("capital capital capital..." instead of " Paris.").

Regression coverage for that silent drop. The guarded failure modes are:
a split expert tensor being passed through unloaded (the bug), a scale losing its
`weight_scale` suffix and being loaded as if it were the weight, a packed-layout
w1/w2/w3 name being wrongly claimed here instead of downstream, and an
unrecognised expert tensor being dropped rather than raising.

The two helpers are covered directly as well as through the loader, because each
decides which of those branches a tensor takes: `_match_split_expert` must return
None for anything it does not own, and `_is_packed_expert` must match a shard only
as a whole `.w1.`-style segment -- relaxing it to a bare substring test silently
sends an unknown expert tensor down the pass-through path instead of raising.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.models.granitemoe import (
    _is_packed_expert,
    _match_split_expert,
    granitemoe_load_split_experts,
)
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
PREFIX = "model.layers.0.block_sparse_moe"


class _RecordingParam:
    """Stands in for a FusedMoE parameter, recording loader invocations.

    Deliberately does NOT accept `return_success`: no loader in the tree does,
    so a caller passing it would raise TypeError here.
    """

    def __init__(self, calls):
        self._calls = calls

    def weight_loader(self, param, loaded_weight, name, shard_id, expert_id):
        self._calls.append(
            {
                "name": name,
                "shard_id": shard_id,
                "expert_id": expert_id,
                "value": loaded_weight,
            }
        )


def _mapping(num_experts=NUM_EXPERTS):
    return FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )


def _run(weights, params_dict=None, calls=None):
    """Drive the loader, returning (passed_through, calls)."""
    calls = [] if calls is None else calls
    if params_dict is None:
        params_dict = _AutoParams(calls)
    passed_through = list(
        granitemoe_load_split_experts(
            weights,
            expert_params_mapping=_mapping(),
            params_dict=params_dict,
        )
    )
    return passed_through, calls


class _AutoParams(dict):
    """params_dict that materialises a recording param for any requested name."""

    def __init__(self, calls):
        super().__init__()
        self._calls = calls

    def __getitem__(self, key):
        if key not in self:
            super().__setitem__(key, _RecordingParam(self._calls))
        return super().__getitem__(key)


class TestGraniteMoeLoadSplitExperts(CustomTestCase):
    def test_split_experts_are_loaded_not_passed_through(self):
        """The bug: these tensors reached the generic path and were dropped."""
        weights = [
            (f"{PREFIX}.experts.3.gate_proj.weight", torch.zeros(2)),
            (f"{PREFIX}.experts.3.up_proj.weight", torch.zeros(2)),
            (f"{PREFIX}.experts.1.down_proj.weight", torch.zeros(2)),
        ]
        passed_through, calls = _run(weights)

        self.assertEqual(passed_through, [], "split experts must not fall through")
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [(c["name"], c["shard_id"], c["expert_id"]) for c in calls],
            [
                (f"{PREFIX}.experts.w13_weight", "w1", 3),
                (f"{PREFIX}.experts.w13_weight", "w3", 3),
                (f"{PREFIX}.experts.w2_weight", "w2", 1),
            ],
        )

    def test_scale_keeps_its_suffix(self):
        """FusedMoE's loader dispatches on substrings of the name it is handed,
        so a scale must arrive as `*_weight_scale`. Handing it `*_weight` would
        load the scale as if it were the weight."""
        weights = [
            (f"{PREFIX}.experts.0.gate_proj.weight_scale", torch.zeros(1)),
            (f"{PREFIX}.experts.0.down_proj.weight_scale", torch.zeros(1)),
        ]
        _, calls = _run(weights)

        self.assertEqual(
            [c["name"] for c in calls],
            [
                f"{PREFIX}.experts.w13_weight_scale",
                f"{PREFIX}.experts.w2_weight_scale",
            ],
        )

    def test_every_expert_and_projection_is_loaded(self):
        """A missing (expert, projection) pair is the silent-drop bug: nothing
        matches and those weights never reach the layer."""
        weights = [
            (f"{PREFIX}.experts.{e}.{proj}.{suffix}", torch.zeros(1))
            for e in range(NUM_EXPERTS)
            for proj in ("gate_proj", "up_proj", "down_proj")
            for suffix in ("weight", "weight_scale")
        ]
        passed_through, calls = _run(weights)

        self.assertEqual(passed_through, [])
        self.assertEqual(len(calls), len(weights))
        self.assertEqual(
            {(c["expert_id"], c["shard_id"]) for c in calls},
            {(e, s) for e in range(NUM_EXPERTS) for s in ("w1", "w2", "w3")},
        )

    def test_non_expert_tensors_pass_through_untouched(self):
        """Attention, router, norm and lm_head must reach the generic path."""
        weights = [
            ("model.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
            ("model.layers.0.self_attn.q_proj.weight_scale", torch.zeros(1)),
            (f"{PREFIX}.router.layer.weight", torch.zeros(1)),
            ("model.layers.0.input_layernorm.weight", torch.zeros(1)),
            ("lm_head.weight", torch.zeros(1)),
        ]
        passed_through, calls = _run(weights)

        self.assertEqual([n for n, _ in passed_through], [n for n, _ in weights])
        self.assertEqual(calls, [])

    def test_packed_layout_experts_pass_through(self):
        """The packed (unquantized) path pre-splits into w1/w2/w3, which load
        downstream. Claiming or rejecting them here breaks the bf16 model."""
        weights = [
            (f"{PREFIX}.experts.0.{shard}.weight", torch.zeros(1))
            for shard in ("w1", "w2", "w3")
        ]
        passed_through, calls = _run(weights)

        self.assertEqual([n for n, _ in passed_through], [n for n, _ in weights])
        self.assertEqual(calls, [])

    def test_unrecognised_expert_tensor_raises(self):
        """The original failure was silent. An expert tensor that matches no
        mapping must now fail the load instead of producing a garbage model."""
        weights = [(f"{PREFIX}.experts.0.mystery_proj.weight", torch.zeros(1))]
        with self.assertRaises(ValueError) as ctx:
            _run(weights)
        self.assertIn("unmatched MoE expert tensor", str(ctx.exception))


class TestMatchSplitExpert(CustomTestCase):
    """Direct coverage for the name -> (param, expert, shard) resolution."""

    def setUp(self):
        self.mapping = _mapping()

    def test_returns_none_when_nothing_matches(self):
        """None is what tells the caller to fall through or raise; a predicate
        that degraded to always-matching would break both branches."""
        self.assertIsNone(
            _match_split_expert("model.layers.0.self_attn.q_proj.weight", self.mapping)
        )

    def test_projection_maps_to_expected_shard(self):
        for proj, expected_param, expected_shard in (
            ("gate_proj", f"{PREFIX}.experts.w13_weight", "w1"),
            ("up_proj", f"{PREFIX}.experts.w13_weight", "w3"),
            ("down_proj", f"{PREFIX}.experts.w2_weight", "w2"),
        ):
            with self.subTest(proj=proj):
                mapped, expert_id, shard_id = _match_split_expert(
                    f"{PREFIX}.experts.2.{proj}.weight", self.mapping
                )
                self.assertEqual(mapped, expected_param)
                self.assertEqual(expert_id, 2)
                self.assertEqual(shard_id, expected_shard)

    def test_empty_mapping_matches_nothing(self):
        self.assertIsNone(
            _match_split_expert(f"{PREFIX}.experts.0.gate_proj.weight", [])
        )


class TestIsPackedExpert(CustomTestCase):
    """Direct coverage for the packed-layout (w1/w2/w3) predicate.

    This predicate decides whether an unmatched expert tensor is passed through
    or raises, so a false positive silently reintroduces the dropped-expert bug.
    """

    def test_packed_shard_names_are_recognized(self):
        for shard in ("w1", "w2", "w3"):
            with self.subTest(shard=shard):
                self.assertTrue(_is_packed_expert(f"{PREFIX}.experts.0.{shard}.weight"))

    def test_split_projection_names_are_not_packed(self):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            with self.subTest(proj=proj):
                self.assertFalse(_is_packed_expert(f"{PREFIX}.experts.0.{proj}.weight"))

    def test_unknown_projection_is_not_packed(self):
        """The name that must reach the raise rather than pass through."""
        self.assertFalse(_is_packed_expert(f"{PREFIX}.experts.0.mystery_proj.weight"))

    def test_shard_substring_requires_dot_delimiters(self):
        """`w1` appearing inside a longer segment is not a packed shard. Without
        the dots this predicate would swallow such a tensor as packed and the
        loader would drop it silently instead of raising."""
        for name in (
            f"{PREFIX}.experts.0.w1_proj.weight",
            f"{PREFIX}.experts.0.gate_w2.weight",
            f"{PREFIX}.experts.0.w3x.weight",
        ):
            with self.subTest(name=name):
                self.assertFalse(_is_packed_expert(name))


if __name__ == "__main__":
    unittest.main()

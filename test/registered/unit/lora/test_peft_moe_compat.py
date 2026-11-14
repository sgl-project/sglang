"""CPU contract tests for the isolated PEFT MoE checkpoint normalizer.

The numerical oracle indexes the saved 2D factors directly, rather than
reproducing the implementation's reshape/permute operations.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.peft_moe import normalize_peft_moe_weights as normalize
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, R, H, I = 3, 2, 5, 3
PREFIX = "base_model.model.model.language_model.layers.0.mlp.experts"
TARGETS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def model_config(**overrides):
    return SimpleNamespace(
        **{
            "num_experts": E,
            "hidden_size": H,
            "moe_intermediate_size": I,
            **overrides,
        }
    )


def adapter_config(**overrides):
    return {
        "r": R,
        "lora_alpha": 16,
        "target_parameters": TARGETS.copy(),
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": True,
        **overrides,
    }


def raw_pair(
    leaf: str,
    *,
    legacy: bool = False,
    inner: bool = False,
    suffix: str = "",
    hidden: int = H,
    intermediate: int = I,
):
    in_dim, out_dim = (
        (hidden, 2 * intermediate) if leaf == "gate_up_proj" else (intermediate, hidden)
    )
    a_dim, b_dim = (out_dim, in_dim) if legacy else (in_dim, out_dim)
    # Distinct expert/rank/output/input values expose permutations that a
    # zeros fixture or expert-constant fixture cannot detect.
    a = torch.arange(E * R * a_dim, dtype=torch.float64).reshape(E * R, a_dim)
    b = torch.arange(b_dim * R * E, dtype=torch.float64).reshape(b_dim, R * E)
    a = a / 17 - 2
    b = b / 13 + 0.25
    wrapper = ".base_layer" if inner else ""
    base = PREFIX + wrapper
    return {
        f"{base}.lora_A{suffix}.weight": a,
        f"{base}.lora_B{suffix}.weight": b,
    }


def indexed_delta(pair, leaf: str, *, legacy: bool):
    """PEFT delta expressed directly in terms of saved scalar indices."""
    a = next(value for key, value in pair.items() if ".lora_A" in key)
    b = next(value for key, value in pair.items() if ".lora_B" in key)
    in_dim, out_dim = (H, 2 * I) if leaf == "gate_up_proj" else (I, H)
    delta = torch.empty(E, out_dim, in_dim, dtype=torch.float64)
    for expert in range(E):
        for output in range(out_dim):
            for input_ in range(in_dim):
                if legacy:
                    value = sum(
                        a[expert * R + rank, output] * b[input_, rank * E + expert]
                        for rank in range(R)
                    )
                else:
                    value = sum(
                        b[output, rank * E + expert] * a[expert * R + rank, input_]
                        for rank in range(R)
                    )
                delta[expert, output, input_] = value
    return delta


class NumericalCompatibilityTests(CustomTestCase):
    def assert_projection(self, weights, pair, leaf, *, legacy, suffix=""):
        a = weights[f"{PREFIX}.{leaf}.lora_A{suffix}.weight"]
        b = weights[f"{PREFIX}.{leaf}.lora_B{suffix}.weight"]
        expected_in, expected_out = (H, 2 * I) if leaf == "gate_up_proj" else (I, H)
        # Helper MUST NOT pre-stack gate/up A: the existing SGLang
        # normalize_gate_up_proj stage owns that operation.
        self.assertEqual(tuple(a.shape), (E, R, expected_in))
        self.assertEqual(tuple(b.shape), (E, expected_out, R))
        self.assertEqual(a.dtype, torch.float64)
        self.assertEqual(b.dtype, torch.float64)
        expected = indexed_delta(pair, leaf, legacy=legacy)
        actual = torch.bmm(b, a)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        if leaf == "gate_up_proj":
            torch.testing.assert_close(
                torch.bmm(b[:, :I], a), expected[:, :I], rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                torch.bmm(b[:, I:], a), expected[:, I:], rtol=1e-12, atol=1e-12
            )

    def test_all_layouts_and_wrapper_orders_preserve_each_expert_delta(self):
        for legacy in (False, True):
            for gate_inner in (False, True):
                with self.subTest(legacy=legacy, gate_inner=gate_inner):
                    gate = raw_pair("gate_up_proj", legacy=legacy, inner=gate_inner)
                    down = raw_pair("down_proj", legacy=legacy, inner=not gate_inner)
                    weights = {**gate, **down}
                    normalize(weights, model_config(), adapter_config())
                    self.assertEqual(len(weights), 4)
                    self.assert_projection(weights, gate, "gate_up_proj", legacy=legacy)
                    self.assert_projection(weights, down, "down_proj", legacy=legacy)

    def test_reverse_config_and_tensor_order_with_named_adapter(self):
        gate = raw_pair("gate_up_proj", inner=True, suffix=".default")
        down = raw_pair("down_proj", suffix=".default")
        weights = dict(reversed(list({**gate, **down}.items())))
        normalize(
            weights,
            model_config(),
            adapter_config(target_parameters=list(reversed(TARGETS))),
        )
        self.assert_projection(
            weights, gate, "gate_up_proj", legacy=False, suffix=".default"
        )
        self.assert_projection(
            weights, down, "down_proj", legacy=False, suffix=".default"
        )
        self.assertTrue(all(".default." in key for key in weights))

    def test_get_text_config_uses_text_dimensions(self):
        cfg = SimpleNamespace(
            num_experts=999,
            hidden_size=999,
            moe_intermediate_size=999,
            get_text_config=lambda: model_config(),
        )
        pair = raw_pair("gate_up_proj", inner=True)
        weights = pair.copy()
        normalize(weights, cfg, adapter_config())
        self.assert_projection(weights, pair, "gate_up_proj", legacy=False)

    def test_single_target_parameter_does_not_require_other_projection(self):
        pair = raw_pair("down_proj")
        weights = pair.copy()
        normalize(
            weights,
            model_config(),
            adapter_config(target_parameters=["mlp.experts.down_proj"]),
        )
        self.assert_projection(weights, pair, "down_proj", legacy=False)

    def test_b_shape_disambiguates_gate_when_hidden_equals_intermediate(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                weights = raw_pair("gate_up_proj", legacy=legacy, intermediate=H)
                normalize(
                    weights, model_config(moe_intermediate_size=H), adapter_config()
                )
                self.assertEqual(
                    tuple(weights[f"{PREFIX}.gate_up_proj.lora_A.weight"].shape),
                    (E, R, H),
                )
                self.assertEqual(
                    tuple(weights[f"{PREFIX}.gate_up_proj.lora_B.weight"].shape),
                    (E, 2 * H, R),
                )

    def test_nested_dictionary_text_config_and_num_local_experts(self):
        pair = raw_pair("down_proj")
        weights = pair.copy()
        normalize(
            weights,
            {
                "hidden_size": 999,
                "text_config": {
                    "num_local_experts": E,
                    "hidden_size": H,
                    "moe_intermediate_size": I,
                },
            },
            adapter_config(),
        )
        self.assert_projection(weights, pair, "down_proj", legacy=False)

    def test_other_declared_targets_do_not_change_present_pair(self):
        pair = raw_pair("down_proj", legacy=True)
        weights = pair.copy()
        normalize(
            weights,
            model_config(),
            adapter_config(target_parameters=TARGETS + ["mlp.experts.router"]),
        )
        self.assert_projection(weights, pair, "down_proj", legacy=True)

    def test_short_and_fully_qualified_parameter_suffixes(self):
        for target in ("experts.down_proj", f"{PREFIX}.down_proj"):
            with self.subTest(target=target):
                pair = raw_pair("down_proj")
                weights = pair.copy()
                normalize(
                    weights,
                    model_config(),
                    adapter_config(target_parameters=[target]),
                )
                self.assert_projection(weights, pair, "down_proj", legacy=False)

    def test_rs_lora_flag_does_not_bake_scaling_into_normalized_weights(self):
        pair = raw_pair("down_proj", legacy=True)
        outputs = []
        for use_rslora in (False, True):
            weights = pair.copy()
            normalize(
                weights,
                model_config(),
                adapter_config(use_rslora=use_rslora),
            )
            self.assert_projection(weights, pair, "down_proj", legacy=True)
            outputs.append(weights)
        for name in outputs[0]:
            torch.testing.assert_close(outputs[0][name], outputs[1][name])

    def test_unrelated_weights_retain_identity_alongside_converted_weights(self):
        weights = raw_pair("down_proj")
        name = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        untouched = torch.randn(R, H)
        weights[name] = untouched
        normalize(weights, model_config(), adapter_config())
        self.assertIs(weights[name], untouched)


class ValidationTests(CustomTestCase):
    def assert_atomic_error(self, weights, *, base=None, config=None):
        original = weights.copy()
        values = {name: value.clone() for name, value in weights.items()}
        with self.assertRaises(ValueError):
            normalize(
                weights,
                model_config() if base is None else base,
                adapter_config() if config is None else config,
            )
        self.assertEqual(list(weights), list(original))
        for name, value in original.items():
            self.assertIs(weights[name], value, name)
            torch.testing.assert_close(weights[name], values[name], rtol=0, atol=0)

    def test_orphan_a_and_orphan_b_fail_without_converting_valid_pair(self):
        for missing in ("lora_A", "lora_B"):
            with self.subTest(missing=missing):
                weights = raw_pair("gate_up_proj", inner=True)
                weights.update(
                    {
                        name: value
                        for name, value in raw_pair("down_proj").items()
                        if missing not in name
                    }
                )
                self.assert_atomic_error(weights)

    def test_bad_matrix_shapes_fail_atomically(self):
        for side, replacement in (
            ("A", torch.zeros(E * R + 1, I)),
            ("A", torch.zeros(E * R, I + 20)),
            ("A", torch.zeros(E * R)),
            ("A", torch.zeros(E, R, I)),
            ("B", torch.zeros(H, R * E + 1)),
            ("B", torch.zeros(H + 20, R * E)),
            ("B", torch.zeros(H)),
            ("B", torch.zeros(E, H, R)),
        ):
            with self.subTest(side=side, shape=tuple(replacement.shape)):
                weights = raw_pair("gate_up_proj", inner=True)
                down = raw_pair("down_proj")
                key = next(name for name in down if f"lora_{side}" in name)
                down[key] = replacement
                weights.update(down)
                self.assert_atomic_error(weights)

    def test_existing_destination_collision_does_not_overwrite(self):
        weights = raw_pair("down_proj")
        weights[f"{PREFIX}.down_proj.lora_A.weight"] = torch.randn(E, R, I)
        self.assert_atomic_error(weights)

    def test_two_wrappers_resolving_to_same_projection_do_not_overwrite(self):
        weights = raw_pair("down_proj", inner=True)
        weights.update(raw_pair("down_proj"))
        self.assert_atomic_error(weights)

    def test_unsupported_rank_and_alpha_patterns_fail_atomically(self):
        for option in ("rank_pattern", "alpha_pattern"):
            with self.subTest(option=option):
                self.assert_atomic_error(
                    raw_pair("down_proj"),
                    config=adapter_config(**{option: {"mlp.experts.down_proj": 4}}),
                )

    def test_unsupported_or_nonmatching_target_parameters_fail(self):
        for targets in (
            [],
            None,
            "mlp.experts.*",
            ["mlp.experts.router"],
            ["feed_forward.experts.down_proj"],
            ["mlp.expert.down_proj"],
        ):
            with self.subTest(targets=targets):
                self.assert_atomic_error(
                    raw_pair("down_proj"),
                    config=adapter_config(target_parameters=targets),
                )

    def test_bad_rank_or_missing_model_dimensions_fail(self):
        for rank in (0, -1):
            with self.subTest(rank=rank):
                self.assert_atomic_error(
                    raw_pair("down_proj"), config=adapter_config(r=rank)
                )
        for omitted in ("num_experts", "hidden_size", "moe_intermediate_size"):
            with self.subTest(omitted=omitted):
                cfg = model_config()
                delattr(cfg, omitted)
                self.assert_atomic_error(raw_pair("down_proj"), base=cfg)

    def test_square_projection_ambiguous_orientation_is_rejected(self):
        for leaf, hidden, intermediate in (
            ("down_proj", H, H),
            ("gate_up_proj", 2 * I, I),
        ):
            with self.subTest(leaf=leaf):
                weights = raw_pair(leaf, hidden=hidden, intermediate=intermediate)
                self.assert_atomic_error(
                    weights,
                    base=model_config(
                        hidden_size=hidden, moe_intermediate_size=intermediate
                    ),
                )


class ExistingFormatTests(CustomTestCase):
    def test_no_raw_keys_is_noop_even_without_model_or_adapter_metadata(self):
        weights = {
            f"{PREFIX}.0.gate_proj.lora_A.weight": torch.randn(R, H),
            f"{PREFIX}.gate_up_proj.lora_A.weight": torch.randn(E, R, H),
            f"{PREFIX}.gate_up_proj.lora_B.weight": torch.randn(E, 2 * I, R),
            f"{PREFIX}.down_proj.lora_B.weight": torch.randn(1, H, R),
            "model.layers.0.mlp.shared_expert.gate_up_proj.lora_A.weight": torch.randn(
                R, H
            ),
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(R, H),
        }
        before = weights.copy()
        normalize(weights, SimpleNamespace(), {})
        self.assertEqual(list(weights), list(before))
        for name, tensor in before.items():
            self.assertIs(weights[name], tensor)

    def test_empty_weight_dictionary_is_noop(self):
        weights = {}
        normalize(weights, SimpleNamespace(), {})
        self.assertEqual(weights, {})


if __name__ == "__main__":
    unittest.main()

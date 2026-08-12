"""Unit tests for the Muse Glimmer MLX model file's load path — no weights, no server.

End-to-end coverage needs the private packaged artifact, so the checkpoint-schema
logic is pinned here with tiny synthetic weights instead:

1. ``flatten_rc_config`` — RC nested-config translation, including the two
   convention conversions (qk_scale_factor gains sqrt(head_dim); NoPE layers
   come from zeros in ``layer_rope_theta``).
2. ``ModelArgs`` validation — derived ``no_rope_layers``/``layer_types``,
   rejection of inconsistent or malformed lists, format-version check.
3. ``sanitize`` — all three accepted weight layouts (raw HF, RC multimodal,
   packaged) plus rejection of incomplete or mislabeled checkpoints. The
   positional RC norm renames and the per-head q/gate interleave are verified
   numerically, since getting either silently wrong still yields a model that
   runs but computes garbage.
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_mlx_ci(est_time=6, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm"

if _HAS_MLX:
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.models.muse_glimmer_mlx import (
        Model,
        ModelArgs,
        flatten_rc_config,
    )

# Tiny architecture: 4 layers so the derived NoPE pattern (last layer NoPE
# with every_n_layers_nope=4) exercises both layer types.
_TINY = dict(
    hidden_size=8,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    intermediate_size=16,
    vocab_size=32,
    every_n_layers_nope=4,
    sliding_window=4,
    max_position_embeddings=64,
)

_RC_TEXT_CONFIG = dict(
    hidden_size=8,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    intermediate_size=16,
    vocab_size=32,
    rms_norm_eps=1e-5,
    post_norm_eps=1e-8,
    max_position_embeddings=64,
    qk_scale_factor=0.5,
    output_multiplier=0.2,
    final_logit_softcapping=20.0,
    sliding_window=4,
    layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
)


def _raw_weights(args):
    """A complete raw HF export with deterministic values."""
    mx.random.seed(0)
    H, D, hid = args.num_attention_heads, args.head_dim, args.hidden_size
    kv = args.num_key_value_heads * D
    weights = {
        "model.embed_tokens.weight": mx.random.normal((args.vocab_size, hid)),
        "model.norm.weight": mx.random.normal((hid,)),
        "lm_head.weight": mx.random.normal((args.vocab_size, hid)),
    }
    for i in range(args.num_hidden_layers):
        p = f"model.layers.{i}."
        weights.update(
            {
                p + "self_attn.q_proj.weight": mx.random.normal((H * D, hid)),
                p + "self_attn.k_proj.weight": mx.random.normal((kv, hid)),
                p + "self_attn.v_proj.weight": mx.random.normal((kv, hid)),
                p + "self_attn.o_proj.weight": mx.random.normal((hid, H * D)),
                p + "self_attn.output_gate_proj.weight": mx.random.normal((H * D, hid)),
                p + "input_layernorm.weight": mx.full((hid,), 0.10),
                p + "post_attn_norm.weight": mx.full((hid,), 0.20),
                p + "post_attention_layernorm.weight": mx.full((hid,), 0.30),
                p + "post_ffn_norm.weight": mx.full((hid,), 0.40),
                p
                + "mlp.gate_proj.weight": mx.random.normal(
                    (args.intermediate_size, hid)
                ),
                p
                + "mlp.up_proj.weight": mx.random.normal((args.intermediate_size, hid)),
                p
                + "mlp.down_proj.weight": mx.random.normal(
                    (hid, args.intermediate_size)
                ),
            }
        )
    return weights


def _rc_weights(args):
    """The same export in the RC multimodal layout (nested prefix, RC names,
    a vision tower to be dropped, no output-gate fusion)."""
    raw = _raw_weights(args)
    rc = {}
    renames = {
        "self_attn.output_gate_proj.weight": "self_attn.gate_proj.weight",
        "post_attn_norm.weight": "post_attention_layernorm.weight",
        "post_attention_layernorm.weight": "pre_feedforward_layernorm.weight",
        "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
    }
    for name, w in raw.items():
        if not name.startswith("model."):
            rc[name] = w  # lm_head stays top-level in the RC layout too
            continue
        rest = name[len("model.") :]
        for raw_suffix, rc_suffix in renames.items():
            if rest.endswith(raw_suffix):
                rest = rest[: -len(raw_suffix)] + rc_suffix
                break
        rc["model.language_model." + rest] = w
    rc["model.vision_tower.patch_embed.weight"] = mx.zeros((4, 4))
    return rc


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestFlattenRcConfig(CustomTestCase):
    def test_field_mapping_and_conversions(self):
        flat = flatten_rc_config({"text_config": dict(_RC_TEXT_CONFIG)})
        self.assertEqual(flat["model_type"], "muse_glimmer")
        # RC qk_scale_factor is against SDPA's 1/sqrt(head_dim).
        self.assertAlmostEqual(flat["qk_scale_factor"], 0.5 * 4**0.5)
        self.assertEqual(flat["output_soft_cap_temp"], 20.0)
        # The current vendor export stores q/k in the NeoX rotary
        # layout; the RC path always reads that convention.
        self.assertIs(flat["rope_is_neox_style"], True)
        # normalize_tok_embeddings must stay at the ModelArgs default (True):
        # the 20260806 export ships the raw table (needs the runtime norm),
        # and on older baked-table exports the scaleless RMS norm is
        # idempotent, so always-on covers both generations.
        self.assertNotIn("normalize_tok_embeddings", flat)
        self.assertIs(ModelArgs.normalize_tok_embeddings, True)
        # Zeros in layer_rope_theta mark NoPE layers.
        self.assertEqual(flat["no_rope_layers"], [1, 1, 1, 0])

    def test_non_silu_activation_rejected(self):
        cfg = dict(_RC_TEXT_CONFIG, hidden_activation="gelu")
        with self.assertRaisesRegex(ValueError, "silu"):
            flatten_rc_config({"text_config": cfg})

    def test_model_args_from_dict_accepts_rc_schema(self):
        args = ModelArgs.from_dict({"text_config": dict(_RC_TEXT_CONFIG)})
        self.assertEqual(args.no_rope_layers, [1, 1, 1, 0])
        self.assertEqual(
            args.layer_types,
            ["sliding_attention"] * 3 + ["full_attention"],
        )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestModelArgsValidation(CustomTestCase):
    def test_derives_nope_and_layer_types(self):
        args = ModelArgs(**_TINY)
        self.assertEqual(args.no_rope_layers, [1, 1, 1, 0])
        self.assertEqual(
            args.layer_types,
            ["sliding_attention"] * 3 + ["full_attention"],
        )

    def test_layer_types_must_match_no_rope_layers(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            ModelArgs(**_TINY, layer_types=["full_attention"] * 4)

    def test_wrong_length_no_rope_layers_rejected(self):
        with self.assertRaisesRegex(ValueError, "entries"):
            ModelArgs(**_TINY, no_rope_layers=[1, 0])

    def test_non_binary_no_rope_flags_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-binary"):
            ModelArgs(**_TINY, no_rope_layers=[1, 1, 2, 0])

    def test_unknown_format_version_rejected(self):
        with self.assertRaisesRegex(ValueError, "muse_glimmer_mlx_format"):
            ModelArgs(**_TINY, muse_glimmer_mlx_format=99)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSanitize(CustomTestCase):
    def _model(self, **overrides):
        return Model(ModelArgs(**dict(_TINY, **overrides)))

    def test_raw_export_folds_norms_and_fuses_gate(self):
        model = self._model()
        raw = _raw_weights(model.args)
        out = model.sanitize(dict(raw))

        # Offset norms gain +1.0; the final norm does not.
        norm = out["model.layers.0.input_layernorm.weight"]
        self.assertTrue(mx.allclose(norm, mx.full(norm.shape, 1.10)))
        self.assertTrue(mx.allclose(out["model.norm.weight"], raw["model.norm.weight"]))

        # q/gate interleave is per-head [q_head; gate_head].
        H, D = model.args.num_attention_heads, model.args.head_dim
        hid = model.args.hidden_size
        fused = out["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(tuple(fused.shape), (2 * H * D, hid))
        per_head = fused.reshape(H, 2 * D, hid)
        q = raw["model.layers.0.self_attn.q_proj.weight"].reshape(H, D, hid)
        g = raw["model.layers.0.self_attn.output_gate_proj.weight"].reshape(H, D, hid)
        self.assertTrue(mx.allclose(per_head[:, :D, :], q))
        self.assertTrue(mx.allclose(per_head[:, D:, :], g))
        self.assertNotIn("model.layers.0.self_attn.output_gate_proj.weight", out)

    def test_sanitized_raw_weights_load_and_forward(self):
        model = self._model()
        out = model.sanitize(_raw_weights(model.args))
        model.load_weights(list(out.items()))
        logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=model.make_cache())
        self.assertEqual(tuple(logits.shape), (1, 3, model.args.vocab_size))
        self.assertTrue(bool(mx.all(mx.isfinite(logits))))

    def test_rc_layout_positional_renames(self):
        model = self._model()
        out = model.sanitize(_rc_weights(model.args))
        # Distinct per-norm constants prove each RC name landed in its
        # positional slot (+1 folded): RC post_attention_layernorm ->
        # raw post_attn_norm (0.20), RC pre_feedforward_layernorm ->
        # raw post_attention_layernorm (0.30).
        for raw_name, value in (
            ("input_layernorm", 1.10),
            ("post_attn_norm", 1.20),
            ("post_attention_layernorm", 1.30),
            ("post_ffn_norm", 1.40),
        ):
            w = out[f"model.layers.0.{raw_name}.weight"]
            self.assertTrue(
                mx.allclose(w, mx.full(w.shape, value)),
                f"{raw_name} expected {value}",
            )
        self.assertFalse(any("vision" in k for k in out))
        self.assertFalse(any("language_model" in k for k in out))

    def test_packaged_artifact_passes_through(self):
        model = self._model(muse_glimmer_mlx_format=1)
        packaged = {"model.embed_tokens.weight": mx.zeros((32, 8))}
        self.assertIs(model.sanitize(packaged), packaged)

    def test_packaged_marker_with_raw_keys_rejected(self):
        model = self._model(muse_glimmer_mlx_format=1)
        raw = _raw_weights(ModelArgs(**_TINY))
        with self.assertRaisesRegex(ValueError, "raw-checkpoint keys"):
            model.sanitize(raw)

    def test_incomplete_raw_checkpoint_rejected(self):
        model = self._model()
        raw = _raw_weights(model.args)
        del raw["model.layers.0.mlp.up_proj.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            model.sanitize(raw)


if __name__ == "__main__":
    unittest.main()

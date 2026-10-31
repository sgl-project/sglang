"""DFlash drafts with SpecForge KDA layers on the target KimiDeltaAttention."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _disable_radix_cache_for_dflash_kda_scan,
)
from sglang.srt.models.dflash import (
    DFlashDraftModel,
    DFlashKDAAttention,
    DFlashMLAAttention,
)
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_kda import committed_row_segments, roll_conv_tails
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

HIDDEN, HEADS, HEAD_DIM, CONV = 16, 2, 8, 4
PROJ = HEADS * HEAD_DIM
KV_LORA, NOPE, ROPE, VDIM = 8, 8, 8, 6


def _specforge_kda_shapes(full_rank_gate):
    """SpecForge's Qwen3DFlashKDAAttention parameters (external checkpoint layout)."""
    shapes = {
        "q_proj.weight": (PROJ, HIDDEN),
        "k_proj.weight": (PROJ, HIDDEN),
        "v_proj.weight": (PROJ, HIDDEN),
        "q_conv1d.weight": (PROJ, CONV),
        "k_conv1d.weight": (PROJ, CONV),
        "v_conv1d.weight": (PROJ, CONV),
        "A_log": (HEADS,),
        "f_a_proj.weight": (HEAD_DIM, HIDDEN),
        "f_b_proj.weight": (PROJ, HEAD_DIM),
        "dt_bias": (PROJ,),
        "b_proj.weight": (HEADS, HIDDEN),
        "o_norm.weight": (HEAD_DIM,),
        "o_proj.weight": (HIDDEN, PROJ),
    }
    if full_rank_gate:
        shapes["g_proj.weight"] = (PROJ, HIDDEN)
    else:
        shapes["g_a_proj.weight"] = (HEAD_DIM, HIDDEN)
        shapes["g_b_proj.weight"] = (PROJ, HEAD_DIM)
    return shapes


def _export(attention_modes=("kda", "gqa"), **linear_attn_config):
    """A SpecForge hybrid KDA export."""
    return dict(
        architectures=["DFlashDraftModel"],
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=len(attention_modes),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        rms_norm_eps=1e-6,
        num_target_layers=2,
        # MLA layers' fields; GQA and KDA layers ignore them.
        rope_theta=1e4,
        max_position_embeddings=4096,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VDIM,
        dflash_config={
            "attention_modes": list(attention_modes),
            "block_size": 4,
            "target_layer_ids": [0],
        },
        linear_attn_config={
            "backend": "fla",
            "context_state": "scan",
            "gate_lower_bound": -5.0,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "short_conv_kernel_size": CONV,
            "use_full_rank_gate": False,
            **linear_attn_config,
        },
    )


class TestDFlashKDA(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def test_attention_mode_validation(self):
        config = parse_dflash_draft_config(draft_hf_config=_export())
        self.assertEqual(config.attention_modes, ("kda", "gqa"))
        self.assertTrue(config.kda.scans_context)
        self.assertEqual(config.kda.gate_lower_bound, -5.0)
        self.assertFalse(config.is_mla)
        # SpecForge's default policy is block-local.
        export = _export()
        del export["linear_attn_config"]["context_state"]
        self.assertFalse(
            parse_dflash_draft_config(draft_hf_config=export).kda.scans_context
        )

        mixed = parse_dflash_draft_config(draft_hf_config=_export(("kda", "mla")))
        self.assertTrue(mixed.is_mla)
        self.assertIsNotNone(mixed.kda)
        self.assertTrue(
            parse_dflash_draft_config(
                draft_hf_config=_export(use_full_rank_gate=True)
            ).kda.full_rank_output_gate
        )

        for export, error in (
            (_export(("kda", "kda")), "at least one gqa/mha/mla"),
            (_export(("kda", "mla", "gqa")), "mix mla with gqa/mha"),
            (_export(use_full_rank_gate="yes"), "use_full_rank_gate"),
            (_export(context_state="bogus"), "context_state"),
            (_export(gate_lower_bound=1.0), "gate_lower_bound"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                parse_dflash_draft_config(draft_hf_config=export)
        # DSpark parses the DFlash fields but has no kda serving path.
        with self.assertRaisesRegex(ValueError, "DSpark drafts do not support kda"):
            parse_dspark_draft_config(draft_hf_config=_export())

    def test_specforge_checkpoint_loads_into_target_layer(self):
        """SpecForge's split KDA tensors land in the target's fused layouts, and
        the GQA layer's q/k/v still land in its own qkv_proj. A full-rank output
        gate takes the no-LoRA layout, whose decay gate is f_b_proj @ f_a_proj."""
        for full_rank_gate in (False, True):
            with self.subTest(full_rank_gate=full_rank_gate):
                model = DFlashDraftModel(
                    Qwen3Config(**_export(use_full_rank_gate=full_rank_gate))
                )
                kda = {
                    name: torch.randn(shape)
                    for name, shape in _specforge_kda_shapes(full_rank_gate).items()
                }
                gqa = {
                    "q_proj.weight": torch.randn(16, HIDDEN),
                    "k_proj.weight": torch.randn(8, HIDDEN),
                    "v_proj.weight": torch.randn(8, HIDDEN),
                }
                model.load_weights(
                    [(f"layers.0.self_attn.{name}", w) for name, w in kda.items()]
                    + [(f"layers.1.self_attn.{name}", w) for name, w in gqa.items()]
                )

                attn = model.layers[0].self_attn
                self.assertIsInstance(attn, DFlashKDAAttention)
                qkvb = [kda[f"{name}_proj.weight"] for name in ("q", "k", "v", "b")]
                expected = {
                    "qkv_conv1d.weight": torch.cat(
                        [kda[f"{qkv}_conv1d.weight"] for qkv in "qkv"]
                    ).unsqueeze(1),
                    "A_log": kda["A_log"].view(1, 1, HEADS, 1),
                    "dt_bias": kda["dt_bias"],
                    "o_norm.weight": kda["o_norm.weight"],
                    "o_proj.weight": kda["o_proj.weight"],
                }
                if full_rank_gate:
                    decay_gate = kda["f_b_proj.weight"] @ kda["f_a_proj.weight"]
                    expected["fused_qkvbfg_proj.weight"] = torch.cat(
                        qkvb + [decay_gate, kda["g_proj.weight"]]
                    )
                    expected["f_a_proj.weight"] = kda["f_a_proj.weight"]
                    expected["f_b_proj.weight"] = kda["f_b_proj.weight"]
                else:
                    expected["fused_qkvbfg_a_proj.weight"] = torch.cat(
                        qkvb + [kda["f_a_proj.weight"], kda["g_a_proj.weight"]]
                    )
                    expected["fused_fg_b_proj.weight"] = torch.stack(
                        [kda["f_b_proj.weight"], kda["g_b_proj.weight"]]
                    )
                params = dict(attn.named_parameters())
                self.assertEqual(set(params), set(expected))
                for name, value in expected.items():
                    torch.testing.assert_close(params[name], value, msg=name)
                # The kernels read the conv weights through the linear-attention layer.
                torch.testing.assert_close(
                    attn.attn.conv_weights, expected["qkv_conv1d.weight"].squeeze(1)
                )
                torch.testing.assert_close(
                    model.layers[1].self_attn.qkv_proj.weight,
                    torch.cat([gqa[f"{qkv}_proj.weight"] for qkv in "qkv"]),
                    rtol=0,
                    atol=0,
                )

    def test_kda_layers_share_a_draft_with_mla_layers(self):
        """KDA layers own no KV, so they serve next to MLA layers, which keep the
        MLA pool. Both read q_proj: each layer's must land in its own parameters."""
        with patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
            model = DFlashDraftModel(Qwen3Config(**_export(("kda", "mla"))))
        kda, mla = model.layers[0].self_attn, model.layers[1].self_attn
        self.assertIsInstance(kda, DFlashKDAAttention)
        self.assertIsInstance(mla, DFlashMLAAttention)
        kda_weights = {
            name: torch.randn(shape)
            for name, shape in _specforge_kda_shapes(full_rank_gate=False).items()
        }
        mla_weights = {
            "q_proj.weight": torch.randn(HEADS * (NOPE + ROPE), HIDDEN),
            "kv_a_proj_with_mqa.weight": torch.randn(KV_LORA + ROPE, HIDDEN),
            "kv_a_layernorm.weight": torch.randn(KV_LORA),
            "kv_b_proj.weight": torch.randn(HEADS * (NOPE + VDIM), KV_LORA),
            "o_proj.weight": torch.randn(HIDDEN, HEADS * VDIM),
        }
        model.load_weights(
            [(f"layers.0.self_attn.{name}", w) for name, w in kda_weights.items()]
            + [(f"layers.1.self_attn.{name}", w) for name, w in mla_weights.items()]
        )
        for name, value in mla_weights.items():
            torch.testing.assert_close(
                dict(mla.named_parameters())[name], value, rtol=0, atol=0, msg=name
            )
        torch.testing.assert_close(
            kda.fused_qkvbfg_a_proj.weight[:PROJ],
            kda_weights["q_proj.weight"],
            rtol=0,
            atol=0,
        )
        # load_weights ran the MLA layer's post-load absorption.
        self.assertIsNotNone(mla.w_kc)

    def test_verified_rows_commit_only_their_prefix(self):
        """Dense [bs, block] verify rows commit their first commit_lens rows to
        the request's slot without a host sync; the rest go to the skipped slot -1."""
        slots, commit_lens, block = torch.tensor([5, 2]), torch.tensor([3, 4]), 4
        query_start_loc, state_indices = committed_row_segments(
            slots, commit_lens, block
        )
        self.assertEqual(query_start_loc.tolist(), [0, 3, 4, 8, 8])
        self.assertEqual(state_indices.tolist(), [5, -1, 2, -1])

        tails, rows = torch.randn(2, CONV - 1, 6), torch.randn(2, block, 6)
        rolled = roll_conv_tails(tails, rows, commit_lens)
        for i, n in enumerate(commit_lens.tolist()):
            expected = torch.cat((tails[i], rows[i, :n]))[-(CONV - 1) :]
            torch.testing.assert_close(rolled[i], expected, rtol=0, atol=0)

    def test_scan_drafts_disable_radix_cache(self):
        """A prefix hit would start drafting from a slot state that never read
        the cached rows, so scan drafts must serve without radix cache."""

        def resolve(export, **server_args):
            args = ServerArgs(
                model_path="dummy",
                speculative_algorithm="DFLASH",
                speculative_draft_model_path="draft",
                speculative_num_draft_tokens=4,
                **server_args,
            )
            with patch(
                "sglang.srt.utils.hf_transformers_utils.get_config",
                return_value=export,
            ):
                _disable_radix_cache_for_dflash_kda_scan(args)
            return resolution_result(args, "disable_radix_cache")

        self.assertTrue(resolve(_export()))
        self.assertFalse(resolve(_export(context_state="reset")))
        with self.assertRaisesRegex(ValueError, "hierarchical"):
            resolve(_export(), enable_hierarchical_cache=True)


if __name__ == "__main__":
    unittest.main()

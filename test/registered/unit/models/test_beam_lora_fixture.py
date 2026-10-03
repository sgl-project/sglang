"""The Beam LoRA fixture: PEFT layout, merged control, and sensitivity."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from sglang.test.beam_fixture import (
    BEAM_FIXTURE_MAX_KL,
    eager_beam_logits,
    make_beam_fixture,
)
from sglang.test.beam_lora_fixture import (
    EXPERT_LAYOUTS,
    LORA_TARGET_GROUPS,
    PEFT_PREFIX,
    make_beam_lora_adapter,
    merge_beam_lora,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PROMPT = [1 + (i * 17) % 127 for i in range(24)]
# An engine that drops an adapter must fail the e2e bound by a wide margin.
MIN_ADAPTER_KL = 20 * BEAM_FIXTURE_MAX_KL


def per_token_kl(expected_logits, actual_logits):
    expected = expected_logits.float().log_softmax(-1)
    actual = actual_logits.float().log_softmax(-1)
    return (expected.exp() * (expected - actual)).sum(-1)


class TestBeamLoRAFixture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-lora-fixture-")
        cls.config, cls.state = make_beam_fixture(cls.checkpoint.name)
        cls.base_logits = eager_beam_logits(cls.config, cls.state, PROMPT)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "checkpoint"):
            cls.checkpoint.cleanup()

    def make_adapter(self, groups, name, **kwargs):
        path = Path(self.checkpoint.name) / "adapters" / name
        return make_beam_lora_adapter(path, self.config, self.state, groups, **kwargs)

    def test_files_follow_the_peft_layout(self):
        config = self.config
        rank = 4
        adapter = self.make_adapter(
            LORA_TARGET_GROUPS, "all", rank=rank, alpha=8, expert_layout="per_expert"
        )
        adapter_config = json.loads(
            (Path(adapter.path) / "adapter_config.json").read_text()
        )
        self.assertEqual(adapter_config["peft_type"], "LORA")
        self.assertEqual((adapter_config["r"], adapter_config["lora_alpha"]), (4, 8))
        # The attention gate stays explicit alongside the MLP gate targets.
        self.assertEqual(
            adapter_config["target_modules"],
            sorted(
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
                + ["down_proj", "embed_tokens", "lm_head", "self_attn.gate_proj"]
            ),
        )
        saved = load_file(str(Path(adapter.path) / "adapter_model.safetensors"))
        self.assertEqual(saved.keys(), adapter.weights.keys())
        for key, factor in saved.items():
            self.assertTrue(key.startswith(PEFT_PREFIX), key)
            self.assertTrue(torch.equal(factor, adapter.weights[key]), key)

        hidden = config.hidden_size
        attention_width = config.num_attention_heads * config.head_dim
        expected_shapes = {
            "model.layers.0.self_attn.q_proj": (
                (rank, hidden),
                (attention_width, rank),
            ),
            "model.layers.0.self_attn.gate_proj": (
                (rank, hidden),
                (attention_width, rank),
            ),
            "model.layers.0.mlp.down_proj": (
                (rank, config.intermediate_size),
                (hidden, rank),
            ),
            "model.layers.1.mlp.experts.15.up_proj": (
                (rank, hidden),
                (config.moe_intermediate_size, rank),
            ),
            "lm_head": ((rank, hidden), (config.vocab_size, rank)),
        }
        for module, (a_shape, b_shape) in expected_shapes.items():
            with self.subTest(module=module):
                self.assertEqual(
                    saved[f"{PEFT_PREFIX}{module}.lora_A.weight"].shape, a_shape
                )
                self.assertEqual(
                    saved[f"{PEFT_PREFIX}{module}.lora_B.weight"].shape, b_shape
                )
        embed = f"{PEFT_PREFIX}model.embed_tokens"
        self.assertEqual(
            saved[f"{embed}.lora_embedding_A"].shape, (rank, config.vocab_size)
        )
        self.assertEqual(saved[f"{embed}.lora_embedding_B"].shape, (hidden, rank))
        # Dense layers have no experts; sparse layers have no dense MLP.
        self.assertFalse(any(".layers.0.mlp.experts." in key for key in saved))
        self.assertFalse(any(".layers.1.mlp.gate_proj." in key for key in saved))

        gate_only = self.make_adapter(["attention_gate"], "gate")
        gate_config = json.loads(
            (Path(gate_only.path) / "adapter_config.json").read_text()
        )
        self.assertEqual(gate_config["target_modules"], ["self_attn.gate_proj"])

        shared = self.make_adapter(
            ["experts"], "shared", rank=rank, expert_layout="shared_outer"
        ).weights
        experts = f"{PEFT_PREFIX}model.layers.2.mlp.experts"
        experts_count, moe_width = config.num_experts, config.moe_intermediate_size
        # Gate/up A and down B are shared across experts.
        self.assertEqual(
            shared[f"{experts}.gate_proj.lora_A.weight"].shape, (1, rank, hidden)
        )
        self.assertEqual(
            shared[f"{experts}.gate_proj.lora_B.weight"].shape,
            (experts_count, moe_width, rank),
        )
        self.assertEqual(
            shared[f"{experts}.down_proj.lora_A.weight"].shape,
            (experts_count, rank, moe_width),
        )
        self.assertEqual(
            shared[f"{experts}.down_proj.lora_B.weight"].shape, (1, hidden, rank)
        )

    def test_merged_weights_add_the_scaled_lora_product(self):
        rank, alpha = 8, 32
        scaling = alpha / rank
        per_expert = self.make_adapter(
            LORA_TARGET_GROUPS, "merge", rank=rank, alpha=alpha
        )
        shared = self.make_adapter(
            ["experts"],
            "merge-shared",
            rank=rank,
            alpha=alpha,
            seed=1,
            expert_layout="shared_outer",
        )

        def expected(base_key, lora_a, lora_b, transpose=False):
            delta = lora_b.float() @ lora_a.float()
            return self.state[base_key].float() + scaling * (
                delta.T if transpose else delta
            )

        def factor(adapter, module, side):
            return adapter.weights[f"{PEFT_PREFIX}{module}.lora_{side}.weight"]

        cases = []
        for module in (
            "model.layers.3.self_attn.v_proj",
            "model.layers.0.self_attn.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.2.mlp.experts.7.down_proj",
            "lm_head",
        ):
            cases.append(
                (
                    per_expert,
                    f"{module}.weight",
                    expected(
                        f"{module}.weight",
                        factor(per_expert, module, "A"),
                        factor(per_expert, module, "B"),
                    ),
                )
            )
        embed = f"{PEFT_PREFIX}model.embed_tokens"
        cases.append(
            (
                per_expert,
                "model.embed_tokens.weight",
                expected(
                    "model.embed_tokens.weight",
                    per_expert.weights[f"{embed}.lora_embedding_A"],
                    per_expert.weights[f"{embed}.lora_embedding_B"],
                    transpose=True,
                ),
            )
        )
        experts = "model.layers.1.mlp.experts"
        for expert in (0, 11):
            cases.append(
                (
                    shared,
                    f"{experts}.{expert}.gate_proj.weight",
                    expected(
                        f"{experts}.{expert}.gate_proj.weight",
                        factor(shared, f"{experts}.gate_proj", "A")[0],
                        factor(shared, f"{experts}.gate_proj", "B")[expert],
                    ),
                )
            )
            cases.append(
                (
                    shared,
                    f"{experts}.{expert}.down_proj.weight",
                    expected(
                        f"{experts}.{expert}.down_proj.weight",
                        factor(shared, f"{experts}.down_proj", "A")[expert],
                        factor(shared, f"{experts}.down_proj", "B")[0],
                    ),
                )
            )

        merged = {
            id(per_expert): merge_beam_lora(self.state, per_expert),
            id(shared): merge_beam_lora(self.state, shared),
        }
        for adapter, key, reference in cases:
            with self.subTest(key=key):
                actual = merged[id(adapter)][key]
                self.assertEqual(actual.dtype, torch.bfloat16)
                # One BF16 rounding of the exact FP32 sum.
                torch.testing.assert_close(actual, reference.bfloat16(), atol=0, rtol=0)
        untouched = merged[id(shared)]
        for key in ("model.layers.1.self_attn.q_proj.weight", "lm_head.weight"):
            self.assertTrue(torch.equal(untouched[key], self.state[key]), key)

    def test_every_group_moves_the_reference_past_the_bound(self):
        cases = [(group, "per_expert") for group in LORA_TARGET_GROUPS]
        cases += [("experts", layout) for layout in EXPERT_LAYOUTS[1:]]
        for group, layout in cases:
            with self.subTest(group=group, layout=layout):
                adapter = self.make_adapter(
                    [group], f"kl-{group}-{layout}", expert_layout=layout
                )
                logits = eager_beam_logits(
                    self.config, merge_beam_lora(self.state, adapter), PROMPT
                )
                self.assertTrue(torch.isfinite(logits).all())
                kl = per_token_kl(logits, self.base_logits)
                print(
                    f"{group}/{layout}: min_kl={kl.min().item():.4f} "
                    f"max_kl={kl.max().item():.4f}",
                    flush=True,
                )
                self.assertGreater(kl[-4:].min().item(), MIN_ADAPTER_KL)


if __name__ == "__main__":
    unittest.main()

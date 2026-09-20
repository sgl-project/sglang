"""CPU contracts for DSpark checkpoint normalization, budgets and vocabularies."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
from sglang.srt.arg_groups.fields.spec import Spec
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.configs.dspark import normalize_speculators_dspark_config
from sglang.srt.models.dspark import (
    DSparkDraftMixin,
    VanillaMarkov,
    validate_dspark_d2t,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
    resolve_markov_candidate_config,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    build_markov_embed_stack,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _config(**updates):
    config = dict(
        vocab_size=8,
        hidden_size=4,
        num_hidden_layers=1,
        block_size=2,
        markov_rank=2,
        markov_head_type="vanilla",
        mask_token_id=7,
        enable_confidence_head=False,
    )
    config.update(updates)
    return SimpleNamespace(**config)


class TestDSparkConfiguration(unittest.TestCase):
    def test_cli_generated_from_optional_fields(self):
        parser = argparse.ArgumentParser()
        fields = [
            "speculative_dspark_markov_topk",
            "speculative_dspark_markov_bias_topk",
        ]
        add_cli_args_from_dataclass(parser, Spec, fields=fields)
        defaults = parser.parse_args([])
        self.assertIsNone(defaults.speculative_dspark_markov_topk)
        self.assertIsNone(defaults.speculative_dspark_markov_bias_topk)
        explicit = parser.parse_args(
            [
                "--speculative-dspark-markov-topk",
                "0",
                "--speculative-dspark-markov-bias-topk",
                "128",
            ]
        )
        self.assertEqual(explicit.speculative_dspark_markov_topk, 0)
        self.assertEqual(explicit.speculative_dspark_markov_bias_topk, 128)

    def test_non_dspark_rejects_explicit_even_zero(self):
        for algorithm in (None, "EAGLE", "DFLASH"):
            for field in ("markov_topk", "markov_bias_topk"):
                cfg = SimpleNamespace(
                    speculative_algorithm=algorithm,
                    speculative_dspark_markov_topk=None,
                    speculative_dspark_markov_bias_topk=None,
                )
                setattr(cfg, f"speculative_dspark_{field}", 0)
                with patch(
                    "sglang.srt.arg_groups.speculative_hook.resolving_view",
                    return_value=cfg,
                ):
                    with self.assertRaisesRegex(ValueError, "requires.*DSPARK"):
                        handle_speculative_decoding(cfg)

    def test_optional_budgets_and_explicit_zero(self):
        config = _config(markov_topk=3, dspark_draft_topk=4, markov_bias_topk=2)
        self.assertEqual(resolve_markov_candidate_config(config).effective_topk, 3)
        disabled = resolve_markov_candidate_config(config, markov_topk=0)
        self.assertEqual(
            (disabled.effective_topk, disabled.effective_bias_topk), (0, 0)
        )
        self.assertEqual(
            resolve_markov_candidate_config(
                config, markov_bias_topk=0
            ).effective_bias_topk,
            0,
        )
        self.assertEqual(
            resolve_markov_candidate_config(
                _config(dspark_draft_topk=4), markov_bias_topk=1
            ).effective_topk,
            4,
        )
        self.assertEqual(resolve_markov_candidate_config(_config()).effective_topk, 0)
        self.assertEqual(
            resolve_markov_candidate_config(
                _config(vocab_size=32), markov_topk=1
            ).effective_bias_topk,
            16,
        )
        union = resolve_markov_candidate_config(
            config, markov_topk=7, markov_bias_topk=7
        )
        self.assertEqual(union.effective_topk + union.effective_bias_topk, 14)
        parsed = parse_dspark_draft_config(
            draft_hf_config=_config(markov_topk=None, dspark_draft_topk=3)
        )
        self.assertEqual(
            resolve_markov_candidate_config(parsed, markov_bias_topk=0).effective_topk,
            3,
        )

    def test_budget_errors(self):
        for invalid in (-1, True, 1.5, "2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_markov_candidate_config(_config(), markov_topk=invalid)
        for kwargs in (
            dict(markov_topk=9, markov_bias_topk=0),
            dict(markov_topk=2, markov_bias_topk=9),
            dict(markov_topk=0, markov_bias_topk=9),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_markov_candidate_config(_config(), **kwargs)

    def test_speculators_query_rows_and_layer_conversion_once(self):
        raw = dict(
            model_type="speculators",
            speculators_model_type="dspark",
            transformer_layer_config=dict(vocab_size=9, hidden_size=4),
            aux_hidden_state_layer_ids=[1, 4],
            block_size=9,
            mask_token_id=8,
            markov_rank=2,
            markov_head_type="vanilla",
            draft_vocab_size=4,
            logit_scale=-2.0,
        )
        normalized = normalize_speculators_dspark_config(raw)
        self.assertEqual(normalized["block_size"], 8)
        self.assertEqual(normalized["target_layer_ids"], [0, 3])
        self.assertEqual(normalized["architectures"], ["Qwen3DSparkModel"])
        self.assertEqual(normalized["draft_vocab_size"], 4)
        self.assertEqual(normalized["logit_scale"], -2.0)
        self.assertFalse(normalized["enable_confidence_head"])
        self.assertFalse(normalized["confidence_head_with_markov"])
        self.assertEqual(normalize_speculators_dspark_config(normalized), normalized)
        self.assertEqual(raw["block_size"], 9)
        raw["sample_from_anchor"] = True
        self.assertEqual(normalize_speculators_dspark_config(raw)["block_size"], 9)
        raw["use_aux_hidden_state"] = False
        with self.assertRaisesRegex(ValueError, "use_aux_hidden_state=True"):
            normalize_speculators_dspark_config(raw)

    def test_mask_uses_input_embedding_domain(self):
        config = _config(vocab_size=9, mask_token_id=8)
        runtime = resolve_runtime_config(
            draft_hf_config=config,
            speculative_num_draft_tokens=3,
            target_vocab_size=8,
            input_vocab_size=9,
        )
        self.assertEqual((runtime.gamma, runtime.verify_num_draft_tokens), (2, 3))
        with self.assertRaisesRegex(ValueError, "input embedding"):
            resolve_runtime_config(
                draft_hf_config=config,
                speculative_num_draft_tokens=3,
                target_vocab_size=8,
                input_vocab_size=8,
            )


class TestDSparkVocabulary(unittest.TestCase):
    def test_confidence_uses_target_predecessor_ids(self):
        head = VanillaMarkov(vocab_size=6, draft_vocab_size=3, markov_rank=1)
        head.configure_target_vocab(6, torch.tensor([3, 0, 2]))
        with torch.no_grad():
            head.markov_w1.weight.copy_(torch.arange(6).view(6, 1))
        embeddings = build_markov_embed_stack(
            anchor_tokens=torch.tensor([1]),
            draft_tokens=torch.tensor([[4, 3]]),
            markov_head=head,
            gamma=2,
        )
        torch.testing.assert_close(embeddings, torch.tensor([[[1.0], [4.0]]]))

    def test_zero_draft_vocabulary_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "vocabulary sizes must be positive"):
            VanillaMarkov(vocab_size=8, draft_vocab_size=0, markov_rank=2)

    def test_mapping_requires_range_shape_and_injectivity(self):
        validate_dspark_d2t(
            torch.tensor([3, 0, 2]), draft_vocab_size=3, target_vocab_size=6
        )
        for offsets in (
            torch.tensor([0, -1, 0]),
            torch.tensor([0, 0, 9]),
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0, 0]),
        ):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                validate_dspark_d2t(offsets, draft_vocab_size=3, target_vocab_size=6)

    def test_dense_reduced_chain_uses_target_predecessors_and_scale_once(self):
        head = VanillaMarkov(
            vocab_size=6, draft_vocab_size=3, markov_rank=1, logit_scale=-2
        )
        head.configure_target_vocab(6, torch.tensor([3, 0, 2]))  # target IDs [3, 1, 4]
        with torch.no_grad():
            head.markov_w1.weight.copy_(torch.arange(6).view(6, 1))
            head.markov_w2.weight.copy_(torch.tensor([[0.0], [1.0], [-1.0]]))
        base = torch.zeros(1, 2, 3)
        tokens, logits = head.sample_block(
            base,
            first_prev_tokens=torch.tensor([1]),
            hidden_states=None,
            sampler=lambda scores, step: scores.argmax(-1),
        )
        self.assertEqual(tokens.tolist(), [[4, 4]])
        torch.testing.assert_close(
            logits[0, 0, [3, 1, 4]], torch.tensor([0.0, -2.0, 2.0])
        )
        torch.testing.assert_close(
            logits[0, 1, [3, 1, 4]], torch.tensor([0.0, -8.0, 8.0])
        )
        self.assertTrue(torch.isneginf(logits[..., [0, 2, 5]]).all())


class _Backbone(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.is_nemotron_35_draft = False
        self.embed_tokens = None

    def load_weights(self, weights):
        self.backbone_weights = list(weights)


class _Draft(DSparkDraftMixin, _Backbone):
    pass


class _Vocab(nn.Module):
    def __init__(self, rows, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, hidden))
        self.org_vocab_size = rows
        self.num_embeddings = rows
        self.embedding_dim = hidden


class TestDSparkCheckpointWeights(unittest.TestCase):
    def _weights(self, *, input_vocab=9, draft_vocab=3):
        return [
            ("model.markov_head.markov_w1.weight", torch.ones(input_vocab, 2)),
            ("model.markov_head.markov_w2.weight", torch.ones(draft_vocab, 2)),
        ]

    def test_own_head_embedding_and_mapping_survive_attach(self):
        model = _Draft(_config(vocab_size=9, draft_vocab_size=3, mask_token_id=8))
        weights = self._weights() + [
            ("embed_tokens.weight", torch.ones(9, 4)),
            ("lm_head.weight", torch.ones(3, 4)),
            ("d2t", torch.tensor([3, 0, 2])),
        ]
        # The input-only mask row need not appear in the predecessor domain.
        weights[0] = ("markov_head.markov_w1.weight", torch.ones(8, 2))
        with patch("sglang.srt.models.dspark.ParallelLMHead", _Vocab), patch(
            "sglang.srt.models.dspark.VocabParallelEmbedding", _Vocab
        ):
            model.load_weights(weights)
        own_embed, own_head = model.embed_tokens, model.lm_head
        model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=_Vocab(8, 4))
        self.assertIs(model.embed_tokens, own_embed)
        self.assertIs(model.lm_head, own_head)
        self.assertEqual(model.markov_head.markov_w1.num_embeddings, 8)
        self.assertEqual(model.map_draft_to_target(torch.arange(3)).tolist(), [3, 1, 4])

    def test_w1_must_cover_all_target_predecessors(self):
        model = _Draft(_config())
        model.load_weights(self._weights(input_vocab=7, draft_vocab=8))
        with self.assertRaisesRegex(ValueError, "cover every target predecessor"):
            model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=_Vocab(8, 4))

    def test_output_vocab_inferred_from_w2_when_input_has_mask_row(self):
        model = _Draft(_config(vocab_size=9, mask_token_id=8))
        with patch("sglang.srt.models.dspark.ParallelLMHead", _Vocab), patch(
            "sglang.srt.models.dspark.VocabParallelEmbedding", _Vocab
        ):
            model.load_weights(
                self._weights(input_vocab=8, draft_vocab=3)
                + [
                    ("embed_tokens.weight", torch.ones(9, 4)),
                    ("lm_head.weight", torch.ones(3, 4)),
                    ("d2t", torch.tensor([3, 0, 2])),
                ]
            )
        model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=_Vocab(8, 4))
        self.assertEqual(
            (model.input_vocab_size, model.target_vocab_size, model.draft_vocab_size),
            (9, 8, 3),
        )
        self.assertEqual(model.markov_head.markov_w2.out_features, 3)

    def test_weight_reload_refreshes_candidate_table(self):
        model = _Draft(_config())
        model.load_weights(self._weights(input_vocab=8, draft_vocab=8))
        model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=_Vocab(8, 4))
        refreshed = []
        model.markov_candidate_sampler = SimpleNamespace(
            refresh_weights=lambda w1, w2, **kwargs: refreshed.append(
                (w1.clone(), w2.clone(), kwargs)
            )
        )
        model.load_weights(
            [
                ("markov_head.markov_w1.weight", torch.full((8, 2), 2.0)),
                ("markov_head.markov_w2.weight", torch.full((8, 2), 3.0)),
            ]
        )
        self.assertEqual(len(refreshed), 1)
        torch.testing.assert_close(refreshed[0][0], torch.full((8, 2), 2.0))
        torch.testing.assert_close(refreshed[0][1], torch.full((8, 2), 3.0))
        self.assertEqual(refreshed[0][2], {"alpha": 1.0, "d2t_offset": None})

    def test_predecessor_shape_reload_requires_recapture(self):
        model = _Draft(_config())
        model.load_weights(self._weights(input_vocab=8, draft_vocab=8))
        model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=_Vocab(8, 4))
        with self.assertRaisesRegex(ValueError, "restart/recapture"):
            model.load_weights(self._weights(input_vocab=9, draft_vocab=8))

    def test_reload_cannot_overwrite_shared_target_embedding(self):
        model = _Draft(_config())
        model.load_weights(self._weights(input_vocab=8, draft_vocab=8))
        target_embed = _Vocab(8, 4)
        with torch.no_grad():
            target_embed.weight.fill_(7.0)
        model.attach_shared_modules(embed_tokens=target_embed, lm_head=_Vocab(8, 4))
        with self.assertRaisesRegex(ValueError, "shared to checkpoint-owned"):
            model.load_weights(
                self._weights(input_vocab=8, draft_vocab=8)
                + [("embed_tokens.weight", torch.zeros(8, 4))]
            )
        torch.testing.assert_close(target_embed.weight, torch.full((8, 4), 7.0))

    def test_missing_or_wrong_markov_weight_is_load_error(self):
        for weights in (self._weights()[:1], self._weights(draft_vocab=4)):
            model = _Draft(_config(vocab_size=9, draft_vocab_size=3))
            with self.assertRaises(ValueError):
                model.load_weights(weights)

    def test_speculators_missing_backbone_weight_is_load_error(self):
        model = _Draft(_config(_sglang_speculators_dspark_normalized=True))
        model.fc = nn.Linear(4, 4, bias=False)
        with self.assertRaisesRegex(ValueError, "missing required backbone.*fc.weight"):
            model.load_weights(self._weights(input_vocab=8, draft_vocab=8))

    def test_training_backbone_aliases_are_normalized_before_loading(self):
        model = _Draft(_config())
        model.load_weights(
            self._weights(input_vocab=8, draft_vocab=8)
            + [
                ("model.midlayer.self_attn.q_proj.weight", torch.ones(4, 4)),
                ("model.encoder.fc.weight", torch.ones(4, 4)),
            ]
        )
        self.assertEqual(
            [name for name, _ in model.backbone_weights],
            ["layers.0.self_attn.q_proj.weight", "encoder.fc.weight"],
        )

    def test_reduced_vocab_cannot_use_shared_full_head(self):
        model = _Draft(_config(vocab_size=8, draft_vocab_size=3))
        model.load_weights(self._weights(input_vocab=8))
        with self.assertRaisesRegex(ValueError, "checkpoint lm_head"):
            model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=_Vocab(8, 4))

    def test_own_embedding_only_keeps_target_head(self):
        model = _Draft(_config())
        with patch("sglang.srt.models.dspark.VocabParallelEmbedding", _Vocab):
            model.load_weights(
                self._weights(input_vocab=8, draft_vocab=8)
                + [("embed_tokens.weight", torch.ones(8, 4))]
            )
        own_embed, target_head = model.embed_tokens, _Vocab(8, 4)
        model.attach_shared_modules(embed_tokens=_Vocab(8, 4), lm_head=target_head)
        self.assertIs(model.embed_tokens, own_embed)
        self.assertIs(model.lm_head, target_head)

    def test_base_and_bias_each_receive_logit_scale_once(self):
        model = _Draft(_config(logit_scale=-2.0))
        with patch("sglang.srt.models.dspark.ParallelLMHead", _Vocab):
            model.load_weights(
                self._weights(input_vocab=8, draft_vocab=8)
                + [("lm_head.weight", torch.ones(8, 4))]
            )
        own_head = model.lm_head
        target_embed = _Vocab(8, 4)
        model.attach_shared_modules(embed_tokens=target_embed, lm_head=_Vocab(8, 4))
        self.assertIs(model.lm_head, own_head)
        self.assertIs(model.embed_tokens, target_embed)
        with patch(
            "sglang.srt.models.dspark.project_through_lm_head",
            lambda hidden, head: hidden @ head.weight.T,
        ), patch(
            "sglang.srt.models.dspark.gather_and_crop_vocab",
            lambda logits, head: logits,
        ):
            base, _ = model.compute_base_logits(torch.ones(1, 4))
        torch.testing.assert_close(base, torch.full((1, 8), -8.0))
        corrected = model.markov_head.apply_step_logits(
            base, token_ids=torch.tensor([0]), hidden_states=None
        )
        torch.testing.assert_close(corrected, torch.full((1, 8), -12.0))


if __name__ == "__main__":
    unittest.main()

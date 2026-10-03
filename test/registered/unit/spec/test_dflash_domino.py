import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.domino_utils import validate_domino_runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _domino_config(**overrides):
    dflash_config = {
        "projector_type": "domino",
        "mask_token_id": 29,
        "shift_label": True,
        "target_layer_ids": [1, 3],
        "pure_draft_prefix_len": 1,
        "gru_hidden_dim": 4,
        "emb_dim": 5,
    }
    dflash_config.update(overrides.pop("dflash_config", {}))
    fields = {
        "num_hidden_layers": 2,
        "num_target_layers": 4,
        "block_size": 16,
        "hidden_size": 8,
        "vocab_size": 31,
        "emb_dim": 5,
        "dflash_config": dflash_config,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _projector_model(projector_type="domino"):
    model = DFlashDraftModel.__new__(DFlashDraftModel)
    nn.Module.__init__(model)
    model.projector_type = projector_type
    model.config = SimpleNamespace(hidden_size=8)
    if projector_type == "domino":
        model.prefix_gru = nn.GRU(8, 4, batch_first=True, bias=False)
        model.embed_proj = nn.Sequential(
            nn.Linear(12, 5, bias=False),
            nn.SiLU(),
            nn.Linear(5, 31, bias=False),
        )
    else:
        model.prefix_gru = None
        model.embed_proj = None
    return model


def _projector_weights(model):
    return {
        "prefix_gru.weight_ih_l0": torch.randn_like(model.prefix_gru.weight_ih_l0),
        "prefix_gru.weight_hh_l0": torch.randn_like(model.prefix_gru.weight_hh_l0),
        "embed_proj.0.weight": torch.randn_like(model.embed_proj[0].weight),
        "embed_proj.2.weight": torch.randn_like(model.embed_proj[2].weight),
    }


class TestDFlashDominoConfig(CustomTestCase):
    def test_top_level_emb_dim_fallback(self):
        config = _domino_config()
        del config.dflash_config["emb_dim"]
        self.assertEqual(parse_dflash_draft_config(draft_hf_config=config).emb_dim, 5)

    def test_invalid_domino_config_fails_fast(self):
        cases = {
            "shift_label": {"shift_label": 1},
            "pure_draft_prefix_len": {"pure_draft_prefix_len": 2},
            "gru_hidden_dim": {"gru_hidden_dim": None},
        }
        for expected, updates in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    parse_dflash_draft_config(
                        draft_hf_config=_domino_config(dflash_config=updates)
                    )

        config = _domino_config(dflash_config={"emb_dim": None}, emb_dim=None)
        with self.assertRaisesRegex(ValueError, "emb_dim"):
            parse_dflash_draft_config(draft_hf_config=config)

        with self.assertRaisesRegex(ValueError, "block_size > 1"):
            parse_dflash_draft_config(draft_hf_config=_domino_config(block_size=1))

    def test_conflicting_emb_dim_fails(self):
        with self.assertRaisesRegex(ValueError, "emb_dim differs"):
            parse_dflash_draft_config(draft_hf_config=_domino_config(emb_dim=6))


class TestDFlashDominoWeights(CustomTestCase):
    def test_projector_weights_load_exactly(self):
        model = _projector_model()
        weights = _projector_weights(model)
        model.load_weights(weights.items())
        for name, expected in weights.items():
            torch.testing.assert_close(
                dict(model.named_parameters())[name], expected, rtol=0, atol=0
            )

    def test_each_required_projector_weight_is_checked(self):
        for missing_name in _projector_weights(_projector_model()):
            with self.subTest(missing_name=missing_name):
                model = _projector_model()
                weights = _projector_weights(model)
                del weights[missing_name]
                with self.assertRaisesRegex(ValueError, missing_name):
                    model.load_weights(weights.items())

    def test_projector_shape_mismatch_fails(self):
        model = _projector_model()
        weights = _projector_weights(model)
        weights["embed_proj.2.weight"] = torch.empty(30, 5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            model.load_weights(weights.items())

    def test_projector_weights_require_domino_config(self):
        model = _projector_model(projector_type="domnio")
        with self.assertRaisesRegex(ValueError, "projector_type"):
            model.load_weights([("prefix_gru.weight_ih_l0", torch.empty(12, 8))])


class TestDFlashDominoRuntimeValidation(CustomTestCase):
    def _modules(self, dtype=torch.bfloat16):
        embedding = nn.Embedding(31, 8, dtype=dtype)
        lm_head = nn.Linear(8, 31, bias=False, dtype=dtype)
        prefix_gru = nn.GRU(8, 4, batch_first=True, bias=False, dtype=dtype)
        embed_proj = nn.Sequential(
            nn.Linear(12, 5, bias=False, dtype=dtype),
            nn.SiLU(),
            nn.Linear(5, 31, bias=False, dtype=dtype),
        )
        return embedding, lm_head, prefix_gru, embed_proj

    def _tp2_modules(self):
        embedding, lm_head, prefix_gru, embed_proj = self._modules()
        embedding = nn.Embedding(16, 8, dtype=torch.bfloat16)
        lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        shard = SimpleNamespace(
            num_added_elements=0,
            org_vocab_start_index=0,
            org_vocab_end_index=16,
            num_org_elements=16,
            num_org_elements_padded=16,
        )
        for module in (embedding, lm_head):
            module.shard_indices = shard
            module.org_vocab_size = 31
            module.tp_size = 2
            module.num_added_embeddings = 0
        return embedding, lm_head, prefix_gru, embed_proj

    def _validate(self, **overrides):
        embedding, lm_head, prefix_gru, embed_proj = overrides.pop(
            "modules", self._modules()
        )
        args = {
            "device": torch.device("cuda"),
            "tp_size": 1,
            "tp_rank": 0,
            "target_vocab_size": 31,
            "draft_vocab_size": 31,
            "hidden_size": 8,
            "target_embedding": embedding,
            "lm_head": lm_head,
            "prefix_gru": prefix_gru,
            "embed_proj": embed_proj,
        }
        args.update(overrides)
        validate_domino_runtime(**args)

    def test_tp_requires_vocab_shard_metadata(self):
        with self.assertRaisesRegex(ValueError, "lm_head shard metadata"):
            self._validate(tp_size=2)

    def test_tp2_vocab_shards_supported(self):
        self._validate(tp_size=2, modules=self._tp2_modules())

    def test_tp2_incomplete_lm_head_shard_fails(self):
        modules = self._tp2_modules()
        modules[1].shard_indices = SimpleNamespace(
            num_added_elements=0,
            num_org_elements_padded=16,
        )
        with self.assertRaisesRegex(ValueError, "shard metadata is missing"):
            self._validate(tp_size=2, modules=modules)

    def test_tp_vocab_shard_must_match_rank(self):
        modules = self._tp2_modules()
        modules[1].shard_indices.org_vocab_start_index = 1
        modules[1].shard_indices.org_vocab_end_index = 17
        with self.assertRaisesRegex(ValueError, "does not match its TP rank"):
            self._validate(tp_size=2, modules=modules)

    def test_tp1_requires_complete_vocab_shard(self):
        modules = self._modules()
        modules[1].shard_indices = SimpleNamespace(
            num_added_elements=0,
            org_vocab_start_index=0,
            org_vocab_end_index=30,
            num_org_elements=30,
            num_org_elements_padded=31,
        )
        modules[1].org_vocab_size = 31
        modules[1].tp_size = 1
        modules[1].num_added_embeddings = 0
        with self.assertRaisesRegex(ValueError, "does not match its TP rank"):
            self._validate(modules=modules)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for srt/speculative/pp_draft_embedding."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from torch import nn

from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.speculative.pp_draft_embedding import (
    find_draft_embedding_param,
    load_draft_embedding_from_checkpoint,
    resolve_target_embed_and_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, HIDDEN = 16, 8


class _Inner(nn.Module):
    def __init__(self, embed: nn.Module):
        super().__init__()
        self.embed_tokens = embed


class _Target(nn.Module):
    """Target with the common getter shape: ``self.model.embed_tokens.weight``."""

    def __init__(self, *, owns_embedding: bool):
        super().__init__()
        self.model = _Inner(
            nn.Embedding(VOCAB, HIDDEN) if owns_embedding else PPMissingLayer()
        )
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight


class _Draft(nn.Module):
    def __init__(self, *, with_embedding: bool = True):
        super().__init__()
        self.model = _Inner(
            nn.Embedding(VOCAB, HIDDEN) if with_embedding else nn.Identity()
        )
        # A nested draft-only embedding that must not be picked over model.embed_tokens.
        self.mtp = _Inner(nn.Embedding(VOCAB, HIDDEN))


def _write_sharded_checkpoint(root: str, embed: torch.Tensor) -> None:
    """Two shards + index; the real embedding sits in shard 2, an MTP decoy in shard 1."""
    decoy = torch.full_like(embed, -1.0)
    save_file(
        {"model.layers.0.mtp.embed_tokens.weight": decoy},
        os.path.join(root, "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"model.embed_tokens.weight": embed, "lm_head.weight": decoy},
        os.path.join(root, "model-00002-of-00002.safetensors"),
    )
    index = {
        "weight_map": {
            "model.layers.0.mtp.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        }
    }
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


class TestResolveTargetEmbedAndHead(CustomTestCase):
    def test_non_first_stage_returns_head_without_embed(self):
        """Regression: a PPMissingLayer embed_tokens used to raise AttributeError and
        abort draft init on the last stage; the head must still be returned."""
        target = _Target(owns_embedding=False)
        embed, head = resolve_target_embed_and_head(target, is_first_pp_rank=False)
        self.assertIsNone(embed)
        self.assertIs(head, target.lm_head.weight)

    def test_first_stage_missing_embedding_is_a_real_error(self):
        target = _Target(owns_embedding=False)
        with self.assertRaises(AttributeError):
            resolve_target_embed_and_head(target, is_first_pp_rank=True)

    def test_owning_stage_shares_both(self):
        target = _Target(owns_embedding=True)
        embed, head = resolve_target_embed_and_head(target, is_first_pp_rank=False)
        self.assertIs(embed, target.model.embed_tokens.weight)
        self.assertIs(head, target.lm_head.weight)


class TestLoadDraftEmbeddingFromCheckpoint(CustomTestCase):
    def test_picks_input_embedding_over_mtp_decoy_and_loads_it(self):
        expected = torch.randn(VOCAB, HIDDEN)
        draft = _Draft()
        with tempfile.TemporaryDirectory() as root:
            _write_sharded_checkpoint(root, expected)
            param = load_draft_embedding_from_checkpoint(draft, root)
        self.assertIs(param, draft.model.embed_tokens.weight)
        torch.testing.assert_close(param.detach(), expected)
        # The draft's nested MTP embedding is neither the target param nor overwritten.
        self.assertFalse(torch.equal(draft.mtp.embed_tokens.weight.detach(), expected))

    def test_no_index_falls_back_to_header_scan(self):
        expected = torch.randn(VOCAB, HIDDEN)
        draft = _Draft()
        with tempfile.TemporaryDirectory() as root:
            save_file(
                {"model.layers.0.mtp.embed_tokens.weight": torch.zeros(VOCAB, HIDDEN)},
                os.path.join(root, "a.safetensors"),
            )
            save_file(
                {"model.embed_tokens.weight": expected},
                os.path.join(root, "b.safetensors"),
            )
            param = load_draft_embedding_from_checkpoint(draft, root)
        torch.testing.assert_close(param.detach(), expected)

    def test_draft_without_embedding_fails_loudly(self):
        draft = _Draft(with_embedding=False)
        # Only the nested draft-only embedding remains; it is still a valid owner.
        self.assertIsNotNone(find_draft_embedding_param(draft))
        draft.mtp = nn.Identity()
        with self.assertRaises(ValueError):
            load_draft_embedding_from_checkpoint(draft, "/nonexistent")


if __name__ == "__main__":
    unittest.main()

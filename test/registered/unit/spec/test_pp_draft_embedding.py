"""Unit tests for srt/speculative/pp_draft_embedding."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from torch import nn

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.pp_draft_embedding import (
    load_draft_embedding_from_checkpoint,
    resolve_target_embed_and_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.layers.vocab_parallel_embedding import (  # noqa: E402
    VocabParallelEmbedding,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

VOCAB, HIDDEN = 16, 8
AUTO = LoadConfig(load_format=LoadFormat.AUTO)
MAIN_KEY = "model.embed_tokens.weight"
# A DeepSeek/GLM-style MTP layer embedding: no mtp/nextn marker in the key.
MTP_LAYER_KEY = "model.layers.61.embed_tokens.weight"


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


class _BuggyGetterTarget(_Target):
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_haed.weight  # typo on purpose


class _Draft(nn.Module):
    def __init__(self, *, with_embedding: bool = True):
        super().__init__()
        self.model = _Inner(
            VocabParallelEmbedding(VOCAB, HIDDEN) if with_embedding else nn.Identity()
        )


def _write_sharded_checkpoint(root: str, embed: torch.Tensor) -> None:
    """Two shards + index; the MTP-layer decoy sorts before the real embedding."""
    decoy = torch.full_like(embed, -1.0)
    shard_a, shard_b = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    save_file({MTP_LAYER_KEY: decoy}, os.path.join(root, shard_a))
    save_file({MAIN_KEY: embed, "lm_head.weight": decoy}, os.path.join(root, shard_b))
    weight_map = {MTP_LAYER_KEY: shard_a, MAIN_KEY: shard_b, "lm_head.weight": shard_b}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)


class TestResolveTargetEmbedAndHead(CustomTestCase):
    def test_non_first_stage_returns_head_without_embed(self):
        """Regression: a PPMissingLayer embed_tokens used to raise AttributeError and
        abort draft init on the last stage; the head must still be returned."""
        target = _Target(owns_embedding=False)
        embed, head = resolve_target_embed_and_head(target)
        self.assertIsNone(embed)
        self.assertIs(head, target.lm_head.weight)

    def test_unrelated_attribute_error_is_not_swallowed(self):
        """A getter bug on a stage that owns its embedding must propagate rather
        than silently redirect to checkpoint loading."""
        target = _BuggyGetterTarget(owns_embedding=True)
        with self.assertRaises(AttributeError):
            resolve_target_embed_and_head(target)


class TestLoadDraftEmbeddingFromCheckpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # The model loader reads the published context once (checksum check).
        cls._override = get_context().override_server_args()
        cls._override.install()

    @classmethod
    def tearDownClass(cls):
        cls._override.restore()

    def setUp(self):
        enter_scope(self, published_topology("test", tp_size=1, pp_size=1))

    def _load(self, draft, root, load_config=AUTO):
        return load_draft_embedding_from_checkpoint(
            draft, root, revision=None, load_config=load_config
        )

    def _assert_loaded(self, param, expected):
        # VocabParallelEmbedding pads the vocab; only the real rows are loaded.
        torch.testing.assert_close(param.detach()[:VOCAB], expected)

    def test_index_picks_input_embedding_over_mtp_layer_key(self):
        expected = torch.randn(VOCAB, HIDDEN)
        draft = _Draft()
        with tempfile.TemporaryDirectory() as root:
            _write_sharded_checkpoint(root, expected)
            param = self._load(draft, root)
        self.assertIs(param, draft.model.embed_tokens.weight)
        self._assert_loaded(param, expected)

    def test_no_index_picks_over_whole_checkpoint_not_first_shard(self):
        """Regression: the first shard holding *an* embedding-like key was chosen;
        an MTP-layer embedding in an earlier shard must lose to the real one."""
        expected = torch.randn(VOCAB, HIDDEN)
        with tempfile.TemporaryDirectory() as root:
            save_file(
                {MTP_LAYER_KEY: torch.zeros(VOCAB, HIDDEN)},
                os.path.join(root, "a.safetensors"),
            )
            save_file({MAIN_KEY: expected}, os.path.join(root, "b.safetensors"))
            param = self._load(_Draft(), root)
        self._assert_loaded(param, expected)

    def test_bin_only_checkpoint_follows_loader_fallback(self):
        """Regression: a checkpoint with only pytorch_model.bin raised
        FileNotFoundError because only safetensors were searched."""
        expected = torch.randn(VOCAB, HIDDEN)
        with tempfile.TemporaryDirectory() as root:
            torch.save({MAIN_KEY: expected}, os.path.join(root, "pytorch_model.bin"))
            param = self._load(_Draft(), root)
        self._assert_loaded(param, expected)

    def test_weight_loader_receives_full_vocab_tensor(self):
        """The TP shard is the parameter's weight_loader's job; the helper must hand
        it the whole checkpoint tensor, not a pre-sliced one."""
        expected = torch.randn(VOCAB, HIDDEN)
        draft = _Draft()
        seen = {}

        def sharding_loader(p, loaded):
            seen["shape"] = tuple(loaded.shape)
            p.data[: loaded.shape[0]].copy_(loaded)

        draft.model.embed_tokens.weight.weight_loader = sharding_loader
        with tempfile.TemporaryDirectory() as root:
            _write_sharded_checkpoint(root, expected)
            self._load(draft, root)
        self.assertEqual(seen["shape"], (VOCAB, HIDDEN))

    def test_load_format_gates_disk_access(self):
        """dummy returns the parameter untouched; a streaming format has no weight
        files to re-open and must fail instead of leaving the embedding random."""
        draft = _Draft()
        param = self._load(
            draft, "/nonexistent", LoadConfig(load_format=LoadFormat.DUMMY)
        )
        self.assertIs(param, draft.model.embed_tokens.weight)
        with self.assertRaises(ValueError):
            self._load(
                _Draft(),
                "/nonexistent",
                LoadConfig(load_format=LoadFormat.REMOTE_INSTANCE),
            )

    def test_draft_without_embedding_fails_loudly(self):
        with self.assertRaises(ValueError):
            self._load(_Draft(with_embedding=False), "/nonexistent")


if __name__ == "__main__":
    unittest.main()

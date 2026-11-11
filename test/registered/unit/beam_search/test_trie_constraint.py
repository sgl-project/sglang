"""Unit tests for the trie artifact loader, valid-child masks, trie-conditioned
top-k scoring, and model-bundled trie discovery."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

import sglang.srt.beam_search.trie_constraint as trie_constraint_module
from sglang.srt.beam_search.trie_config import (
    TRIE_OUTPUT_HEAD_CONFIG_KEY,
    TRIE_OUTPUT_HEAD_FILENAME,
    discover_trie_output_head_config,
    load_trie_output_head_config,
)
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Trie (codes relative to each codebook, size 4):
#   root 0 -> level1 {1, 3};  root 2 -> level1 {0}
#   (0, 1) -> {0, 2};  (0, 3) -> {1};  (2, 0) -> {2, 3}
TOKEN_START = 10
CODEBOOK_SIZE = 4
VOCAB_SIZE = 30
VALID_CODES = {
    0: [[0, 2]],
    1: {0: [1, 3], 2: [0]},
    2: {(0, 1): [0, 2], (0, 3): [1], (2, 0): [2, 3]},
}


def _tensors():
    return {
        "root_token_ids": torch.tensor([0, 2], dtype=torch.int16),
        "level1_token_ids": torch.tensor([1, 3, 0], dtype=torch.int16),
        "level1_offsets": torch.tensor([0, 2, 3], dtype=torch.int32),
        "level2_offsets": torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        "level2_token_ids": torch.tensor([0, 2, 1, 2, 3], dtype=torch.int16),
    }


def _metadata():
    return {
        "format_version": "1",
        "token_start": str(TOKEN_START),
        "codebook_size": str(CODEBOOK_SIZE),
        "num_codebooks": "3",
    }


class _ArtifactTestCase(CustomTestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)
        self._count = 0

    def write_artifact(self, *, tensors=None, metadata=None, directory=None):
        """Writes an artifact under a fresh directory (the config loader is
        cached by path) and returns the tensor path."""
        if directory is None:
            self._count += 1
            directory = self.root / f"artifact{self._count}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / TRIE_OUTPUT_HEAD_FILENAME
        save_file(
            _tensors() if tensors is None else tensors,
            str(path),
            metadata=_metadata() if metadata is None else metadata,
        )
        return str(path)

    def load(self, **kwargs):
        path = self.write_artifact(**kwargs)
        config = load_trie_output_head_config(path)
        return config, BeamTrieConstraint.load(config, "cpu")

    def load_sparse(self, config):
        with mock.patch.object(
            trie_constraint_module, "_MAX_DENSE_LEVEL2_MASK_BYTES", 0
        ):
            return BeamTrieConstraint.load(config, "cpu")


class TestValidChildMask(_ArtifactTestCase):
    def test_masks_at_every_depth_on_dense_and_sparse_paths(self):
        config, dense = self.load()
        sparse = self.load_sparse(config)
        self.assertIsNotNone(dense.level2_masks)
        self.assertIsNone(dense.level2_sparse)
        self.assertIsNone(sparse.level2_masks)
        self.assertIsNotNone(sparse.level2_sparse)

        cases = (
            (torch.empty((1, 0), dtype=torch.int64), [[True, False, True, False]]),
            (
                torch.tensor([[0], [2]]),
                [[False, True, False, True], [True, False, False, False]],
            ),
            (
                torch.tensor([[0, 1], [0, 3], [2, 0]]),
                [
                    [True, False, True, False],
                    [False, True, False, False],
                    [False, False, True, True],
                ],
            ),
        )
        for name, constraint in (("dense", dense), ("sparse", sparse)):
            self.assertEqual(constraint.level_cardinalities, (2, 3, 5))
            for depth, (prefixes, expected) in enumerate(cases):
                with self.subTest(path=name, depth=depth):
                    self.assertEqual(
                        constraint.valid_child_mask(prefixes, depth).tolist(),
                        expected,
                    )

    def test_sparse_level2_rejects_unknown_prefix(self):
        config, _ = self.load()
        sparse = self.load_sparse(config)
        # (0, 0) is not a trie prefix; keys sort below, between, and above it.
        for prefix in ([0, 0], [0, 2], [3, 3]):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(ValueError, "unknown AB pair"):
                    sparse.valid_child_mask(torch.tensor([prefix]), 2)

    def test_prefix_shape_is_validated(self):
        _, constraint = self.load()
        with self.assertRaises(ValueError):
            constraint.valid_child_mask(torch.zeros((2, 0), dtype=torch.int64), 0)
        with self.assertRaises(ValueError):
            constraint.valid_child_mask(torch.zeros((2, 2), dtype=torch.int64), 1)
        with self.assertRaises(ValueError):
            constraint.valid_child_mask(torch.zeros((2, 3), dtype=torch.int64), 3)

    def test_advance_prefixes_appends_codes_relative_to_depth(self):
        _, constraint = self.load()
        advanced = constraint.advance_prefixes(
            torch.tensor([[0], [2]]),
            parent_idx=torch.tensor([1, 0]),
            next_tokens=torch.tensor([14, 17]),
            depth=1,
        )
        self.assertEqual(advanced.tolist(), [[2, 0], [0, 3]])

    def test_compact_signed_array_preserves_values_above_int32(self):
        maximum = np.iinfo(np.int32).max + 1
        compact = trie_constraint_module._compact_signed_array(
            np.array([maximum], dtype=np.int64), maximum
        )
        self.assertEqual(compact.dtype, np.int64)
        self.assertEqual(compact.item(), maximum)


class TestTopkLogprobs(_ArtifactTestCase):
    def prefixes_for(self, depth):
        return {
            0: torch.empty((1, 0), dtype=torch.int64),
            1: torch.tensor([[0], [2], [0]]),
            2: torch.tensor([[0, 1], [0, 3], [2, 0]]),
        }[depth]

    def test_negative_infinity_padding_stays_on_trie(self):
        config, dense = self.load()
        sparse = self.load_sparse(config)
        for name, constraint in (("dense", dense), ("sparse", sparse)):
            for depth in (0, 1, 2):
                prefixes = self.prefixes_for(depth)
                valid = constraint.valid_child_mask(prefixes, depth)
                rows = prefixes.shape[0]
                logits = torch.randn(rows, VOCAB_SIZE, generator=_generator(depth))
                # More candidates than any row has valid children; only the
                # root level pads past the codebook width.
                values, tokens = constraint.topk_logprobs(
                    [logits], prefixes, depth, num_candidates=CODEBOOK_SIZE + 2
                )
                width = CODEBOOK_SIZE + 2 if depth == 0 else CODEBOOK_SIZE
                self.assertEqual(values.shape, (rows, width))
                codes = tokens - config.token_start_for_depth(depth)
                for row in range(rows):
                    with self.subTest(path=name, depth=depth, row=row):
                        num_valid = int(valid[row].sum())
                        self.assertTrue(valid[row, codes[row]].all())
                        self.assertTrue(torch.isfinite(values[row, :num_valid]).all())
                        self.assertTrue(torch.isneginf(values[row, num_valid:]).all())

    def test_dense_scores_normalize_over_full_vocabulary(self):
        _, constraint = self.load()
        logits = torch.randn(3, VOCAB_SIZE, generator=_generator(7))
        prefixes = self.prefixes_for(1)
        values, tokens = constraint.topk_logprobs(
            [logits], prefixes, depth=1, num_candidates=2
        )
        full = torch.log_softmax(logits, dim=-1)
        for row in range(3):
            for value, token in zip(values[row], tokens[row]):
                if torch.isfinite(value):
                    self.assertAlmostEqual(
                        value.item(), full[row, token].item(), places=5
                    )

    def test_compact_pieces_match_dense_full_vocab_pieces(self):
        config, dense = self.load()
        sparse = self.load_sparse(config)
        generator = _generator(11)
        hidden = torch.randn(3, 5, generator=generator)
        weight = torch.randn(VOCAB_SIZE, 5, generator=generator) * 3
        full_logits = hidden @ weight.T
        identity = lambda x: x
        project = lambda h, w: h @ w.T

        for name, constraint in (("dense", dense), ("sparse", sparse)):
            for depth in (0, 1, 2):
                prefixes = self.prefixes_for(depth)
                rows = prefixes.shape[0]
                depths = torch.full((rows,), depth, dtype=torch.int64)
                head = BeamTrieOutputHead(config, VOCAB_SIZE, chunk_size=7)
                head_output = head.project_mixed(
                    hidden[:rows], weight, depths, project, identity
                )
                self.assertEqual(tuple(head_output.logits.shape), (rows, CODEBOOK_SIZE))
                # Split into two pieces to cover piece/normalizer alignment.
                split = max(rows - 1, 1) if rows > 1 else 1
                bounds = [(0, split)] + ([(split, rows)] if split < rows else [])
                compact = constraint.topk_logprobs(
                    [head_output.logits[a:b] for a, b in bounds],
                    prefixes,
                    depth,
                    num_candidates=3,
                    normalizers=[head_output.normalizer[a:b] for a, b in bounds],
                )
                expected = constraint.topk_logprobs(
                    [full_logits[:rows]], prefixes, depth, num_candidates=3
                )
                with self.subTest(path=name, depth=depth):
                    torch.testing.assert_close(compact[0], expected[0])
                    torch.testing.assert_close(compact[1], expected[1])

    def test_compact_pieces_require_matching_normalizers(self):
        _, constraint = self.load()
        compact = torch.zeros(3, CODEBOOK_SIZE)
        prefixes = self.prefixes_for(1)
        with self.assertRaisesRegex(ValueError, "normalizer"):
            constraint.topk_logprobs([compact], prefixes, 1, num_candidates=2)
        with self.assertRaisesRegex(ValueError, "align"):
            constraint.topk_logprobs(
                [compact], prefixes, 1, num_candidates=2, normalizers=[]
            )
        with self.assertRaisesRegex(ValueError, "normalizer rows"):
            constraint.topk_logprobs(
                [compact],
                prefixes,
                1,
                num_candidates=2,
                normalizers=[torch.zeros(2, 2)],
            )

    def test_row_count_must_match_prefixes(self):
        _, constraint = self.load()
        with self.assertRaisesRegex(ValueError, "do not match"):
            constraint.topk_logprobs(
                [torch.zeros(2, VOCAB_SIZE)], self.prefixes_for(1), 1, 2
            )


def _generator(seed):
    return torch.Generator().manual_seed(seed)


class TestArtifactValidation(_ArtifactTestCase):
    def assert_rejects(self, pattern, **changes):
        tensors = _tensors()
        tensors.update(changes)
        path = self.write_artifact(tensors=tensors)
        config = load_trie_output_head_config(path)
        with self.assertRaisesRegex(ValueError, pattern):
            BeamTrieConstraint.load(config, "cpu")

    def test_accepts_the_reference_artifact(self):
        self.load()

    def test_rejects_malformed_offsets(self):
        for name, offsets, pattern in (
            ("level1_offsets", [0, 2, 2 + 2], "level1_offsets"),
            ("level1_offsets", [1, 2, 3], "level1_offsets"),
            ("level1_offsets", [0, 3, 2], "level1_offsets"),
            ("level1_offsets", [0, 3], "level1_offsets"),
            ("level2_offsets", [0, 2, 3, 4], "level2_offsets"),
            ("level2_offsets", [0, 3, 2, 5], "level2_offsets"),
        ):
            with self.subTest(name=name, offsets=offsets):
                self.assert_rejects(
                    pattern, **{name: torch.tensor(offsets, dtype=torch.int32)}
                )

    def test_rejects_empty_groups(self):
        self.assert_rejects(
            "empty group",
            level1_offsets=torch.tensor([0, 0, 3], dtype=torch.int32),
        )
        self.assert_rejects(
            "empty group",
            level2_offsets=torch.tensor([0, 2, 2, 5], dtype=torch.int32),
        )

    def test_rejects_unsorted_or_duplicate_values(self):
        self.assert_rejects(
            "root_token_ids must be sorted",
            root_token_ids=torch.tensor([2, 0], dtype=torch.int16),
        )
        self.assert_rejects(
            "root_token_ids must be sorted",
            root_token_ids=torch.tensor([2, 2], dtype=torch.int16),
        )
        self.assert_rejects(
            "B children must be sorted and unique",
            level1_token_ids=torch.tensor([3, 1, 0], dtype=torch.int16),
        )
        self.assert_rejects(
            "B children must be sorted and unique",
            level1_token_ids=torch.tensor([1, 1, 0], dtype=torch.int16),
        )
        self.assert_rejects(
            "C children must be sorted",
            level2_token_ids=torch.tensor([2, 0, 1, 2, 3], dtype=torch.int16),
        )

    def test_rejects_out_of_range_codes(self):
        for name, values in (
            ("root_token_ids", [0, CODEBOOK_SIZE]),
            ("root_token_ids", [-1, 2]),
            ("level1_token_ids", [1, CODEBOOK_SIZE, 0]),
            ("level2_token_ids", [0, 2, 1, 2, CODEBOOK_SIZE]),
        ):
            with self.subTest(name=name, values=values):
                self.assert_rejects(
                    f"{name} contains an out-of-range",
                    **{name: torch.tensor(values, dtype=torch.int16)},
                )

    def test_rejects_non_integer_or_multidimensional_tensors(self):
        self.assert_rejects(
            "one-dimensional integer",
            root_token_ids=torch.tensor([0.0, 2.0]),
        )
        self.assert_rejects(
            "one-dimensional integer",
            root_token_ids=torch.tensor([[0, 2]], dtype=torch.int16),
        )

    def test_rejects_missing_tensor(self):
        tensors = _tensors()
        del tensors["level2_token_ids"]
        path = self.write_artifact(tensors=tensors)
        config = load_trie_output_head_config(path)
        with self.assertRaisesRegex(ValueError, "level2_token_ids"):
            BeamTrieConstraint.load(config, "cpu")


class TestTrieConfig(_ArtifactTestCase):
    def declared_model(self, **artifact):
        model_root = self.root / f"model{self._count}"
        self._count += 1
        model_root.mkdir()
        self.write_artifact(directory=model_root, **artifact)
        return model_root

    def discover(self, hf_config, model_root, vocab_size=VOCAB_SIZE, **kwargs):
        kwargs.setdefault("is_draft_model", False)
        return discover_trie_output_head_config(
            hf_config, str(model_root), vocab_size, **kwargs
        )

    def test_returns_none_when_not_declared_or_disabled(self):
        model_root = self.declared_model()
        self.assertIsNone(self.discover(PretrainedConfig(), model_root))
        disabled = PretrainedConfig(**{TRIE_OUTPUT_HEAD_CONFIG_KEY: False})
        self.assertIsNone(self.discover(disabled, model_root))

    def test_returns_none_for_draft_model_even_without_artifact(self):
        declared = PretrainedConfig(**{TRIE_OUTPUT_HEAD_CONFIG_KEY: True})
        empty_root = self.root / "empty"
        empty_root.mkdir()
        self.assertIsNone(self.discover(declared, empty_root, is_draft_model=True))
        self.assertIsNone(
            self.discover(declared, self.declared_model(), is_draft_model=True)
        )

    def test_declared_model_resolves_its_root_artifact(self):
        model_root = self.declared_model()
        declared = PretrainedConfig(**{TRIE_OUTPUT_HEAD_CONFIG_KEY: True})
        config = self.discover(declared, model_root)
        self.assertEqual(config.token_start, TOKEN_START)
        self.assertEqual(config.codebook_size, CODEBOOK_SIZE)
        self.assertEqual(config.token_end, TOKEN_START + 3 * CODEBOOK_SIZE)
        self.assertEqual(
            config.tensor_path, str(model_root / TRIE_OUTPUT_HEAD_FILENAME)
        )

    def test_declared_model_without_artifact_raises(self):
        declared = PretrainedConfig(**{TRIE_OUTPUT_HEAD_CONFIG_KEY: True})
        empty_root = self.root / "empty2"
        (empty_root / "nested").mkdir(parents=True)
        # An artifact below the model root is not the model's artifact.
        self.write_artifact(directory=empty_root / "nested")
        with self.assertRaises(OSError):
            self.discover(declared, empty_root)

    def test_token_range_must_fit_vocabulary(self):
        declared = PretrainedConfig(**{TRIE_OUTPUT_HEAD_CONFIG_KEY: True})
        model_root = self.declared_model()
        token_end = TOKEN_START + 3 * CODEBOOK_SIZE
        self.assertIsNotNone(self.discover(declared, model_root, token_end))
        with self.assertRaisesRegex(ValueError, "exceeds the model vocabulary"):
            self.discover(declared, model_root, token_end - 1)

    def test_rejects_invalid_metadata(self):
        for name, changes in (
            ("format_version", {"format_version": "2"}),
            ("num_codebooks", {"num_codebooks": "2"}),
            ("num_codebooks", {"num_codebooks": "4"}),
            ("codebook_size", {"codebook_size": "0"}),
            ("token_start", {"token_start": "-1"}),
        ):
            with self.subTest(name=name, changes=changes):
                path = self.write_artifact(metadata={**_metadata(), **changes})
                with self.assertRaises(ValueError):
                    load_trie_output_head_config(path)

    def test_rejects_missing_or_non_integer_metadata(self):
        metadata = _metadata()
        del metadata["token_start"]
        path = self.write_artifact(metadata=metadata)
        with self.assertRaisesRegex(ValueError, "integer 'token_start'"):
            load_trie_output_head_config(path)
        path = self.write_artifact(metadata={**_metadata(), "codebook_size": "x"})
        with self.assertRaisesRegex(ValueError, "integer 'codebook_size'"):
            load_trie_output_head_config(path)

    def test_rejects_missing_artifact_file(self):
        with self.assertRaisesRegex(ValueError, "do not exist"):
            load_trie_output_head_config(str(self.root / "nope.safetensors"))


if __name__ == "__main__":
    unittest.main()

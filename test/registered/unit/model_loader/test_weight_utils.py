"""Unit tests for srt/model_loader/weight_utils.py."""

import json
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    probe_safetensors_weight_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


def _write_safetensors(folder, filename, weight_name, dtype):
    size = 1 if dtype.startswith("F8_") else 2
    header = json.dumps(
        {weight_name: {"dtype": dtype, "shape": [1], "data_offsets": [0, size]}}
    ).encode()
    with open(os.path.join(folder, filename), "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + b"\0" * size)


class TestProbeSafetensorsWeightDtype(CustomTestCase):
    _WEIGHT_NAME = "model.layers.0.attn.wo_a.weight"
    _SUFFIX = ".wo_a.weight"

    def test_local_indexed_and_single_file_checkpoints(self):
        for filename, indexed, dtype in (
            ("model-00001-of-00001.safetensors", True, "BF16"),
            ("model.safetensors", False, "F8_E4M3"),
        ):
            with self.subTest(indexed=indexed), tempfile.TemporaryDirectory() as folder:
                if indexed:
                    _write_index(folder, {self._WEIGHT_NAME: filename})
                _write_safetensors(folder, filename, self._WEIGHT_NAME, dtype)

                self.assertEqual(
                    probe_safetensors_weight_dtype(
                        folder, self._SUFFIX, revision="revision", cache_dir="/models"
                    ),
                    dtype,
                )

    def test_local_unindexed_shards_are_scanned_until_match(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_safetensors(
                folder, "model-00001.safetensors", "model.other.weight", "BF16"
            )
            _write_safetensors(
                folder, "model-00002.safetensors", self._WEIGHT_NAME, "BF16"
            )

            self.assertEqual(
                probe_safetensors_weight_dtype(folder, self._SUFFIX), "BF16"
            )

    @patch("huggingface_hub.parse_safetensors_file_metadata")
    @patch("transformers.utils.hub.cached_file")
    @patch("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    def test_remote_checkpoint_reads_one_matching_shard_header(self, cached, parse):
        second_name = "model.layers.1.attn.wo_a.weight"
        with tempfile.TemporaryDirectory() as folder:
            _write_index(
                folder,
                {
                    self._WEIGHT_NAME: "model-00001-of-00002.safetensors",
                    second_name: "model-00002-of-00002.safetensors",
                },
            )
            cached.side_effect = [os.path.join(folder, INDEX_NAME), None]
            parse.return_value = SimpleNamespace(
                tensors={self._WEIGHT_NAME: SimpleNamespace(dtype="BF16")}
            )

            dtype = probe_safetensors_weight_dtype(
                "org/model", self._SUFFIX, revision="resolved", cache_dir="/models"
            )

        self.assertEqual(dtype, "BF16")
        self.assertEqual(
            cached.call_args_list,
            [
                call(
                    "org/model",
                    INDEX_NAME,
                    revision="resolved",
                    cache_dir="/models",
                    _raise_exceptions_for_missing_entries=False,
                ),
                call(
                    "org/model",
                    "model-00001-of-00002.safetensors",
                    revision="resolved",
                    cache_dir="/models",
                    local_files_only=True,
                    _raise_exceptions_for_missing_entries=False,
                ),
            ],
        )
        parse.assert_called_once_with(
            "org/model",
            "model-00001-of-00002.safetensors",
            revision="resolved",
        )


class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_shard_raises(self):
        # Index lists two shards, only one on disk (interrupted download).
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        present = _touch(self.folder, "model-00001-of-00002.safetensors")

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[present],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_complete_checkpoint_filters_non_indexed(self):
        # All indexed shards present; a non-indexed duplicate is still filtered out.
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        shard1 = _touch(self.folder, "model-00001-of-00002.safetensors")
        shard2 = _touch(self.folder, "model-00002-of-00002.safetensors")
        extra = _touch(self.folder, "consolidated.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[shard1, shard2, extra],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(sorted(result), sorted([shard1, shard2]))

    def test_missing_shard_outside_allow_patterns_is_ignored(self):
        # Cosmos3-style checkpoints use one root index for multiple subfolder
        # weight sources. Loading the transformer source should not require the
        # vision encoder shard to already be present; the secondary source
        # downloads and loads it separately.
        _write_index(
            self.folder,
            {
                "llm": "transformer/diffusion_pytorch_model.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        transformer = _touch(
            self.folder, "transformer/diffusion_pytorch_model.safetensors"
        )

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[transformer],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
            allow_patterns=["transformer/*.safetensors"],
        )
        self.assertEqual(result, [transformer])

    def test_missing_shard_inside_allow_patterns_raises(self):
        _write_index(
            self.folder,
            {
                "llm1": "transformer/model-00001-of-00002.safetensors",
                "llm2": "transformer/model-00002-of-00002.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        transformer = _touch(
            self.folder, "transformer/model-00001-of-00002.safetensors"
        )

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[transformer],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
                allow_patterns=["transformer/*.safetensors"],
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


if __name__ == "__main__":
    unittest.main()

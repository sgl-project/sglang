"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
)
from sglang.srt.utils import runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.muse_spark_utils import try_cached_muse_spark_model
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
        shard = "transformer/diffusion_pytorch_model.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with self.subTest(folder=folder):
                transformer = os.path.join(folder, shard)
                result = filter_duplicate_safetensors_files(
                    hf_weights_files=[transformer],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
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
        shard = "transformer/model-00001-of-00002.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with (
                self.subTest(folder=folder),
                self.assertRaisesRegex(
                    RuntimeError, r"model-00002-of-00002\.safetensors"
                ),
            ):
                filter_duplicate_safetensors_files(
                    hf_weights_files=[os.path.join(folder, shard)],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )

    def test_local_index_allows_subset_of_existing_shards(self):
        _write_index(
            self.folder,
            {"w1": "first.safetensors", "w2": "second.safetensors"},
        )
        first = _touch(self.folder, "first.safetensors")
        _touch(self.folder, "second.safetensors")

        result = filter_duplicate_safetensors_files([first], self.folder, INDEX_NAME)

        self.assertEqual(result, [first])

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestMaybeAddMtpSafetensors(CustomTestCase):
    def test_remote_mtp_guards(self):
        folder = "s3://bucket/model"
        model = f"{folder}/model.safetensors"
        mtp = f"{folder}/mtp.safetensors"
        cases = (
            ("Glm4MoeForCausalLM", 1, [model, mtp], [model, mtp], False),
            ("Glm4MoeForCausalLM", 1, [model], [model], True),
            ("Glm4MoeForCausalLM", 0, [model], [model, mtp], False),
            ("LlamaForCausalLM", 1, [model], [model, mtp], False),
        )
        for arch, nextn, selected, available, may_list in cases:
            with (
                self.subTest(arch=arch, nextn=nextn, selected=selected),
                patch.object(
                    runai_utils, "list_safetensors", return_value=available
                ) as listing,
            ):
                config = SimpleNamespace(
                    architectures=[arch], num_nextn_predict_layers=nextn
                )
                self.assertEqual(
                    maybe_add_mtp_safetensors(selected, folder, INDEX_NAME, config),
                    selected,
                )
                if not may_list:
                    listing.assert_not_called()

    def test_local_unindexed_mtp(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _touch(folder, "model.safetensors")
            mtp = _touch(folder, "mtp.safetensors")
            config = SimpleNamespace(
                architectures=["Glm4MoeLiteForCausalLMNextN"],
                num_nextn_predict_layers=1,
            )
            with patch.object(runai_utils, "list_safetensors") as listing:
                self.assertEqual(
                    maybe_add_mtp_safetensors([model], folder, INDEX_NAME, config),
                    [model, mtp],
                )
                listing.assert_not_called()


class TestMuseSparkCachedModel(CustomTestCase):
    """A partial HF snapshot must not replace a downloadable model ID."""

    def test_cached_checkpoint_completeness(self):
        repo = "test/model"
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "snapshot"
            snapshot.mkdir()
            for name in (
                "config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "l4_200k_base",
                "chat_template.jinja",
                "generation_config.json",
                "hf_quant_config.json",
            ):
                (snapshot / name).write_text("{}")
            shard = snapshot / "model-00001-of-00002.safetensors"
            shard.write_bytes(b"weights")
            other = snapshot / "model-00002-of-00002.safetensors"
            index = snapshot / INDEX_NAME
            index.write_text(
                json.dumps({"weight_map": {"a": shard.name, "b": other.name}})
            )

            with (
                patch.dict(os.environ, {"DEFAULT_MODEL_CACHE_DIR": ""}),
                patch(
                    "sglang.test.muse_spark_utils.find_local_repo_dir",
                    return_value=str(snapshot),
                ),
            ):
                resolve = lambda: try_cached_muse_spark_model(repo)
                # An interrupted download still needs the Hub fallback.
                self.assertEqual(resolve(), repo)
                other.write_bytes(b"weights")
                self.assertEqual(resolve(), str(snapshot))

                # Required processor assets and weight symlinks must be usable.
                (snapshot / "processor_config.json").unlink()
                self.assertEqual(resolve(), repo)
                (snapshot / "processor_config.json").write_text("{}")
                other.unlink()
                other.symlink_to(snapshot / "missing-blob")
                self.assertEqual(resolve(), repo)
                other.unlink()
                other.touch()
                self.assertEqual(resolve(), repo)
                other.write_bytes(b"weights")
                for data in ("{", "{}", '{"weight_map": {}}'):
                    index.write_text(data)
                    self.assertEqual(resolve(), repo)

    def test_single_file_draft_and_default_cache(self):
        # DFlash has config + weights, with no tokenizer or processor of its own.
        repo = "test/draft"
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / repo
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            with patch.dict(os.environ, {"DEFAULT_MODEL_CACHE_DIR": folder}):
                self.assertEqual(try_cached_muse_spark_model(repo, draft=True), repo)
                (snapshot / "model.safetensors").write_bytes(b"weights")
                self.assertEqual(
                    try_cached_muse_spark_model(repo, draft=True), str(snapshot)
                )

    def test_cache_miss_keeps_model_id(self):
        with (
            patch.dict(os.environ, {"DEFAULT_MODEL_CACHE_DIR": ""}),
            patch(
                "sglang.test.muse_spark_utils.find_local_repo_dir", return_value=None
            ),
        ):
            self.assertEqual(
                try_cached_muse_spark_model("test/missing", draft=True), "test/missing"
            )

    def test_incomplete_default_cache_does_not_hide_hf_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "test/model").mkdir(parents=True)
            snapshot = Path(folder) / "hf-snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            with (
                patch.dict(os.environ, {"DEFAULT_MODEL_CACHE_DIR": folder}),
                patch(
                    "sglang.test.muse_spark_utils.find_local_repo_dir",
                    return_value=str(snapshot),
                ),
            ):
                self.assertEqual(
                    try_cached_muse_spark_model("test/model", draft=True), str(snapshot)
                )


class TestInitializeDummyWeights(CustomTestCase):
    def test_nvfp4_dummy_scales_are_neutral(self):
        model = torch.nn.Linear(8, 4, bias=False)
        for name, shape in (
            ("w13_weight_scale_2", (4, 2)),
            ("w2_weight_scale_2", (4,)),
        ):
            model.register_parameter(
                name, torch.nn.Parameter(torch.empty(shape), requires_grad=False)
            )

        initialize_dummy_weights(model, low=0.25, high=0.75)

        self.assertTrue(torch.equal(model.w13_weight_scale_2, torch.ones(4, 2)))
        self.assertTrue(torch.equal(model.w2_weight_scale_2, torch.ones(4)))
        self.assertTrue(torch.all((model.weight >= 0.25) & (model.weight <= 0.75)))


if __name__ == "__main__":
    unittest.main()

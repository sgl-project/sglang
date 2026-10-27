"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import (
    _NP_CACHE_INDEX_VERSION,
    filter_duplicate_safetensors_files,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
    np_cache_weights_iterator,
)
from sglang.srt.utils import runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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


class TestNpCacheWeightNames(CustomTestCase):
    """np_cache must not let a checkpoint choose where its cache files go.

    A tensor name is data read out of the checkpoint. It used to be joined onto
    the cache directory as a file name, so a name containing ``..`` placed the
    .npy file outside that directory on the write pass and read it back from
    there on later loads.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _dump(self, state_dict):
        """Run the iterator over a one-shard .bin holding ``state_dict``."""
        hf_folder = os.path.join(self.folder, "checkpoint")
        os.makedirs(hf_folder, exist_ok=True)
        bin_file = os.path.join(hf_folder, "pytorch_model.bin")
        torch.save(state_dict, bin_file)
        loaded = dict(np_cache_weights_iterator(hf_folder, None, hf_folder, [bin_file]))
        return hf_folder, loaded

    def test_traversing_tensor_name_stays_inside_the_cache_dir(self):
        outside = os.path.join(self.folder, "outside")
        os.makedirs(outside, exist_ok=True)
        victim = os.path.join(outside, "victim")
        with open(victim, "w") as f:
            f.write("untouched")

        name = "../../outside/victim"
        hf_folder, loaded = self._dump({name: torch.zeros(4)})

        with open(victim) as f:
            self.assertEqual(f.read(), "untouched")
        self.assertEqual(
            sorted(os.listdir(outside)),
            ["victim"],
            "np_cache wrote outside its own directory",
        )
        # the weight is still served under its real name
        self.assertEqual(list(loaded), [name])
        self.assertEqual(loaded[name].shape, (4,))

    def test_separator_in_tensor_name_round_trips(self):
        # A name with a separator used to need a matching subdirectory to exist
        # and raised FileNotFoundError when it did not.
        name = "encoder/layer.0/weight"
        _, loaded = self._dump({name: torch.ones(2)})
        self.assertEqual(list(loaded), [name])
        self.assertTrue(torch.equal(loaded[name], torch.ones(2)))

    def test_stale_index_is_rebuilt_not_trusted(self):
        hf_folder, _ = self._dump({"w": torch.zeros(1)})
        index_file = os.path.join(hf_folder, "np", "weight_names.json")

        # the pre-fix on-disk format: a bare list of checkpoint-chosen paths
        with open(index_file, "w") as f:
            json.dump(["../../escaped"], f)

        bin_file = os.path.join(hf_folder, "pytorch_model.bin")
        loaded = dict(np_cache_weights_iterator(hf_folder, None, hf_folder, [bin_file]))
        self.assertEqual(list(loaded), ["w"])
        with open(index_file) as f:
            self.assertEqual(json.load(f)["version"], _NP_CACHE_INDEX_VERSION)

    def test_index_naming_a_path_is_rejected(self):
        hf_folder, _ = self._dump({"w": torch.zeros(1)})
        index_file = os.path.join(hf_folder, "np", "weight_names.json")

        # current layout version, but an entry that points out of np/
        with open(index_file, "w") as f:
            json.dump(
                {
                    "version": _NP_CACHE_INDEX_VERSION,
                    "weights": [["w", "../../escaped.npy"]],
                },
                f,
            )

        bin_file = os.path.join(hf_folder, "pytorch_model.bin")
        with self.assertRaises(ValueError):
            dict(np_cache_weights_iterator(hf_folder, None, hf_folder, [bin_file]))

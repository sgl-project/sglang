"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    maybe_filter_nextn_shards,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
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

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestMaybeFilterNextnShards(CustomTestCase):
    """A combined target+MTP checkpoint: 47 shards, main layers 0..77, and the
    NextN head (layer 78 with eh_proj/enorm/hnorm/shared_head) in shards 45-47.
    Mirrors palmyra-x6-...-mtp-v2."""

    N = 47

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name
        self.shards = [
            f"model-{i:05d}-of-{self.N:05d}.safetensors" for i in range(1, self.N + 1)
        ]
        wm = {}
        # main body: layers 0..77 spread across shards 1..44
        for layer in range(78):
            shard = self.shards[layer % 44]
            wm[f"model.layers.{layer}.self_attn.q_proj.weight"] = shard
        wm["model.embed_tokens.weight"] = self.shards[0]  # shard 1 (shared)
        wm["lm_head.weight"] = self.shards[44]  # shard 45 (shared)
        wm["model.norm.weight"] = self.shards[44]  # shard 45
        # NextN head = layer 78, in shards 45/46/47, with the marker modules
        wm["model.layers.78.eh_proj.weight"] = self.shards[44]
        wm["model.layers.78.enorm.weight"] = self.shards[44]
        wm["model.layers.78.hnorm.weight"] = self.shards[45]
        wm["model.layers.78.self_attn.q_proj.weight"] = self.shards[45]
        wm["model.layers.78.mlp.experts.0.down_proj.weight"] = self.shards[46]
        wm["model.layers.78.shared_head.norm.weight"] = self.shards[46]
        self.weight_map = wm
        _write_index(self.folder, wm)
        self.all_files = [_touch(self.folder, s) for s in self.shards]

    def tearDown(self):
        self._tmp.cleanup()

    def _cfg(self, arch, **extra):
        return SimpleNamespace(architectures=[arch], **extra)

    def test_draft_load_keeps_only_mtp_shards(self):
        result = maybe_filter_nextn_shards(
            self.all_files,
            self.folder,
            INDEX_NAME,
            self._cfg("GlmMoeDsaForCausalLMNextN"),
        )
        expected = sorted(
            os.path.join(self.folder, self.shards[i]) for i in (44, 45, 46)
        )  # shards 45,46,47
        self.assertEqual(sorted(result), expected)

    def test_target_load_is_untouched(self):
        # Non-NextN architecture (the target) must load every shard.
        result = maybe_filter_nextn_shards(
            self.all_files, self.folder, INDEX_NAME, self._cfg("GlmMoeDsaForCausalLM")
        )
        self.assertEqual(result, self.all_files)

    def test_num_hidden_layers_rewrite_does_not_mislead(self):
        # Some archs reset num_hidden_layers (e.g. to the nextn count). Because we
        # discover the MTP layer from the checkpoint markers, a bogus config value
        # cannot make us drop the real head shards.
        result = maybe_filter_nextn_shards(
            self.all_files,
            self.folder,
            INDEX_NAME,
            self._cfg("GlmMoeDsaForCausalLMNextN", num_hidden_layers=1),
        )
        expected = sorted(
            os.path.join(self.folder, self.shards[i]) for i in (44, 45, 46)
        )
        self.assertEqual(sorted(result), expected)

    def test_no_markers_returns_unchanged(self):
        # Checkpoint whose MTP modules aren't the known markers -> safe no-op.
        with tempfile.TemporaryDirectory() as d:
            wm = {f"model.layers.{i}.w": f"s{i}.safetensors" for i in range(3)}
            _write_index(d, wm)
            files = [_touch(d, f"s{i}.safetensors") for i in range(3)]
            result = maybe_filter_nextn_shards(
                files, d, INDEX_NAME, self._cfg("SomethingForCausalLMNextN")
            )
            self.assertEqual(result, files)

    def test_separately_added_mtp_file_is_kept(self):
        # A mtp.safetensors not described by the index (GLM4Moe packaging bug,
        # appended by maybe_add_mtp_safetensors) must survive filtering.
        extra = _touch(self.folder, "mtp.safetensors")
        result = maybe_filter_nextn_shards(
            self.all_files + [extra],
            self.folder,
            INDEX_NAME,
            self._cfg("GlmMoeDsaForCausalLMNextN"),
        )
        self.assertIn(extra, result)

    def test_no_index_returns_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            files = [_touch(d, "model.safetensors")]
            result = maybe_filter_nextn_shards(
                files, d, INDEX_NAME, self._cfg("FooForCausalLMNextN")
            )
            self.assertEqual(result, files)


if __name__ == "__main__":
    unittest.main()

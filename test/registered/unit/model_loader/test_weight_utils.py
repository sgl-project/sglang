import json
import os
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub import HfApi, hf_hub_download

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import get_safetensors_weight_files
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class TestWeightDiscovery(unittest.TestCase):
    SUBDIR_MODEL = "ModelCloud/Llama-3.2-1B-Instruct-GPTQ-subdirectories-safetensors"
    ONE_LEVEL_MODEL = "ModelCloud/Llama-3.2-1B-gptqmodel-ci-4bit"

    def _prepare_weights(self, model_path: str):
        loader = DefaultModelLoader(LoadConfig())
        with patch(
            "sglang.srt.model_loader.loader.get_global_server_args", return_value=None
        ):
            _, weight_files, use_safetensors = loader._prepare_weights(
                model_path, revision=None, fall_back_to_pt=False
            )
        self.assertTrue(use_safetensors)
        return weight_files

    def test_subdirectory_safetensors_model_uses_index_weight_map(self):
        index_path = hf_hub_download(
            repo_id=self.SUBDIR_MODEL,
            filename="model.safetensors.index.json",
        )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for relative_path in set(weight_map.values()):
                abs_path = os.path.join(tmpdir, relative_path)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                open(abs_path, "a").close()

            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)

            open(os.path.join(tmpdir, "unrelated.safetensors"), "a").close()

            weight_files = self._prepare_weights(tmpdir)

            self.assertEqual(
                weight_files,
                sorted(
                    {
                        os.path.join(tmpdir, relative_path)
                        for relative_path in weight_map.values()
                    }
                ),
            )

    def test_one_level_safetensors_model_does_not_recurse_without_index(self):
        repo_files = set(HfApi().list_repo_files(repo_id=self.ONE_LEVEL_MODEL))
        self.assertIn("model.safetensors", repo_files)
        self.assertNotIn("model.safetensors.index.json", repo_files)

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model.safetensors"), "a").close()
            os.makedirs(os.path.join(tmpdir, "nested"))
            open(os.path.join(tmpdir, "nested", "unrelated.safetensors"), "a").close()

            weight_files = self._prepare_weights(tmpdir)

            self.assertEqual(weight_files, [os.path.join(tmpdir, "model.safetensors")])


class TestSafetensorsWeightFileDiscovery(unittest.TestCase):
    def test_prefers_index_weight_map_for_both_loaders(self):
        weight_map = {
            "model.layer0.weight": "shards/model-00001-of-00002.safetensors",
            "model.layer1.weight": "shards/model-00002-of-00002.safetensors",
            "model.layer1.bias": "shards/model-00002-of-00002.safetensors",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for relative_path in set(weight_map.values()):
                abs_path = os.path.join(tmpdir, relative_path)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                open(abs_path, "a").close()

            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)

            open(os.path.join(tmpdir, "model.safetensors"), "a").close()

            expected = sorted(
                {
                    os.path.join(tmpdir, relative_path)
                    for relative_path in weight_map.values()
                }
            )
            self.assertEqual(
                get_safetensors_weight_files(tmpdir, "model.safetensors.index.json"),
                expected,
            )

    def test_without_index_only_discovers_top_level_safetensors_for_both_loaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model.safetensors"), "a").close()
            os.makedirs(os.path.join(tmpdir, "nested"))
            open(os.path.join(tmpdir, "nested", "ignored.safetensors"), "a").close()

            expected = [os.path.join(tmpdir, "model.safetensors")]
            self.assertEqual(
                get_safetensors_weight_files(tmpdir, "model.safetensors.index.json"),
                expected,
            )


if __name__ == "__main__":
    unittest.main()

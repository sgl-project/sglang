"""Model-config parser auto-selection by checkpoint layout — local dirs, no network."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import huggingface_hub.constants as hub_constants

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.utils.hf_transformers import config as config_mod
from sglang.srt.utils.hf_transformers import processor as processor_mod


class _Picked(Exception):
    pass


class _RecordingParser:
    def __init__(self, name):
        self.name = name

    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        raise _Picked(self.name)


def _picked_parser(files):
    with tempfile.TemporaryDirectory() as d:
        for name in files:
            with open(os.path.join(d, name), "w") as f:
                f.write("{}")
        return _config_route(d)


def _config_route(model, revision=None):
    with mock.patch.object(config_mod, "get_model_config_parser", _RecordingParser):
        try:
            config_mod.get_config(model, trust_remote_code=False, revision=revision)
        except _Picked as picked:
            return picked.args[0]
    raise AssertionError("get_config did not reach a model-config parser")


def _processor_route(model, revision=None):
    with (
        mock.patch.object(
            processor_mod, "load_mistral_config", side_effect=_Picked("mistral")
        ),
        mock.patch.object(
            processor_mod.AutoConfig, "from_pretrained", side_effect=_Picked("hf")
        ),
    ):
        try:
            processor_mod.get_processor(model, revision=revision)
        except _Picked as picked:
            return picked.args[0]
    raise AssertionError("get_processor did not reach a config loader")


class TestMistralCheckpointFormat(CustomTestCase):
    def test_native_weights_with_config_json_use_mistral_parser(self):
        files = [
            "params.json",
            "config.json",
            "consolidated.safetensors.index.json",
            "consolidated-00001-of-00002.safetensors",
            "consolidated-00002-of-00002.safetensors",
        ]
        self.assertEqual(_picked_parser(files), "mistral")

    def test_params_only_checkpoint_uses_mistral_parser(self):
        files = ["params.json", "consolidated.safetensors"]
        self.assertEqual(_picked_parser(files), "mistral")

    def test_hf_layout_uses_hf_parser(self):
        files = ["config.json", "model-00001-of-00001.safetensors"]
        self.assertEqual(_picked_parser(files), "hf")

    def test_dual_layout_keeps_hf_parser(self):
        files = [
            "params.json",
            "config.json",
            "consolidated.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        self.assertEqual(_picked_parser(files), "hf")


class TestMistralHubCheckpointFormat(CustomTestCase):
    """A Hub cache holding only config.json must not decide the format."""

    REPO = "org/model"
    REVISION = "a" * 40

    def _routes(self, remote_files):
        with tempfile.TemporaryDirectory() as cache:
            snapshot = os.path.join(
                cache, "models--org--model", "snapshots", self.REVISION
            )
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as f:
                f.write("{}")
            info = SimpleNamespace(
                siblings=[SimpleNamespace(rfilename=f) for f in remote_files]
            )
            with (
                mock.patch.object(hub_constants, "HF_HUB_CACHE", cache),
                mock.patch.object(hub_constants, "HF_HUB_OFFLINE", False),
                mock.patch("huggingface_hub.HfApi.model_info", return_value=info),
                mock.patch(
                    "huggingface_hub.HfApi.file_exists",
                    side_effect=lambda repo, filename, **kw: filename in remote_files,
                ),
            ):
                return (
                    _config_route(self.REPO, self.REVISION),
                    _processor_route(self.REPO, self.REVISION),
                )

    def test_config_only_cache_of_native_repo_uses_mistral(self):
        remote = [
            "params.json",
            "config.json",
            "consolidated-00001-of-00002.safetensors",
        ]
        self.assertEqual(self._routes(remote), ("mistral", "mistral"))

    def test_config_only_cache_of_hf_repo_uses_hf(self):
        remote = ["config.json", "model-00001-of-00002.safetensors"]
        self.assertEqual(self._routes(remote), ("hf", "hf"))


if __name__ == "__main__":
    unittest.main()

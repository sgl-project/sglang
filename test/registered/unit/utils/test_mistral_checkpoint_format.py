"""Model-config parser auto-selection by checkpoint layout — local dirs, no network."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.utils.hf_transformers import config as config_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Picked(Exception):
    pass


def _config_route(model, revision=None):
    with mock.patch.object(config_mod, "get_model_config_parser") as parser:
        parser.side_effect = lambda name: SimpleNamespace(
            parse=mock.Mock(side_effect=_Picked(name))
        )
        try:
            config_mod.get_config(model, trust_remote_code=False, revision=revision)
        except _Picked as picked:
            return picked.args[0]
    raise AssertionError("get_config did not reach a model-config parser")


class TestMistralCheckpointFormat(CustomTestCase):
    def test_local_checkpoint_routes(self):
        for files, expected in (
            (
                (
                    "params.json",
                    "config.json",
                    "consolidated.safetensors.index.json",
                    "consolidated-00001-of-00002.safetensors",
                    "consolidated-00002-of-00002.safetensors",
                ),
                "mistral",
            ),
            (("params.json", "consolidated.safetensors"), "mistral"),
            (("config.json", "model-00001-of-00001.safetensors"), "hf"),
            (
                (
                    "params.json",
                    "config.json",
                    "consolidated.safetensors",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ),
                "hf",
            ),
        ):
            with self.subTest(files=files), tempfile.TemporaryDirectory() as checkpoint:
                for name in files:
                    with open(os.path.join(checkpoint, name), "w") as f:
                        f.write("{}")
                self.assertEqual(_config_route(checkpoint), expected)


if __name__ == "__main__":
    unittest.main()

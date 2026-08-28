"""`get_model_config()` caches, and the key is the path the record carried.

Two movements of a `model_path` reach this cache, and only the first one means
the cached configuration describes the wrong checkpoint:

- the record's own path moves, because the GGUF and ModelScope handlers declare
  a local path for a Hub reference;
- `ModelConfig` moves its own path to the local pull directory when the weights
  sit behind an object-store URI, while still describing the checkpoint the
  record asked for.

So the key is the path the record carried when the cache was filled, and these
cases pin both movements against it, on a raw record and on a resolved one.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import EnvField, envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

_OBJECT_STORE_URI = "gs://sglang-test-bucket/mini-llama/"


class TestTheModelConfigCache(CustomTestCase):
    def setUp(self):
        # Resolution writes the environment on the way through
        # (`SGLANG_USE_CUDA_IPC_TRANSPORT` among others) and flips the
        # descriptor-level flag `EnvField.set()` keeps, which `os.environ` does
        # not carry. The cases that resolve put both back.
        fields = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    fields[name] = field
        state = (
            dict(os.environ),
            {name: field._set_to_none for name, field in fields.items()},
        )
        self.addCleanup(self._restore, state)

    @staticmethod
    def _restore(state):
        saved_environ, saved_none_flags = state
        os.environ.clear()
        os.environ.update(saved_environ)
        for name, was_none in saved_none_flags.items():
            getattr(type(envs), name)._set_to_none = was_none

    def _checkpoint(self) -> str:
        directory = tempfile.mkdtemp(prefix="model_config_cache_")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        with open(os.path.join(directory, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return directory

    def _pulled_to(self, directory: str) -> None:
        """The launcher has already pulled the metadata for the URI.

        Both entry points answer with the local directory, which is what a
        second process on the same host sees: the download runs once, in the
        launcher, and everything after it resolves the path.
        """
        for name in ("download_and_get_path", "get_path"):
            original = getattr(runai_utils.ObjectStorageModel, name)
            self.addCleanup(setattr, runai_utils.ObjectStorageModel, name, original)
            setattr(
                runai_utils.ObjectStorageModel,
                name,
                classmethod(lambda cls, model_path, _dir=directory: _dir),
            )

    def _resolved(self, **kwargs) -> ServerArgs:
        # device="cuda" keeps the golden path host-independent: an
        # accelerator-less runner resolves only the base platform, where
        # get_device() raises.
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("random_seed", 42)
        server_args = ServerArgs(**kwargs)
        server_args.resolve_once()
        return server_args

    def test_a_configuration_that_repoints_itself_stays_cached(self):
        """The object-store shape: the configuration's own path moved."""
        pulled = self._checkpoint()
        self._pulled_to(pulled)

        server_args = self._resolved(
            model_path=_OBJECT_STORE_URI, load_format="runai_streamer"
        )
        cached = server_args.__dict__["_model_config"]
        self.assertIsInstance(cached, ModelConfig)
        # The record still carries the URI the operator typed, and the
        # configuration carries the directory it read the metadata from.
        self.assertEqual(server_args.model_path, _OBJECT_STORE_URI)
        self.assertEqual(cached.model_path, pulled)

        self.assertIs(server_args.get_model_config(), cached)

    def test_a_declared_model_path_rebuilds_the_configuration(self):
        """The GGUF and ModelScope shape: the record's own path moved."""
        first_checkpoint = self._checkpoint()
        second_checkpoint = self._checkpoint()

        server_args = ServerArgs(model_path=first_checkpoint, device="cuda")
        first = server_args.get_model_config()
        self.assertEqual(first.model_path, first_checkpoint)

        server_args._declare(
            "test_a_declared_model_path_rebuilds_the_configuration",
            model_path=second_checkpoint,
        )
        second = server_args.get_model_config()
        self.assertIsNot(second, first)
        self.assertEqual(second.model_path, second_checkpoint)

    def test_the_cache_refills_on_a_resolved_record(self):
        """A rebuild has to be storable wherever the key can invalidate.

        The record is read-only once resolution has finished, and the cache is
        the record's own bookkeeping, so the guard lets the refill through.
        """
        first_checkpoint = self._checkpoint()
        second_checkpoint = self._checkpoint()

        server_args = self._resolved(model_path=first_checkpoint)
        copy_ = server_args.replace_resolved(
            "test_the_cache_refills_on_a_resolved_record",
            model_path=second_checkpoint,
        )

        rebuilt = copy_.get_model_config()
        self.assertEqual(rebuilt.model_path, second_checkpoint)
        self.assertIs(copy_.get_model_config(), rebuilt)
        # The parent keeps the configuration it resolved with.
        self.assertEqual(server_args.get_model_config().model_path, first_checkpoint)

    def test_a_supplied_configuration_is_handed_back(self):
        """A configuration nothing in here built carries no key, so nothing
        invalidates it."""
        server_args = ServerArgs(model_path=self._checkpoint(), device="cuda")
        stand_in = SimpleNamespace(model_path="somewhere/else")
        server_args._model_config = stand_in

        self.assertIs(server_args.get_model_config(), stand_in)


if __name__ == "__main__":
    unittest.main()

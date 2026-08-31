"""The model-source axis, on PR CI.

Four ways a model path can name something that is not a local directory, and
until now only one of them was checked before merge:

- an object-store URI (``s3://`` / ``gs://`` / ``az://``), covered by
  ``test_model_config_cache.py`` and, end to end, by a ``nightly`` test;
- a Hub reference to a ``.gguf`` file;
- a ModelScope repo id;
- a remote-connector URL, which is any other ``scheme://`` and is reached by a
  different arm of ``ModelConfig`` than the object-store one.

The last three had no registered test at all. That is how a change to
``get_model_config()``'s cache semantics went green through PR CI and broke two
days later in the nightly: the axis it broke was not being looked at.

None of this needs a network. The GGUF arm asks one resolver for a local path,
the ModelScope arm returns any path that already exists on disk untouched and
otherwise goes through two imports that can be stood in for, and the
remote-connector arm goes through one factory. Each case stubs exactly that
seam and checks what the handler declares -- and, where the path moves, that
the model-configuration cache notices.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import sglang.srt.connector as connector_module
from sglang.srt.arg_groups.model_path_hook import (
    handle_modelscope_paths,
    resolve_hf_gguf_model_path,
)
from sglang.srt.arg_groups.overrides import model_config_of, resolving_view
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

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

_GGUF_REFERENCE = "owner/repo"
_MODELSCOPE_REPO = "org/model"
_REMOTE_URL = "redis://host:6379/mini-llama"


class _ModelSourceCase(CustomTestCase):
    def _directory(self) -> str:
        directory = tempfile.mkdtemp(prefix="model_source_")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return directory

    def _checkpoint(self) -> str:
        directory = self._directory()
        with open(os.path.join(directory, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return directory

    def _gguf_file(self) -> str:
        path = os.path.join(self._directory(), "model.gguf")
        open(path, "w").close()
        return path


class TestTheGgufArm(_ModelSourceCase):
    """`resolve_hf_gguf_model_path` turns a Hub reference into a local file."""

    def _resolving_to(self, resolved):
        """Stand in for the one Hub call, keyed on what it is asked about."""
        table = resolved if isinstance(resolved, dict) else None

        def _resolve(model, revision=None):
            if table is not None:
                return table.get(model)
            return resolved

        return mock.patch(
            "sglang.srt.utils.hf_transformers_utils.resolve_hf_gguf_reference",
            side_effect=_resolve,
        )

    def test_a_hub_reference_declares_the_local_path(self):
        local = self._gguf_file()
        server_args = ServerArgs(model_path=_GGUF_REFERENCE, device="cuda")
        with self._resolving_to(local):
            resolve_hf_gguf_model_path(server_args)

        self.assertEqual(resolving_view(server_args).model_path, local)
        # The record itself still carries what the operator typed.
        self.assertEqual(server_args.model_path, _GGUF_REFERENCE)

    def test_the_tokenizer_follows_only_when_it_was_the_same_reference(self):
        local = self._gguf_file()
        together = ServerArgs(
            model_path=_GGUF_REFERENCE, tokenizer_path=_GGUF_REFERENCE, device="cuda"
        )
        with self._resolving_to(local):
            resolve_hf_gguf_model_path(together)
        self.assertEqual(resolving_view(together).tokenizer_path, local)

        apart = ServerArgs(
            model_path=_GGUF_REFERENCE, tokenizer_path="somewhere/else", device="cuda"
        )
        with self._resolving_to({_GGUF_REFERENCE: local}):
            resolve_hf_gguf_model_path(apart)
        self.assertEqual(resolving_view(apart).tokenizer_path, "somewhere/else")

    def test_a_draft_gguf_is_resolved_on_its_own(self):
        target, draft = self._gguf_file(), self._gguf_file()
        server_args = ServerArgs(
            model_path=_GGUF_REFERENCE,
            speculative_draft_model_path="owner/draft",
            device="cuda",
        )
        with self._resolving_to({_GGUF_REFERENCE: target, "owner/draft": draft}):
            resolve_hf_gguf_model_path(server_args)

        view = resolving_view(server_args)
        self.assertEqual(view.model_path, target)
        self.assertEqual(view.speculative_draft_model_path, draft)

    def test_a_reference_that_is_not_a_gguf_declares_nothing(self):
        server_args = ServerArgs(model_path=_GGUF_REFERENCE, device="cuda")
        with self._resolving_to(None):
            resolve_hf_gguf_model_path(server_args)
        self.assertEqual(resolving_view(server_args).model_path, _GGUF_REFERENCE)

    def test_the_declared_path_invalidates_the_model_configuration(self):
        """The point of pinning the declaration: a configuration built before
        it describes the Hub reference, not the file that was downloaded."""
        first, second = self._checkpoint(), self._checkpoint()
        server_args = ServerArgs(model_path=first, device="cuda")
        before = model_config_of(server_args)
        self.assertEqual(before.model_path, first)

        with self._resolving_to(second):
            resolve_hf_gguf_model_path(server_args)

        after = model_config_of(server_args)
        self.assertIsNot(after, before)
        self.assertEqual(after.model_path, second)


class TestTheModelScopeArm(_ModelSourceCase):
    """`handle_modelscope_paths` resolves repo ids against the local cache."""

    def _modelscope(self, cache_root: str, downloads: dict):
        """Stand in for the two modules the handler imports on a cache miss."""
        calls = []

        def _snapshot_download(path, cache_dir=None, revision=None, **kwargs):
            calls.append((path, cache_dir, revision, kwargs.get("ignore_patterns")))
            return downloads[path]

        hub = types.ModuleType("modelscope.hub.snapshot_download")
        hub.snapshot_download = _snapshot_download
        file_utils = types.ModuleType("modelscope.utils.file_utils")
        file_utils.get_model_cache_root = lambda: cache_root
        modules = {
            "modelscope": types.ModuleType("modelscope"),
            "modelscope.hub": types.ModuleType("modelscope.hub"),
            "modelscope.hub.snapshot_download": hub,
            "modelscope.utils": types.ModuleType("modelscope.utils"),
            "modelscope.utils.file_utils": file_utils,
        }
        return mock.patch.dict(sys.modules, modules), calls

    def test_a_path_already_on_disk_is_left_alone(self):
        """And nothing is imported to decide that -- the arm has to stay usable
        on a host with no modelscope installed."""
        local = self._directory()
        server_args = ServerArgs(model_path=local, tokenizer_path=local, device="cuda")
        imported = {name for name in sys.modules if name.startswith("modelscope")}

        handle_modelscope_paths(server_args)

        view = resolving_view(server_args)
        self.assertEqual(view.model_path, local)
        self.assertEqual(view.tokenizer_path, local)
        self.assertEqual(
            imported, {name for name in sys.modules if name.startswith("modelscope")}
        )

    def test_a_repo_id_resolves_against_the_modelscope_cache(self):
        cache_root = self._directory()
        os.makedirs(os.path.join(cache_root, _MODELSCOPE_REPO))
        patch, _ = self._modelscope(cache_root, {})
        server_args = ServerArgs(
            model_path=_MODELSCOPE_REPO, tokenizer_path=_MODELSCOPE_REPO, device="cuda"
        )
        with patch:
            handle_modelscope_paths(server_args)

        cached = os.path.join(cache_root, _MODELSCOPE_REPO)
        view = resolving_view(server_args)
        self.assertEqual(view.model_path, cached)
        self.assertEqual(view.tokenizer_path, cached)

    def test_a_cache_miss_downloads_and_the_tokenizer_skips_the_weights(self):
        downloaded = self._directory()
        patch, calls = self._modelscope(
            self._directory(), {_MODELSCOPE_REPO: downloaded}
        )
        server_args = ServerArgs(
            model_path=_MODELSCOPE_REPO, tokenizer_path=_MODELSCOPE_REPO, device="cuda"
        )
        with patch:
            handle_modelscope_paths(server_args)

        view = resolving_view(server_args)
        self.assertEqual(view.model_path, downloaded)
        self.assertEqual(view.tokenizer_path, downloaded)
        # The tokenizer download does not drag the weights along with it.
        self.assertEqual(
            [call[3] for call in calls], [None, ["*.bin", "*.safetensors"]]
        )

    def test_the_download_directory_is_searched_before_the_hub(self):
        download_dir = self._directory()
        os.makedirs(os.path.join(download_dir, _MODELSCOPE_REPO))
        patch, calls = self._modelscope(self._directory(), {})
        server_args = ServerArgs(
            model_path=_MODELSCOPE_REPO,
            tokenizer_path=_MODELSCOPE_REPO,
            download_dir=download_dir,
            device="cuda",
        )
        with patch:
            handle_modelscope_paths(server_args)

        self.assertEqual(
            resolving_view(server_args).model_path,
            os.path.join(download_dir, _MODELSCOPE_REPO),
        )
        self.assertEqual(calls, [])

    def test_a_draft_repo_id_is_resolved_with_its_own_revision(self):
        cache_root = self._directory()
        drafted = self._directory()
        patch, calls = self._modelscope(cache_root, {"org/draft": drafted})
        local = self._directory()
        server_args = ServerArgs(
            model_path=local,
            tokenizer_path=local,
            speculative_draft_model_path="org/draft",
            speculative_draft_model_revision="v2",
            device="cuda",
        )
        with patch:
            handle_modelscope_paths(server_args)

        self.assertEqual(
            resolving_view(server_args).speculative_draft_model_path, drafted
        )
        self.assertEqual([call[2] for call in calls], ["v2"])


class TestTheRemoteConnectorArm(_ModelSourceCase):
    """`ModelConfig` repoints itself for any other ``scheme://``.

    `redis://` is the shape the object-store arm does not claim, so it is the
    one that reaches `_maybe_pull_model_tokenizer_from_remote`.
    """

    def _connected_to(self, directory):
        state = {}

        class _Client:
            def pull_files(self, allow_pattern=None):
                state["allow_pattern"] = allow_pattern

            def get_local_dir(self):
                return directory

        return (
            mock.patch.object(
                connector_module, "create_remote_connector", return_value=_Client()
            ),
            state,
        )

    def test_the_configuration_reads_from_the_pulled_directory(self):
        pulled = self._checkpoint()
        patch, state = self._connected_to(pulled)
        with patch:
            config = ModelConfig(model_path=_REMOTE_URL)

        self.assertEqual(config.model_path, pulled)
        # The weights stay where they are; only the metadata was pulled.
        self.assertEqual(config.model_weights, _REMOTE_URL)
        self.assertEqual(state["allow_pattern"], ["*config.json"])

    def test_the_record_keeps_the_url_and_the_cache_stays_keyed_on_it(self):
        """Same movement the object-store arm makes: the configuration's path
        moves, the record's does not, and the cache key follows the record."""
        pulled = self._checkpoint()
        patch, _ = self._connected_to(pulled)
        server_args = ServerArgs(model_path=_REMOTE_URL, device="cuda")
        with patch:
            config = model_config_of(server_args)

        self.assertEqual(server_args.model_path, _REMOTE_URL)
        self.assertEqual(config.model_path, pulled)
        self.assertIs(model_config_of(server_args), config)


if __name__ == "__main__":
    unittest.main()

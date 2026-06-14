"""Unit tests for the Hub-first / local-cache-fallback resolution of
``model_index.json`` in ``maybe_download_model_index``.

Regression coverage for the case where a model is already cached locally but the
Hub cannot be reached or denies access (e.g. a gated repo with no token, or an
offline run): the loader must fall back to the cached copy with a warning
instead of failing outright.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub.errors import EntryNotFoundError

from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils as hfu

VALID_INDEX = {"_class_name": "FluxPipeline", "_diffusers_version": "0.30.0"}


class TestModelIndexCacheFallback(unittest.TestCase):
    def _write_index(self, payload=VALID_INDEX):
        fd, path = tempfile.mkstemp(suffix="_model_index.json")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_hub_success_returns_fresh_config(self):
        """When the Hub is reachable, use the freshly resolved file."""
        path = self._write_index()
        with patch.object(hfu, "maybe_load_overlay_model_index", return_value=None), patch.object(
            hfu, "hf_hub_download", return_value=path
        ):
            cfg = hfu.maybe_download_model_index("org/SomeModel")
        self.assertEqual(cfg["_class_name"], "FluxPipeline")
        self.assertEqual(cfg["pipeline_name"], "FluxPipeline")

    def test_gated_but_cached_falls_back_with_warning(self):
        """Hub access denied + a cached copy present -> use cache and warn."""
        path = self._write_index()
        with patch.object(hfu, "maybe_load_overlay_model_index", return_value=None), patch.object(
            hfu, "hf_hub_download", side_effect=Exception("403 Client Error: gated repo")
        ), patch("huggingface_hub.try_to_load_from_cache", return_value=path), patch.object(
            hfu.logger, "warning"
        ) as mock_warning:
            cfg = hfu.maybe_download_model_index("org/GatedModel")
        self.assertEqual(cfg["_class_name"], "FluxPipeline")
        mock_warning.assert_called_once()

    def test_hub_failure_without_cache_raises(self):
        """Hub access denied + nothing cached -> surface the error."""
        with patch.object(hfu, "maybe_load_overlay_model_index", return_value=None), patch.object(
            hfu, "hf_hub_download", side_effect=Exception("403 Client Error: gated repo")
        ), patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            with self.assertRaises(ValueError):
                hfu.maybe_download_model_index("org/UncachedGatedModel")

    def test_entry_not_found_propagates_for_single_model_repo(self):
        """A repo with no model_index.json must NOT be masked by the cache
        fallback — the EntryNotFoundError has to propagate so the caller can try
        the single-model (config.json) path."""
        with patch.object(
            hfu, "hf_hub_download", side_effect=EntryNotFoundError("no model_index.json")
        ):
            with self.assertRaises(EntryNotFoundError):
                hfu._resolve_remote_model_index_path("org/SingleModelRepo")


if __name__ == "__main__":
    unittest.main()

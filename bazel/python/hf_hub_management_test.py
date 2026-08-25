import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def install_namespace_packages() -> None:
    root = (
        Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / "python/sglang"
    )
    for name, path in (
        ("sglang", root),
        ("sglang.srt", root / "srt"),
        ("sglang.srt.utils", root / "srt" / "utils"),
        (
            "sglang.srt.utils.hf_transformers",
            root / "srt" / "utils" / "hf_transformers",
        ),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


sys.modules.pop("torch", None)
install_namespace_packages()

import huggingface_hub  # noqa: E402

from sglang.srt.utils.hf_transformers import hub  # noqa: E402


class HfHubManagementTest(unittest.TestCase):
    def test_hub_import_is_torch_free(self) -> None:
        self.assertTrue(huggingface_hub.__version__)
        self.assertNotIn("torch", sys.modules)

    def test_local_resolution_does_not_call_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text("{}")
            with mock.patch.object(
                hub, "snapshot_download", side_effect=AssertionError("network")
            ):
                self.assertEqual(hub.download_from_hf(directory), directory)
                self.assertEqual(
                    hub._resolve_local_or_cached_file(directory, "config.json"),
                    str(config),
                )

    def test_gguf_detection_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.bin"
            model.write_bytes(b"GGUFpayload")
            sidecar = Path(directory) / "config.json"
            sidecar.write_text("{}")
            self.assertTrue(hub.check_gguf_file(model))
            self.assertEqual(
                hub.gguf_sidecar_dir(model, "config.json"), Path(directory)
            )

    @mock.patch(
        "huggingface_hub.hf_hub_download",
        return_value="/cache/model.gguf",
    )
    def test_full_gguf_reference_is_resolved_without_listing(self, download) -> None:
        self.assertEqual(
            hub.resolve_hf_gguf_reference(
                "owner/repo/subdir/model.gguf", revision="revision"
            ),
            "/cache/model.gguf",
        )
        download.assert_called_once_with(
            "owner/repo", "subdir/model.gguf", revision="revision"
        )


if __name__ == "__main__":
    unittest.main()

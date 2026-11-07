import tempfile
import unittest
from pathlib import Path
from unittest import mock

import huggingface_hub

from sglang.srt.utils import model_file_verifier
from sglang.srt.utils.model_file_verifier import (
    IntegrityError,
    generate_checksums,
    verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_diffusion_style_repo(root: Path) -> None:
    (root / "transformer").mkdir()
    (root / "vae").mkdir()
    (root / ".cache").mkdir()
    (root / "model_index.json").write_text("index")
    (root / "README.md").write_text("ignored")
    (root / "transformer" / "config.json").write_text("transformer config")
    (root / "vae" / "config.json").write_text("vae config")
    (root / ".cache" / "state.json").write_text("hidden")


class _FakeHfFileSystem:
    def find(self, repo_id, detail=True):
        files = {
            f"{repo_id}/model_index.json": "index",
            f"{repo_id}/transformer/config.json": "transformer config",
            f"{repo_id}/vae/config.json": "vae config",
            f"{repo_id}/README.md": "ignored",
            f"{repo_id}/.gitattributes": "ignored",
        }
        return {
            name: {"name": name, "type": "file", "size": 1, "sha256": content}
            for name, content in files.items()
        }


class TestModelFileVerifierRecursive(CustomTestCase):
    def test_recursive_manifest_detects_tampered_subfolder_file(self):
        """Two components can both ship a config.json; each must be checked by its
        repo-relative path, so editing one of them fails verification."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            _make_diffusion_style_repo(root)
            manifest = Path(tmp) / "checksums.json"

            generate_checksums(
                source=str(root), output_path=str(manifest), recursive=True
            )
            verify(model_path=str(root), checksums_source=str(manifest), recursive=True)

            (root / "vae" / "config.json").write_text("tampered")
            with self.assertRaisesRegex(IntegrityError, "vae/config.json"):
                verify(
                    model_path=str(root),
                    checksums_source=str(manifest),
                    recursive=True,
                )

    def test_default_manifest_stays_flat(self):
        """Without --recursive the manifest keeps the LM behaviour: top-level files
        only, so existing checksum files and callers are unaffected."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            _make_diffusion_style_repo(root)

            manifest = generate_checksums(
                source=str(root), output_path=str(Path(tmp) / "flat.json")
            )
            recursive = generate_checksums(
                source=str(root),
                output_path=str(Path(tmp) / "recursive.json"),
                recursive=True,
            )

        self.assertEqual(sorted(manifest.files), ["model_index.json"])
        self.assertEqual(
            sorted(recursive.files),
            ["model_index.json", "transformer/config.json", "vae/config.json"],
        )

    def test_hub_listing_is_recursive_with_repo_relative_paths(self):
        with mock.patch.object(huggingface_hub, "HfFileSystem", _FakeHfFileSystem):
            infos = model_file_verifier._load_file_infos_from_hf(
                repo_id="org/repo", recursive=True
            )

        self.assertEqual(
            sorted(infos),
            ["model_index.json", "transformer/config.json", "vae/config.json"],
        )
        self.assertEqual(infos["vae/config.json"].sha256, "vae config")


if __name__ == "__main__":
    unittest.main()

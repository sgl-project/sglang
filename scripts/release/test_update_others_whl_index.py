import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import update_others_whl_index as indexer


class UpdateOthersWhlIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = pathlib.Path(self.temp_dir.name)
        self.root_index = self.repo_dir / "index.html"
        self.root_index.write_text(
            '<!DOCTYPE html>\n<a href="cu130/">cu130</a><br>\n',
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def update_asset(
        self,
        *,
        asset_url="https://github.com/sgl-project/whl/releases/download/build-2/archive.zip",
        filename="archive.zip",
        tag="build-2",
        sha256="2" * 64,
    ):
        root_changed = indexer.update_root_index(self.repo_dir)
        others_changed = indexer.update_others_index(
            repo_dir=self.repo_dir,
            asset_url=asset_url,
            filename=filename,
            tag=tag,
            sha256=sha256,
        )
        return root_changed, others_changed

    def test_first_update_creates_flat_index_and_root_link(self):
        self.assertEqual(self.update_asset(), (True, True))

        root_content = self.root_index.read_text(encoding="utf-8")
        others_content = (self.repo_dir / "others" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertEqual(root_content.count(indexer.ROOT_LINK), 1)
        self.assertTrue(others_content.startswith(indexer.OTHERS_HEADER))
        self.assertIn(">archive.zip</a> (build-2)<br>", others_content)
        self.assertIn(f"#sha256={'2' * 64}", others_content)

    def test_repeated_update_is_unchanged(self):
        self.update_asset()
        root_before = self.root_index.read_bytes()
        others_path = self.repo_dir / "others" / "index.html"
        others_before = others_path.read_bytes()

        self.assertEqual(self.update_asset(), (False, False))
        self.assertEqual(self.root_index.read_bytes(), root_before)
        self.assertEqual(others_path.read_bytes(), others_before)

    def test_newest_asset_is_inserted_first(self):
        self.update_asset(
            asset_url="https://github.com/sgl-project/whl/releases/download/build-1/old.zip",
            filename="old.zip",
            tag="build-1",
            sha256="1" * 64,
        )
        self.update_asset()

        content = (self.repo_dir / "others" / "index.html").read_text(encoding="utf-8")
        self.assertLess(content.index("archive.zip"), content.index("old.zip"))

    def test_html_sensitive_values_are_escaped(self):
        self.update_asset(
            asset_url=(
                "https://github.com/sgl-project/whl/releases/download/"
                "build-2/archive.zip?source=a&mirror=b"
            ),
            filename='archive & "symbols".zip',
            tag="build-2",
        )

        content = (self.repo_dir / "others" / "index.html").read_text(encoding="utf-8")
        self.assertIn("source=a&amp;mirror=b", content)
        self.assertIn("archive &amp; &quot;symbols&quot;.zip", content)

    def test_duplicate_root_links_are_normalized(self):
        self.root_index.write_text(
            f"{indexer.ROOT_LINK}\n{indexer.ROOT_LINK}\n", encoding="utf-8"
        )

        self.assertTrue(indexer.update_root_index(self.repo_dir))
        content = self.root_index.read_text(encoding="utf-8")
        self.assertEqual(content.count(indexer.ROOT_LINK), 1)

    def test_malformed_existing_others_index_is_rejected(self):
        others_dir = self.repo_dir / "others"
        others_dir.mkdir()
        (others_dir / "index.html").write_text(
            "<h1>Unexpected index</h1>\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "expected SGLang header"):
            self.update_asset()


if __name__ == "__main__":
    unittest.main()

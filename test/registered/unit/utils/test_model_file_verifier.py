"""Unit tests for utils/model_file_verifier.py — no server, no model loading.

Covers the pure-logic surface of the model integrity verifier:
  * Manifest / FileInfo (de)serialization, including the deprecated
    ``checksums`` on-disk format and its DeprecationWarning.
  * ``compute_sha256`` streaming hash correctness (multi-chunk reads).
  * ``_compute_manifest_from_folder`` (missing files skipped).
  * ``_discover_files`` dotfile + IGNORE_PATTERNS glob filtering.
  * ``_compare_manifests`` / ``verify`` failure modes (missing, sha
    mismatch) and the success path.
  * ``generate_checksums`` end-to-end round-trip and empty-dir error.
  * ``_load_checksums`` reading a manifest file from disk.

The HuggingFace code paths (``_load_file_infos_from_hf`` etc.) require
network + ``huggingface_hub`` and are intentionally out of scope here; the
local-filesystem logic is what production integrity checks depend on.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import hashlib
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

from sglang.srt.utils.model_file_verifier import (
    IGNORE_PATTERNS,
    FileInfo,
    IntegrityError,
    Manifest,
    _compare_manifests,
    _compute_manifest_from_folder,
    _discover_files,
    _load_checksums,
    compute_sha256,
    generate_checksums,
    verify,
)
from sglang.test.test_utils import CustomTestCase


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestManifestSerialization(CustomTestCase):
    """Manifest.from_dict / to_dict round-trip and legacy-format handling."""

    def test_to_dict_shape(self):
        m = Manifest(files={"a.bin": FileInfo(sha256="ab", size=3)})
        d = m.to_dict()
        self.assertEqual(d, {"files": {"a.bin": {"sha256": "ab", "size": 3}}})

    def test_from_dict_new_format(self):
        d = {"files": {"a.bin": {"sha256": "ab", "size": 3}}}
        m = Manifest.from_dict(d)
        self.assertEqual(m.files["a.bin"], FileInfo(sha256="ab", size=3))

    def test_round_trip_preserves_data(self):
        m = Manifest(
            files={
                "model.safetensors": FileInfo(sha256=_sha(b"x"), size=1),
                "config.json": FileInfo(sha256=_sha(b"yy"), size=2),
            }
        )
        m2 = Manifest.from_dict(m.to_dict())
        self.assertEqual(m2.files, m.files)

    def test_from_dict_deprecated_checksums_format(self):
        """Legacy ``{"checksums": {name: sha}}`` maps to FileInfo(size=-1)."""
        legacy = {"checksums": {"a.bin": "deadbeef", "b.bin": "cafe"}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = Manifest.from_dict(legacy)
        self.assertEqual(m.files["a.bin"], FileInfo(sha256="deadbeef", size=-1))
        self.assertEqual(m.files["b.bin"], FileInfo(sha256="cafe", size=-1))
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "expected a DeprecationWarning for the legacy checksums format",
        )

    def test_legacy_checksums_key_takes_priority_when_both_present(self):
        """``from_dict`` checks ``checksums`` *first*, so a mixed dict that
        still carries the deprecated key is parsed via the legacy branch
        (size=-1) even when a modern ``files`` block is also present.

        This pins down the current key-precedence ordering; a regression
        that reordered the branches would silently change how mixed
        manifests are interpreted.
        """
        d = {
            "files": {"a.bin": {"sha256": "new", "size": 7}},
            "checksums": {"a.bin": "old"},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            m = Manifest.from_dict(d)
        self.assertEqual(m.files["a.bin"], FileInfo(sha256="old", size=-1))


class TestComputeSha256(CustomTestCase):
    """compute_sha256 must equal hashlib over the whole file, incl. >64KiB."""

    def _write(self, tmp, name, data):
        p = Path(tmp) / name
        p.write_bytes(data)
        return p

    def test_small_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, "s.bin", b"hello sglang")
            self.assertEqual(compute_sha256(file_path=p), _sha(b"hello sglang"))

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, "e.bin", b"")
            self.assertEqual(compute_sha256(file_path=p), _sha(b""))

    def test_multi_chunk_file(self):
        # Larger than the 64 KiB read window to exercise the streaming loop.
        data = os.urandom(64 * 1024 * 2 + 123)
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, "big.bin", data)
            self.assertEqual(compute_sha256(file_path=p), _sha(data))


class TestDiscoverFiles(CustomTestCase):
    """_discover_files: sorted, dotfile-skipping, IGNORE_PATTERNS glob-filtered."""

    def test_ignore_patterns_contract(self):
        """Pin the non-weight files the verifier must never track."""
        for name in (".DS_Store", "LICENSE", "README.md", "NOTICE"):
            self.assertIn(name, IGNORE_PATTERNS)
        self.assertIn("*.lock", IGNORE_PATTERNS)

    def _mkdir_with(self, tmp, names):
        for n in names:
            (Path(tmp) / n).write_bytes(b"x")

    def test_sorted_and_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._mkdir_with(tmp, ["b.bin", "a.bin", "c.json"])
            self.assertEqual(_discover_files(Path(tmp)), ["a.bin", "b.bin", "c.json"])

    def test_dotfiles_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._mkdir_with(tmp, ["model.bin", ".hidden", ".DS_Store"])
            self.assertEqual(_discover_files(Path(tmp)), ["model.bin"])

    def test_ignore_patterns_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._mkdir_with(
                tmp,
                [
                    "model.bin",
                    "weights.lock",  # *.lock
                    "LICENSE",
                    "LICENSE.txt",  # LICENSE.*
                    "README.md",  # README.*
                    "NOTICE",
                ],
            )
            self.assertEqual(_discover_files(Path(tmp)), ["model.bin"])

    def test_subdirectories_not_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "sub" / "nested.bin").write_bytes(b"x")
            (Path(tmp) / "top.bin").write_bytes(b"x")
            self.assertEqual(_discover_files(Path(tmp)), ["top.bin"])


class TestComputeManifestFromFolder(CustomTestCase):
    def test_hashes_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.bin").write_bytes(b"aaa")
            (Path(tmp) / "b.bin").write_bytes(b"bbbb")
            man = _compute_manifest_from_folder(
                model_path=Path(tmp), filenames=["a.bin", "b.bin"], max_workers=2
            )
            self.assertEqual(man.files["a.bin"], FileInfo(sha256=_sha(b"aaa"), size=3))
            self.assertEqual(man.files["b.bin"], FileInfo(sha256=_sha(b"bbbb"), size=4))

    def test_missing_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "present.bin").write_bytes(b"z")
            man = _compute_manifest_from_folder(
                model_path=Path(tmp),
                filenames=["present.bin", "absent.bin"],
                max_workers=2,
            )
            self.assertIn("present.bin", man.files)
            self.assertNotIn("absent.bin", man.files)


class TestCompareManifests(CustomTestCase):
    def test_identical_ok(self):
        exp = Manifest(files={"a": FileInfo("h", 1)})
        _compare_manifests(expected=exp, actual=Manifest(files={"a": FileInfo("h", 1)}))

    def test_missing_file_raises(self):
        exp = Manifest(files={"a": FileInfo("h", 1)})
        with self.assertRaises(IntegrityError) as ctx:
            _compare_manifests(expected=exp, actual=Manifest(files={}))
        self.assertIn("missing", str(ctx.exception))
        self.assertIn("a", str(ctx.exception))

    def test_sha_mismatch_raises(self):
        exp = Manifest(files={"a": FileInfo("h" * 20, 1)})
        act = Manifest(files={"a": FileInfo("g" * 20, 1)})
        with self.assertRaises(IntegrityError) as ctx:
            _compare_manifests(expected=exp, actual=act)
        self.assertIn("mismatch", str(ctx.exception))

    def test_multiple_errors_aggregated(self):
        exp = Manifest(files={"a": FileInfo("h" * 20, 1), "b": FileInfo("k" * 20, 2)})
        act = Manifest(files={"a": FileInfo("x" * 20, 1)})  # b missing, a mismatch
        with self.assertRaises(IntegrityError) as ctx:
            _compare_manifests(expected=exp, actual=act)
        msg = str(ctx.exception)
        self.assertIn("a", msg)
        self.assertIn("b", msg)

    def test_extra_actual_file_is_ignored(self):
        """Files present only in actual do not fail verification."""
        exp = Manifest(files={"a": FileInfo("h", 1)})
        act = Manifest(files={"a": FileInfo("h", 1), "extra": FileInfo("z", 9)})
        _compare_manifests(expected=exp, actual=act)


class TestVerifyAndGenerateRoundTrip(CustomTestCase):
    def _make_model_dir(self, tmp):
        (Path(tmp) / "model.safetensors").write_bytes(b"weights-data")
        (Path(tmp) / "config.json").write_bytes(b'{"k": 1}')
        (Path(tmp) / "README.md").write_bytes(b"ignored")  # filtered out

    def test_generate_then_verify_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_model_dir(tmp)
            ckpt = Path(tmp) / "checksums.json"
            man = generate_checksums(source=tmp, output_path=str(ckpt))
            # README.md must be excluded by IGNORE_PATTERNS.
            self.assertNotIn("README.md", man.files)
            self.assertEqual(set(man.files), {"model.safetensors", "config.json"})
            # File on disk parses back to an identical manifest.
            self.assertEqual(_load_checksums(str(ckpt)).files, man.files)
            # Full verify passes against the same directory.
            verify(model_path=tmp, checksums_source=str(ckpt))

    def test_verify_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_model_dir(tmp)
            ckpt = Path(tmp) / "checksums.json"
            generate_checksums(source=tmp, output_path=str(ckpt))
            # Corrupt a tracked file after checksum generation.
            (Path(tmp) / "model.safetensors").write_bytes(b"tampered!!")
            with self.assertRaises(IntegrityError):
                verify(model_path=tmp, checksums_source=str(ckpt))

    def test_verify_detects_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_model_dir(tmp)
            ckpt = Path(tmp) / "checksums.json"
            generate_checksums(source=tmp, output_path=str(ckpt))
            (Path(tmp) / "config.json").unlink()
            with self.assertRaises(IntegrityError):
                verify(model_path=tmp, checksums_source=str(ckpt))

    def test_generate_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Only ignored files present -> no model files discovered.
            (Path(tmp) / "README.md").write_bytes(b"x")
            (Path(tmp) / ".DS_Store").write_bytes(b"x")
            with self.assertRaises(IntegrityError):
                generate_checksums(source=tmp, output_path=str(Path(tmp) / "c.json"))

    def test_generated_file_is_valid_sorted_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_model_dir(tmp)
            ckpt = Path(tmp) / "checksums.json"
            generate_checksums(source=tmp, output_path=str(ckpt))
            data = json.loads(ckpt.read_text())
            self.assertIn("files", data)
            names = list(data["files"].keys())
            self.assertEqual(names, sorted(names))  # sort_keys=True


if __name__ == "__main__":
    unittest.main()

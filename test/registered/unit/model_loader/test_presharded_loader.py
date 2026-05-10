"""Unit tests for PreshardedModelLoader's pure helpers.

These tests exercise the deterministic pieces of the presharding algorithm
(tensor hashing, plan construction, file naming, dedup, file-size cap, and
per-rank workload balance) without needing a GPU or distributed setup.
"""

import json
import os
import tempfile
import unittest

import torch

from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestPreshardedHashTensor(unittest.TestCase):
    def test_same_content_same_hash(self):
        a = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        b = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        self.assertEqual(
            PreshardedModelLoader._hash_tensor(a),
            PreshardedModelLoader._hash_tensor(b),
        )

    def test_different_content_different_hash(self):
        a = torch.arange(100, dtype=torch.float32)
        b = a.clone()
        b[0] = 999
        self.assertNotEqual(
            PreshardedModelLoader._hash_tensor(a),
            PreshardedModelLoader._hash_tensor(b),
        )

    def test_dtype_changes_hash(self):
        a = torch.zeros(8, dtype=torch.float32)
        b = torch.zeros(16, dtype=torch.float16)  # same byte content (zeros)
        self.assertNotEqual(
            PreshardedModelLoader._hash_tensor(a),
            PreshardedModelLoader._hash_tensor(b),
        )

    def test_shape_changes_hash(self):
        a = torch.zeros(16, dtype=torch.float32)
        b = torch.zeros((4, 4), dtype=torch.float32)
        self.assertNotEqual(
            PreshardedModelLoader._hash_tensor(a),
            PreshardedModelLoader._hash_tensor(b),
        )

    def test_empty_tensor_is_hashable(self):
        a = torch.empty(0, dtype=torch.float32)
        digest = PreshardedModelLoader._hash_tensor(a)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 40)  # sha1 hex


class TestPreshardedFilename(unittest.TestCase):
    def test_common_filename(self):
        self.assertEqual(
            PreshardedModelLoader._make_filename(0, (0, 1, 2, 3), is_common=True),
            "model-00000-common.safetensor",
        )

    def test_rank_filename_three_digit_padding(self):
        self.assertEqual(
            PreshardedModelLoader._make_filename(5, (1, 3, 5, 7), is_common=False),
            "model-00005-rank-001,003,005,007.safetensor",
        )

    def test_file_id_zero_padding(self):
        self.assertTrue(
            PreshardedModelLoader._make_filename(42, (0,), is_common=False).startswith(
                "model-00042-"
            )
        )


class TestBuildDumpPlan(unittest.TestCase):
    def _write_manifests(self, tmp_dir, manifests):
        for r, m in manifests.items():
            with open(os.path.join(tmp_dir, f"manifest_{r:05d}.json"), "w") as f:
                json.dump(m, f)

    def test_dedup_across_ranks(self):
        # Same content (same checksum) on all 4 ranks → single file marked common.
        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "checksum": "deadbeef",
                "size": 1024,
                "dtype": "torch.float32",
                "shape": [256],
            }
            manifests = {r: {"shared.weight": shared} for r in range(4)}
            self._write_manifests(tmp, manifests)
            plan = PreshardedModelLoader._build_dump_plan(
                world_size=4, tmp_dir=tmp, max_file_bytes=10**12
            )
        self.assertEqual(len(plan["files"]), 1)
        self.assertTrue(plan["files"][0]["is_common"])
        self.assertIn("common", plan["files"][0]["filename"])
        # Each rank should still have a read entry pointing at the file.
        for r in range(4):
            reads = plan["rank_to_reads"][str(r)]
            self.assertEqual(len(reads), 1)
            self.assertEqual(reads[0]["name"], "shared.weight")
            self.assertEqual(reads[0]["filename"], plan["files"][0]["filename"])
            self.assertEqual(reads[0]["stored_key"], "deadbeef")
        self.assertIn("rank_checksums", plan)
        self.assertEqual(set(plan["rank_checksums"].keys()), {"0", "1", "2", "3"})

    def test_per_rank_unique_tensors(self):
        # Each rank has its own tensor (different content). 4 distinct files,
        # filenames should be rank-{rrr}.
        with tempfile.TemporaryDirectory() as tmp:
            manifests = {
                r: {
                    "layer.weight": {
                        "checksum": f"hash_{r}",
                        "size": 2048,
                        "dtype": "torch.float32",
                        "shape": [512],
                    }
                }
                for r in range(4)
            }
            self._write_manifests(tmp, manifests)
            plan = PreshardedModelLoader._build_dump_plan(
                world_size=4, tmp_dir=tmp, max_file_bytes=10**12
            )
        self.assertEqual(len(plan["files"]), 4)
        for f in plan["files"]:
            self.assertFalse(f["is_common"])
            self.assertEqual(len(f["rank_list"]), 1)
            # Writer is the only rank in the rank_list.
            self.assertEqual(f["writer_rank"], f["rank_list"][0])
            self.assertIn(f"-rank-{f['rank_list'][0]:03d}.safetensor", f["filename"])

    def test_partial_share_has_correct_rank_list(self):
        # Tensor shared by ranks 1,3,5,7 only.
        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "checksum": "shared_hash",
                "size": 1024,
                "dtype": "torch.float32",
                "shape": [256],
            }
            manifests = {
                r: ({"x": shared} if r in (1, 3, 5, 7) else {}) for r in range(8)
            }
            self._write_manifests(tmp, manifests)
            plan = PreshardedModelLoader._build_dump_plan(
                world_size=8, tmp_dir=tmp, max_file_bytes=10**12
            )
        self.assertEqual(len(plan["files"]), 1)
        f = plan["files"][0]
        self.assertFalse(f["is_common"])
        self.assertEqual(f["rank_list"], [1, 3, 5, 7])
        self.assertIn(f["writer_rank"], (1, 3, 5, 7))
        self.assertEqual(f["filename"], "model-00000-rank-001,003,005,007.safetensor")
        # Only the 4 sharing ranks have read entries.
        for r in range(8):
            reads = plan["rank_to_reads"].get(str(r), [])
            if r in (1, 3, 5, 7):
                self.assertEqual(len(reads), 1)
            else:
                self.assertEqual(len(reads), 0)

    def test_max_file_size_caps_files(self):
        # Two tensors of 1 MiB each shared by all 2 ranks; cap = 1.5 MiB →
        # two files.
        with tempfile.TemporaryDirectory() as tmp:
            t1 = {
                "checksum": "h1",
                "size": 1024 * 1024,
                "dtype": "torch.float32",
                "shape": [256, 1024],
            }
            t2 = dict(t1, checksum="h2")
            manifests = {0: {"a": t1, "b": t2}, 1: {"a": t1, "b": t2}}
            self._write_manifests(tmp, manifests)
            plan = PreshardedModelLoader._build_dump_plan(
                world_size=2,
                tmp_dir=tmp,
                max_file_bytes=int(1.5 * 1024 * 1024),
            )
        # Both tensors share the same rank_list (0,1). With balanced packing,
        # each writer (rank 0 and rank 1) gets one tensor → 2 files.
        self.assertEqual(len(plan["files"]), 2)
        for f in plan["files"]:
            self.assertEqual(len(f["tensors"]), 1)
            # rank_list is full world ⇒ marked common
            self.assertTrue(f["is_common"])

    def test_workload_balanced_within_rank_list(self):
        # 4 same-size tensors all shared by ranks (0,1,2,3) → with balanced
        # packing each writer rank should get exactly one tensor.
        with tempfile.TemporaryDirectory() as tmp:
            tensors = {
                f"t{i}": {
                    "checksum": f"h{i}",
                    "size": 1024,
                    "dtype": "torch.float32",
                    "shape": [256],
                }
                for i in range(4)
            }
            manifests = {r: dict(tensors) for r in range(4)}
            self._write_manifests(tmp, manifests)
            plan = PreshardedModelLoader._build_dump_plan(
                world_size=4,
                tmp_dir=tmp,
                max_file_bytes=10**12,
            )
        # 4 tensors, 4 writers, 4 files (one per writer).
        self.assertEqual(len(plan["files"]), 4)
        writers = sorted(f["writer_rank"] for f in plan["files"])
        self.assertEqual(writers, [0, 1, 2, 3])

    def test_round_trip_dump_and_read_back(self):
        # End-to-end on disk for world_size=1: build manifests for tensors,
        # construct plan, write safetensors per the plan, read back and
        # verify both checksums and bit-identity.
        from safetensors.torch import safe_open, save_file

        with tempfile.TemporaryDirectory() as tmp:
            tensors = {
                "embed.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
                "norm.weight": torch.full((16,), 0.5, dtype=torch.float32),
                "head.weight": torch.linspace(-1, 1, 32, dtype=torch.float32),
            }
            manifest_dir = os.path.join(tmp, "manifests")
            presharded_dir = os.path.join(tmp, "presharded")
            os.makedirs(manifest_dir)
            os.makedirs(presharded_dir)

            checksums = {
                name: PreshardedModelLoader._hash_tensor(t)
                for name, t in tensors.items()
            }
            manifest = {
                name: {
                    "checksum": checksums[name],
                    "size": t.numel() * t.element_size(),
                    "dtype": str(t.dtype),
                    "shape": list(t.shape),
                }
                for name, t in tensors.items()
            }
            with open(os.path.join(manifest_dir, "manifest_00000.json"), "w") as f:
                json.dump(manifest, f)

            plan = PreshardedModelLoader._build_dump_plan(
                world_size=1, tmp_dir=manifest_dir, max_file_bytes=10**12
            )

            # Single rank → all tensors live in one common file.
            self.assertEqual(len(plan["files"]), 1)
            f = plan["files"][0]
            self.assertTrue(f["is_common"])

            # Write the file with stored_keys mapped to tensor content.
            to_save = {}
            for t_entry in f["tensors"]:
                name = t_entry["rank_to_names"]["0"][0]
                to_save[t_entry["stored_key"]] = tensors[name]
            save_file(to_save, os.path.join(presharded_dir, f["filename"]))

            # Read back: verify each tensor's checksum and content.
            with safe_open(
                os.path.join(presharded_dir, f["filename"]), framework="pt"
            ) as fh:
                for r in plan["rank_to_reads"]["0"]:
                    loaded = fh.get_tensor(r["stored_key"])
                    self.assertEqual(
                        PreshardedModelLoader._hash_tensor(loaded), r["stored_key"]
                    )
                    torch.testing.assert_close(loaded, tensors[r["name"]])

    def test_dedup_with_multiple_names_per_rank(self):
        # Same checksum can appear under MULTIPLE param names on the same
        # rank (e.g., k_scale and v_scale both default to 1.0). Both names
        # must end up in rank_to_reads.
        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "checksum": "scale_hash",
                "size": 4,
                "dtype": "torch.float32",
                "shape": [],
            }
            manifests = {
                0: {
                    "layers.0.attn.k_scale": shared,
                    "layers.0.attn.v_scale": shared,
                    "layers.1.attn.k_scale": shared,
                },
                1: {
                    "layers.0.attn.k_scale": shared,
                    "layers.0.attn.v_scale": shared,
                    "layers.1.attn.k_scale": shared,
                },
            }
            self._write_manifests(tmp, manifests)
            plan = PreshardedModelLoader._build_dump_plan(
                world_size=2, tmp_dir=tmp, max_file_bytes=10**12
            )
        # All 6 (rank,name) pairs must be readable, even though there is one
        # underlying tensor stored on disk.
        self.assertEqual(len(plan["files"]), 1)
        for r in (0, 1):
            reads = plan["rank_to_reads"][str(r)]
            names = sorted(rd["name"] for rd in reads)
            self.assertEqual(
                names,
                [
                    "layers.0.attn.k_scale",
                    "layers.0.attn.v_scale",
                    "layers.1.attn.k_scale",
                ],
            )
            # All point at the same stored_key (deduplicated content).
            self.assertEqual({rd["stored_key"] for rd in reads}, {"scale_hash"})

    def test_collision_size_mismatch_raises(self):
        # Same checksum but different sizes ⇒ plan builder rejects.
        with tempfile.TemporaryDirectory() as tmp:
            manifests = {
                0: {
                    "x": {
                        "checksum": "same",
                        "size": 1024,
                        "dtype": "torch.float32",
                        "shape": [256],
                    }
                },
                1: {
                    "x": {
                        "checksum": "same",
                        "size": 2048,
                        "dtype": "torch.float32",
                        "shape": [512],
                    }
                },
            }
            self._write_manifests(tmp, manifests)
            with self.assertRaises(RuntimeError):
                PreshardedModelLoader._build_dump_plan(
                    world_size=2, tmp_dir=tmp, max_file_bytes=10**12
                )

    def test_rank_checksum_deterministic(self):
        # rank_checksums must be reproducible from the same manifest input
        # and depend on (name, content-SHA) pairs of every tensor a rank
        # owns. Permuting the manifest's insertion order must not change
        # the rank checksum.
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            base_entries = {
                "alpha.weight": {
                    "checksum": "h_alpha",
                    "size": 16,
                    "dtype": "torch.float32",
                    "shape": [4],
                },
                "beta.weight": {
                    "checksum": "h_beta",
                    "size": 16,
                    "dtype": "torch.float32",
                    "shape": [4],
                },
            }
            self._write_manifests(tmp_a, {0: dict(base_entries)})
            # Insertion-order-permuted copy.
            permuted = {k: base_entries[k] for k in reversed(list(base_entries))}
            self._write_manifests(tmp_b, {0: permuted})
            plan_a = PreshardedModelLoader._build_dump_plan(
                world_size=1, tmp_dir=tmp_a, max_file_bytes=10**12
            )
            plan_b = PreshardedModelLoader._build_dump_plan(
                world_size=1, tmp_dir=tmp_b, max_file_bytes=10**12
            )
            self.assertEqual(plan_a["rank_checksums"], plan_b["rank_checksums"])

    def test_rank_checksum_distinguishes_content(self):
        # Changing one tensor's content-SHA must change the rank checksum.
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            entries_a = {
                "x.weight": {
                    "checksum": "ha",
                    "size": 16,
                    "dtype": "torch.float32",
                    "shape": [4],
                },
            }
            entries_b = {
                "x.weight": {
                    "checksum": "hb",  # different content
                    "size": 16,
                    "dtype": "torch.float32",
                    "shape": [4],
                },
            }
            self._write_manifests(tmp_a, {0: entries_a})
            self._write_manifests(tmp_b, {0: entries_b})
            plan_a = PreshardedModelLoader._build_dump_plan(
                world_size=1, tmp_dir=tmp_a, max_file_bytes=10**12
            )
            plan_b = PreshardedModelLoader._build_dump_plan(
                world_size=1, tmp_dir=tmp_b, max_file_bytes=10**12
            )
            self.assertNotEqual(
                plan_a["rank_checksums"]["0"],
                plan_b["rank_checksums"]["0"],
            )

    def test_presharded_ready_sentinel(self):
        # Loader treats a dir as a valid presharded ckpt only when the
        # READY sentinel exists. A bare checksum.json (or partial files)
        # is NOT enough.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(PreshardedModelLoader._presharded_ready(tmp))
            # checksum.json alone is not sufficient.
            with open(
                os.path.join(tmp, PreshardedModelLoader.CHECKSUM_FILENAME), "w"
            ) as f:
                f.write("{}")
            self.assertFalse(PreshardedModelLoader._presharded_ready(tmp))
            # READY makes it ready.
            with open(
                os.path.join(tmp, PreshardedModelLoader.READY_FILENAME), "w"
            ) as f:
                f.write("{}")
            self.assertTrue(PreshardedModelLoader._presharded_ready(tmp))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for --weight-loader-drop-cache-after-load with mmap safetensors loads.

posix_fadvise(DONTNEED) skips pages that are still mmapped, so a shard is only
released once no tensor of it is alive. Residency is read with mincore(2).
"""

import ctypes
import mmap
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.srt.model_loader.weight_utils import (
    _drop_unmapped_file_caches,
    buffered_multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SHARDS, TENSORS, NUMEL = 4, 4, 64 << 10  # 4 shards x 4 x 256 KiB fp32
PAGE = mmap.PAGESIZE


_LIBC = None
if sys.platform.startswith("linux"):
    try:
        _LIBC = ctypes.CDLL(None, use_errno=True)
        _LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    except (OSError, AttributeError):
        _LIBC = None


def _resident_fraction(path: str) -> float:
    """Fraction of the file's pages in the page cache. The probe mapping is
    closed before returning, so it never shows up in /proc/self/maps."""
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    try:
        buf = (ctypes.c_char * len(m)).from_buffer(m)
        vec = (ctypes.c_ubyte * ((len(m) + PAGE - 1) // PAGE))()
        rc = _LIBC.mincore(ctypes.addressof(buf), len(m), vec)
        del buf
        if rc != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        return sum(b & 1 for b in vec) / len(vec)
    finally:
        m.close()


def _evict(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


@unittest.skipUnless(_LIBC is not None, "needs Linux mincore and /proc/self/maps")
class TestDropCacheAfterLoad(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.files, self.ref = [], {}
        for s in range(SHARDS):
            tensors = {
                f"s{s}.t{i}": torch.full((NUMEL,), s * TENSORS + i + 0.5)
                for i in range(TENSORS)
            }
            self.ref.update(tensors)
            path = os.path.join(self._tmp.name, f"model-{s:05d}.safetensors")
            safetensors.torch.save_file(tensors, path)
            self.files.append(path)
        for path in self.files:
            _evict(path)
        if any(_resident_fraction(p) > 0 for p in self.files):
            self.skipTest("page cache cannot be dropped on this filesystem")

    def tearDown(self):
        self._tmp.cleanup()

    def _consume(self, weights_iter):
        """Copy every tensor into a param, as weight loaders do, and keep s0.t0
        alive until shard 2 starts, like a model that stashes a weight for a
        later fused copy. Returns residency per shard at each shard start and
        after the end."""
        params = {k: torch.empty_like(v) for k, v in self.ref.items()}
        rows, cur, stash = [], -1, None
        for name, w in weights_iter:
            shard = int(name.split(".")[0][1:])
            if shard != cur:
                cur = shard
                if shard == 2:
                    stash = None
                rows.append([_resident_fraction(p) for p in self.files])
            if name == "s0.t0":
                stash = w
            params[name].copy_(w)
        del w, stash
        rows.append([_resident_fraction(p) for p in self.files])
        for k, v in self.ref.items():
            self.assertTrue(torch.equal(params[k], v), k)
        return rows

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_consumed_shards_are_dropped_once_unmapped(self, _):
        iterators = {
            "serial": lambda: safetensors_weights_iterator(
                self.files, drop_cache_after_load=True
            ),
            "buffered": lambda: buffered_multi_thread_safetensors_weights_iterator(
                self.files, max_workers=2, drop_cache_after_load=True
            ),
        }
        for kind, make_iter in iterators.items():
            with self.subTest(kind):
                for path in self.files:
                    _evict(path)
                rows = self._consume(make_iter())
                # Start of shard 2: shard 0 is stashed and shard 1's last tensor
                # is still held, so both stay mapped and cached.
                self.assertGreater(rows[2][0], 0.9)
                self.assertGreater(rows[2][1], 0.9)
                # Start of shard 3: both were released at the previous boundary.
                self.assertEqual(rows[3][:2], [0.0, 0.0])
                # After the end every shard but the last held one is dropped.
                self.assertEqual(rows[4][:3], [0.0, 0.0, 0.0])


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc/self/maps")
class TestDropUnmappedFileCaches(CustomTestCase):
    def test_non_utf8_mapped_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(os.fsencode(tmpdir), b"\xff.bin")
            try:
                with open(path, "wb") as f:
                    f.write(b"\0" * PAGE)
            except (OSError, UnicodeError):
                self.skipTest("filesystem rejects non-UTF-8 names")
            pending = [os.fsdecode(path)]
            with patch(
                "sglang.srt.model_loader.weight_utils._drop_file_cache_after_load"
            ) as drop_cache:
                with open(path, "rb") as f:
                    m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                _drop_unmapped_file_caches(pending)
                self.assertEqual(len(pending), 1)
                drop_cache.assert_not_called()
                m.close()
                _drop_unmapped_file_caches(pending)
            self.assertEqual(pending, [])
            drop_cache.assert_called_once_with(os.fsdecode(path))

    def test_maps_check_failure_drops_all_pending(self):
        pending = ["a.safetensors", "b.safetensors"]
        with (
            patch(
                "sglang.srt.model_loader.weight_utils.open",
                side_effect=OSError("no maps"),
                create=True,
            ),
            patch(
                "sglang.srt.model_loader.weight_utils._drop_file_cache_after_load"
            ) as drop_cache,
        ):
            _drop_unmapped_file_caches(pending)
        self.assertEqual(pending, [])
        self.assertEqual(
            [c.args[0] for c in drop_cache.call_args_list],
            ["a.safetensors", "b.safetensors"],
        )


if __name__ == "__main__":
    unittest.main()

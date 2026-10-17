"""Unit tests for evict_hf_cache.evict_least_recently_used.

Stdlib only, like the script under test:

    python -m unittest discover -s scripts/ci/utils -p 'test_evict_hf_cache.py'
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evict_hf_cache as ehc  # noqa: E402

HOUR = 3600.0
GIB = 1 << 30


def _make_repo(cache_dir: Path, name: str, last_used: float) -> None:
    blob = cache_dir / name / "blobs" / "0123abcd"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\0" * 4096)
    # Children first: creating an entry bumps its parent directory's mtime.
    for path in (blob, blob.parent, blob.parent.parent):
        os.utime(path, (last_used, last_used))


class EvictLeastRecentlyUsedTest(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.cache_dir)
        self.now = time.time()
        self.removed = []

    def _evict(self, *, target_free_gib: int, min_idle_hours: float):
        def remove(path: Path) -> None:
            self.removed.append(path.name)
            shutil.rmtree(path)

        # Each eviction frees exactly 1 GiB, so the target counts evictions.
        return ehc.evict_least_recently_used(
            cache_dir=self.cache_dir,
            target_free_bytes=target_free_gib * GIB,
            min_idle_seconds=min_idle_hours * HOUR,
            now=self.now,
            free_bytes_fn=lambda: len(self.removed) * GIB,
            remove_fn=remove,
        )

    def test_evicts_least_recently_used_first_and_stops_at_target(self):
        _make_repo(self.cache_dir, "models--org--mid", self.now - 20 * HOUR)
        _make_repo(self.cache_dir, "models--org--oldest", self.now - 30 * HOUR)
        _make_repo(self.cache_dir, "models--org--newest", self.now - 10 * HOUR)

        self._evict(target_free_gib=2, min_idle_hours=2)

        self.assertEqual(self.removed, ["models--org--oldest", "models--org--mid"])
        self.assertTrue((self.cache_dir / "models--org--newest").is_dir())

    def test_never_evicts_a_repo_used_within_min_idle(self):
        """A host's other runner may be loading a recent repo; deleting it
        mid-load fails that job, so recency wins over the free-space target."""
        _make_repo(self.cache_dir, "models--org--stale", self.now - 30 * HOUR)
        _make_repo(self.cache_dir, "models--org--in-use", self.now - 0.5 * HOUR)
        _make_repo(self.cache_dir, ".locks", self.now - 30 * HOUR)

        self._evict(target_free_gib=100, min_idle_hours=2)

        self.assertEqual(self.removed, ["models--org--stale"])
        self.assertTrue((self.cache_dir / "models--org--in-use").is_dir())
        self.assertTrue((self.cache_dir / ".locks").is_dir())


if __name__ == "__main__":
    unittest.main()

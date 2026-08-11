import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.mem_cache.multimodal_cache import (
    EmbeddingResult,
    MultiModalStaticCache,
)
from sglang.srt.multimodal.cache import (
    CacheReservation,
    MultimodalPreprocessCache,
    build_artifact_key,
    estimate_cache_size_bytes,
    parse_content_hash,
    snapshot_media,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMediaIdentity(unittest.TestCase):
    def test_hash_format_is_strict_and_normalized(self):
        digest = "AB" * 32
        self.assertEqual(
            parse_content_hash(f"sha256:{digest}"), f"sha256:{digest.lower()}"
        )
        for invalid in (
            "",
            digest,
            "md5:" + digest,
            "sha256:1234",
            "sha256:" + "z" * 64,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_content_hash(invalid)

    def test_same_bytes_have_same_identity_across_input_forms(self):
        payload = b"strict-media-identity"
        data_url = (
            "data:application/octet-stream;base64," + base64.b64encode(payload).decode()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(payload)
            snapshots = [
                snapshot_media(payload),
                snapshot_media(data_url),
                snapshot_media(str(path)),
            ]
        self.assertEqual(len({item.content_digest for item in snapshots}), 1)
        self.assertTrue(all(item.data == payload for item in snapshots))

    def test_same_path_with_new_contents_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"first")
            first = snapshot_media(str(path))
            path.write_bytes(b"second")
            second = snapshot_media(str(path))
        self.assertNotEqual(first.content_digest, second.content_digest)

    def test_same_url_with_new_contents_misses(self):
        with patch(
            "sglang.srt.utils.get_image_bytes", side_effect=[b"first", b"second"]
        ):
            first = snapshot_media("https://example.com/image.png")
            second = snapshot_media("https://example.com/image.png")
        self.assertNotEqual(first.content_digest, second.content_digest)

    def test_pil_and_noncontiguous_tensor_are_snapshotted(self):
        image = Image.new("RGBA", (3, 2), (1, 2, 3, 4))
        first = snapshot_media(image)
        image.putpixel((0, 0), (9, 9, 9, 9))
        self.assertNotEqual(first.content_digest, snapshot_media(image).content_digest)

        tensor = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4).transpose(1, 2)
        tensor_snapshot = snapshot_media(tensor)
        self.assertTrue(tensor_snapshot.data.is_contiguous())
        self.assertTrue(torch.equal(tensor_snapshot.data, tensor))

    def test_pil_palette_and_transparency_are_part_of_identity(self):
        first = Image.new("P", (2, 2), color=0)
        second = first.copy()
        first.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        second.putpalette([0, 255, 0] + [0, 0, 0] * 255)
        self.assertNotEqual(
            snapshot_media(first).content_digest,
            snapshot_media(second).content_digest,
        )

        second.putpalette(first.getpalette())
        first.info["transparency"] = 0
        second.info["transparency"] = 1
        self.assertNotEqual(
            snapshot_media(first).content_digest,
            snapshot_media(second).content_digest,
        )

    def test_artifact_key_includes_processor_and_kwargs(self):
        digest = snapshot_media(b"image").content_digest
        base = build_artifact_key(
            digest,
            modality="image",
            processor_fingerprint="processor-a",
            preprocess_kwargs={"antialias": True},
        )
        self.assertNotEqual(
            base,
            build_artifact_key(
                digest,
                modality="image",
                processor_fingerprint="processor-b",
                preprocess_kwargs={"antialias": True},
            ),
        )
        self.assertNotEqual(
            base,
            build_artifact_key(
                digest,
                modality="image",
                processor_fingerprint="processor-a",
                preprocess_kwargs={"antialias": False},
            ),
        )


class TestMultimodalPreprocessCache(unittest.TestCase):
    def test_byte_and_entry_bounded_lru(self):
        cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=6, max_entries=2)
        self.assertTrue(cache.put("a", b"aaa"))
        self.assertTrue(cache.put("b", b"bbb"))
        self.assertEqual(cache.get("a"), b"aaa")
        self.assertTrue(cache.put("c", b"ccc"))
        self.assertNotIn("b", cache)
        self.assertIn("a", cache)
        self.assertIn("c", cache)
        self.assertEqual(cache.current_size_bytes, 6)

    def test_gpu_backed_values_are_not_implicitly_copied(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        value = torch.zeros(1, device="cuda")
        cache = MultimodalPreprocessCache[str, torch.Tensor](max_size_bytes=1024)
        self.assertIsNone(estimate_cache_size_bytes(value))
        self.assertFalse(cache.put("gpu", value))

    def test_async_singleflight(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            calls = 0
            started = asyncio.Event()
            release = asyncio.Event()

            async def compute():
                nonlocal calls
                calls += 1
                started.set()
                await release.wait()
                return b"artifact"

            first = asyncio.create_task(cache.get_or_compute("key", compute))
            await started.wait()
            second = asyncio.create_task(cache.get_or_compute("key", compute))
            await asyncio.sleep(0)
            release.set()
            owner, joiner = await asyncio.gather(first, second)
            self.assertEqual(calls, 1)
            self.assertFalse(owner.hit)
            self.assertTrue(joiner.joined)
            self.assertEqual(cache.get("key"), b"artifact")

        asyncio.run(run())

    def test_clear_does_not_repopulate_from_inflight_work(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            started = asyncio.Event()
            release = asyncio.Event()

            async def compute():
                started.set()
                await release.wait()
                return b"old-generation"

            task = asyncio.create_task(cache.get_or_compute("key", compute))
            await started.wait()
            cache.clear()
            release.set()
            self.assertEqual((await task).value, b"old-generation")
            self.assertNotIn("key", cache)

        asyncio.run(run())

    def test_reserve_many_batches_owners_and_joins(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            reservations = cache.reserve_many(["a", "b", "a"])
            owners = [
                item
                for item in reservations
                if isinstance(item, CacheReservation) and item.owner
            ]
            self.assertEqual([item.key for item in owners], ["a", "b"])

            cache.fulfill(owners[0], b"value-a")
            cache.fulfill(owners[1], b"value-b")
            self.assertEqual(await cache.wait(reservations[2]), b"value-a")
            self.assertEqual(cache.get("b"), b"value-b")

        asyncio.run(run())

    def test_disabled_cache_does_not_join_or_retain(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=0)
            reservations = cache.reserve_many(["a", "a"])
            self.assertTrue(
                all(
                    isinstance(item, CacheReservation) and item.owner
                    for item in reservations
                )
            )
            for item in reservations:
                cache.fulfill(item, b"value")
            self.assertEqual(len(cache), 0)
            self.assertEqual(cache.stats()["singleflight_joins"], 0)

        asyncio.run(run())

    def test_clear_starts_a_new_singleflight_generation(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            started = asyncio.Event()
            release = asyncio.Event()

            async def compute_old():
                started.set()
                await release.wait()
                return b"old"

            async def compute_new():
                return b"new"

            old_task = asyncio.create_task(cache.get_or_compute("key", compute_old))
            await started.wait()
            cache.clear()
            new_result = await cache.get_or_compute("key", compute_new)
            release.set()
            old_result = await old_task

            self.assertEqual(old_result.value, b"old")
            self.assertEqual(new_result.value, b"new")
            self.assertEqual(cache.get("key"), b"new")

        asyncio.run(run())

    def test_clear_starts_a_new_reservation_generation(self):
        cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
        old = cache.reserve_many(["key"])[0]
        cache.clear()
        new = cache.reserve_many(["key"])[0]

        self.assertTrue(old.owner)
        self.assertTrue(new.owner)
        self.assertIsNot(old.future, new.future)
        cache.fulfill(old, b"old")
        self.assertNotIn("key", cache)
        cache.fulfill(new, b"new")
        self.assertEqual(cache.get("key"), b"new")


class TestMultimodalEmbeddingCacheLease(unittest.TestCase):
    @staticmethod
    def _embedding(value: int) -> EmbeddingResult:
        return EmbeddingResult(embedding=torch.tensor([value], dtype=torch.int64))

    def test_lease_pins_entry_until_consumed(self):
        cache = MultiModalStaticCache(max_size=8)
        self.assertTrue(cache.set(1, self._embedding(1)))
        self.assertEqual(cache.acquire_many("lease", [1], ttl_s=300), [True])
        self.assertFalse(cache.set(2, self._embedding(2)))
        self.assertEqual(cache.consume("lease", 1).embedding.item(), 1)
        self.assertTrue(cache.set(2, self._embedding(2)))
        self.assertFalse(cache.has(1))

    def test_duplicate_hashes_are_consumed_individually(self):
        cache = MultiModalStaticCache(max_size=16)
        cache.set(1, self._embedding(1))
        self.assertEqual(
            cache.acquire_many("lease", [1, 1, None], ttl_s=300),
            [True, True, False],
        )
        self.assertIsNotNone(cache.consume("lease", 1))
        self.assertTrue(cache.lease_contains("lease", 1))
        self.assertIsNotNone(cache.consume("lease", 1))
        self.assertFalse(cache.lease_contains("lease", 1))

    def test_expiry_and_clear_release_pins(self):
        cache = MultiModalStaticCache(max_size=16)
        cache.set(1, self._embedding(1))
        with patch(
            "sglang.srt.mem_cache.multimodal_cache.time.monotonic",
            side_effect=[10.0, 10.0, 12.0],
        ):
            cache.acquire_many("expired", [1], ttl_s=1)
            self.assertFalse(cache.lease_contains("expired", 1))

        cache.acquire_many("cleared", [1], ttl_s=300)
        cache.clear()
        self.assertEqual(cache.lease_stats(), (0, 0))
        self.assertEqual(len(cache), 0)

    def test_admitted_lease_does_not_expire_while_request_is_queued(self):
        cache = MultiModalStaticCache(max_size=16)
        cache.set(1, self._embedding(1))
        with patch(
            "sglang.srt.mem_cache.multimodal_cache.time.monotonic",
            side_effect=[10.0, 10.0, 10.5, 12.0, 12.0],
        ):
            cache.acquire_many("admitted", [1], ttl_s=1)
            self.assertTrue(cache.admit_lease("admitted"))
            self.assertTrue(cache.lease_contains("admitted", 1))
            self.assertEqual(cache.consume("admitted", 1).embedding.item(), 1)


if __name__ == "__main__":
    unittest.main()

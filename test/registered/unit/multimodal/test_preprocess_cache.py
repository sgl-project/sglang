import asyncio
import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.cache import (
    CacheMiss,
    MultimodalPreprocessCache,
    build_artifact_key,
    build_processor_fingerprint,
    estimate_cache_size_bytes,
    parse_content_hash,
    resolve_multimodal_item_hash,
    snapshot_media,
)
from sglang.srt.server_args import ServerArgs
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
        # Keep the encoded data URL above common filesystem filename limits;
        # probing it as a local path must not raise ENAMETOOLONG.
        payload = b"strict-media-identity" * 32
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

    def test_wrapped_image_input_snapshots_the_image_not_the_wrapper(self):
        image = Image.new("RGB", (2, 2), (1, 2, 3))
        direct = snapshot_media(image)
        wrapped = snapshot_media({"type": "image", "image": image})
        self.assertEqual(direct.content_digest, wrapped.content_digest)

    def test_same_path_with_new_contents_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"first")
            first = snapshot_media(str(path))
            path.write_bytes(b"second")
            second = snapshot_media(str(path))
        self.assertNotEqual(first.content_digest, second.content_digest)

    def test_relative_local_path_uses_file_bytes(self):
        payload = b"relative-image-bytes"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(payload)
            previous = Path.cwd()
            try:
                os.chdir(directory)
                snapshot = snapshot_media("image.png")
            finally:
                os.chdir(previous)

        self.assertEqual(snapshot.data, payload)
        self.assertEqual(
            snapshot.content_digest, snapshot_media(payload).content_digest
        )

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

        same_bytes_new_shape = tensor.contiguous().reshape(2, 2, 6)
        self.assertNotEqual(
            tensor_snapshot.content_digest,
            snapshot_media(same_bytes_new_shape).content_digest,
        )
        self.assertNotEqual(
            snapshot_media(torch.tensor([1], dtype=torch.int32)).content_digest,
            snapshot_media(torch.tensor([1], dtype=torch.int64)).content_digest,
        )

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

    def test_artifact_key_canonicalization_is_type_preserving(self):
        digest = snapshot_media(b"image").content_digest

        def key(kwargs):
            return build_artifact_key(
                digest,
                modality="image",
                processor_fingerprint="processor",
                preprocess_kwargs=kwargs,
            )

        # These pairs used to collapse to the same JSON representation. A
        # processor is allowed to distinguish them, so sharing an artifact
        # would be a correctness bug rather than a harmless cache collision.
        self.assertNotEqual(key({1: "value"}), key({"1": "value"}))
        self.assertNotEqual(key({"value": [1, 2]}), key({"value": (1, 2)}))
        self.assertNotEqual(key({"value": 1}), key({"value": True}))
        self.assertNotEqual(
            key({"value": np.array([1, 2], dtype=np.int32)}),
            key({"value": np.array([1, 3], dtype=np.int32)}),
        )
        self.assertEqual(
            key({"first": 1, "second": 2}),
            key({"second": 2, "first": 1}),
        )

    def test_artifact_key_rejects_lossy_unknown_values(self):
        digest = snapshot_media(b"image").content_digest
        with self.assertRaisesRegex(ValueError, "Unsupported value"):
            build_artifact_key(
                digest,
                modality="image",
                processor_fingerprint="processor",
                preprocess_kwargs={"value": object()},
            )

    def test_processor_fingerprint_changes_with_output_affecting_config(self):
        class Processor:
            def __init__(self, backend):
                self.backend = backend

            def preprocess_fingerprint_payload(self):
                return {"backend": self.backend, "antialias": True}

        class Config:
            def to_dict(self):
                return {"model_type": "vlm", "architectures": ["VLM"]}

        config = Config()
        args = ServerArgs(
            model_path="dummy",
            revision="model-revision",
            disable_fast_image_processor=False,
            mm_process_config={"image": {"max_pixels": 1024}},
        )
        base = build_processor_fingerprint(Processor("gpu"), config, args)

        changed_backend = build_processor_fingerprint(Processor("cpu"), config, args)
        changed_args = ServerArgs(
            model_path="dummy",
            revision="model-revision",
            disable_fast_image_processor=False,
            mm_process_config={"image": {"max_pixels": 2048}},
        )
        changed_config = build_processor_fingerprint(
            Processor("gpu"), config, changed_args
        )
        self.assertNotEqual(base, changed_backend)
        self.assertNotEqual(base, changed_config)

    def test_item_hash_namespace_covers_identity_and_processor_output(self):
        digest = snapshot_media(b"image").content_digest
        first = build_artifact_key(
            digest,
            modality="image",
            processor_fingerprint="processor-a",
        )
        second = build_artifact_key(
            digest,
            modality="image",
            processor_fingerprint="processor-b",
        )
        self.assertNotEqual(
            resolve_multimodal_item_hash(existing_hash=1, namespace=first),
            resolve_multimodal_item_hash(existing_hash=1, namespace=second),
        )
        self.assertNotEqual(
            resolve_multimodal_item_hash(existing_hash=1, namespace=first),
            resolve_multimodal_item_hash(existing_hash=2, namespace=first),
        )
        with self.assertRaises(ValueError):
            resolve_multimodal_item_hash(existing_hash=-1, namespace=first)

    def test_multimodal_data_item_uses_shared_feature_hash(self):
        feature = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        expected = resolve_multimodal_item_hash(feature=feature)
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=feature)

        item.set_pad_value()

        self.assertEqual(item.hash, expected)


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

    def test_compatible_lookup_is_atomic_and_does_not_count_bypass_as_miss(self):
        cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
        cache.put("key", b"metadata-only")

        self.assertIsNone(cache.get_if_present("key", lambda value: False))
        self.assertEqual((cache.hits, cache.misses), (0, 0))
        self.assertEqual(
            cache.get_if_present("key", lambda value: value.startswith(b"metadata")),
            b"metadata-only",
        )
        self.assertEqual((cache.hits, cache.misses), (1, 0))

    def test_claimed_miss_rejects_an_incompatible_racing_entry(self):
        cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
        cache.put("key", b"metadata-only")

        miss = cache.lookup_or_claim_many(
            ["key"], predicate=lambda key, value: value == b"full-feature"
        )[0]

        self.assertIsInstance(miss, CacheMiss)
        self.assertTrue(miss.should_compute)
        self.assertNotIn("key", cache)
        self.assertEqual(cache.current_size_bytes, 0)

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

    def test_cancelled_singleflight_joiner_does_not_cancel_owner(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            started = asyncio.Event()
            release = asyncio.Event()

            async def compute():
                started.set()
                await release.wait()
                return b"artifact"

            owner = asyncio.create_task(cache.get_or_compute("key", compute))
            await started.wait()
            joiner = asyncio.create_task(cache.get_or_compute("key", compute))
            await asyncio.sleep(0)
            joiner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await joiner

            release.set()
            result = await owner
            self.assertEqual(result.value, b"artifact")
            self.assertEqual(cache.get("key"), b"artifact")

        asyncio.run(run())

    def test_cancelled_singleflight_owner_does_not_cancel_joiner(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            started = asyncio.Event()
            release = asyncio.Event()

            async def compute():
                started.set()
                await release.wait()
                return b"artifact"

            owner = asyncio.create_task(cache.get_or_compute("key", compute))
            await started.wait()
            joiner = asyncio.create_task(cache.get_or_compute("key", compute))
            await asyncio.sleep(0)
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner

            release.set()
            result = await joiner
            self.assertEqual(result.value, b"artifact")
            self.assertTrue(result.joined)
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

    def test_lookup_or_claim_many_batches_owned_and_joined_misses(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            results = cache.lookup_or_claim_many(["a", "b", "a"])
            misses_to_compute = [
                item
                for item in results
                if isinstance(item, CacheMiss) and item.should_compute
            ]
            self.assertEqual([item.key for item in misses_to_compute], ["a", "b"])

            cache.complete_miss(misses_to_compute[0], b"value-a")
            cache.complete_miss(misses_to_compute[1], b"value-b")
            self.assertEqual(await cache.wait_for_miss(results[2]), b"value-a")
            self.assertEqual(cache.get("b"), b"value-b")

        asyncio.run(run())

    def test_cancelled_miss_waiter_does_not_cancel_computing_caller(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
            computing_miss = cache.lookup_or_claim_many(["key"])[0]
            waiting_miss = cache.lookup_or_claim_many(["key"])[0]
            self.assertTrue(computing_miss.should_compute)
            self.assertFalse(waiting_miss.should_compute)

            waiter = asyncio.create_task(cache.wait_for_miss(waiting_miss))
            await asyncio.sleep(0)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter

            cache.complete_miss(computing_miss, b"artifact")
            self.assertEqual(computing_miss.future.result(), b"artifact")
            self.assertEqual(cache.get("key"), b"artifact")

        asyncio.run(run())

    def test_disabled_cache_does_not_join_or_retain(self):
        async def run():
            cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=0)
            misses = cache.lookup_or_claim_many(["a", "a"])
            self.assertTrue(
                all(
                    isinstance(item, CacheMiss) and item.should_compute
                    for item in misses
                )
            )
            for item in misses:
                cache.complete_miss(item, b"value")
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

    def test_clear_starts_a_new_cache_miss_generation(self):
        cache = MultimodalPreprocessCache[str, bytes](max_size_bytes=1024)
        old = cache.lookup_or_claim_many(["key"])[0]
        cache.clear()
        new = cache.lookup_or_claim_many(["key"])[0]

        self.assertTrue(old.should_compute)
        self.assertTrue(new.should_compute)
        self.assertIsNot(old.future, new.future)
        cache.complete_miss(old, b"old")
        self.assertNotIn("key", cache)
        cache.complete_miss(new, b"new")
        self.assertEqual(cache.get("key"), b"new")


if __name__ == "__main__":
    unittest.main()

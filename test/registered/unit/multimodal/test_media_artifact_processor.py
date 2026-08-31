import asyncio
import base64
import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Optional

from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.cache import MultimodalPreprocessCache, snapshot_media
from sglang.srt.multimodal.media_artifacts import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _Artifact:
    content_digest: str
    artifact_key: str
    feature_hash: int
    feature: Optional[bytes]

    @property
    def has_feature(self) -> bool:
        return self.feature is not None

    def cache_value(self):
        return self

    def cache_size_items(self):
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.feature,
        )


@dataclass(frozen=True)
class _FutureMediaInput:
    url: str
    content_hash: Optional[str] = None
    frame_sampling: int = 2


class _Processor(MediaArtifactCacheMixin):
    artifact_modality = Modality.IMAGE
    artifact_option_defaults = {"detail": "auto", "frame_sampling": 2}

    def __init__(self):
        self.processor_fingerprint = "processor"
        self.trust_mm_content_hashes = False
        self.mm_preprocess_cache = MultimodalPreprocessCache(1024 * 1024)
        self.mm_processor_executor = None
        self.io_executor = ThreadPoolExecutor(max_workers=4)
        self.batches = []

    def decode_media_snapshot(self, snapshot, modality):
        self.assert_artifact_modality(modality)
        return snapshot.data

    @staticmethod
    def assert_artifact_modality(modality):
        if modality != Modality.IMAGE:
            raise AssertionError(f"unexpected modality: {modality}")

    def prepare_artifact_batch(
        self, entries: list[MediaArtifactInput]
    ) -> list[_Artifact]:
        self.batches.append(entries)
        return [
            _Artifact(
                content_digest=entry.content_digest,
                artifact_key=entry.artifact_key,
                feature_hash=int(entry.content_digest[-16:], 16),
                feature=entry.media,
            )
            for entry in entries
        ]

    def close(self):
        self.io_executor.shutdown()


class TestMediaArtifactProcessor(unittest.TestCase):
    def test_default_image_decoder_rejects_lazy_pil_failure(self):
        malformed_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJSwAAAABJRU5ErkJggg=="
        )
        processor = MediaArtifactCacheMixin()
        processor.gpu_image_decode = False

        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            processor.decode_media_snapshot(
                snapshot_media(malformed_png), Modality.IMAGE
            )

        lazy_image = Image.open(io.BytesIO(malformed_png))
        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            snapshot_media(lazy_image)

    def test_unknown_model_option_is_part_of_artifact_identity(self):
        processor = _Processor()
        digest = snapshot_media(b"image").content_digest
        try:
            base = processor._artifact_key(digest, _FutureMediaInput(url="image.png"))
            self.assertEqual(
                base,
                processor._artifact_key(
                    digest,
                    {
                        "url": "image.png",
                        "content_hash": digest,
                        "frame_sampling": 2,
                    },
                ),
            )
            self.assertNotEqual(
                base,
                processor._artifact_key(
                    digest,
                    {"url": "image.png", "future_model_knob": "different"},
                ),
            )
            self.assertNotEqual(
                base,
                processor._artifact_key(
                    digest,
                    _FutureMediaInput(url="image.png"),
                    modality=Modality.VIDEO,
                ),
            )
        finally:
            processor.close()

    def test_non_image_models_can_override_identity_and_decode_hooks(self):
        class _VideoProcessor(_Processor):
            artifact_modality = Modality.VIDEO

            def snapshot_media_source(self, source, modality):
                self.assert_video_modality(modality)
                return snapshot_media(source.encode())

            def decode_media_snapshot(self, snapshot, modality):
                self.assert_video_modality(modality)
                return snapshot.data

            @staticmethod
            def assert_video_modality(modality):
                if modality != Modality.VIDEO:
                    raise AssertionError(f"unexpected modality: {modality}")

        processor = _VideoProcessor()
        try:
            artifacts = asyncio.run(processor.prepare_media_artifacts(["clip.mp4"]))
        finally:
            processor.close()

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(
            artifacts[0].content_digest, snapshot_media(b"clip.mp4").content_digest
        )

    def test_partial_hits_and_duplicate_misses_are_shared_by_contract(self):
        processor = _Processor()
        first_digest = snapshot_media(b"first").content_digest
        first_key = processor._artifact_key(first_digest, b"first")
        first = _Artifact(first_digest, first_key, 1, b"first")
        processor.mm_preprocess_cache.put(first_key, first)

        try:
            artifacts = asyncio.run(
                processor.prepare_media_artifacts(
                    [b"first", b"second", b"second", b"first"]
                )
            )
        finally:
            processor.close()

        self.assertEqual(len(processor.batches), 1)
        self.assertEqual(len(processor.batches[0]), 1)
        self.assertEqual(
            [artifact.content_digest for artifact in artifacts],
            [
                first_digest,
                snapshot_media(b"second").content_digest,
                snapshot_media(b"second").content_digest,
                first_digest,
            ],
        )
        self.assertIs(artifacts[0], artifacts[3])
        self.assertIs(artifacts[1], artifacts[2])

    def test_adapter_cannot_change_validated_artifact_identity(self):
        processor = _Processor()

        async def wrong_identity(entries):
            artifact = processor.prepare_artifact_batch(entries)[0]
            return [replace(artifact, artifact_key="sha256:" + "0" * 64)]

        processor._run_preprocess_and_build_artifact_batch = wrong_identity
        try:
            with self.assertRaisesRegex(ValueError, "changed the media artifact key"):
                asyncio.run(processor.prepare_media_artifacts([b"image"]))
        finally:
            processor.close()

    def test_trusted_hit_uses_identity_without_reading_source(self):
        processor = _Processor()
        processor.trust_mm_content_hashes = True
        digest = snapshot_media(b"cached").content_digest
        key = processor._artifact_key(digest, "unread-source")
        artifact = _Artifact(digest, key, 1, b"cached")
        processor.mm_preprocess_cache.put(key, artifact)

        try:
            artifacts = asyncio.run(
                processor.prepare_media_artifacts(
                    ["unread-source"], content_hashes=[digest]
                )
            )
        finally:
            processor.close()

        self.assertEqual(artifacts, [artifact])
        self.assertEqual(processor.batches, [])

    def test_cached_artifact_must_match_content_identity(self):
        processor = _Processor()
        digest = snapshot_media(b"fresh").content_digest
        key = processor._artifact_key(digest, b"fresh")
        processor.mm_preprocess_cache.put(
            key,
            _Artifact(
                snapshot_media(b"stale").content_digest,
                key,
                1,
                b"stale",
            ),
        )
        try:
            artifacts = asyncio.run(processor.prepare_media_artifacts([b"fresh"]))
        finally:
            processor.close()

        self.assertEqual(artifacts[0].content_digest, digest)
        self.assertEqual(artifacts[0].feature, b"fresh")
        self.assertEqual(len(processor.batches), 1)


if __name__ == "__main__":
    unittest.main()

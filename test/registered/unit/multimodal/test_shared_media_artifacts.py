import asyncio
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
from test_media_artifact_processor import _Processor

from sglang.srt.multimodal.cache.identity import (
    build_artifact_configuration,
    snapshot_media,
    snapshot_witness,
)
from sglang.srt.multimodal.cache.shared_preprocess_cache import SharedPreprocessCache


class TestSharedMediaArtifacts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.processors = []

    def tearDown(self):
        for processor in self.processors:
            processor.close()
        self.directory.cleanup()

    def processor(self, configuration=b"processor-settings"):
        processor = _Processor()
        processor.processor_fingerprint_description = configuration
        processor.mm_preprocess_cache = SharedPreprocessCache(
            self.directory.name, 1024 * 1024
        )
        self.processors.append(processor)
        return processor

    def test_cross_processor_partial_hits_and_order(self):
        first, second = self.processor(), self.processor()
        cold = asyncio.run(first.prepare_media_artifacts([b"a", b"b", b"a"]))
        hot = asyncio.run(second.prepare_media_artifacts([b"b", b"c", b"a"]))
        self.assertEqual([a.feature for a in hot], [b"b", b"c", b"a"])
        self.assertEqual([len(batch) for batch in first.batches], [2])
        self.assertEqual([len(batch) for batch in second.batches], [1])
        self.assertEqual(cold[0], hot[2])

    def test_forced_content_and_configuration_key_collision(self):
        first, second = self.processor(b"config-a"), self.processor(b"config-b")
        for processor in (first, second):
            processor._artifact_key = lambda *args, **kwargs: "forced-collision"
        with patch(
            "sglang.srt.multimodal.cache.identity._digest_bytes",
            return_value="sha256:" + "0" * 64,
        ):
            a = asyncio.run(first.prepare_media_artifacts([b"a", b"b"]))
            b = asyncio.run(second.prepare_media_artifacts([b"a"]))
            hot = asyncio.run(first.prepare_media_artifacts([b"b", b"a"]))
        self.assertEqual([x.feature for x in a], [b"a", b"b"])
        self.assertEqual([x.feature for x in hot], [b"b", b"a"])
        self.assertEqual(b[0].feature, b"a")
        self.assertEqual(len(first.batches), 1)
        self.assertEqual(len(second.batches), 1)
        self.assertEqual(first.mm_preprocess_cache.cache_info()["entries"], 3)

    def test_trusted_caller_hash_still_checks_real_content(self):
        processor = self.processor()
        processor.trust_mm_content_hashes = True
        digest = asyncio.run(processor.prepare_media_artifacts([b"a"]))[
            0
        ].content_digest
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            asyncio.run(
                processor.prepare_media_artifacts([b"b"], content_hashes=[digest])
            )

    def test_witness_distinguishes_type_shape_palette_and_configuration(self):
        arrays = [np.zeros((2, 3), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8)]
        values = [*arrays, torch.from_numpy(arrays[0]), b"\0" * 6]
        palette_a = Image.new("P", (2, 2))
        palette_b = palette_a.copy()
        palette_a.putpalette([0, 0, 0] * 256)
        palette_b.putpalette([255, 0, 0] * 256)
        values += [palette_a, palette_b]
        self.assertEqual(
            len({snapshot_witness(snapshot_media(value)) for value in values}),
            len(values),
        )
        with patch(
            "sglang.srt.multimodal.cache.identity._digest_bytes",
            return_value="collision",
        ):
            self.assertNotEqual(
                build_artifact_configuration(
                    b"config", "image", {"x": torch.tensor([1])}
                ),
                build_artifact_configuration(
                    b"config", "image", {"x": torch.tensor([2])}
                ),
            )

    def test_shared_directory_requires_explicit_positive_budget(self):
        from sglang.srt.arg_groups.serving_hook import handle_multimodal
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy", mm_preprocess_cache_dir=self.directory.name
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            handle_multimodal(args)
        args.mm_preprocess_cache_size_mb = 3
        handle_multimodal(args)

    def test_clear_during_snapshot_does_not_repopulate(self):
        import threading

        processor = self.processor()
        entered, release = threading.Event(), threading.Event()
        snapshot = processor.snapshot_media_source

        def blocked_snapshot(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("snapshot was not released")
            return snapshot(*args)

        async def run():
            with patch.object(
                processor, "snapshot_media_source", side_effect=blocked_snapshot
            ):
                task = asyncio.create_task(processor.prepare_media_artifacts([b"a"]))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                    processor.mm_preprocess_cache.clear()
                finally:
                    release.set()
                self.assertEqual((await task)[0].feature, b"a")
                self.assertEqual(
                    processor.mm_preprocess_cache.cache_info()["entries"], 0
                )

        asyncio.run(run())

    def test_clear_invalidates_other_processor(self):
        first, second = self.processor(), self.processor()
        asyncio.run(first.prepare_media_artifacts([b"a"]))
        second.mm_preprocess_cache.clear()
        asyncio.run(first.prepare_media_artifacts([b"a"]))
        self.assertEqual(len(first.batches), 2)


if __name__ == "__main__":
    unittest.main()

"""Checks for the Qwen-VL rank-local image preprocessing protocol."""

import datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image, ImageDraw
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from sglang.srt.multimodal.qwen_vl_tp_preprocess import (
    DEFERRED_IMAGE_PATCH_WIDTH_KEY,
    agree_on_image_misses,
    encode_tp_images,
    prepare_deferred_image,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _test_ranks(rank, rendezvous):
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=datetime.timedelta(seconds=25),
    )
    try:
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=rank,
            ranks=[0, 1],
            cpu_group=dist.group.WORLD,
            device_group=dist.group.WORLD,
        )
        items = [
            SimpleNamespace(
                cache_key=bytes([i - 1]) * 32,
                feature=torch.full((3, size, size), i, dtype=torch.uint8),
                image_grid_thw=torch.ones((1, 3), dtype=torch.long),
                offsets=[(0, 0)],
                model_specific_data={DEFERRED_IMAGE_PATCH_WIDTH_KEY: 1},
            )
            for i, size in enumerate((2, 8, 4), start=1)
        ]
        ordered = {(bytes([i]) * 32, 1): (item, 1) for i, item in enumerate(items)}
        keys = list(ordered)
        misses = agree_on_image_misses(ordered, {keys[rank]}, group)
        assert list(misses) == keys[:2]

        divergent_order = dict(reversed(ordered.items())) if rank else ordered
        try:
            agree_on_image_misses(divergent_order, set(), group)
        except RuntimeError as exc:
            assert "disagree on the ordered image cache keys" in str(exc), exc
        else:
            raise AssertionError("The ranks accepted a different image order")

        local_images = []
        fail_owner = False

        def processor(images, **kwargs):
            value = int(images[0][0, 0, 0])
            if fail_owner and value == 2:
                raise ValueError("Injected owner failure")
            local_images.append(value)
            return {
                "pixel_values": torch.full((1, 1), float(value)),
                "image_grid_thw": torch.ones((1, 3), dtype=torch.long),
            }

        def encode(packed, *, grid_thw):
            return packed

        args = (
            items,
            encode,
            lambda: processor,
            {},
            torch.device("cpu"),
            torch.float32,
            group,
        )
        out = encode_tp_images(*args)
        assert [value.item() for value in out] == [1, 2, 3]
        assert local_images == ([2] if rank == 0 else [3, 1])
        local_images.clear()
        result = encode_tp_images([items[1]], *args[1:])
        assert result[0].item() == 2
        assert local_images == ([] if rank == 0 else [2])

        fail_owner = True
        try:
            encode_tp_images(*args)
        except RuntimeError as exc:
            assert "A TP rank could not prepare" in str(exc), exc
        else:
            raise AssertionError("The peer did not observe the owner failure")
    finally:
        dist.destroy_process_group()


class TestQwenVLTPPreprocess(unittest.TestCase):
    def test_grid_and_pixels_match_huggingface(self):
        processor = Qwen2VLImageProcessor()
        for size in ((512, 512), (1020, 1024), (1024, 1028), (768, 2048)):
            with self.subTest(size=size):
                image = Image.new("RGB", size, (17, 31, 254))
                ImageDraw.Draw(image).rectangle(
                    (1, 19, size[0] // 3, size[1] * 2 // 3), fill=(230, 85, 6)
                )
                pixels, grid, width = prepare_deferred_image(image, processor, {})
                original = processor(images=[image], return_tensors="pt", device="cpu")
                deferred = processor(images=[pixels], return_tensors="pt", device="cpu")
                torch.testing.assert_close(
                    grid, original.image_grid_thw, rtol=0, atol=0
                )
                self.assertEqual(width, original.pixel_values.shape[1])
                torch.testing.assert_close(
                    deferred.pixel_values, original.pixel_values, rtol=0, atol=0
                )

    def test_rank_consensus_order_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_test_ranks, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()

"""Image layout and cache identity regressions for DeepSeek-V4 vision."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from PIL import Image

import sglang.srt.managers.mm_schedule as mm_schedule
from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.multimodal.processors.deepseek_v4_vl import (
    IMAGE_PLACEHOLDER,
    DeepseekV4VLImageProcessor,
    build_image_block,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestDeepseekV4Processor(CustomTestCase):
    def process(self, text, *, grid=(12, 12), image=None):
        processor = object.__new__(DeepseekV4VLImageProcessor)
        processor.image_token_id = 100
        processor.mm_tokens = None
        processor._tokenizer = SimpleNamespace(
            encode=Mock(side_effect=lambda s, **kw: [1] * len(s))
        )
        processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(
                input_text=text, images=[image if image is not None else object()]
            )
        )
        if image is None:
            h, w = grid
            processor._load_image = Mock(
                return_value=(
                    torch.zeros(h * w, 3, 14, 14, dtype=torch.bfloat16),
                    h,
                    w,
                    (h + 2) // 3,
                    (w + 2) // 3,
                )
            )
        else:
            processor.patch_size = 14
            processor.downsample_ratio = 3
            processor.max_n_token = 384
            processor.min_pixels = 147456
            processor.max_wh_ratio = 8
        output = asyncio.run(processor.process_mm_data_async([], text, None, 8192))
        return output

    def test_compression_padding_changes_cache_identity(self):
        a = self.process("a" + IMAGE_PLACEHOLDER)
        b = self.process("ab" + IMAGE_PLACEHOLDER)
        self.assertNotEqual(a.mm_items[0].hash, b.mm_items[0].hash)
        self.assertEqual(
            a.mm_items[0].hash, self.process("a" + IMAGE_PLACEHOLDER).mm_items[0].hash
        )

    def equal_length_layouts(self):
        # Both inputs survive production preprocessing unchanged: 756 patches
        # and 105 block tokens, but different 2D positions and newline layouts.
        return [
            self.process(IMAGE_PLACEHOLDER, image=Image.new("RGB", size, "red"))
            for size in [(378, 392), (252, 588)]
        ]

    def test_equal_length_grids_have_distinct_cache_and_prefix_identity(self):
        """Feature-only hashing aliases valid grids despite the length guard."""
        a, b = self.equal_length_layouts()
        x, y = a.mm_items[0], b.mm_items[0]
        self.assertEqual(x.feature.shape, (756, 3, 14, 14))
        self.assertTrue(torch.equal(x.feature, y.feature))
        self.assertEqual(hash_feature(x.feature), hash_feature(y.feature))
        self.assertEqual(len(a.input_ids), 105)
        self.assertEqual(len(b.input_ids), 105)
        self.assertNotEqual(
            x.model_specific_data["types"], y.model_specific_data["types"]
        )
        for item in (x, y):
            item.set_pad_value()
        self.assertNotEqual(x.hash, y.hash)
        # These pad values replace the image spans in radix-cache token keys.
        self.assertNotEqual(x.pad_value, y.pad_value)

    def test_equal_length_grids_do_not_share_embeddings(self):
        """Both batch deduplication and warm-cache hits must distinguish grids."""
        a, b = [out.mm_items[0] for out in self.equal_length_layouts()]
        for item in (a, b):
            item.set_pad_value()

        def encode(items):
            return torch.cat(
                [
                    torch.tensor(item.model_specific_data["types"]).float().unsqueeze(1)
                    for item in items
                ]
            )

        embedder = Mock(side_effect=encode)

        def get(items):
            lengths = [len(item.model_specific_data["types"]) for item in items]
            ids = torch.cat(
                [torch.full((n,), item.pad_value) for item, n in zip(items, lengths)]
            )
            return mm_schedule.get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=torch.tensor([item.pad_value for item in items]),
                input_ids=ids,
                items_size=list(range(len(items) + 1)),
                prefix_length=[0] * len(items),
                extend_length=lengths,
                items_offset_list=[item.offsets for item in items],
            )[0]

        for batched in (False, True):
            with self.subTest(batched=batched), patch.object(
                mm_schedule, "embedding_cache", MultiModalStaticCache(1 << 20)
            ):
                embedder.reset_mock()
                actual = get([a, b]) if batched else torch.cat([get([a]), get([b])])
                torch.testing.assert_close(actual, encode([a, b]), atol=0, rtol=0)
                calls = embedder.call_count
                torch.testing.assert_close(get([b]), encode([b]), atol=0, rtol=0)
                self.assertEqual(embedder.call_count, calls)

    def test_offsets_cover_exactly_the_emitted_block(self):
        out = self.process("abc" + IMAGE_PLACEHOLDER + "tail")
        item = out.mm_items[0]
        start, end = item.offsets[0]
        self.assertEqual((start, end + 1), (3, len(out.input_ids) - 4))
        self.assertEqual(end - start + 1, len(item.model_specific_data["types"]))

    def test_layout_permutation_covers_every_patch_once(self):
        for h in range(1, 12):
            for w in range(1, 12):
                for start in range(4):
                    types, perm = build_image_block(h, w, start)
                    self.assertEqual(sorted(perm), list(range(h * w)))
                    self.assertEqual(types.count(2), h * w)
                    self.assertEqual(types[3 - start], 0)
                    self.assertEqual(types[-1], 4)


if __name__ == "__main__":
    unittest.main()

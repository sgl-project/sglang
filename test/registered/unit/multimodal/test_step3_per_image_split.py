"""Step3-VL splits per image, and splitting changes nothing it computes.

This drives the real pieces end to end -- the real ``Step3VLProcessor``, the
real ``get_new_expanded_mm_items``, and the real
``Step3VLForConditionalGeneration.get_image_feature`` -- with only the vision
tower stubbed, so no weights are needed. What it holds to:

* one item per image comes out, where a packed request produces one bundled
  item for the whole request;
* every per-item slice reassembles into exactly what the packed request held;
* the embedding the model builds is bit-identical either way.

Step3 is the one processor in tree whose features pass through HF
``BatchFeature``, which stacks a list of tensors instead of carrying it, so it
is also the one that needs segmented storage rather than a plain list.
"""

import re
import unittest

import torch
from PIL import Image

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items, hash_feature
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.step3_vl import Step3VLForConditionalGeneration
from sglang.srt.multimodal.processors.step3_vl import (
    Step3VLImageProcessor,
    Step3VLProcessor,
)
from sglang.srt.multimodal.segmented_features import SegmentedFeatures
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

SPECIAL_TOKENS = [
    "<im_patch>",
    "<im_start>",
    "<im_end>",
    "<patch_start>",
    "<patch_end>",
    "<patch_newline>",
]
VOCAB = {token: 100 + i for i, token in enumerate(SPECIAL_TOKENS)}
IMAGE_TOKEN_ID = VOCAB["<im_patch>"]
_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


class _Tokenizer:
    """Enough tokenizer to run the processor: the special tokens and a filler.

    Only the count and position of the image token matter here -- that is what
    the placeholder offsets are read from -- so ordinary text becomes one
    filler id per word.
    """

    def get_vocab(self):
        return dict(VOCAB)

    def convert_tokens_to_ids(self, token):
        return VOCAB[token]

    def __call__(self, text):
        batch = []
        for line in text:
            ids = []
            for chunk in _SPLIT.split(line):
                if not chunk:
                    continue
                if chunk in VOCAB:
                    ids.append(VOCAB[chunk])
                else:
                    ids.extend([7] * max(1, len(chunk.split())))
            batch.append(ids)
        return {"input_ids": batch}


class _StubVisionTower:
    """The model's slicing and merge logic, with the ViT replaced.

    The stub keeps what equivalence depends on: one row per output token, and
    a value derived from the image, so two different images cannot produce the
    same feature and a misrouted slice cannot pass unnoticed.
    """

    HIDDEN = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class vision_model:
        dtype = torch.float32

    def _get_vision_model_output(self, pixels):
        tokens = 169 if pixels.shape[-1] == 728 else 81
        per_image = pixels.flatten(1).mean(1)[:, None, None]
        ramp = torch.arange(
            1,
            tokens * self.HIDDEN + 1,
            device=pixels.device,
            dtype=pixels.dtype,
        )
        return per_image * ramp.view(1, tokens, self.HIDDEN)

    def _process_image_features(self, features):
        return features

    _flatten_embeddings = Step3VLForConditionalGeneration._flatten_embeddings
    get_image_feature = Step3VLForConditionalGeneration.get_image_feature


class TestStep3PerImageSplit(CustomTestCase):
    def setUp(self):
        self.processor = Step3VLProcessor(config=None, tokenizer=_Tokenizer())
        # The two lookups the base processor installs for these keys; the
        # collector below is the real one.
        self.collector = Step3VLImageProcessor.__new__(Step3VLImageProcessor)
        self.collector.ATTR_NAME_TO_MODALITY = {
            key: Modality.IMAGE
            for key in (
                "pixel_values",
                "num_patches",
                "patch_pixel_values",
                "patch_newline_mask",
            )
        }
        self.collector.FEATURE_NAMES = ["pixel_values"]

    def _process(self, images):
        text = "start " + " middle ".join(["<im_patch>"] * len(images)) + " end"
        return self.processor(text=[text], images=images, return_tensors="pt")

    def _split(self, data, input_ids):
        items = self.collector.collect_mm_items_from_processor_output(dict(data))
        self.assertEqual(len(items), 1)
        items[0].offsets = self.collector.get_mm_items_offset(
            input_ids.flatten(), IMAGE_TOKEN_ID
        )
        return get_new_expanded_mm_items(items)

    def _packed(self, batch):
        """What the processor produced before it kept the parts."""
        data = dict(batch)
        data["pixel_values"] = torch.cat(batch["pixel_values"].parts(), dim=0)
        return data

    # -- the producer ------------------------------------------------------

    def test_batch_feature_carries_the_parts_instead_of_stacking_them(self):
        batch = self._process(
            [Image.new("RGB", (800, 600)), Image.new("RGB", (640, 640))]
        )
        features = batch["pixel_values"]
        self.assertIsInstance(features, SegmentedFeatures)
        self.assertEqual(features.num_parts, 2)
        # A plain list of two equal-shaped tensors would have come back out of
        # BatchFeature stacked into one rank-5 tensor instead.
        self.assertEqual(len(features.shape), 4)

    def test_the_parts_are_the_tensors_the_processor_built(self):
        batch = self._process(
            [Image.new("RGB", (800, 600)), Image.new("RGB", (400, 1600))]
        )
        packed = self._packed(batch)
        self.assertTrue(
            torch.equal(batch["pixel_values"].dense(), packed["pixel_values"])
        )

    # -- the split ---------------------------------------------------------

    def test_one_item_per_image(self):
        images = [
            Image.new("RGB", (800, 600), (40, 30, 20)),
            Image.new("RGB", (1200, 400), (10, 200, 30)),
            Image.new("RGB", (500, 500), (90, 90, 250)),
        ]
        batch = self._process(images)
        items = self._split(batch, batch["input_ids"])
        self.assertEqual(len(items), 3)

        # Step3 wraps every crop in boundary tokens, so an image's placeholders
        # arrive as num_patches + 1 separate spans rather than one.
        counts = batch["num_patches"].tolist()
        self.assertEqual([len(item.offsets) for item in items], [c + 1 for c in counts])
        self.assertEqual(
            [span for item in items for span in item.offsets],
            self.collector.get_mm_items_offset(
                batch["input_ids"].flatten(), IMAGE_TOKEN_ID
            ),
        )

    def test_every_slice_reassembles_into_the_packed_request(self):
        images = [
            Image.new("RGB", (800, 600), (40, 30, 20)),
            Image.new("RGB", (1200, 400), (10, 200, 30)),
            Image.new("RGB", (500, 500), (90, 90, 250)),
        ]
        batch = self._process(images)
        items = self._split(batch, batch["input_ids"])

        self.assertTrue(
            torch.equal(
                torch.cat([item.feature for item in items], dim=0),
                self._packed(batch)["pixel_values"],
            )
        )
        for key in ("num_patches", "patch_pixel_values", "patch_newline_mask"):
            rejoined = torch.cat(
                [item.model_specific_data[key] for item in items], dim=0
            )
            self.assertTrue(torch.equal(rejoined, batch[key]), key)

    def test_each_item_leaves_owning_its_own_rows(self):
        batch = self._process(
            [Image.new("RGB", (800, 600)), Image.new("RGB", (640, 640))]
        )
        for item in self._split(batch, batch["input_ids"]):
            feature = item.feature
            self.assertEqual(
                feature.untyped_storage().nbytes(),
                feature.numel() * feature.element_size(),
            )

    def test_identical_images_hash_alike_and_different_ones_do_not(self):
        """The reason to split at all: one cache entry per image."""
        repeated = Image.new("RGB", (700, 700), (5, 5, 5))
        batch = self._process(
            [repeated, repeated, Image.new("RGB", (700, 700), (9, 9, 9))]
        )
        items = self._split(batch, batch["input_ids"])
        self.assertEqual(len(items), 3)

        hashes = [hash_feature(item.feature) for item in items]
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    # -- the consumer ------------------------------------------------------

    def test_the_model_builds_the_same_embedding_either_way(self):
        for label, images in (
            ("one image", [Image.new("RGB", (800, 600), (40, 30, 20))]),
            (
                "mixed shapes",
                [
                    Image.new("RGB", (800, 600), (40, 30, 20)),
                    Image.new("RGB", (1200, 400), (10, 200, 30)),
                    Image.new("RGB", (500, 500), (90, 90, 250)),
                ],
            ),
            (
                "no crops",
                [
                    Image.new("RGB", (700, 700), (5, 5, 5)),
                    Image.new("RGB", (700, 700), (9, 9, 9)),
                ],
            ),
            (
                "four tall",
                [Image.new("RGB", (400, 1600), (i * 30, 10, 10)) for i in range(4)],
            ),
        ):
            with self.subTest(label):
                batch = self._process(images)
                model = _StubVisionTower()
                packed = model.get_image_feature(
                    self._split(self._packed(batch), batch["input_ids"])
                )
                split = model.get_image_feature(self._split(batch, batch["input_ids"]))
                self.assertTrue(torch.equal(packed, split))

    def test_the_embedding_covers_exactly_the_placeholder_tokens(self):
        images = [
            Image.new("RGB", (800, 600), (40, 30, 20)),
            Image.new("RGB", (1200, 400), (10, 200, 30)),
        ]
        batch = self._process(images)
        items = self._split(batch, batch["input_ids"])
        embedding = _StubVisionTower().get_image_feature(items)
        placeholders = sum(
            end - start + 1 for item in items for start, end in item.offsets
        )
        self.assertEqual(embedding.shape[0], placeholders)


if __name__ == "__main__":
    unittest.main()

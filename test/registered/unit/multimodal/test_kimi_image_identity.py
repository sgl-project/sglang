"""Kimi image identities: full digest, config and grid folding, wide pads."""

import hashlib
import json
import unittest
from array import array
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    MultiModalityDataPaddingPatternTokenPairs,
    embed_mm_inputs,
    get_new_expanded_mm_items,
)
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.cache import snapshot_media
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    KimiK3DeferredPreprocessing,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.kimi_common import (
    KimiGridMMDataMixin,
    KimiLoadedImage,
    kimi_image_identity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# Two single-element features whose legacy 30-bit pad sentinels collide.
_LEGACY_COLLIDING_FEATURES = (23990, 59582)


class _IdentityOnly(KimiGridMMDataMixin):
    def __init__(self, preprocess_config):
        payload = json.dumps(preprocess_config, sort_keys=True).encode()
        self.processor_fingerprint = "sha256:" + hashlib.sha256(payload).hexdigest()
        self.hf_config = None


def _image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (28, 28, 3), dtype=np.uint8))


def _image_item(grid, feature=None, fmt=MultimodalInputFormat.NORMAL):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=feature if feature is not None else torch.zeros(1),
        model_specific_data={"image_grid_thw": torch.tensor([grid])},
    )
    item.format = fmt
    return item


class _LoadingProcessor(KimiGridMMDataMixin, BaseMultimodalProcessor):
    uses_wide_image_identity = True


class TestKimiImageIdentity(CustomTestCase):
    def test_wide_padding_reaches_embedding_scatter(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                item = _image_item([1, 4, 4])
                item.set_identity(kimi_image_identity("sha256:" + "12" * 32, [1, 4, 4]))
                item.offsets = [(1, 6)]
                item.feature = None
                item.precomputed_embeddings = (
                    torch.arange(18, device=device).reshape(6, 3).float()
                )
                mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=9)
                padded = (
                    MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
                        array("q", [1, 9, 9, 9, 9, 9, 9, 2]), mm_inputs
                    )
                )
                embedding = torch.nn.Embedding(10, 3, device=device)
                for prefix, length in ((0, 8), (3, 4)):
                    inputs = torch.tensor(
                        padded[prefix : prefix + length], device=device
                    )
                    actual, _ = embed_mm_inputs(
                        [mm_inputs],
                        [prefix],
                        [length],
                        inputs,
                        embedding,
                        data_embedding_func_mapping={Modality.IMAGE: lambda _: None},
                    )
                    expected = (
                        item.precomputed_embeddings
                        if prefix == 0
                        else item.precomputed_embeddings[2:6]
                    )
                    self.assertTrue(
                        torch.equal(actual[1:7] if prefix == 0 else actual, expected)
                    )

    def test_loader_hashes_the_same_snapshot_it_decodes(self):
        encoded = BytesIO()
        _image(0).save(encoded, format="JPEG")
        payload = encoded.getvalue()
        loaded = _LoadingProcessor._load_single_item(payload, Modality.IMAGE)
        self.assertIsInstance(loaded, KimiLoadedImage)
        self.assertEqual(loaded.content_digest, snapshot_media(payload).content_digest)
        item = _image_item([1, 4, 4])
        with patch(
            "sglang.srt.multimodal.processors.kimi_common.snapshot_media",
            side_effect=AssertionError("decoded images must not be copied for hashing"),
        ):
            _IdentityOnly({}).assign_kimi_image_identities([item], [loaded])
        self.assertIsNotNone(item.identity)

    def test_hf_grid_alias_has_the_same_identity(self):
        processor = _IdentityOnly({})
        canonical = _image_item([1, 4, 4])
        alias = _image_item([1, 4, 4])
        alias.model_specific_data["grid_thws"] = alias.model_specific_data.pop(
            "image_grid_thw"
        )
        processor.assign_kimi_image_identities([canonical, alias], [_image(0)] * 2)
        self.assertEqual(canonical.identity, alias.identity)

    def test_hf_bundled_grid_alias_splits_before_identity(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(12).reshape(12, 1),
            offsets=[(0, 0), (2, 3)],
            model_specific_data={"grid_thws": torch.tensor([[1, 2, 2], [1, 2, 4]])},
            format=MultimodalInputFormat.PROCESSOR_OUTPUT,
        )
        items = get_new_expanded_mm_items([item])
        self.assertEqual([len(entry.feature) for entry in items], [4, 8])
        _IdentityOnly({}).assign_kimi_image_identities(items, None)
        self.assertNotEqual(items[0].identity, items[1].identity)

    def test_epd_embeddings_receive_content_and_grid_identity(self):
        processor = _IdentityOnly({})
        processor.uses_wide_image_identity = True
        processor.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
        )
        identities = []
        for grid, value in (([1, 4, 4], 1), ([1, 2, 8], 1), ([1, 4, 4], 2)):
            output = processor._build_kimi_mm_data_from_grids(
                [9],
                {Modality.IMAGE: torch.full((4, 2), value)},
                image_token_id=9,
                img_grid_thw=[grid],
            )
            item = output.mm_items[0]
            self.assertIsNotNone(item.identity)
            identities.append(item.identity)
        self.assertEqual(len(set(identities)), 3)

    def test_prepared_inputs_do_not_alias_other_formats(self):
        """equal tensors need different cache keys when the encoder contract differs"""
        processor = _IdentityOnly({})
        feature = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        reference = _image_item(
            [1, 4, 4], feature=feature, fmt=MultimodalInputFormat.PROCESSOR_OUTPUT
        )
        processor.assign_kimi_image_identities([reference], None)
        cache = MultiModalStaticCache(1024)
        cached = EmbeddingResult(embedding=torch.ones(4, 4))
        cache.set(reference.cache_key, cached)

        changed_format = _image_item(
            [1, 4, 4], feature=feature, fmt=MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        )
        changed_format.feature = None
        changed_format.precomputed_embeddings = feature
        processor.assign_kimi_image_identities([changed_format], None)
        self.assertIsNone(cache.get_single(changed_format.cache_key))

        repeat = _image_item(
            [1, 4, 4],
            feature=feature.clone(),
            fmt=MultimodalInputFormat.PROCESSOR_OUTPUT,
        )
        processor.assign_kimi_image_identities([repeat], None)
        self.assertIs(cache.get_single(repeat.cache_key), cached)

    def test_prepared_inputs_keep_deferred_preprocessing_config(self):
        processor = _IdentityOnly({})
        feature = torch.zeros(3, 28, 28, dtype=torch.uint8)
        config = KimiK3DeferredPreprocessing(
            backend="cpu",
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            transparent_bg_config=None,
            resize_config={
                "new_height": 28,
                "new_width": 28,
                "pad_height": 0,
                "pad_width": 0,
            },
        )
        cache = MultiModalStaticCache(1024)
        for index, deferred in enumerate(
            (
                config,
                replace(config, image_mean=[0.0] * 3),
                replace(config, backend="gpu"),
            )
        ):
            with self.subTest(config=deferred):
                item = _image_item([1, 2, 2], feature=feature)
                item.model_specific_data[DEFERRED_PREPROCESSING_KEY] = deferred
                processor.assign_kimi_image_identities([item], None)
                if index == 0:
                    cache.set(
                        item.cache_key, EmbeddingResult(embedding=torch.ones(1, 4))
                    )
                else:
                    self.assertIsNone(cache.get_single(item.cache_key))

    def test_prepared_inputs_use_common_identity_without_snapshot(self):
        """prepared features stay on their device instead of taking the raw-media snapshot path"""
        processor = _IdentityOnly({})
        item = _image_item(
            [1, 4, 4],
            feature=torch.arange(16).reshape(4, 4),
            fmt=MultimodalInputFormat.PROCESSOR_OUTPUT,
        )
        expected = _image_item(
            [1, 4, 4],
            feature=item.feature.clone(),
            fmt=MultimodalInputFormat.PROCESSOR_OUTPUT,
        )
        expected.bind_processor_fingerprint(processor.processor_fingerprint)
        expected.set_pad_value()
        with patch(
            "sglang.srt.multimodal.processors.kimi_common.snapshot_media",
            side_effect=AssertionError(
                "prepared inputs must not be snapshotted to CPU"
            ),
        ):
            processor.assign_kimi_image_identities([item], None)
        self.assertEqual(item.cache_key, expected.cache_key)
        self.assertEqual(item.pad_values, expected.pad_values)

    def test_same_pixels_different_grid_gives_different_key(self):
        processor = _IdentityOnly({"in_patch_limit": 16384})
        image = _image(0)
        items = [_image_item([1, 4, 4]), _image_item([1, 4, 8])]
        processor.assign_kimi_image_identities(items, [image, image])
        self.assertNotEqual(items[0].identity, items[1].identity)
        self.assertNotEqual(items[0].hash, items[1].hash)
        self.assertTrue(set(items[0].pad_values).isdisjoint(items[1].pad_values))

    def test_same_pixels_different_config_gives_different_key(self):
        image = _image(0)
        keys = []
        for limit in (16384, 4096):
            item = _image_item([1, 4, 4])
            _IdentityOnly({"in_patch_limit": limit}).assign_kimi_image_identities(
                [item], [image]
            )
            keys.append(item)
        self.assertNotEqual(keys[0].identity, keys[1].identity)
        self.assertTrue(set(keys[0].pad_values).isdisjoint(keys[1].pad_values))

    def test_same_pixels_grid_and_config_are_stable(self):
        items = [_image_item([1, 4, 4]), _image_item([1, 4, 4])]
        for item in items:
            _IdentityOnly({"in_patch_limit": 16384}).assign_kimi_image_identities(
                [item], [_image(0)]
            )
        self.assertEqual(items[0].identity, items[1].identity)
        self.assertEqual(items[0].pad_values, items[1].pad_values)

    def test_identity_keeps_full_digest(self):
        item = _image_item([1, 4, 4])
        _IdentityOnly({}).assign_kimi_image_identities([item], [_image(0)])
        self.assertEqual(len(item.identity), 32)
        self.assertEqual(item.hash, int.from_bytes(item.identity[:8], "big"))
        self.assertEqual(len(item.pad_values), 4)
        for value in item.pad_values:
            self.assertGreaterEqual(value, MM_PAD_SHIFT_VALUE)
            self.assertLess(value, 2**63)

    def test_known_legacy_collision_is_separated(self):
        legacy = []
        generic = []
        for value in _LEGACY_COLLIDING_FEATURES:
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.tensor([value], dtype=torch.int64),
            )
            item.set_pad_value()
            old_hash = int.from_bytes(
                hashlib.sha256(item.feature.numpy().tobytes()).digest()[:8], "big"
            )
            legacy.append(MM_PAD_SHIFT_VALUE + old_hash % (1 << 30))
            generic.append(item.pad_value)
        self.assertEqual(legacy[0], legacy[1])
        self.assertNotEqual(generic[0], generic[1])

        items = [
            _image_item(
                [1, 4, 4],
                feature=torch.tensor([value], dtype=torch.int64),
                fmt=MultimodalInputFormat.PROCESSOR_OUTPUT,
            )
            for value in _LEGACY_COLLIDING_FEATURES
        ]
        _IdentityOnly({}).assign_kimi_image_identities(items, None)
        self.assertNotEqual(items[0].identity, items[1].identity)
        self.assertTrue(set(items[0].pad_values).isdisjoint(items[1].pad_values))

    def test_many_distinct_images_have_disjoint_pad_values(self):
        seen = set()
        for seed in range(256):
            item = _image_item([1, 4, 4])
            _IdentityOnly({}).assign_kimi_image_identities([item], [_image(seed)])
            self.assertTrue(seen.isdisjoint(item.pad_values))
            seen.update(item.pad_values)

    def test_skip_hash_keeps_legacy_contract(self):
        item = _image_item([1, 4, 4])
        with envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(True):
            _IdentityOnly({}).assign_kimi_image_identities([item], [_image(0)])
        self.assertIsNone(item.identity)
        self.assertIsNone(item.pad_values)

    def test_set_hash_replaces_wide_identity(self):
        item = _image_item([1, 4, 4])
        item.set_identity(kimi_image_identity("sha256:" + "ab" * 32, [1, 4, 4]))
        old_identity = item.identity
        item.set_hash(7)
        self.assertNotEqual(item.identity, old_identity)
        expected = _image_item([1, 4, 4])
        expected.hash = 7
        expected.set_pad_value()
        self.assertEqual(item.padding_values(), expected.padding_values())
        self.assertEqual(len(item.padding_values()), 4)

    def test_rejects_short_identity(self):
        with self.assertRaises(ValueError):
            _image_item([1, 4, 4]).set_identity(b"\x00" * 8)

    def test_wide_pads_fill_spans_and_placeholders(self):
        item = _image_item([1, 4, 4])
        item.set_identity(kimi_image_identity("sha256:" + "cd" * 32, [1, 4, 4]))
        item.offsets = [(1, 6)]
        mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=9)
        padded = MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            array("q", [0, 9, 9, 9, 9, 9, 9, 2]), mm_inputs
        )
        values = item.pad_values
        self.assertEqual(list(padded[1:7]), [values[i % 4] for i in range(6)])
        self.assertEqual(padded[0], 0)
        self.assertEqual(padded[7], 2)
        self.assertEqual(item.padding_sequence(6), list(padded[1:7]))

    def test_wide_pads_fill_token_pairs_without_mutating_input(self):
        item = _image_item([1, 4, 4])
        item.set_identity(kimi_image_identity("sha256:" + "cd" * 32, [1, 4, 4]))
        mm_inputs = MultimodalInputs(mm_items=[item])
        tokens = array("q", [0, 10, 9, 9, 9, 9, 9, 9, 11, 2])
        original = tokens[:]
        padded = MultiModalityDataPaddingPatternTokenPairs([(10, 11)]).pad_input_tokens(
            tokens, mm_inputs
        )
        self.assertEqual(list(padded[2:8]), item.padding_sequence(6))
        self.assertEqual(list(padded[:2]), [0, 10])
        self.assertEqual(list(padded[8:]), [11, 2])
        self.assertEqual(mm_inputs.data_offsets, [1])
        self.assertEqual(tokens, original)

    def test_wide_identity_survives_msgspec_roundtrip(self):
        item = MultimodalDataItem(modality=Modality.IMAGE)
        item.set_identity(kimi_image_identity("sha256:" + "ef" * 32, [1, 2, 2]))
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(item), type=MultimodalDataItem
        )
        self.assertEqual(decoded.identity, item.identity)
        self.assertEqual(decoded.pad_values, item.pad_values)
        self.assertEqual(decoded.pad_value, item.pad_value)
        self.assertEqual(decoded.hash, item.hash)


class TestKimiCallerIdentityRejection(CustomTestCase):
    def test_rejects_mm_hashes(self):
        request = GenerateReqInput(text="x", image_data=["image"], mm_hashes=["07"])
        with self.assertRaisesRegex(ValueError, "mm_hashes"):
            KimiGridMMDataMixin.reject_caller_image_identity(["image"], request)

    def test_rejects_identity_fields_in_image_dicts(self):
        request = GenerateReqInput(text="x", image_data=["image"])
        for key in ("hash", "pad_value", "pad_values", "identity"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                KimiGridMMDataMixin.reject_caller_image_identity(
                    [{"format": "processor_output", key: 1}], request
                )

    def test_accepts_plain_images(self):
        request = GenerateReqInput(text="x", image_data=["image"])
        KimiGridMMDataMixin.reject_caller_image_identity(
            ["image", {"format": "processor_output"}], request
        )


if __name__ == "__main__":
    unittest.main()

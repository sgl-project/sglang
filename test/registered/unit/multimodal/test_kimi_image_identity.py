"""Kimi image identities: full digest, config and grid folding, wide pads."""

import json
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
import torch
from PIL import Image

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    GenerateReqInput,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.multimodal.processors.kimi_cache_config import (
    uses_kimi_wide_image_pads,
    validate_kimi_wide_pad_config,
)
from sglang.srt.multimodal.processors.kimi_common import (
    KimiGridMMDataMixin,
    kimi_image_identity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Two single-element features whose legacy 30-bit pad sentinels collide.
_LEGACY_COLLIDING_FEATURES = (23990, 59582)


class _IdentityOnly(KimiGridMMDataMixin):
    def __init__(self, preprocess_config):
        self.processor_fingerprint = json.dumps(preprocess_config, sort_keys=True)
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


def _server_args(**overrides):
    args = dict(
        kv_events_config=None,
        hicache_storage_backend=None,
        disaggregation_mode="null",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class TestKimiImageIdentity(CustomTestCase):
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

    def test_distinct_images_cannot_share_pad_value(self):
        legacy = []
        for value in _LEGACY_COLLIDING_FEATURES:
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.tensor([value], dtype=torch.int64),
            )
            item.set_pad_value()
            legacy.append(item.pad_value)
        self.assertEqual(legacy[0], legacy[1])

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

    def test_set_hash_clears_wide_identity(self):
        item = _image_item([1, 4, 4])
        item.set_identity(kimi_image_identity("sha256:" + "ab" * 32, [1, 4, 4]))
        item.set_hash(7)
        self.assertIsNone(item.identity)
        self.assertIsNone(item.pad_values)
        self.assertEqual(item.padding_values(), (item.pad_value,))

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


class TestKimiWidePadConfig(CustomTestCase):
    def test_refuses_narrow_token_id_consumers(self):
        cases = (
            _server_args(kv_events_config="{}"),
            _server_args(hicache_storage_backend="file"),
            _server_args(disaggregation_mode="prefill"),
            _server_args(disaggregation_mode="decode"),
        )
        for server_args in cases:
            with self.subTest(server_args=server_args):
                with self.assertRaises(ValueError):
                    validate_kimi_wide_pad_config(server_args)
        with self.assertRaises(ValueError):
            validate_kimi_wide_pad_config(
                _server_args(), hicache_storage_backend="file"
            )

    def test_allows_default_and_skip_hash(self):
        validate_kimi_wide_pad_config(_server_args())
        with envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(True):
            validate_kimi_wide_pad_config(_server_args(kv_events_config="{}"))

    def test_architecture_gate(self):
        self.assertTrue(
            uses_kimi_wide_image_pads(
                ["KimiK3ForConditionalGeneration"], ModelImpl.SGLANG
            )
        )
        self.assertTrue(
            uses_kimi_wide_image_pads(
                ["KimiK25ForConditionalGeneration"], ModelImpl.SGLANG
            )
        )
        self.assertFalse(
            uses_kimi_wide_image_pads(
                ["KimiK3ForConditionalGeneration"], ModelImpl.TRANSFORMERS
            )
        )
        self.assertFalse(uses_kimi_wide_image_pads(["LlamaForCausalLM"], "sglang"))
        self.assertFalse(uses_kimi_wide_image_pads(None, ModelImpl.SGLANG))

    def test_runtime_storage_attach_is_refused(self):
        attached = []
        fake = SimpleNamespace(
            enable_hierarchical_cache=True,
            is_fully_idle=lambda: True,
            tree_cache=SimpleNamespace(
                attach_storage_backend=lambda **kwargs: attached.append(kwargs)
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["KimiK3ForConditionalGeneration"]
                )
            ),
            server_args=_server_args(),
        )
        request = AttachHiCacheStorageReqInput(hicache_storage_backend="file")
        with patch.object(
            scheduler_module,
            "get_resolved_model_impl",
            return_value=ModelImpl.SGLANG,
        ):
            result = scheduler_module.Scheduler.attach_hicache_storage_wrapped(
                fake, request
            )
        self.assertFalse(result.success)
        self.assertIn("int64", result.message)
        self.assertEqual(attached, [])


if __name__ == "__main__":
    unittest.main()

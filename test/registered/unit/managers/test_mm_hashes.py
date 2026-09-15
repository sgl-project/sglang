"""Tests for caller-supplied mm_hashes plumbing.

Verifies the contract that:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. MultimodalDataItem.set_pad_value() honors a pre-set hash and does NOT
     overwrite it via hash_feature().
  3. The derived pad_value is deterministic across requests with identical
     mm_hashes — the property external KV routers depend on.

The wiring step that copies GenerateReqInput.mm_hashes into per-item
MultimodalDataItem.hash lives in tokenizer_manager.py and is exercised by
the e2e serve tests; this file pins the unit-level invariants the wiring
relies on.
"""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    _compute_pad_value,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def rust_hash_fixtures():
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.utils import ImageData

    digest = "sha256:" + "ab" * 32
    other = "sha256:" + "cd" * 32
    image = "https://fixtures/image.png"
    single = {"text": "one", "image_data": image}
    batch = {"text": ["one", "two"], "image_data": [image, image]}
    bodies = [
        single,
        {**single, "mm_hashes": ["a1b2", "not-hex"]},
        {**single, "mm_hashes": [], "mm_content_hashes": [None]},
        {**single, "mm_content_hashes": [digest]},
        {**single, "mm_content_hashes": ["sha256:" + "AB" * 32]},
        {**single, "image_data": {"url": image, "content_hash": digest}},
        {**single, "image_data": {"url": image, "content_hash": ""}},
        {
            **single,
            "image_data": {"url": image, "content_hash": digest},
            "mm_content_hashes": ["sha256:" + "AB" * 32],
            "mm_hashes": ["0x2a"],
        },
        {**batch, "mm_hashes": ["01", "02"], "mm_content_hashes": [digest, None]},
        {
            **batch,
            "image_data": [[image], [image, image]],
            "mm_hashes": ["01", ["02", "03"]],
            "mm_content_hashes": [[digest], [None, other]],
        },
        {
            **batch,
            "mm_hashes": [["01"], ["02"]],
            "mm_content_hashes": [[digest], [other]],
            "sampling_params": {"n": 3},
        },
        {**single, "mm_hashes": ["01"], "sampling_params": {"n": 3}},
        {**batch, "mm_hashes": []},
        {**batch, "mm_content_hashes": [digest]},
        {**batch, "mm_hashes": [["01", "02"], ["03"]]},
        {**batch, "image_data": [[image, image], [image]], "mm_hashes": ["01", "02"]},
        {**single, "mm_content_hashes": []},
        {**single, "mm_content_hashes": [digest, other]},
        {**single, "mm_content_hashes": ["ab" * 32]},
        {**single, "mm_content_hashes": ["sha256:" + "a" * 63]},
        {**single, "mm_content_hashes": ["sha256:" + "x" * 64]},
        {
            **single,
            "image_data": {"url": image, "content_hash": digest},
            "mm_content_hashes": [other],
        },
    ]

    def image_objects(value):
        if isinstance(value, list):
            return [image_objects(item) for item in value]
        if isinstance(value, dict):
            return ImageData(**value)
        return value

    cases = []
    for body in bodies:
        case = {"body": body}
        request = GenerateReqInput(**copy.deepcopy(body))
        # OpenAI image_url objects reach TokenizerManager as ImageData.
        request.image_data = image_objects(request.image_data)
        try:
            request.normalize_batch_and_arguments()
            parents = (
                [request]
                if request.is_single
                else [request[index] for index in range(request.batch_size)]
            )
            expected = []
            for item in parents:
                if not isinstance(item.image_data, list):
                    item.image_data = [item.image_data]
                TokenizerManager._normalize_mm_content_hashes(item)
                expected.extend(
                    {
                        "mm_hashes": item.mm_hashes or [],
                        "mm_content_hashes": item.mm_content_hashes,
                    }
                    for _ in range(request.parallel_sample_num)
                )
            case["expected"] = expected
        except ValueError as error:
            case["error"] = str(error)
        cases.append(case)
    return cases


class TestMmHashesContract(CustomTestCase):
    def test_rust_hash_fixture_matches_python_normalization(self):
        fixture = (
            Path(__file__).resolve().parents[4]
            / "rust/sglang-server/testdata/mm_hashes_python.json"
        )
        self.assertEqual(json.loads(fixture.read_text()), rust_hash_fixtures())

    def test_generate_req_input_accepts_mm_hashes(self):
        """GenerateReqInput exposes mm_hashes as an optional field."""
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeefcafe1234"],
        )
        self.assertEqual(req.mm_hashes, ["deadbeefcafe1234"])

    def test_generate_req_input_defaults_mm_hashes_to_none(self):
        """Absent mm_hashes preserves existing (None) behavior."""
        req = GenerateReqInput(text="hi")
        self.assertIsNone(req.mm_hashes)

    def test_content_hashes_are_distinct_from_feature_hashes(self):
        content_hash = "sha256:" + "ab" * 32
        req = GenerateReqInput(
            text="hi",
            image_data=["http://example.com/img.png"],
            mm_hashes=["deadbeef"],
            mm_content_hashes=[content_hash],
        )
        self.assertEqual(req.mm_hashes, ["deadbeef"])
        self.assertEqual(req.mm_content_hashes, [content_hash])

    def test_batched_hashes_follow_each_request(self):
        req = GenerateReqInput(
            text=["one", "two"],
            image_data=[["a"], ["b", "c"]],
            mm_hashes=["01", ["02", "03"]],
            mm_content_hashes=[
                ["sha256:" + "11" * 32],
                ["sha256:" + "22" * 32, "sha256:" + "33" * 32],
            ],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req[0].mm_hashes, ["01"])
        self.assertEqual(req[1].mm_hashes, ["02", "03"])
        self.assertEqual(len(req[1].mm_content_hashes), 2)

    def test_set_pad_value_honors_preset_hash(self):
        """set_pad_value() must use a pre-set hash without recomputing."""
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xDEADBEEF)
        # If hash_feature is invoked, the test fails — we patch it to
        # raise so any accidental recompute is loud.
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=AssertionError(
                "hash_feature must NOT be called when hash is preset"
            ),
        ):
            item.set_pad_value()
        self.assertEqual(item.hash, 0xDEADBEEF)
        self.assertEqual(item.pad_value, _compute_pad_value(0xDEADBEEF))

    def test_set_pad_value_is_deterministic_across_items(self):
        """Two items with the same preset hash must derive the same pad_value."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0x123456789ABCDEF0)
        # No feature payload — set_pad_value uses the preset hash.
        a.set_pad_value()
        b.set_pad_value()
        self.assertEqual(a.pad_value, b.pad_value)
        self.assertEqual(a.hash, b.hash)

    def test_set_pad_value_distinguishes_different_preset_hashes(self):
        """Distinct preset hashes must produce distinct pad_values."""
        a = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        b = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        a.set_pad_value()
        b.set_pad_value()
        self.assertNotEqual(a.pad_value, b.pad_value)

    def test_set_hash_updates_an_existing_pad_value(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        item.set_pad_value()

        item.set_hash(0xBBBB)

        self.assertEqual(item.hash, 0xBBBB)
        self.assertEqual(item.pad_value, _compute_pad_value(0xBBBB))


if __name__ == "__main__":
    unittest.main()

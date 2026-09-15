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

import asyncio
import pickle
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    _compute_pad_value,
)
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMmHashesContract(CustomTestCase):
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


class TestMultimodalHashExecutor(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    def setUp(self):
        self.executor = MultimodalHashExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)

    async def test_cpu_hashes_match_scheduler_and_survive_pickle(self):
        """Offloading must preserve the prefix-cache identity for every CPU layout."""
        features = [
            torch.arange(24).reshape(4, 6).t(),
            torch.arange(8, dtype=torch.bfloat16),
            np.arange(24).reshape(4, 6)[:, ::2],
            [torch.arange(3), [torch.arange(7), torch.arange(2)]],
        ]
        for feature in features:
            for field in ("feature", "precomputed_embeddings"):
                with self.subTest(type=type(feature), field=field):
                    expected = MultimodalDataItem(
                        modality=Modality.IMAGE, **{field: feature}
                    )
                    expected.set_pad_value()
                    item = MultimodalDataItem(
                        modality=Modality.IMAGE, **{field: feature}
                    )
                    await self.executor.set_pad_values([item])
                    received = pickle.loads(pickle.dumps(item))
                    with patch(
                        "sglang.srt.managers.mm_utils.hash_feature",
                        side_effect=AssertionError("scheduler rehashed a feature"),
                    ):
                        received.set_pad_value()
                    self.assertEqual(
                        (received.hash, received.pad_value),
                        (expected.hash, expected.pad_value),
                    )

    async def test_overrides_and_device_features_stay_on_caller_thread(self):
        caller_thread = threading.get_ident()
        overridden = MultimodalDataItem(modality=Modality.IMAGE, hash=1234)
        padded = MultimodalDataItem(modality=Modality.IMAGE, hash=99, pad_value=1)
        device_item = MultimodalDataItem(
            modality=Modality.IMAGE, feature=torch.empty(1, device="meta")
        )

        def device_hash(feature):
            self.assertEqual(threading.get_ident(), caller_thread)
            self.assertIs(feature, device_item.feature)
            return 42

        with patch("sglang.srt.managers.mm_utils.hash_feature", device_hash):
            await self.executor.set_pad_values([overridden, padded, device_item])
        self.assertEqual(overridden.hash, 1234)
        self.assertEqual((padded.hash, padded.pad_value), (99, 1))
        self.assertEqual(device_item.hash, 42)

    async def test_precomputed_identity_survives_msgpack_router_hops(self):
        from array import array

        from sglang.srt.managers.io_struct import (
            TokenizedEmbeddingReqInput,
            msgpack_decode,
            msgpack_encode,
        )
        from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
        from sglang.srt.sampling.sampling_params import SamplingParams

        item = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.arange(8))
        await self.executor.set_pad_values([item])
        request = TokenizedEmbeddingReqInput(
            rid="hash-roundtrip",
            input_text="",
            input_ids=array("q", [1]),
            mm_inputs=MultimodalProcessorOutput(mm_items=[item]),
            token_type_ids=None,
            sampling_params=SamplingParams(),
        )
        for _ in range(2):
            request = msgpack_decode(msgpack_encode(request))
        received = request.mm_inputs.mm_items[0]
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=AssertionError("scheduler must reuse frontend hash"),
        ):
            received.set_pad_value()
        self.assertEqual(
            (received.hash, received.pad_value), (item.hash, item.pad_value)
        )

    async def test_skip_hash_preserves_existing_semantics(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(4))
        with (
            envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(True),
            patch(
                "sglang.srt.managers.mm_utils.hash_feature",
                side_effect=AssertionError("skip mode read a feature"),
            ),
        ):
            await self.executor.set_pad_values([item])
        self.assertIsNotNone(item.pad_value)

    async def test_cancelled_native_reader_keeps_capacity_until_finished(self):
        """Repeated cancellation cannot admit another reader or free its input."""
        started = asyncio.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        loop = asyncio.get_running_loop()
        first = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(4))
        second = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.zeros(4))
        reads = []

        def blocking_hash(feature):
            reads.append(feature)
            if feature is first.feature:
                loop.call_soon_threadsafe(started.set)
                if not release.wait(5):
                    raise TimeoutError("test did not release hash worker")
            return hash_feature(feature)

        with patch("sglang.srt.managers.mm_utils.hash_feature", blocking_hash):
            task = asyncio.create_task(self.executor.set_pad_values([first]))
            await asyncio.wait_for(started.wait(), 3)
            # Reaching here while the hash blocks also proves loop responsiveness.
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            queued = asyncio.create_task(self.executor.set_pad_values([second]))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(len(reads), 1)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNone(second.hash)
            await self.executor.set_pad_values([second])
        self.assertEqual(len(reads), 2)
        self.assertIsNotNone(second.hash)

    async def test_error_releases_capacity_and_shutdown_rejects_new_work(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(4))
        with patch(
            "sglang.srt.managers.mm_utils.hash_feature",
            side_effect=ValueError("invalid feature"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid feature"):
                await self.executor.set_pad_values([item])
        await asyncio.wait_for(self.executor.set_pad_values([item]), 3)
        self.executor.shutdown()
        self.executor.shutdown()
        item.hash = item.pad_value = None
        with self.assertRaisesRegex(RuntimeError, "shutdown"):
            await self.executor.set_pad_values([item])


if __name__ == "__main__":
    unittest.main()

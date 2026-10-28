"""Legacy routing hashes must not replace content-verified cache identities."""

import asyncio
import pickle
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.mm_utils import hash_feature_digest
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
)
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMmHashesContract(CustomTestCase):
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

    def test_preset_hash_does_not_override_feature_contents(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE, hash=0xDEADBEEF, feature=feature
            )
            for feature in (torch.zeros(4), torch.ones(4))
        ]
        for item in items:
            item.set_pad_value()
            self.assertEqual(item.hash, 0xDEADBEEF)
            self.assertEqual(len(item.padding_values()), 4)
            self.assertEqual(item.pad_value, item.padding_values()[0])
        self.assertNotEqual(items[0].identity, items[1].identity)
        self.assertNotEqual(items[0].padding_values(), items[1].padding_values())

    def test_set_hash_updates_an_existing_pad_value(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        item.set_pad_value()

        item.set_hash(0xBBBB)

        self.assertEqual(item.hash, 0xBBBB)
        expected = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        expected.set_pad_value()
        self.assertEqual(item.padding_values(), expected.padding_values())

    def test_wide_padding_is_shared_by_all_modalities(self):
        identities = set()
        for modality in Modality.all():
            with self.subTest(modality=modality):
                a = MultimodalDataItem(modality=modality, hash=7)
                b = MultimodalDataItem(modality=modality, hash=7 + (1 << 30))
                a.set_pad_value()
                b.set_pad_value()
                self.assertNotEqual(a.pad_value, b.pad_value)
                repeated = MultimodalDataItem(modality=modality, hash=7)
                repeated.set_pad_value()
                self.assertEqual(a.padding_values(), repeated.padding_values())
                self.assertNotIn(a.identity, identities)
                identities.add(a.identity)
                self.assertEqual(
                    a.padding_sequence(6), list(a.pad_values) + list(a.pad_values[:2])
                )
                for value in a.padding_values():
                    self.assertGreaterEqual(value, MM_PAD_SHIFT_VALUE)
                    self.assertLess(value, 1 << 63)

    def test_set_hash_and_constructor_normalize_identically(self):
        for value in (-1, 1 << 64, (1 << 128) + 7):
            with self.subTest(value=value):
                a = MultimodalDataItem(modality=Modality.IMAGE, hash=value)
                b = MultimodalDataItem(modality=Modality.IMAGE)
                a.set_pad_value()
                b.set_hash(value)
                self.assertEqual(a.hash, b.hash)
                self.assertEqual(a.padding_values(), b.padding_values())


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
                        "sglang.srt.managers.mm_utils.hash_feature_digest",
                        side_effect=AssertionError("scheduler rehashed a feature"),
                    ):
                        received.set_pad_value()
                    self.assertEqual(
                        (received.hash, received.padding_values()),
                        (expected.hash, expected.padding_values()),
                    )

    async def test_overrides_and_device_features_stay_on_caller_thread(self):
        caller_thread = threading.get_ident()
        overridden = MultimodalDataItem(modality=Modality.IMAGE, hash=1234)
        padded = MultimodalDataItem(modality=Modality.IMAGE)
        padded.set_identity(b"p" * 32)
        expected_padding = padded.padding_values()
        device_item = MultimodalDataItem(
            modality=Modality.IMAGE, feature=torch.empty(1, device="meta")
        )

        def device_hash(payload):
            if payload[5] is device_item.feature:
                self.assertEqual(threading.get_ident(), caller_thread)
                return b"d" * 32
            return hash_feature_digest(payload)

        with patch("sglang.srt.managers.mm_utils.hash_feature_digest", device_hash):
            await self.executor.set_pad_values([overridden, padded, device_item])
        self.assertEqual(overridden.hash, 1234)
        self.assertEqual(padded.padding_values(), expected_padding)
        self.assertEqual(device_item.identity, b"d" * 32)

    async def test_skip_hash_preserves_existing_semantics(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(4))
        with (
            envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(True),
            patch(
                "sglang.srt.managers.mm_utils.hash_feature_digest",
                side_effect=AssertionError("skip mode read a feature"),
            ),
        ):
            await self.executor.set_pad_values([item])
        self.assertIsNotNone(item.pad_value)

    async def test_gpu_metadata_stays_on_caller_thread(self):
        caller_thread = threading.get_ident()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.ones(4),
            model_specific_data={"grid": torch.empty(3, device="meta")},
        )

        def digest(payload):
            self.assertEqual(threading.get_ident(), caller_thread)
            self.assertIs(payload[7], item.model_specific_data)
            return b"m" * 32

        with patch("sglang.srt.managers.mm_utils.hash_feature_digest", digest):
            await self.executor.set_pad_values([item])
        self.assertEqual(item.identity, b"m" * 32)

    async def test_cancelled_native_reader_keeps_capacity_until_finished(self):
        """Repeated cancellation cannot admit another reader or free its input."""
        started = asyncio.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        loop = asyncio.get_running_loop()
        first = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(4))
        second = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.zeros(4))
        reads = []

        def blocking_hash(payload):
            reads.append(payload[5])
            if payload[5] is first.feature:
                loop.call_soon_threadsafe(started.set)
                if not release.wait(5):
                    raise TimeoutError("test did not release hash worker")
            return hash_feature_digest(payload)

        with patch("sglang.srt.managers.mm_utils.hash_feature_digest", blocking_hash):
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
            "sglang.srt.managers.mm_utils.hash_feature_digest",
            side_effect=ValueError("invalid feature"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid feature"):
                await self.executor.set_pad_values([item])
        await asyncio.wait_for(self.executor.set_pad_values([item]), 3)
        self.executor.shutdown()
        self.executor.shutdown()
        item.clear_identity()
        with self.assertRaisesRegex(RuntimeError, "shutdown"):
            await self.executor.set_pad_values([item])


if __name__ == "__main__":
    unittest.main()

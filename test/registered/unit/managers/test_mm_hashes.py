"""Tests for caller-supplied mm_hashes plumbing.

Verifies the contract that:
  1. GenerateReqInput.mm_hashes is an optional list of hex strings.
  2. MultimodalDataItem.set_pad_value() honors a pre-set hash and does NOT
     overwrite it via hash_feature().
  3. The derived pad_value is deterministic across requests with identical
     mm_hashes — the property external KV routers depend on.
  4. TokenizerManager rebuilds processor-built padded_input_ids after it
     applies the caller hashes, so the ids carry the caller-derived pads.
"""

import asyncio
import pickle
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        self.assertEqual(len(item.padding_values()), 4)
        self.assertEqual(item.pad_value, item.padding_values()[0])

    def test_set_hash_updates_an_existing_pad_value(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=0xAAAA)
        item.set_pad_value()

        item.set_hash(0xBBBB)

        self.assertEqual(item.hash, 0xBBBB)
        expected = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        expected.set_pad_value()
        self.assertEqual(item.padding_values(), expected.padding_values())

    def test_wide_padding_is_shared_by_all_modalities(self):
        expected = (
            4445679232235744898,
            1252851680943950380,
            4490111419971165939,
            1999168716371743145,
        )
        for modality in Modality.all():
            with self.subTest(modality=modality):
                a = MultimodalDataItem(modality=modality, hash=7)
                b = MultimodalDataItem(modality=modality, hash=7 + (1 << 30))
                a.set_pad_value()
                b.set_pad_value()
                self.assertNotEqual(a.pad_value, b.pad_value)
                self.assertEqual(a.padding_values(), expected)
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

    def test_caller_hashes_repad_processor_padded_input_ids(self):
        """Caller mm_hashes must reach the padded ids a processor precomputed;
        the scheduler reuses them, so a stale pad leaves no slot for the embedding."""
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

        expanded_ids = [1, 2, 9, 9, 9, 4]
        item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(2, 4)])
        item.set_hash(0xAAAA)
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.model_config = Mock()
        tm.model_config.hf_config.architectures = []
        tm.max_req_input_len = 128
        tm.mm_processor = Mock(prefer_tokenized_input=True)
        tm.mm_processor.process_mm_data_async = AsyncMock(
            return_value=MultimodalProcessorOutput(
                mm_items=[item],
                input_ids=expanded_ids,
                padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
                    expanded_ids, [item]
                ),
            )
        )
        tm._validate_one_request = Mock()
        tm._create_tokenized_object = Mock()
        obj = GenerateReqInput(
            input_ids=[1, 2, 9, 4], image_data=["image"], mm_hashes=["bbbb"]
        )

        # The mock processor has no hash executor to await.
        with envs.SGLANG_MM_PRECOMPUTE_HASH.override(False):
            asyncio.run(tm._tokenize_one_request(obj))

        mm_inputs = tm._create_tokenized_object.call_args.args[4]
        expected = MultimodalDataItem(modality=Modality.IMAGE, hash=0xBBBB)
        expected.set_pad_value()
        self.assertEqual(
            mm_inputs.padded_input_ids, [1, 2, *expected.padding_sequence(3), 4]
        )


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
                        (received.hash, received.padding_values()),
                        (expected.hash, expected.padding_values()),
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

"""Unit tests for caller multimodal routing labels on KV events.

`mm_routing_labels` attaches one opaque label to each multimodal item. SGLang
keeps its own item identity: a label never changes hashes, pad values or
reuse. The KV event recorder maps each labeled item's pad values to the first
label seen and reports per-block `[start, length, label]` runs on BlockStored
(`mm_routing_spans`), so an external router can translate those blocks into
its own routing representation.
"""

import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.disaggregation.encoder.receiver import WaitingZmqRequest
from sglang.srt.disaggregation.kv_events import BlockStored, EventBatch
from sglang.srt.managers.io_struct import (
    MAX_MM_ROUTING_LABEL_BYTES,
    GenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache import events
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

X = (11_000_000_001, 12_000_000_002, 13_000_000_003, 14_000_000_004)
Y = (21_000_000_001, 22_000_000_002, 23_000_000_003, 24_000_000_004)
LABEL_U = "9e3779b97f4a7c15"
LABEL_V = "0123456789abcdef"


def item_tokens(values, count, offset=0):
    return [values[(offset + index) % len(values)] for index in range(count)]


def stored(block_hash, parent, token_ids, spans=None):
    return BlockStored(
        block_hashes=[block_hash],
        parent_block_hash=parent,
        token_ids=token_ids,
        block_size=len(token_ids),
        lora_id=None,
        mm_routing_spans=spans,
    )


class TestMmRoutingLabels(CustomTestCase):
    def setUp(self):
        self._saved = events.MM_ROUTING_LABELS
        events.MM_ROUTING_LABELS = events.MmRoutingLabels(capacity=8)
        self.registry = events.MM_ROUTING_LABELS

    def tearDown(self):
        events.MM_ROUTING_LABELS = self._saved

    def test_spans_mark_labeled_runs_per_block(self):
        self.registry.register(X, LABEL_U)
        self.registry.register(Y, LABEL_V)
        block0 = list(range(100, 110)) + item_tokens(X, 6)
        block1 = item_tokens(X, 10, offset=6) + item_tokens(Y, 6)
        self.assertEqual(
            events.mm_routing_spans(block0 + block1, 16, 2),
            [[(10, 6, LABEL_U)], [(0, 10, LABEL_U), (10, 6, LABEL_V)]],
        )

    def test_unlabeled_or_text_only_blocks_have_no_spans(self):
        self.assertIsNone(events.mm_routing_spans(list(range(16)), 16, 1))
        self.registry.register(X, LABEL_U)
        self.assertIsNone(events.mm_routing_spans(list(range(16)), 16, 1))
        self.assertIsNone(events.mm_routing_spans(item_tokens(Y, 16), 16, 1))

    def test_first_label_wins_and_capacity_bounds_registry(self):
        self.registry.register(X, LABEL_U)
        self.registry.register(X, LABEL_V)
        self.assertEqual(self.registry.get(X[0]), LABEL_U)
        self.registry.register(Y, LABEL_V)  # 8 entries: at capacity
        self.registry.register((31_000_000_001,), "ffff000000000001")
        self.assertIsNone(self.registry.get(X[0]))  # least recently used

    def test_register_maps_each_labeled_item_pad_value(self):
        labeled = MultimodalDataItem(
            modality=Modality.IMAGE, hash=0xDEADBEEF, routing_label=LABEL_U
        )
        unlabeled = MultimodalDataItem(modality=Modality.IMAGE, hash=0xFEEDFACE)
        for item in (labeled, unlabeled):
            item.set_pad_value()
        events.enable_mm_routing_labels()
        events.register_mm_routing_labels([labeled, unlabeled])
        for value in labeled.padding_values():
            self.assertEqual(self.registry.get(value), LABEL_U)
        for value in unlabeled.padding_values():
            self.assertIsNone(self.registry.get(value))

    def test_registration_waits_for_an_enabled_event_recorder(self):
        # Only the rank that records KV events reads labels; the other
        # tensor-parallel ranks build disabled recorders and register nothing.
        labeled = MultimodalDataItem(
            modality=Modality.IMAGE, hash=0xDEADBEEF, routing_label=LABEL_U
        )
        labeled.set_pad_value()
        events.KVCacheEventRecorder(enabled=False, page_size=16)
        events.register_mm_routing_labels([labeled])
        self.assertFalse(self.registry)

        events.KVCacheEventRecorder(enabled=True, page_size=16)
        events.register_mm_routing_labels([labeled])
        for value in labeled.padding_values():
            self.assertEqual(self.registry.get(value), LABEL_U)

    def test_label_does_not_change_item_identity(self):
        labeled = MultimodalDataItem(
            modality=Modality.IMAGE, hash=0xDEADBEEF, routing_label=LABEL_U
        )
        plain = MultimodalDataItem(modality=Modality.IMAGE, hash=0xDEADBEEF)
        for item in (labeled, plain):
            item.set_pad_value()
        self.assertEqual(labeled.hash, plain.hash)
        self.assertEqual(labeled.padding_values(), plain.padding_values())

    def test_batched_labels_follow_each_request(self):
        req = GenerateReqInput(
            text=["one", "two"],
            image_data=[["a"], ["b", "c"]],
            mm_routing_labels=["01", ["02", "03"]],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req[0].mm_routing_labels, ["01"])
        self.assertEqual(req[1].mm_routing_labels, ["02", "03"])

    def test_batched_labels_cover_every_modality(self):
        cases = [
            dict(video_data=["a.mp4", "b.mp4"], mm_routing_labels=[["A"], ["B"]]),
            dict(audio_data=["a.wav", "b.wav"], mm_routing_labels=[["A"], "B"]),
            dict(
                image_data=[["a.jpg"], ["b.jpg"]],
                audio_data=["a.wav", "b.wav"],
                mm_routing_labels=[["I", "A"], ["J", "B"]],
            ),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                req = GenerateReqInput(text=["one", "two"], **kwargs)
                req.normalize_batch_and_arguments()
                expected = [
                    labels if isinstance(labels, list) else [labels]
                    for labels in kwargs["mm_routing_labels"]
                ]
                self.assertEqual([req[i].mm_routing_labels for i in (0, 1)], expected)

    def test_batched_labels_need_one_entry_per_request(self):
        req = GenerateReqInput(
            text=["one", "two"], image_data=[["a"], ["b"]], mm_routing_labels=[["A"]]
        )
        with self.assertRaises(ValueError):
            req.normalize_batch_and_arguments()

    def test_parallel_sampling_repeats_each_requests_labels(self):
        req = GenerateReqInput(
            text="one",
            image_data=["a"],
            video_data=["v"],
            mm_routing_labels=["I", "V"],
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.mm_routing_labels, [["I", "V"], ["I", "V"]])

    def test_label_size_is_bounded_in_bytes(self):
        limit = MAX_MM_ROUTING_LABEL_BYTES
        GenerateReqInput(
            text="one", image_data=["a"], mm_routing_labels=["x" * limit]
        ).normalize_batch_and_arguments()
        # Two-byte characters: under the limit in characters, over it in bytes.
        for label in ("x" * (limit + 1), "\u00e9" * (limit // 2 + 1)):
            with self.subTest(chars=len(label)):
                req = GenerateReqInput(
                    text="one", image_data=["a"], mm_routing_labels=[label]
                )
                with self.assertRaises(ValueError):
                    req.normalize_batch_and_arguments()
        batch = GenerateReqInput(
            text=["one", "two"],
            image_data=[["a"], ["b"]],
            mm_routing_labels=[["A"], ["x" * (limit + 1)]],
        )
        with self.assertRaises(ValueError):
            batch.normalize_batch_and_arguments()

    def test_attach_follows_item_order_and_skips_count_mismatch(self):
        items = [
            MultimodalDataItem(modality=Modality.IMAGE),
            MultimodalDataItem(modality=Modality.VIDEO),
        ]
        events.attach_mm_routing_labels(items, [LABEL_U])
        self.assertEqual([item.routing_label for item in items], [None, None])
        events.attach_mm_routing_labels(items, [LABEL_U, ""])
        self.assertEqual([item.routing_label for item in items], [LABEL_U, None])

    def test_encoder_assembly_in_the_scheduler_attaches_labels(self):
        # With encoder disaggregation that delivers embeddings to the
        # scheduler, items are built there; labels ride on the tokenized request.
        items = [
            MultimodalDataItem(modality=Modality.IMAGE),
            MultimodalDataItem(modality=Modality.VIDEO),
        ]
        waiting = object.__new__(WaitingZmqRequest)
        waiting.mm_processor = SimpleNamespace(
            prefer_tokenized_input=False,
            get_validated_mm_data=lambda prompt, embedding, **meta: SimpleNamespace(
                mm_items=items, input_ids=[1, 2, 3]
            ),
        )
        waiting.recv_req = SimpleNamespace(
            input_text="prompt",
            input_ids=None,
            mm_inputs=None,
            mm_routing_labels=[LABEL_U, LABEL_V],
        )
        waiting.recv_embedding_data = SimpleNamespace(get_mm_extra_meta=lambda: {})
        waiting._pool_slot_id = None
        waiting.embedding_pool = None
        waiting._finish_assemble(None)
        self.assertEqual([item.routing_label for item in items], [LABEL_U, LABEL_V])

    def test_radix_cache_insert_reports_spans(self):
        # The real producer path: insert -> record_store -> BlockStored spans.
        cache = RadixCache.create_simulated(page_size=16, enable_kv_cache_events=True)
        cache.take_events()
        self.registry.register(X, LABEL_U)
        tokens = list(range(100, 110)) + item_tokens(X, 22)
        cache.insert(InsertParams(key=RadixKey(tokens), value=None))
        stored = [e for e in cache.take_events() if isinstance(e, BlockStored)]
        spans = [
            block_spans
            for event in stored
            for block_spans in (
                event.mm_routing_spans or [[]] * len(event.block_hashes)
            )
        ]
        self.assertEqual(spans, [[(10, 6, LABEL_U)], [(0, 16, LABEL_U)]])

    def test_coalescing_long_chains_keeps_spans_aligned(self):
        recorder = events.KVCacheEventRecorder(enabled=True, page_size=1)
        recorder.enqueue(stored(0, None, [X[0]], [[(0, 1, LABEL_U)]]))
        for block in range(1, 1000):
            recorder.enqueue(stored(block, block - 1, [block]))
        recorder.enqueue(stored(1000, 999, [Y[0]], [[(0, 1, LABEL_V)]]))
        (event,) = recorder.take()
        self.assertEqual(len(event.mm_routing_spans), len(event.block_hashes))
        self.assertEqual(event.mm_routing_spans[0], [(0, 1, LABEL_U)])
        self.assertEqual(event.mm_routing_spans[1000], [(0, 1, LABEL_V)])
        self.assertTrue(all(spans == [] for spans in event.mm_routing_spans[1:1000]))

    def test_enqueue_coalescing_keeps_spans_aligned(self):
        recorder = events.KVCacheEventRecorder(enabled=True, page_size=16)
        recorder.enqueue(stored(1, None, list(range(16))))
        recorder.enqueue(stored(2, 1, item_tokens(X, 16), [[(0, 16, LABEL_U)]]))
        recorder.enqueue(stored(3, 2, list(range(16))))
        (event,) = recorder.take()
        self.assertEqual(event.block_hashes, [1, 2, 3])
        self.assertEqual(event.mm_routing_spans, [[], [(0, 16, LABEL_U)], []])

    def test_wire_format_is_an_optional_map_key(self):
        with_spans = stored(2, None, item_tokens(X, 4), [[(0, 4, LABEL_U)]])
        without = stored(3, None, [1, 2, 3, 4])
        batch = EventBatch(ts=0.0, events=[with_spans, without])
        decoded = msgspec.msgpack.decode(msgspec.msgpack.Encoder().encode(batch))
        self.assertEqual(decoded[1][0]["mm_routing_spans"], [[[0, 4, LABEL_U]]])
        self.assertNotIn("mm_routing_spans", decoded[1][1])  # omitted when None


if __name__ == "__main__":
    unittest.main()

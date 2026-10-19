"""Regression tests for multimodal Tensor view amplification in pickle IPC."""

import errno
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import mm_utils
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestInlineMultimodalStorage(CustomTestCase):
    def _request(self, items):
        return SimpleNamespace(mm_inputs=SimpleNamespace(mm_items=items))

    def _prepare(self, req, use_shm=False, skip_tokenizer_init=False):
        with (
            patch.object(
                mm_utils, "_get_is_default_transport", return_value=not use_shm
            ),
            patch.object(
                mm_utils,
                "get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=skip_tokenizer_init),
            ),
        ):
            return mm_utils.wrap_shm_features(req)

    def _expanded_request(self, count, dtype=torch.float32):
        patches, width = 16, 128
        source = (
            torch.arange(count * patches * width, dtype=torch.float32)
            .reshape(count * patches, width)
            .to(dtype)
        )
        bundled = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=source,
            offsets=[(i * 4, i * 4 + 3) for i in range(count)],
            model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]] * count)},
        )
        items = mm_utils.get_new_expanded_mm_items([bundled])
        return source, self._request(items)

    def test_image_views_have_linear_wire_size_and_preserve_values(self):
        for dtype in (torch.float32, torch.bfloat16):
            for count in (4, 16):
                with self.subTest(dtype=dtype, count=count):
                    source, req = self._expanded_request(count, dtype)
                    self._prepare(req)
                    wire = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
                    logical_bytes = source.numel() * source.element_size()
                    self.assertLess(len(wire), logical_bytes * 2 + 16384)
                    received = pickle.loads(wire)
                    actual = torch.cat([x.feature for x in received.mm_inputs.mm_items])
                    self.assertTrue(torch.equal(source, actual))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(
                        [x.offsets for x in received.mm_inputs.mm_items],
                        [[(i * 4, i * 4 + 3)] for i in range(count)],
                    )

    def test_nested_embeddings_and_metadata_roundtrip_without_parent_storage(self):
        parent = torch.arange(8192, dtype=torch.float32).reshape(64, 128)
        view = parent[3:5, ::2]
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            precomputed_embeddings=[view, (view,)],
            model_specific_data={"nested": {"same_view": view}},
        )
        req = self._prepare(self._request([item]))
        wire = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertLess(len(wire), 8192)
        received = pickle.loads(wire).mm_inputs.mm_items[0]
        actual = received.precomputed_embeddings[0]
        self.assertTrue(torch.equal(actual, view))
        self.assertIs(actual, received.precomputed_embeddings[1][0])
        self.assertIs(actual, received.model_specific_data["nested"]["same_view"])
        self.assertTrue(torch.equal(parent[3:5, ::2], view))

    def test_full_storage_tensors_are_not_copied(self):
        source = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        transposed = source.T
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=source,
            precomputed_embeddings=transposed,
        )
        self._prepare(self._request([item]))
        self.assertIs(item.feature, source)
        self.assertIs(item.precomputed_embeddings, transposed)

    def test_empty_view_does_not_serialize_its_parent(self):
        empty = torch.ones(8192)[0:0]
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=empty)
        req = self._prepare(self._request([item]))
        wire = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertLess(len(wire), 4096)
        self.assertEqual(pickle.loads(wire).mm_inputs.mm_items[0].feature.numel(), 0)

    def test_shm_allocation_failure_uses_compact_inline_storage(self):
        source, req = self._expanded_request(8)
        with patch.object(
            mm_utils, "ShmPointerMMData", side_effect=OSError(errno.ENOSPC, "test")
        ):
            self._prepare(req, use_shm=True)
        wire = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertLess(len(wire), source.numel() * source.element_size() * 2 + 16384)
        received = pickle.loads(wire)
        self.assertTrue(
            torch.equal(
                source, torch.cat([x.feature for x in received.mm_inputs.mm_items])
            )
        )

    def test_successful_shm_does_not_clone_feature_views(self):
        source = torch.arange(8192, dtype=torch.float32).reshape(64, 128)
        view = source[2:4]
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=view)
        req = self._request([item])
        try:
            with patch.object(
                torch.Tensor,
                "clone",
                side_effect=AssertionError("unexpected feature copy"),
            ):
                self._prepare(req, use_shm=True)
            self.assertIsInstance(item.feature, mm_utils.ShmPointerMMData)
            wire = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
            self.assertLess(len(wire), 4096)
            received = pickle.loads(wire).mm_inputs.mm_items[0].feature
            self.assertTrue(torch.equal(received.materialize(), view))
        finally:
            if isinstance(item.feature, mm_utils.ShmPointerMMData):
                item.feature.close_and_unlink()

    def test_skip_tokenizer_init_inline_features_are_also_compact(self):
        source, req = self._expanded_request(8)
        self._prepare(req, use_shm=True, skip_tokenizer_init=True)
        wire = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertLess(len(wire), source.numel() * source.element_size() * 2 + 16384)

    def test_non_cpu_tensors_are_untouched(self):
        tensor = torch.empty((16, 128), device="meta")[2:4]
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=tensor)
        self._prepare(self._request([item]))
        self.assertIs(item.feature, tensor)


if __name__ == "__main__":
    unittest.main(verbosity=2)

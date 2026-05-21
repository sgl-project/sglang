"""Unit tests for srt/multimodal/processors/base_processor.py — no server, no model weights."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import dataclasses
import enum
import re
import sys
import types
from unittest import mock

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase

# These stub classes mirror the real schedule_batch types and are injected into
# the sys.modules stub so that base_processor.py uses the same classes as the tests.
class Modality(enum.Enum):
    IMAGE = enum.auto()
    MULTI_IMAGES = enum.auto()
    VIDEO = enum.auto()
    AUDIO = enum.auto()


@dataclasses.dataclass
class MultimodalDataItem:
    modality: Modality
    offsets: list | None = None
    precomputed_embeddings: object | None = None
    feature: object | None = None

    def set(self, name, value):
        setattr(self, name, value)


class MultimodalInputFormat(enum.Enum):
    PROCESSOR_OUTPUT = enum.auto()
    PRECOMPUTED_EMBEDDING = enum.auto()


def _build_stub_modules() -> dict:
    schedule_batch = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.Modality = Modality
    schedule_batch.MultimodalDataItem = MultimodalDataItem
    schedule_batch.MultimodalInputFormat = MultimodalInputFormat

    server_args = types.ModuleType("sglang.srt.server_args")

    class _Args:
        rl_on_policy_target = None

    server_args.get_global_server_args = lambda: _Args()

    utils = types.ModuleType("sglang.srt.utils")
    utils.envs = types.SimpleNamespace(
        SGLANG_USE_CUDA_IPC_TRANSPORT=types.SimpleNamespace(get=lambda: False)
    )
    utils.is_cpu = lambda: True
    utils.is_npu = lambda: False
    utils.is_xpu = lambda: False
    utils.load_audio = lambda data, sr=None: data
    utils.load_image = lambda data: (data, (0, 0))
    utils.load_video = lambda data, frame_count_limit=None: data
    utils.logger = types.SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )

    cuda_ipc = types.ModuleType("sglang.srt.utils.cuda_ipc_transport_utils")
    cuda_ipc.MM_FEATURE_CACHE_SIZE = 0
    cuda_ipc.MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL = 0

    class CudaIpcTensorTransportProxy:
        def __init__(self, data=None, info_data=None, sync_buffer_meta=None):
            self.data = data
            self.info_data = info_data
            self.sync_buffer_meta = sync_buffer_meta

    class MmItemMemoryPool:
        def __init__(self, *args, **kwargs):
            pass

        def return_a_slice_tensor_with_flag(self, _):
            return None, None

    cuda_ipc.CudaIpcTensorTransportProxy = CudaIpcTensorTransportProxy
    cuda_ipc.MmItemMemoryPool = MmItemMemoryPool

    return {
        "sglang.srt.managers.schedule_batch": schedule_batch,
        "sglang.srt.server_args": server_args,
        "sglang.srt.utils": utils,
        "sglang.srt.utils.cuda_ipc_transport_utils": cuda_ipc,
    }


with mock.patch.dict(sys.modules, _build_stub_modules()):
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultiModalProcessorOutput,
        BaseMultimodalProcessor,
        MultimodalSpecialTokens,
    )


class TestBaseMultiModalProcessorOutput(CustomTestCase):
    def test_organize_results_orders_modalities(self):
        out = BaseMultiModalProcessorOutput(
            input_text="x",
            images=[{"i": 1}],
            videos=[{"v": 2}],
            audios=[{"a": 3}],
        )
        organized = out.organize_results()
        self.assertEqual(
            [m for m, _ in organized], [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]
        )

    def test_organize_results_empty(self):
        out = BaseMultiModalProcessorOutput(input_text="x")
        self.assertEqual(out.organize_results(), [])

    def test_organize_results_images_only(self):
        out = BaseMultiModalProcessorOutput(input_text="x", images=[{"i": 1}, {"i": 2}])
        organized = out.organize_results()
        self.assertEqual([m for m, _ in organized], [Modality.IMAGE, Modality.IMAGE])


class TestMultimodalSpecialTokens(CustomTestCase):
    def test_get_modality_of_token_exact_match(self):
        tokens = MultimodalSpecialTokens(
            image_token="<image>",
            video_token="<video>",
            audio_token="<audio>",
        )
        self.assertEqual(tokens.get_modality_of_token("<image>"), Modality.IMAGE)
        self.assertEqual(tokens.get_modality_of_token("<video>"), Modality.VIDEO)
        self.assertEqual(tokens.get_modality_of_token("<audio>"), Modality.AUDIO)

    def test_get_modality_of_token_regex_match(self):
        tokens = MultimodalSpecialTokens(
            image_token="<image>",
            image_token_regex=re.compile(r"(?:<image>)+"),
        )
        self.assertEqual(tokens.get_modality_of_token("<image><image>"), Modality.IMAGE)

    def test_get_modality_of_token_no_match(self):
        tokens = MultimodalSpecialTokens(image_token="<image>")
        self.assertIsNone(tokens.get_modality_of_token("<not-image>"))

    def test_parse_regex_builds_regexes(self):
        tokens = MultimodalSpecialTokens(image_token="<image>", video_token="<video>")
        self.assertIsNone(tokens.image_token_regex)
        self.assertIsNone(tokens.video_token_regex)
        tokens.parse_regex()
        self.assertIsNotNone(tokens.image_token_regex)
        self.assertIsNotNone(tokens.video_token_regex)
        self.assertTrue(tokens.image_token_regex.match("<image>"))

    def test_get_token_id_by_modality(self):
        tokens = MultimodalSpecialTokens(
            image_token_id=11, video_token_id=22, audio_token_id=33
        )
        self.assertEqual(tokens.get_token_id_by_modality(Modality.IMAGE), 11)
        self.assertEqual(tokens.get_token_id_by_modality(Modality.MULTI_IMAGES), 11)
        self.assertEqual(tokens.get_token_id_by_modality(Modality.VIDEO), 22)
        self.assertEqual(tokens.get_token_id_by_modality(Modality.AUDIO), 33)

    def test_combined_regex_splits_prompt(self):
        tokens = MultimodalSpecialTokens(image_token="<image>", video_token="<video>")
        tokens.parse_regex()
        pat = tokens.get_combined_regex()
        parts = re.split(pat, "a<image>b<video>c")
        # The split keeps delimiters (capturing group), so tokens appear in list.
        self.assertIn("<image>", parts)
        self.assertIn("<video>", parts)

    def test_get_combined_regex_is_cached(self):
        tokens = MultimodalSpecialTokens(image_token="<image>")
        tokens.parse_regex()
        first = tokens.get_combined_regex()
        second = tokens.get_combined_regex()
        self.assertIs(first, second)


class TestValidateMmData(CustomTestCase):
    def test_validate_mm_data_all_none(self):
        BaseMultimodalProcessor.validate_mm_data(None, None, None)

    def test_validate_mm_data_type_error(self):
        with self.assertRaises(TypeError):
            BaseMultimodalProcessor.validate_mm_data(image_data={"x": 1})

    def test_validate_mm_data_precomputed_requires_single_item(self):
        bad = [{"format": "processor_output"}, "extra"]
        with self.assertRaises(ValueError):
            BaseMultimodalProcessor.validate_mm_data(image_data=bad)

    def test_validate_mm_data_allows_single_precomputed(self):
        good = [{"format": "precomputed_embedding", "feature": np.zeros((1,))}]
        BaseMultimodalProcessor.validate_mm_data(image_data=good)


class TestMmOffsets(CustomTestCase):
    def test_get_mm_items_offset_docstring_example(self):
        input_ids = torch.tensor([1, 2, 3, 3, 3, 4, 3, 3])
        offsets = BaseMultimodalProcessor.get_mm_items_offset(input_ids, mm_token_id=3)
        self.assertEqual(offsets, [(2, 4), (6, 7)])

    def test_get_mm_items_offset_none_present(self):
        input_ids = torch.tensor([1, 2, 4, 5])
        offsets = BaseMultimodalProcessor.get_mm_items_offset(input_ids, mm_token_id=3)
        self.assertEqual(offsets, [])

    def test_get_mm_items_offset_single_run_edges(self):
        input_ids = torch.tensor([0, 3, 3, 1, 2, 3, 0])
        offsets = BaseMultimodalProcessor.get_mm_items_offset(input_ids, mm_token_id=3)
        self.assertEqual(offsets, [(1, 2), (5, 5)])

    def test_get_mm_items_offset_by_pair_single_pair(self):
        input_ids = torch.tensor([9, 7, 8, 8, 6, 10])
        offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
            input_ids, mm_start_id=7, mm_end_id=6
        )
        self.assertEqual(offsets, [(2, 3)])

    def test_get_mm_items_offset_by_pair_no_pairs(self):
        input_ids = torch.tensor([1, 2, 3])
        offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
            input_ids, mm_start_id=7, mm_end_id=6
        )
        self.assertEqual(offsets, [])

    def test_get_mm_items_offset_by_pair_multiple_pairs(self):
        input_ids = torch.tensor([7, 1, 6, 2, 7, 3, 4, 6])
        offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
            input_ids, mm_start_id=7, mm_end_id=6
        )
        self.assertEqual(offsets, [(1, 1), (5, 6)])


class _MinimalProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(
        self, image_data, audio_data, input_text, request_obj, **kwargs
    ):
        raise NotImplementedError()


class TestProcessLoadedMmData(CustomTestCase):
    def _make_minimal(self):
        # Create an instance without calling BaseMultimodalProcessor.__init__ (heavy).
        obj = object.__new__(_MinimalProcessor)
        return obj

    def test_process_loaded_mm_data_image_raw(self):
        obj = self._make_minimal()
        is_precomputed, imgs, vids, auds = obj._process_loaded_mm_data(
            Modality.IMAGE, raw_data="x.png", result=object()
        )
        self.assertFalse(is_precomputed)
        self.assertEqual(len(imgs), 1)
        self.assertEqual(vids, [])
        self.assertEqual(auds, [])

    def test_process_loaded_mm_data_image_precomputed_dict(self):
        obj = self._make_minimal()
        raw = {"format": "precomputed_embedding"}
        is_precomputed, imgs, vids, auds = obj._process_loaded_mm_data(
            Modality.IMAGE, raw_data=raw, result={"feature": np.zeros((1,))}
        )
        self.assertTrue(is_precomputed)
        self.assertEqual(len(imgs), 1)
        self.assertEqual(vids, [])
        self.assertEqual(auds, [])

    def test_process_loaded_mm_data_audio_and_video(self):
        obj = self._make_minimal()
        is_precomputed_v, imgs_v, vids_v, auds_v = obj._process_loaded_mm_data(
            Modality.VIDEO, raw_data="v.mp4", result=torch.zeros((1, 2, 3))
        )
        self.assertFalse(is_precomputed_v)
        self.assertEqual(imgs_v, [])
        self.assertEqual(len(vids_v), 1)
        self.assertEqual(auds_v, [])

        is_precomputed_a, imgs_a, vids_a, auds_a = obj._process_loaded_mm_data(
            Modality.AUDIO, raw_data="a.wav", result=np.zeros((10,))
        )
        self.assertFalse(is_precomputed_a)
        self.assertEqual(imgs_a, [])
        self.assertEqual(vids_a, [])
        self.assertEqual(len(auds_a), 1)

    def test_process_loaded_mm_data_image_list_extends(self):
        obj = self._make_minimal()
        is_precomputed, imgs, vids, auds = obj._process_loaded_mm_data(
            Modality.IMAGE, raw_data="img", result=[object(), object()]
        )
        self.assertFalse(is_precomputed)
        self.assertEqual(len(imgs), 2)
        self.assertEqual(vids, [])
        self.assertEqual(auds, [])


if __name__ == "__main__":
    import unittest

    unittest.main()

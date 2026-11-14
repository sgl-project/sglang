"""Embedding template tokenization survives multimodal processing and caching."""

import asyncio
import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch

from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImagePreprocessArtifact,
    QwenVLImageProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _AutoSpecialTokenizer:
    """A tokenizer whose default inserts BOS even when the template omits it."""

    bos_token = "<s>"

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return ([1] if add_special_tokens else []) + [10, 11, 12, 99]

    def __call__(self, text, return_tensors, add_special_tokens=True):
        return SimpleNamespace(
            input_ids=torch.tensor([self.encode(text, add_special_tokens)])
        )


class _HFProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, text, padding, return_tensors, add_special_tokens=True, **kwargs
    ):
        return {
            "input_ids": torch.tensor(
                [self.tokenizer.encode(text[0], add_special_tokens)]
            )
        }


def _request(add_special_tokens=None):
    return EmbeddingReqInput(
        embedding_add_special_tokens=add_special_tokens,
        video_data=None,
        audio_data=None,
    )


def _processor(cls):
    processor = object.__new__(cls)
    processor._tokenizer = _AutoSpecialTokenizer()
    processor._processor = _HFProcessor(processor._tokenizer)
    processor._tokenizer_auto_adds_specials = True
    processor.mm_processor_executor = None
    processor.image_config = None
    processor.FEATURE_NAMES = ()
    processor.mm_feature_transport = "cpu"
    processor.precompute_hash_before_cpu_transfer = False
    processor.IM_TOKEN_ID = 11
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token="<img>",
        image_token_id=11,
        image_token_regex=re.compile(r"<img>|<image>"),
    )
    processor.collect_mm_items_from_processor_output = Mock(return_value=[])
    processor._finalize_mm_items = Mock(side_effect=lambda items, **kwargs: items)
    processor.hf_config = SimpleNamespace(
        model_type="qwen2_vl", vision_config=SimpleNamespace(spatial_merge_size=1)
    )
    processor.model_type = "qwen2_vl"
    processor.video_config = None
    processor.vision_start_token_id = 10
    processor.vision_end_token_id = 12
    processor._mark_cuda_ipc_features_for_deferred_reconstruction = Mock()
    processor._get_grid_from_output_or_items = Mock(return_value=None)
    processor._compute_image_only_mrope_positions_from_offsets = Mock(
        side_effect=lambda input_len, **kwargs: (
            torch.arange(input_len).expand(3, -1),
            torch.tensor([0]),
        )
    )
    return processor


def _loaded(items):
    return SimpleNamespace(
        input_text="<img> text",
        input_ids=None,
        images=[object()] if items else [],
        videos=None,
        organize_results=lambda: items,
    )


class TestEmbeddingSpecialTokens(unittest.TestCase):
    def test_base_fallback_tokenization_preserves_request_policy(self):
        # Cover both text-only and preprocessed media without supplied IDs.
        processor = _processor(PixtralProcessor)
        for items in ([], [(Modality.IMAGE, {"format": "processor_output"})]):
            for policy in (None, False, True, False, None):
                with self.subTest(items=items, policy=policy):
                    kwargs = {} if policy is None else {"add_special_tokens": policy}
                    _, ids, _ = processor.process_and_combine_mm_data(
                        _loaded(items), processor.mm_tokens, **kwargs
                    )
                    expected = [10, 11, 12, 99]
                    if policy is not False:
                        expected.insert(0, 1)
                    self.assertEqual(ids.tolist(), expected)

    def test_pixtral_request_policy_reaches_final_hf_tokenization(self):
        processor = _processor(PixtralProcessor)
        processor.load_mm_data = AsyncMock(
            return_value=_loaded([(Modality.IMAGE, object())])
        )
        for policy in (None, False, True, None):
            with self.subTest(policy=policy):
                output = asyncio.run(
                    processor.process_mm_data_async(
                        ["image"], "<img> text", _request(policy)
                    )
                )
                self.assertEqual(output.input_ids[0], 10 if policy is False else 1)
                self.assertEqual(processor._tokenizer.calls[-1][1], policy is not False)

    def test_qwen_uncached_request_policy_reaches_final_hf_tokenization(self):
        processor = _processor(QwenVLImageProcessor)
        processor.load_mm_data = AsyncMock(
            return_value=_loaded([(Modality.IMAGE, object())])
        )
        processor.mm_preprocess_cache = SimpleNamespace(enabled=False)
        processor.uses_media_artifacts_without_cache = False
        for policy in (None, False, True, None):
            with self.subTest(policy=policy):
                output = asyncio.run(
                    processor.process_mm_data_async(
                        ["image"], "<img> text", _request(policy)
                    )
                )
                self.assertEqual(output.input_ids[0], 10 if policy is False else 1)
                self.assertEqual(processor._tokenizer.calls[-1][1], policy is not False)

    def test_qwen_artifact_cache_separates_prompt_tokenization_policies(self):
        processor = _processor(QwenVLImageProcessor)
        processor.mm_preprocess_cache = SimpleNamespace(enabled=True)
        artifact = QwenVLImagePreprocessArtifact(
            content_digest="image-content",
            artifact_key="image-artifact",
            feature_hash=17,
            feature=torch.ones(1, 2),
            model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 1]])},
        )
        processor.prepare_media_artifacts = AsyncMock(return_value=[artifact])
        processor._prepare_mm_items_for_transport = Mock(
            side_effect=lambda items: items
        )
        processor._process_mm_data_uncached = AsyncMock(
            side_effect=AssertionError("image artifact should remain reusable")
        )
        processor._cached_image_prompt_template.cache_clear()

        async def run_requests():
            return await asyncio.gather(
                *(
                    processor.process_mm_data_async(
                        ["image"], "<image> text", _request(policy)
                    )
                    for policy in (False, None, True, False, None)
                )
            )

        outputs = asyncio.run(run_requests())
        self.assertEqual([output.input_ids[0] for output in outputs], [10, 1, 1, 10, 1])
        self.assertEqual(
            [output.mm_items[0].offsets for output in outputs],
            [[(1, 1)], [(2, 2)], [(2, 2)], [(1, 1)], [(2, 2)]],
        )
        # Each policy is tokenized once; repeats reuse the correct cached result.
        self.assertEqual(len(processor._tokenizer.calls), 3)
        self.assertTrue(
            all(text == "<img> text" for text, _ in processor._tokenizer.calls)
        )
        self.assertEqual(processor._cached_image_prompt_template.cache_info().hits, 2)
        processor._cached_image_prompt_template.cache_clear()


if __name__ == "__main__":
    unittest.main()

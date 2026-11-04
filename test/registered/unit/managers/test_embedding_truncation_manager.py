import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.req_types import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEmbeddingTruncationManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            chunked_prefill_size=-1, max_prefill_tokens=100
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 16
        self.manager.num_reserved_tokens = 1
        self.manager.max_req_input_len = 14
        self.manager.allow_auto_truncate = True
        self.manager.validate_total_tokens = False
        self.manager.is_generation = False
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None,
            pointer_head=None,
            hf_config=SimpleNamespace(architectures=[]),
            is_embedding_gemma=False,
        )
        self.manager.mm_processor = None
        self.manager.tokenizer = Mock(is_fast=True)
        self.manager.async_dynamic_batch_tokenizer = None
        self.manager._create_tokenized_object = Mock(
            side_effect=lambda obj, text, ids, embeds, mm, types: SimpleNamespace(
                input_ids=ids, mm_inputs=mm, token_type_ids=types
            )
        )

    @staticmethod
    def request(**kwargs):
        obj = EmbeddingReqInput(**kwargs)
        obj.normalize_batch_and_arguments()
        return obj

    async def test_none_cannot_be_overridden_by_global_auto_truncation(self):
        ids = list(range(18))
        req = self.request(input_ids=ids, embedding_truncate="NONE")
        with self.assertRaisesRegex(ValueError, "budget of 13.*truncate=NONE"):
            await self.manager._tokenize_one_request(req)
        self.assertEqual(ids, list(range(18)))
        self.manager._create_tokenized_object.assert_not_called()

    async def test_embedding_uses_scheduler_budget_at_exact_boundary(self):
        for policy in ("END", "START", "NONE"):
            req = self.request(input_ids=list(range(13)), embedding_truncate=policy)
            result = await self.manager._tokenize_one_request(req)
            self.assertEqual(result.input_ids, list(range(13)))
        for policy, expected in (
            ("END", list(range(13))),
            ("START", list(range(1, 14))),
        ):
            req = self.request(input_ids=list(range(14)), embedding_truncate=policy)
            result = await self.manager._tokenize_one_request(req)
            self.assertEqual(result.input_ids, expected)

    async def test_minus_one_and_zero_follow_request_policy(self):
        for policy, expected in (
            ("END", list(range(13))),
            ("START", list(range(1, 14))),
        ):
            req = self.request(
                input_ids=list(range(14)),
                embedding_truncate=policy,
                embedding_max_tokens=-1,
            )
            result = await self.manager._tokenize_one_request(req)
            self.assertEqual(result.input_ids, expected)
        for policy, message in (
            ("END", "input is empty"),
            ("START", "input is empty"),
            ("NONE", "budget of 0.*truncate=NONE"),
        ):
            req = self.request(
                input_ids=[1, 2, 3], embedding_truncate=policy, embedding_max_tokens=0
            )
            self.manager._create_tokenized_object.reset_mock()
            with self.assertRaisesRegex(ValueError, message):
                await self.manager._tokenize_one_request(req)
            self.manager._create_tokenized_object.assert_not_called()
        req = self.request(
            input_ids=[1], embedding_truncate="NONE", embedding_max_tokens=-1
        )
        with self.assertRaisesRegex(ValueError, "budget of -1.*truncate=NONE"):
            await self.manager._tokenize_one_request(req)

    async def test_none_above_server_budget_only_restricts_actual_input(self):
        req = self.request(
            input_ids=[1, 2], embedding_truncate="NONE", embedding_max_tokens=100
        )
        self.assertEqual(
            (await self.manager._tokenize_one_request(req)).input_ids, [1, 2]
        )
        req = self.request(
            input_ids=list(range(14)),
            embedding_truncate="NONE",
            embedding_max_tokens=100,
        )
        with self.assertRaisesRegex(ValueError, "budget of 13.*truncate=NONE"):
            await self.manager._tokenize_one_request(req)

    def test_empty_input_rejection_only_applies_to_scoped_embedding_requests(self):
        for policy in ("END", "START", "NONE"):
            req = EmbeddingReqInput(text="", embedding_truncate=policy)
            with self.assertRaisesRegex(ValueError, "input is empty"):
                self.manager._truncate_embedding_input(req, [])
        for req in (EmbeddingReqInput(text=""), GenerateReqInput(text="")):
            self.assertEqual(self.manager._truncate_embedding_input(req, [])[0], [])

    async def test_zero_budget_is_rejected_in_batch_before_inference(self):
        req = self.request(
            text=["a", "b"], embedding_truncate="START", embedding_max_tokens=0
        )
        self.manager._tokenize_texts = AsyncMock(return_value=([[1, 2], [3, 4]], None))
        with self.assertRaisesRegex(ValueError, "input is empty"):
            await self.manager._batch_tokenize_and_process(2, req)
        self.manager._create_tokenized_object.assert_not_called()

    def test_existing_requests_do_not_opt_into_new_policy(self):
        ids = list(range(18))
        for req in (EmbeddingReqInput(input_ids=ids), GenerateReqInput(input_ids=ids)):
            actual, _, _ = self.manager._truncate_embedding_input(req, ids)
            self.assertIs(actual, ids)
            self.assertEqual(actual, list(range(18)))

    def test_policy_propagates_to_each_batch_item(self):
        for cross_encoder in (False, True):
            req = self.request(
                text=[["q", "a"], ["q", "b"]] if cross_encoder else ["a", "b"],
                is_cross_encoder_request=cross_encoder,
                embedding_truncate="START",
                embedding_max_tokens=4,
                embedding_add_special_tokens=False,
            )
            for i in range(2):
                self.assertEqual(req[i].embedding_truncate, "START")
                self.assertEqual(req[i].embedding_max_tokens, 4)
                self.assertFalse(req[i].embedding_add_special_tokens)

    async def test_batch_tokenization_truncates_each_sample_with_aligned_types(self):
        req = self.request(
            text=["a", "b"], embedding_truncate="START", embedding_max_tokens=3
        )
        self.manager._tokenize_texts = AsyncMock(
            return_value=(
                [[0, 1, 2, 3], [4, 5, 6, 7, 8]],
                [[0, 0, 1, 1], [1, 1, 1, 0, 0]],
            )
        )
        result = await self.manager._batch_tokenize_and_process(2, req)
        self.assertEqual([item.input_ids for item in result], [[1, 2, 3], [6, 7, 8]])
        self.assertEqual(
            [item.token_type_ids for item in result], [[0, 1, 1], [1, 0, 0]]
        )

    async def test_mm_processor_expansion_precedes_budget_and_metadata_is_aligned(self):
        original = MultimodalProcessorOutput(
            input_ids=list(range(12)),
            padded_input_ids=list(range(100, 112)),
            token_type_ids=torch.tensor([[0] * 12]),
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, offsets=[(1, 3)])],
        )
        processor = SimpleNamespace(
            prefer_tokenized_input=True,
            process_mm_data_async=AsyncMock(return_value=original),
            finalize_cache_identities=Mock(),
        )
        self.manager.mm_processor = processor
        req = self.request(
            input_ids=[1, 2],
            image_data=["image"],
            embedding_truncate="END",
            embedding_max_tokens=8,
        )
        result = await self.manager._tokenize_one_request(req)
        self.assertEqual(result.input_ids, list(range(8)))
        self.assertEqual(result.token_type_ids, [0] * 8)
        self.assertEqual(result.mm_inputs.padded_input_ids, list(range(100, 108)))
        self.assertEqual(result.mm_inputs.token_type_ids.shape, (1, 8))
        self.assertEqual(original.input_ids, list(range(12)))
        self.assertEqual(original.token_type_ids.shape, (1, 12))
        processor.process_mm_data_async.assert_awaited_once()
        processor.finalize_cache_identities.assert_called_once_with(original)

    async def test_mm_batch_cannot_skip_processor_when_batch_encoding_enabled(self):
        req = self.request(
            text=["with image", "text only"],
            image_data=[["image"], []],
            embedding_truncate="END",
        )
        self.manager._tokenize_one_request = AsyncMock(side_effect=["first", "second"])
        self.manager._tokenize_texts = AsyncMock()
        self.assertEqual(
            await self.manager._batch_tokenize_and_process(2, req), ["first", "second"]
        )
        self.assertEqual(self.manager._tokenize_one_request.await_count, 2)
        self.manager._tokenize_texts.assert_not_called()

    async def test_rendered_template_special_token_flag_reaches_single_and_batch(self):
        for texts in ("rendered", ["rendered 1", "rendered 2"]):
            req = self.request(
                text=texts, embedding_truncate="END", embedding_add_special_tokens=False
            )
            self.manager._tokenize_texts = AsyncMock(
                return_value=([1, 2], None)
                if isinstance(texts, str)
                else ([[1, 2], [3, 4]], None)
            )
            if isinstance(texts, str):
                await self.manager._tokenize_one_request(req)
            else:
                await self.manager._batch_tokenize_and_process(2, req)
            self.manager._tokenize_texts.assert_awaited_once_with(
                texts, False, add_special_tokens=False
            )

    async def test_special_tokens_are_request_local_for_fast_slow_and_async_tokenizers(
        self,
    ):
        for mode in ("fast", "slow", "async"):
            with self.subTest(mode=mode):
                tokenizer = Mock(is_fast=mode != "slow", eos_token_id=99)
                tokenizer.return_value = {"input_ids": [[1, 2]]}
                tokenizer.encode.return_value = [1, 2]
                self.manager.tokenizer = tokenizer
                async_tokenizer = SimpleNamespace(
                    encode=AsyncMock(return_value={"input_ids": [1, 2]})
                )
                self.manager.async_dynamic_batch_tokenizer = (
                    async_tokenizer if mode == "async" else None
                )
                await self.manager._tokenize_texts("template", add_special_tokens=False)
                method = (
                    async_tokenizer.encode
                    if mode == "async"
                    else tokenizer.encode
                    if mode == "slow"
                    else tokenizer
                )
                self.assertEqual(method.call_args.kwargs, {"add_special_tokens": False})
                await self.manager._tokenize_texts("raw text")
                self.assertEqual(method.call_args.kwargs, {})

    async def test_template_does_not_get_extra_embedding_gemma_eos(self):
        self.manager.model_config.is_embedding_gemma = True
        self.manager.tokenizer.eos_token_id = 99
        self.manager.tokenizer.return_value = {"input_ids": [[1, 2]]}
        self.assertEqual((await self.manager._tokenize_texts("raw"))[0], [1, 2, 99])
        self.assertEqual(
            (await self.manager._tokenize_texts("rendered", add_special_tokens=False))[
                0
            ],
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()

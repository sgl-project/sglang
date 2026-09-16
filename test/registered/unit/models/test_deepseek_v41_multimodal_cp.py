"""V4.1 image/text CP input contracts; vision and model compute are mocked."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import MM_PAD_SHIFT_VALUE, DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_cp_test_utils import cp_context, simulated_collective
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


MODEL = "sglang.srt.models.deepseek_v4"
RUNNER = "sglang.srt.model_executor.runner.eager_runner"
IMAGE_ID = 129264


class TestDSV41MultimodalCP(CustomTestCase):
    def test_image_spans_cross_ranks_before_shard_and_gather(self):
        # Two image spans with distinct cache hashes; first request is text-only.
        original = torch.tensor(
            [
                7,
                8,
                9,
                MM_PAD_SHIFT_VALUE + 11,
                MM_PAD_SHIFT_VALUE + 11,
                10,
                MM_PAD_SHIFT_VALUE + 23,
                MM_PAD_SHIFT_VALUE + 23,
                12,
            ]
        )
        normalized = torch.tensor(
            [7, 8, 9, IMAGE_ID, IMAGE_ID, 10, IMAGE_ID, IMAGE_ID, 12]
        )
        full = torch.arange(36, dtype=torch.float32).reshape(9, 4)
        # Distinct image features expose using text embeddings or wrong row order.
        full[3:5] += 1000
        full[6:8] += 2000
        for size in (2, 4):
            for rank in range(size):
                with (
                    self.subTest(size=size, rank=rank),
                    cp_context(size, rank) as (strategy, batch),
                ):
                    batch.input_ids = original.clone()
                    batch.mm_inputs = [None, MultimodalInputs(mm_items=[])]
                    model = NS(
                        vision=object(),
                        config=NS(image_token_id=IMAGE_ID),
                        get_input_embeddings=Mock(
                            side_effect=AssertionError(
                                "Raw image hashes entered text embeddings"
                            )
                        ),
                        _prepare_mm_embeddings=Mock(return_value=full),
                        capture_aux_hidden_states=False,
                        pp_group=NS(is_last_rank=True),
                        lm_head=object(),
                        logits_processor=Mock(return_value="ok"),
                    )
                    model.prepare_language_model_inputs = lambda ids, fb, emb: (
                        DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, fb, emb
                        )
                    )

                    def body(ids, positions, fb, input_embeds):
                        model._prepare_mm_embeddings.assert_called_once_with(
                            batch.input_ids, batch
                        )
                        n = len(normalized[rank::size])
                        torch.testing.assert_close(ids[:n], normalized[rank::size])
                        torch.testing.assert_close(input_embeds[:n], full[rank::size])
                        torch.testing.assert_close(
                            positions[:n], batch.positions[rank::size]
                        )
                        self.assertFalse(
                            (fb.input_ids_global >= MM_PAD_SHIFT_VALUE).any().item()
                        )
                        return input_embeds

                    model.model = body
                    with (
                        simulated_collective(strategy, batch, full),
                        patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
                    ):
                        result = EagerRunner._execute_extend_cp(
                            NS(model_runner=NS(model=model)), batch, {}
                        )
                    self.assertEqual(result, "ok")
                    torch.testing.assert_close(
                        model.logits_processor.call_args.args[0], normalized
                    )
                    torch.testing.assert_close(
                        model.logits_processor.call_args.args[1], full
                    )
                    torch.testing.assert_close(batch.input_ids, original)
                    self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_chunk_prefix_metadata_and_scheduler_hashes_survive_embedder(self):
        for prefixes, lengths in (([0, 0], [3, 6]), ([16384, 127], [3, 6])):
            with self.subTest(prefixes=prefixes):
                ids = torch.tensor([7, 8, 9] + [MM_PAD_SHIFT_VALUE + 17] * 6)
                original = ids.clone()
                image = MultimodalInputs(mm_items=[])
                batch = NS(
                    mm_inputs=[None, image],
                    extend_prefix_lens_cpu=prefixes,
                    extend_seq_lens_cpu=lengths,
                )
                full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
                embedding = object()
                model = NS(get_input_embeddings=lambda: embedding)

                def embed(**kwargs):
                    self.assertEqual(kwargs["extend_prefix_lens"], prefixes)
                    self.assertEqual(kwargs["extend_seq_lens"], lengths)
                    self.assertIs(kwargs["mm_inputs_list"][1], image)
                    self.assertEqual(kwargs["mm_inputs_list"][0].mm_items, [])
                    self.assertIs(kwargs["input_embedding"], embedding)
                    self.assertNotEqual(kwargs["input_ids"].data_ptr(), ids.data_ptr())
                    kwargs["input_ids"].zero_()
                    return full, {}

                with patch(MODEL + ".embed_mm_inputs", side_effect=embed) as mocked:
                    result = DeepseekV4ForCausalLM._prepare_mm_embeddings(
                        model, ids, batch
                    )
                mocked.assert_called_once()
                self.assertIs(result, full)
                self.assertIs(batch.mm_input_embeds, full)
                torch.testing.assert_close(ids, original)

    def test_vision_enabled_text_batch_skips_image_encoder(self):
        for mm_inputs in (None, [None, None], []):
            with self.subTest(mm_inputs=mm_inputs):
                ids = torch.tensor([4, 5, 6])
                model = NS(
                    vision=object(),
                    config=NS(image_token_id=IMAGE_ID),
                    _prepare_mm_embeddings=Mock(),
                )
                batch = NS(forward_mode=ForwardMode.EXTEND, mm_inputs=mm_inputs)
                result, embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
                    model, ids, batch
                )
                torch.testing.assert_close(result, ids)
                self.assertIsNone(embeds)
                model._prepare_mm_embeddings.assert_not_called()

    def test_decode_idle_and_verify_preserve_vocab_ids(self):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                ids = torch.tensor([4, IMAGE_ID, 6])
                model = NS(
                    vision=object(),
                    config=NS(image_token_id=IMAGE_ID),
                    _prepare_mm_embeddings=Mock(),
                )
                batch = NS(forward_mode=mode, mm_inputs=None)
                result, embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
                    model, ids, batch
                )
                self.assertIs(result, ids)
                self.assertIsNone(embeds)
                model._prepare_mm_embeddings.assert_not_called()

    def test_image_embedding_failure_does_not_mutate_scheduler_ids(self):
        ids = torch.tensor([7, MM_PAD_SHIFT_VALUE + 12, 8])
        original = ids.clone()
        batch = NS(
            mm_inputs=[MultimodalInputs(mm_items=[])],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[3],
        )
        model = NS(get_input_embeddings=lambda: object())

        def fail(**kwargs):
            kwargs["input_ids"].zero_()
            raise RuntimeError("vision failure")

        with patch(MODEL + ".embed_mm_inputs", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "vision failure"):
                DeepseekV4ForCausalLM._prepare_mm_embeddings(model, ids, batch)
        torch.testing.assert_close(ids, original)
        self.assertFalse(hasattr(batch, "mm_input_embeds"))


class TestDSV41MultimodalInputs(CustomTestCase):
    def setUp(self):
        self.ids = torch.tensor(
            [7, MM_PAD_SHIFT_VALUE + 12, MM_PAD_SHIFT_VALUE + 12, 9, 10]
        )
        self.original = self.ids.clone()
        self.embeds = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        self.model = NS(
            vision=object(),
            config=NS(image_token_id=129264),
            _prepare_mm_embeddings=Mock(return_value=self.embeds),
        )
        self.batch = NS(
            input_ids=self.ids,
            forward_mode=ForwardMode.EXTEND,
            mm_inputs=[MultimodalInputs(mm_items=[])],
        )

    def test_prepare_global_embeddings_and_normalized_ids(self):
        ids, embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            self.model, self.ids, self.batch
        )
        self.assertEqual(ids.tolist(), [7, 129264, 129264, 9, 10])
        self.assertIs(embeds, self.embeds)
        self.model._prepare_mm_embeddings.assert_called_once_with(self.ids, self.batch)
        self.assertTrue(torch.equal(self.ids, self.original))

    def test_reject_preembedded_images(self):
        with self.assertRaisesRegex(ValueError, "Cannot combine"):
            DeepseekV4ForCausalLM.prepare_language_model_inputs(
                self.model, self.ids, self.batch, self.embeds
            )

    def test_text_only_model_keeps_existing_embeddings(self):
        self.model.vision = None
        ids, embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            self.model, self.ids, self.batch, self.embeds
        )
        self.assertIs(ids, self.ids)
        self.assertIs(embeds, self.embeds)
        self.model._prepare_mm_embeddings.assert_not_called()

    def test_text_subclass_without_vision_module(self):
        del self.model.vision
        ids, embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            self.model, self.ids, self.batch, self.embeds
        )
        self.assertIs(ids, self.ids)
        self.assertIs(embeds, self.embeds)

    def test_decode_keeps_vocabulary_ids(self):
        self.batch.forward_mode = ForwardMode.DECODE
        ids, embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            self.model, self.ids, self.batch
        )
        self.assertIs(ids, self.ids)
        self.assertIsNone(embeds)
        self.model._prepare_mm_embeddings.assert_not_called()

    def test_embedding_does_not_mutate_scheduler_hashes(self):
        self.batch.extend_prefix_lens_cpu = [0]
        self.batch.extend_seq_lens_cpu = [5]
        self.model.get_input_embeddings = lambda: None

        def embed(**kwargs):
            kwargs["input_ids"].zero_()
            return (self.embeds, {})

        with patch("sglang.srt.models.deepseek_v4.embed_mm_inputs", side_effect=embed):
            result = DeepseekV4ForCausalLM._prepare_mm_embeddings(
                self.model, self.ids, self.batch
            )
        self.assertTrue(torch.equal(self.ids, self.original))
        self.assertIs(result, self.batch.mm_input_embeds)

    def test_cp_runner_prepares_before_sharding_and_uses_model_ids_for_logits(self):
        normalized = torch.tensor([7, 129264, 129264, 9, 10])
        self.batch.positions = torch.arange(5)
        self.model.prepare_language_model_inputs = lambda ids, batch, emb: (
            DeepseekV4ForCausalLM.prepare_language_model_inputs(
                self.model, ids, batch, emb
            )
        )
        self.model.get_input_embeddings = Mock(
            side_effect=AssertionError("Raw hashes must not enter text embedding")
        )
        self.model.model = Mock(return_value=self.embeds[1::4])
        self.model.capture_aux_hidden_states = False
        self.model.pp_group = NS(is_last_rank=True)
        self.model.lm_head = object()
        self.model.logits_processor = Mock(return_value="ok")

        @contextmanager
        def shard(embeds, positions, batch, ids):
            self.assertIs(embeds, self.embeds)
            self.assertTrue(torch.equal(ids, normalized))
            yield (embeds[1::4], positions[1::4], ids[1::4])

        runner = NS(model_runner=NS(model=self.model))
        module = "sglang.srt.model_executor.runner.eager_runner"
        with (
            patch(module + ".cp_shard_model_inputs", side_effect=shard),
            patch(module + ".cp_gather_after_forward", return_value=self.embeds),
            patch(module + ".torch.cuda.current_stream", return_value=None),
        ):
            result = EagerRunner._execute_extend_cp(runner, self.batch, {})
        self.assertEqual(result, "ok")
        args, kwargs = self.model.model.call_args
        self.assertEqual(args[0].tolist(), [129264])
        self.assertTrue(torch.equal(kwargs["input_embeds"], self.embeds[1::4]))
        self.assertTrue(
            torch.equal(self.model.logits_processor.call_args.args[0], normalized)
        )
        self.assertTrue(torch.equal(self.batch.input_ids, self.original))


if __name__ == "__main__":
    unittest.main()

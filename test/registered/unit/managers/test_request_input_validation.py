"""CPU regressions for metadata validation before native request fan-out."""

import unittest

import msgspec
import torch

from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    SessionParams,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSessionParamsValidation(CustomTestCase):
    def test_valid_dictionary_and_explicit_nulls(self):
        values = dict(
            id="session",
            rid="request",
            offset=0,
            replace=False,
            drop_previous_output=True,
        )
        for params in ({}, values, {key: None for key in values}):
            with self.subTest(params=params):
                session = SessionParams(**params)
                self.assertEqual(
                    msgspec.msgpack.decode(
                        msgspec.msgpack.encode(session), type=SessionParams
                    ),
                    session,
                )
                req = GenerateReqInput(text="hello", session_params=params)
                req.validate_input_types()
                self.assertIs(req.session_params, params)

    def test_invalid_nested_field_types(self):
        for name, values in {
            "id": (1, [], {}),
            "rid": (1, [], {}),
            "offset": (True, 0.5, "0", []),
            "replace": (1, "true", []),
            "drop_previous_output": (0, "false", {}),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"session_params.{name}"):
                        SessionParams(**{name: value})
                    req = GenerateReqInput(text="hello", session_params={name: value})
                    with self.assertRaisesRegex(ValueError, f"session_params.{name}"):
                        req.normalize_batch_and_arguments()
                    self.assertNotIn("is_single", req.__dict__)

    def test_invalid_container_and_unknown_fields(self):
        for value in ([], [{"id": "session"}], "session", 1, {"unknown": True}):
            with self.subTest(value=value):
                req = GenerateReqInput(text="hello", session_params=value)
                with self.assertRaisesRegex(ValueError, "session_params"):
                    req.validate_input_types()

    def test_empty_token_session_continuation(self):
        req = GenerateReqInput(input_ids=[], session_params={"id": "session"})
        req.validate_input_types()
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.input_ids, [])


class TestNativeRequestShapeValidation(CustomTestCase):
    def test_raw_positional_values_rejected_before_mutation(self):
        values = (0, "bad", {}, {"embeds": [[1.0]], "positions": [0]}, [], [None])
        for cls in (GenerateReqInput, EmbeddingReqInput):
            for value in values:
                with self.subTest(request=cls.__name__, value=value):
                    req = cls(text="hello", positional_embed_overrides=value)
                    with self.assertRaisesRegex(
                        ValueError, "positional_embed_overrides"
                    ):
                        req.validate_input_types()
                    with self.assertRaisesRegex(
                        ValueError, "positional_embed_overrides"
                    ):
                        req.normalize_batch_and_arguments()
                    self.assertIsNone(req.rid)
                    self.assertNotIn("is_single", req.__dict__)

    def test_scalar_lora_id_rejects_list_for_each_prompt_form(self):
        for cls in (GenerateReqInput, EmbeddingReqInput):
            prompts = ({"text": "hello"}, {"input_ids": [1, 2]})
            for prompt in prompts:
                for lora_id in ([], [None], ["adapter"], ["one", "two"]):
                    with self.subTest(
                        request=cls.__name__, prompt=prompt, lora_id=lora_id
                    ):
                        req = cls(**prompt, lora_id=lora_id)
                        with self.assertRaisesRegex(ValueError, "lora_id"):
                            req.normalize_batch_and_arguments()
        req = GenerateReqInput(input_embeds=[[0.0, 1.0]], lora_id=["adapter"])
        with self.assertRaisesRegex(ValueError, "lora_id"):
            req.validate_input_types()

    def test_invalid_batch_metadata_rejected(self):
        for cls in (GenerateReqInput, EmbeddingReqInput):
            cases = (
                {"lora_id": ["only-one"]},
                {"lora_id": ["valid", []]},
                {"lora_path": ["only-one"]},
                {"lora_path": ["valid", {}]},
                {"positional_embed_overrides": [None]},
                {"positional_embed_overrides": [None, {}]},
                {"sampling_params": []},
                {"sampling_params": [{}]},
                {"sampling_params": [{}, None]},
            )
            for kwargs in cases:
                with self.subTest(request=cls.__name__, kwargs=kwargs):
                    req = cls(text=["one", "two"], **kwargs)
                    with self.assertRaises(ValueError):
                        req.normalize_batch_and_arguments()
                    self.assertEqual(req.text, ["one", "two"])
                    self.assertIsNone(req.rid)

    def test_sampling_validation_precedes_parallel_expansion(self):
        for params in ({"n": 2.5}, {"n": True}, {"beam_width": "2"}, {"regex": []}):
            with self.subTest(params=params):
                req = GenerateReqInput(text="hello", sampling_params=params)
                with self.assertRaises(ValueError):
                    req.normalize_batch_and_arguments()
                self.assertEqual(req.text, "hello")
                self.assertNotIn("parallel_sample_num", req.__dict__)

    def test_null_parallel_sample_count_preserves_default(self):
        for text, params in (
            ("hello", {"n": None}),
            (["one", "two"], [{"n": None}, {"n": 1}]),
        ):
            with self.subTest(text=text):
                req = GenerateReqInput(text=text, sampling_params=params)
                req.normalize_batch_and_arguments()
                self.assertEqual(req.parallel_sample_num, 1)

    def test_lora_path_singleton_list_normalizes_to_scalar(self):
        for cls in (GenerateReqInput, EmbeddingReqInput):
            for value, expected in ((["adapter"], "adapter"), ([None], None)):
                with self.subTest(request=cls.__name__, value=value):
                    req = cls(text="hello", lora_path=value)
                    req.validate_input_types()
                    self.assertIs(req.lora_path, value)
                    req.normalize_batch_and_arguments()
                    self.assertEqual(req.lora_path, expected)

    def test_batch_scalar_lora_ids_are_broadcast(self):
        for cls in (GenerateReqInput, EmbeddingReqInput):
            req = cls(text=["one", "two"], lora_id="adapter")
            req.normalize_batch_and_arguments()
            self.assertEqual([req[i].lora_id for i in range(2)], ["adapter", "adapter"])

    def test_parallel_sampling_preserves_each_metadata_item(self):
        embeds = PositionalEmbeds(torch.zeros(1, 4), [0])
        positional = [embeds, None]
        lora_ids = ["adapter", None]
        req = GenerateReqInput(
            text=["one", "two"],
            lora_id=lora_ids,
            lora_path=["path", None],
            positional_embed_overrides=positional,
            sampling_params={"n": 3},
        )
        req.validate_input_types()
        self.assertNotIn("batch_size", req.__dict__)
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_id, lora_ids * 3)
        self.assertEqual(req.lora_path, ["path", None] * 3)
        self.assertEqual(len(positional), 2)
        for i in range(6):
            item = req[i]
            self.assertEqual(item.lora_id, lora_ids[i % 2])
            self.assertIs(item.positional_embed_overrides, positional[i % 2])

    def test_scalar_parallel_sampling_broadcasts_metadata(self):
        embeds = PositionalEmbeds(torch.zeros(1, 4), [0])
        req = GenerateReqInput(
            text="hello",
            sampling_params={"n": 2},
            lora_id="adapter",
            lora_path=["path"],
            positional_embed_overrides=embeds,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertEqual(req[i].lora_id, "adapter")
            self.assertEqual(req[i].lora_path, "path")
            self.assertIs(req[i].positional_embed_overrides, embeds)

    def test_batch_of_one_accepts_per_item_metadata(self):
        for cls in (GenerateReqInput, EmbeddingReqInput):
            req = cls(
                text=["hello"], lora_id=["adapter"], positional_embed_overrides=[None]
            )
            req.normalize_batch_and_arguments()
            self.assertEqual(req[0].lora_id, "adapter")
            self.assertIsNone(req[0].positional_embed_overrides)


class TestPositionalEmbedsValidation(CustomTestCase):
    def test_invalid_tensors_fail_as_value_errors(self):
        for embeds in (
            [],
            [1],
            [[1.0, 2.0]],
            {},
            "bad",
            torch.zeros(4),
            torch.zeros(1, 1, 4),
            [torch.zeros(2), torch.zeros(3)],
        ):
            with self.subTest(embeds=embeds):
                with self.assertRaisesRegex(ValueError, "positional_embed_overrides"):
                    PositionalEmbeds(embeds, [0])

    def test_invalid_position_types(self):
        for positions in (None, "0", [0.5], [False], ["0"]):
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(ValueError, "positions"):
                    PositionalEmbeds(torch.zeros(1, 4), positions)

    def test_normalized_tensor_and_hidden_dimension(self):
        for embeds in (
            torch.zeros(2, 4),
            [torch.zeros(4), torch.zeros(4)],
            [torch.zeros(1, 4), torch.zeros(1, 4)],
        ):
            pe = PositionalEmbeds(embeds, [0, 1])
            pe.validate_hidden_dim(4)
            self.assertEqual(tuple(pe.embeds.shape), (2, 4))
            with self.assertRaisesRegex(ValueError, "hidden_dim.*hidden_size"):
                pe.validate_hidden_dim(8)
        PositionalEmbeds(torch.zeros(0, 4), []).validate_hidden_dim(4)

    def test_mutated_struct_revalidated_at_admission(self):
        pe = PositionalEmbeds(torch.zeros(1, 4), [0])
        pe.positions = ["invalid"]
        req = GenerateReqInput(text="hello", positional_embed_overrides=pe)
        with self.assertRaisesRegex(ValueError, "positions"):
            req.validate_input_types()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from datasets import Dataset

from sglang.test import simple_eval_mmmu_vlm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_subject(subject: str, split: str) -> Dataset:
    return Dataset.from_dict({"id": [f"validation_{subject}_1"], "question": ["q"]})


class TestLoadSubjectDatasets(CustomTestCase):
    def test_every_subject_loaded_and_tagged(self):
        with mock.patch.object(
            simple_eval_mmmu_vlm,
            "load_dataset",
            side_effect=lambda name, subject, split: _fake_subject(subject, split),
        ):
            datasets = MMMUVLMEval._load_subject_datasets(["Art", "Math"])

        self.assertEqual([d["__subject__"] for d in datasets], [["Art"], ["Math"]])

    def test_a_subject_that_fails_to_load_raises_and_is_named(self):
        def load(name, subject, split):
            if subject == "Math":
                raise ConnectionError("hub unreachable")
            return _fake_subject(subject, split)

        with mock.patch.object(simple_eval_mmmu_vlm, "load_dataset", side_effect=load):
            with self.assertRaises(RuntimeError) as ctx:
                MMMUVLMEval._load_subject_datasets(["Art", "Math", "Physics"])

        message = str(ctx.exception)
        self.assertIn("1 of 3", message)
        self.assertIn("Math (ConnectionError: hub unreachable)", message)
        self.assertNotIn("Art", message)

    def test_each_failure_keeps_only_the_first_line_of_a_long_error(self):
        def load(name, subject, split):
            raise OSError("first line\n" + "traceback detail\n" * 50)

        with mock.patch.object(simple_eval_mmmu_vlm, "load_dataset", side_effect=load):
            with self.assertRaises(RuntimeError) as ctx:
                MMMUVLMEval._load_subject_datasets(["Art", "Math"])

        message = str(ctx.exception)
        self.assertIn("Art (OSError: first line)", message)
        self.assertIn("Math (OSError: first line)", message)
        self.assertNotIn("traceback detail", message)


class TestBriefError(CustomTestCase):
    def test_long_first_line_is_cut_to_the_limit(self):
        brief = simple_eval_mmmu_vlm._brief_error(ValueError("x" * 500), max_chars=50)
        self.assertEqual(brief, "ValueError: " + "x" * 47 + "...")

    def test_empty_message_keeps_the_type(self):
        self.assertEqual(simple_eval_mmmu_vlm._brief_error(KeyError()), "KeyError: ")


class TestParseOptions(CustomTestCase):
    def test_list_literal_string(self):
        self.assertEqual(
            MMMUVLMEval._parse_options("['1/2', \"it's 3\"]"), ["1/2", "it's 3"]
        )

    def test_list_is_returned_as_is(self):
        self.assertEqual(MMMUVLMEval._parse_options(["a", "b"]), ["a", "b"])

    def test_empty_or_malformed_is_none(self):
        for raw in ("[]", "['a',", "not a list", "'just a string'", "{'a': 1}"):
            with self.subTest(raw=raw):
                self.assertIsNone(MMMUVLMEval._parse_options(raw))

    def test_code_in_options_is_not_executed(self):
        with mock.patch("os.system") as system:
            self.assertIsNone(
                MMMUVLMEval._parse_options("__import__('os').system('true')")
            )
        system.assert_not_called()


if __name__ == "__main__":
    unittest.main()
